"""MJWarp engine integration and per-world cable lifecycle."""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from . import kernels as k
from .model import CONTROL_MODES, FLOAT_DEFAULTS, CableModel, load_model


class CableSimulation:
    """A batch of independent worlds with identical cable topology.

    ``data.ctrl``, ``data.qpos`` and ``state.values`` are device arrays. The
    latter stores the existing twelve cable diagnostics (native routes use
    the first eight). No compiled C++ cable library is needed.
    """

    def __init__(
        self, source: str | Path | CableModel, *, nworld: int = 1,
        device: str | None = None, nconmax: int = 64, njmax: int = 256,
    ):
        if nworld < 1:
            raise ValueError("nworld must be positive")
        if nconmax < 1 or njmax < 1:
            raise ValueError("contact and constraint capacities must be positive")
        self.loaded = source if isinstance(source, CableModel) else load_model(source)
        self.cpu_model = self.loaded.model
        self.nworld = nworld
        self.ncable = len(self.loaded.cables)
        self.device = wp.get_device(device)
        self.timestep = float(self.cpu_model.opt.timestep)
        self._graph = None
        self._visual_points = None
        self._visual_counts = None
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(self.cpu_model, nworld=nworld, nconmax=nconmax, njmax=njmax)
            if not hasattr(self.model, "callback"):
                raise RuntimeError("This backend requires MJWarp with Model.callback support; install the documented warp extra")
            self._allocate()
            if self.ncable:
                self.model.callback.passive = self._passive
                self.model.callback.act_gain = self._actuator
                self.model.callback.sensor = self._sensor
            self.forward()

    def _allocate(self):
        m = self.cpu_model
        parameters = []
        nsite = max([len(c.site_ids) for c in self.loaded.cables] + [1])
        sites = np.full((self.ncable, nsite), -1, dtype=np.int32)
        roles = np.zeros_like(sites)
        bindings = []
        for index, spec in enumerate(self.loaded.cables):
            c = spec.config
            p = k.Parameters()
            for name, default in FLOAT_DEFAULTS.items():
                if name != "visual_width":
                    setattr(p, name, float(c.get(name, default)))
            p.surface = int(c.get("route_mode", "native") == "surface")
            p.tendon = spec.tendon_id
            p.actuator = spec.actuator_id
            p.geom = spec.geom_id
            p.analytic = int(p.surface and p.geom >= 0 and p.capstan_mu == 0.0)
            p.geom_body = int(m.geom_bodyid[p.geom]) if p.geom >= 0 else 0
            p.nsite = len(spec.site_ids)
            p.mode = CONTROL_MODES[c.get("ctrl_mode", "target_contraction")]
            p.joint_qpos = -1
            p.joint_dof = -1
            p.joint_zero = 0.0
            if p.mode == 4:
                joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, c["spool_joint"])
                p.joint_qpos = int(m.jnt_qposadr[joint])
                p.joint_dof = int(m.jnt_dofadr[joint])
                p.joint_zero = float(m.qpos0[p.joint_qpos])
            p.reaction = int(c.get("spool_reaction_torque", "false") == "true")
            p.reserve_sign = -1.0 if c.get("spool_reserve_direction") == "negative" else 1.0
            p.friction_direction = {"forward": 0, "reverse": 1, "velocity": 2}[c.get("capstan_direction", "forward")]
            p.ctrl_low, p.ctrl_high = -1.0e30, 1.0e30
            if p.actuator >= 0 and m.actuator_ctrllimited[p.actuator]:
                p.ctrl_low, p.ctrl_high = map(float, m.actuator_ctrlrange[p.actuator])
            p.force_low, p.force_high = -1.0e30, 1.0e30
            if p.actuator >= 0 and m.actuator_forcelimited[p.actuator]:
                p.force_low, p.force_high = map(float, m.actuator_forcerange[p.actuator])
            home = c.get("home_length", "auto_initial")
            # Native mode uses model reference length, even after pose edits.
            p.home_length = float(m.tendon_length0[p.tendon]) if home == "auto_initial" and not p.surface else (-1.0 if home == "auto_initial" else float(home))
            if p.geom >= 0:
                p.radius, p.half_length = map(float, m.geom_size[p.geom, :2])
            parameters.append(p)
            sites[index, :p.nsite] = spec.site_ids
            if p.nsite:
                roles[index, :p.nsite] = np.rint(m.site_user[spec.site_ids, int(c.get("site_role_user_index", 0))]).astype(int)
            for sensor in spec.sensor_ids:
                bindings.append((index, int(m.sensor_adr[sensor]), int(m.sensor_dim[sensor])))
        body_dofs = np.zeros((m.nbody, m.nv), dtype=np.int32)
        for body in range(1, m.nbody):
            ancestor = body
            while ancestor:
                body_dofs[body, m.dof_bodyid == ancestor] = 1
                ancestor = int(m.body_parentid[ancestor])
        self.topology = k.Topology()
        self.topology.parameters = wp.array(parameters, dtype=k.Parameters)
        self.topology.sites = wp.array(sites, dtype=int)
        self.topology.roles = wp.array(roles, dtype=int)
        self.topology.site_body = wp.array(m.site_bodyid, dtype=int)
        self.topology.body_root = wp.array(m.body_rootid, dtype=int)
        self.topology.dof_body = wp.array(m.dof_bodyid, dtype=int)
        self.topology.body_dofs = wp.array(body_dofs, dtype=int)
        self.bindings = wp.array(np.asarray(bindings, dtype=np.int32).reshape(-1, 3), dtype=int)
        self.state = k.State()
        shape = (self.nworld, self.ncable)
        self.state.values = wp.zeros((*shape, 12), dtype=float)
        for name in ("integrated", "filtered", "home"):
            setattr(self.state, name, wp.zeros(shape, dtype=float))
        for name in ("latch", "initialized", "branch", "count"):
            setattr(self.state, name, wp.zeros(shape, dtype=int))
        capacity = max([4 if p.analytic else (98 if p.geom >= 0 else p.nsite)
                        for p in parameters] + [1])
        points = (*shape, capacity)
        for name in ("points", "directions", "velocities"):
            setattr(self.state, name, wp.zeros(points, dtype=wp.vec3))
        for name in ("bodies", "roles", "wrap"):
            setattr(self.state, name, wp.zeros(points, dtype=int))
        self.state.tensions = wp.zeros(points, dtype=float)
        self.state.force = wp.zeros((self.nworld, m.nv), dtype=float)
        self.state.arc = wp.zeros(shape, dtype=wp.vec4)

    def _passive(self, model, data):
        shape = (self.nworld, self.ncable)
        wp.launch(k.route, shape, inputs=[
            self.topology, self.state, data.site_xpos, data.geom_xpos, data.geom_xmat,
            data.subtree_com, data.cvel, data.ten_length, data.ten_velocity,
        ], device=self.device, block_dim=32)
        wp.launch(k.evaluate, shape, inputs=[self.topology, self.state, data.ctrl, data.qpos], device=self.device)
        self.state.force.zero_()
        wp.launch(k.apply_forces, shape, inputs=[
            self.topology, self.state, data.qpos, data.subtree_com, data.cdof, self.cpu_model.nv,
        ], device=self.device, block_dim=32)
        wp.launch(k.add_forces, (self.nworld, self.cpu_model.nv), inputs=[self.state.force, data.qfrc_passive], device=self.device)

    def _actuator(self, model, data):
        wp.launch(k.actuator_forces, (self.nworld, self.ncable), inputs=[self.topology, self.state, data.actuator_force], device=self.device)

    def _sensor(self, model, data, stage):
        if int(stage) == int(mujoco.mjtStage.mjSTAGE_VEL) and self.bindings.shape[0]:
            wp.launch(k.sensors, (self.nworld, self.bindings.shape[0]), inputs=[self.state, self.bindings, data.sensordata], device=self.device)

    def forward(self):
        """Refresh geometry, forces and diagnostics without advancing time."""
        with wp.ScopedDevice(self.device):
            mjw.forward(self.model, self.data)

    def _step(self):
        mjw.step(self.model, self.data)
        if self.ncable:
            wp.launch(k.advance, (self.nworld, self.ncable), inputs=[
                self.topology, self.state, self.data.ctrl, self.data.qpos, self.timestep,
            ], device=self.device)

    def step(self, count: int = 1):
        """Advance physics and then cable control history exactly once per step."""
        if count < 0:
            raise ValueError("step count must be nonnegative")
        with wp.ScopedDevice(self.device):
            for _ in range(count):
                if self._graph is None:
                    self._step()
                else:
                    wp.capture_launch(self._graph)

    def capture_step(self):
        """Capture one step on CUDA after at least one warm-up step.

        The graph reads current control arrays on each replay. Device arrays
        must not be replaced after capture; update their contents instead.
        """
        if not self.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA device")
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._step()
        self._graph = capture.graph

    def reset(self, worlds=None):
        """Reset all worlds, or selected world indices, including cable history."""
        mask = np.zeros(self.nworld, dtype=bool)
        if worlds is None:
            mask[:] = True
        else:
            indices = np.asarray(worlds, dtype=int).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= self.nworld):
                raise IndexError("reset world index out of range")
            mask[indices] = True
        with wp.ScopedDevice(self.device):
            active = wp.array(mask, dtype=bool)
            mjw.reset_data(self.model, self.data, reset=active)
            if self.ncable:
                wp.launch(k.reset_cables, (self.nworld, self.ncable), inputs=[self.state, active], device=self.device)
            self.forward()

    def set_control(self, controls):
        """Upload a control vector or a (nworld, nu) batch. For training, write data.ctrl directly."""
        controls = np.asarray(controls, dtype=np.float32)
        if controls.shape == (self.cpu_model.nu,):
            controls = np.broadcast_to(controls, (self.nworld, self.cpu_model.nu))
        if controls.shape != (self.nworld, self.cpu_model.nu) or not np.all(np.isfinite(controls)):
            raise ValueError(f"controls must be finite with shape ({self.nworld}, {self.cpu_model.nu})")
        self.data.ctrl.assign(np.ascontiguousarray(controls))

    def get_data(self, world: int = 0, result: mujoco.MjData | None = None):
        """Copy one world to CPU for inspection or rendering."""
        if not 0 <= world < self.nworld:
            raise IndexError("world index out of range")
        if result is None:
            result = mujoco.MjData(self.cpu_model)
        mjw.get_data_into(result, self.cpu_model, self.data, world_id=world)
        return result

    def get_route_points(self, world: int = 0):
        """Sample display routes for one world, returning CPU points/counts.

        Analytic cylinders allocate only four physical force application points.
        Dense display samples are generated on demand into separate buffers;
        this method never changes physics, control history or a captured graph.
        Geometry describes the most recent force evaluation, like sensors.
        """
        if not 0 <= world < self.nworld:
            raise IndexError("world index out of range")
        with wp.ScopedDevice(self.device):
            if self._visual_points is None:
                capacity = max(98, self.state.points.shape[2])
                self._visual_points = wp.zeros((self.ncable, capacity), dtype=wp.vec3)
                self._visual_counts = wp.zeros(self.ncable, dtype=int)
            if self.ncable:
                wp.launch(k.sample_visual_routes, self._visual_points.shape, inputs=[
                    self.topology, self.state, self.data.geom_xpos, self.data.geom_xmat,
                    world, self._visual_points, self._visual_counts,
                ], device=self.device)
            return self._visual_points.numpy().copy(), self._visual_counts.numpy().copy()

    def check(self):
        """Explicit host synchronization for numerical/route/capacity checks."""
        if not np.all(np.isfinite(self.data.qpos.numpy())) or not np.all(np.isfinite(self.data.qvel.numpy())):
            raise RuntimeError("Non-finite MJWarp state")
        values = self.state.values.numpy()
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Non-finite cable state")
        invalid = np.argwhere(values[:, :, 8] == 2)
        if invalid.size:
            world, cable = invalid[0]
            raise RuntimeError(f"Invalid surface route in world {world}, cable {self.loaded.cables[cable].name}; cable force was disabled")
        if hasattr(self.data, "overflow") and np.any(self.data.overflow.numpy()):
            raise RuntimeError("MJWarp buffer overflow; increase nconmax/njmax")
