"""Load-dependent hinge friction using cable body-force sensors.

Each world retains its own previous solved bearing load. The native MJWarp
constraint solver supplies the friction torque, including sticking at rest.
"""

import math

import mujoco
import numpy as np
import warp as wp


DEFAULTS = {
    "radial_gain": 0.0, "axial_gain": 0.0, "static_extra": 0.0,
    "stribeck_velocity": 0.02, "radial_peak_gain": 0.0,
    "radial_peak_angle": 0.7, "radial_peak_width": 0.2,
}


def validate_hinges(model, hinges, cables):
    from .model import UnsupportedCableError

    used = set()
    body_sensors = {c.config.get("body_force_sensor") for c in cables if c.config.get("route_mode") == "surface"}
    for c in hinges:
        unknown = set(c) - set(DEFAULTS) - {"name", "joint", "body_force_sensors"}
        if unknown:
            raise UnsupportedCableError(f"{c['name']}: unknown hinge options: {sorted(unknown)}")
        for name, default in DEFAULTS.items():
            value = float(c.get(name, default))
            if not math.isfinite(value) or value < 0 or (name in ("stribeck_velocity", "radial_peak_width") and value == 0):
                raise ValueError(f"{c['name']}: invalid {name}")
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, c.get("joint", ""))
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{c['name']}: joint must name a hinge")
        if joint in used or model.body_jntnum[model.jnt_bodyid[joint]] != 1:
            raise ValueError(f"{c['name']}: use one hinge instance per single-joint body")
        used.add(joint)
        if any(name not in body_sensors for name in c.get("body_force_sensors", "").split()):
            raise UnsupportedCableError(f"{c['name']}: body_force_sensors must name converted cable body-force sensors")
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR:
        raise UnsupportedCableError("Hinge load history requires sensors enabled")


@wp.struct
class Parameters:
    joint: int
    body: int
    dof: int
    qpos: int
    base: float
    radial_gain: float
    axial_gain: float
    static_extra: float
    stribeck_velocity: float
    radial_peak_gain: float
    radial_peak_angle: float
    radial_peak_width: float


@wp.kernel
def limits(
    parameters: wp.array[Parameters], loads: wp.array2d[wp.vec2],
    qpos: wp.array2d[float], qvel: wp.array2d[float],
    nefc: wp.array[int], efc_type: wp.array2d[int], efc_id: wp.array2d[int],
    friction: wp.array2d[float],
):
    world, hinge = wp.tid()
    p = parameters[hinge]
    load = loads[world, hinge]
    coordinate = (qpos[world, p.qpos] - p.radial_peak_angle) / p.radial_peak_width
    gain = p.radial_gain + p.radial_peak_gain * wp.exp(-0.5 * coordinate * coordinate)
    speed = qvel[world, p.dof] / p.stribeck_velocity
    limit = p.base + gain * load[0] + p.axial_gain * load[1] + p.static_extra * wp.exp(-speed * speed)
    for row in range(wp.min(nefc[world], friction.shape[1])):
        if efc_type[world, row] == 1 and efc_id[world, row] == p.dof:
            friction[world, row] = limit


@wp.kernel
def bearing_loads(
    parameters: wp.array[Parameters], descendants: wp.array2d[int],
    cable_ids: wp.array[int], body_forces: wp.array3d[wp.vec3],
    cfrc_int: wp.array2d[wp.spatial_vector], xaxis: wp.array2d[wp.vec3],
    pending: wp.array2d[wp.vec2],
):
    world, hinge = wp.tid()
    p = parameters[hinge]
    if p.radial_gain == 0.0 and p.axial_gain == 0.0 and p.static_extra == 0.0 and p.radial_peak_gain == 0.0:
        pending[world, hinge] = wp.vec2(0.0)
        return
    force = wp.spatial_bottom(cfrc_int[world, p.body])
    for body in range(descendants.shape[1]):
        if descendants[hinge, body] != 0:
            for i in range(cable_ids.shape[0]):
                force -= body_forces[world, cable_ids[i], body]
    axis = xaxis[world, p.joint]
    axial = wp.dot(force, axis)
    pending[world, hinge] = wp.vec2(wp.length(force - axial * axis), wp.abs(axial))


@wp.kernel
def commit(pending: wp.array2d[wp.vec2], loads: wp.array2d[wp.vec2]):
    world, hinge = wp.tid()
    loads[world, hinge] = pending[world, hinge]


@wp.kernel
def reset(loads: wp.array2d[wp.vec2], pending: wp.array2d[wp.vec2], mask: wp.array[bool]):
    world, hinge = wp.tid()
    if mask[world]:
        loads[world, hinge] = wp.vec2(0.0)
        pending[world, hinge] = wp.vec2(0.0)


class HingeFriction:
    def __init__(self, loaded, model, nworld):
        m = loaded.model
        parameters, descendants = [], np.zeros((len(loaded.hinges), m.nbody), dtype=np.int32)
        sensors = set()
        # Keep a constraint row available even when the base bound is zero.
        # Its bound is replaced per world in the passive stage before solving.
        friction = model.dof_frictionloss.numpy().copy()
        for i, c in enumerate(loaded.hinges):
            p = Parameters()
            p.joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, c["joint"])
            p.body = int(m.jnt_bodyid[p.joint])
            p.dof = int(m.jnt_dofadr[p.joint])
            p.qpos = int(m.jnt_qposadr[p.joint])
            p.base = float(m.dof_frictionloss[p.dof])
            for name, default in DEFAULTS.items():
                setattr(p, name, float(c.get(name, default)))
            friction[0, p.dof] = max(p.base, 1.0e-12)
            parameters.append(p)
            sensors.update(c.get("body_force_sensors", "").split())
            for body in range(m.nbody):
                ancestor = body
                while ancestor:
                    if ancestor == p.body:
                        descendants[i, body] = 1
                        break
                    ancestor = int(m.body_parentid[ancestor])
        model.dof_frictionloss.assign(friction)
        model.opt.run_rne_postconstraint = True
        self.parameters = wp.array(parameters, dtype=Parameters)
        self.descendants = wp.array(descendants, dtype=int)
        self.cable_ids = wp.array([i for i, c in enumerate(loaded.cables) if c.config.get("body_force_sensor") in sensors], dtype=int)
        self.shape = (nworld, len(parameters))
        self.loads = wp.zeros(self.shape, dtype=wp.vec2)
        self.pending = wp.zeros(self.shape, dtype=wp.vec2)

    def update_limits(self, data):
        wp.launch(limits, self.shape, inputs=[self.parameters, self.loads, data.qpos, data.qvel, data.nefc, data.efc.type, data.efc.id, data.efc.frictionloss])

    def measure(self, data, state):
        wp.launch(bearing_loads, self.shape, inputs=[self.parameters, self.descendants, self.cable_ids, state.body_forces, data.cfrc_int, data.xaxis, self.pending])

    def advance(self):
        wp.launch(commit, self.shape, inputs=[self.pending, self.loads])

    def reset(self, mask):
        wp.launch(reset, self.shape, inputs=[self.loads, self.pending, mask])

    @staticmethod
    def enabled(model):
        return not (model.opt.disableflags & (int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)))
