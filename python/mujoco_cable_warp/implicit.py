"""Linearly implicit elasticity with frozen routing and material slip state.

For extension gradients G and active elastic stiffnesses k, add h^2 G' k G
to the acceleration metric and -h G' k G v to the force. This is a single
backward-Euler linearization, not a nonlinear implicit capstan solve. The
current-step return mapping, unilateral branch and geometry remain explicit.
"""

import mujoco
import numpy as np
import warp as wp

from .kernels import State, Topology
from .model import UnsupportedCableError


@wp.func
def point_jacobian(
    topology: Topology, world: int, body: int, dof: int,
    point: wp.vec3, com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
):
    result = wp.vec3(0.0)
    if body > 0 and topology.body_dofs[body, dof] != 0:
        root = topology.body_root[topology.dof_body[dof]]
        motion = cdof[world, dof]
        result = wp.spatial_bottom(motion) + wp.cross(
            wp.spatial_top(motion), point - com[world, root]
        )
    return result


@wp.kernel
def linearize(
    topology: Topology, state: State, enabled: wp.array[int],
    com: wp.array2d[wp.vec3], cdof: wp.array2d[wp.spatial_vector],
    ten_j: wp.array2d[float], ten_adr: wp.array[int],
    ten_nnz: wp.array[int], ten_col: wp.array[int],
    gradients: wp.array4d[float], stiffness: wp.array3d[float],
):
    world, cable, region, dof = wp.tid()
    p = topology.parameters[cable]
    gradient = float(0.0)
    k = float(0.0)
    if enabled[cable] != 0 and state.values[world, cable, 8] != 2.0:
        if p.friction_direction == 4:
            segment = region
            if p.native_surface != 0:
                segment = 3 * region
                if region == 1:
                    segment = 2
                elif region == 2:
                    segment = 5
            tension = state.tensions[world, cable, segment]
            if tension > 0.0:
                k = p.stiffness * state.home[world, cable] / wp.max(
                    state.material_lengths[world, cable][region], 1.0e-5
                )
                for edge in range(state.count[world, cable] - 1):
                    use = edge == region
                    if p.native_surface != 0:
                        use = ((region == 0 and edge == 0)
                               or (region == 1 and (edge == 2 or edge == 3))
                               or (region == 2 and edge == 5))
                    if use:
                        start = point_jacobian(
                            topology, world, state.bodies[world, cable, edge],
                            dof, state.points[world, cable, edge], com, cdof,
                        )
                        end = point_jacobian(
                            topology, world, state.bodies[world, cable, edge + 1],
                            dof, state.points[world, cable, edge + 1], com, cdof,
                        )
                        gradient += wp.dot(
                            end - start, state.directions[world, cable, edge]
                        )
        elif region == 0 and state.values[world, cable, 5] > 0.0:
            tension = state.values[world, cable, 5]
            if (tension < p.max_tension and -tension > p.force_low
                    and -tension < p.force_high):
                k = p.stiffness
                extension = state.values[world, cable, 4]
                if p.taut_transition > 0.0:
                    k *= wp.clamp(extension / p.taut_transition, 0.0, 1.0)
                for offset in range(ten_nnz[p.tendon]):
                    index = ten_adr[p.tendon] + offset
                    if ten_col[index] == dof:
                        gradient = ten_j[world, index]
    gradients[world, cable, region, dof] = gradient
    if dof == 0:
        stiffness[world, cable, region] = k


@wp.kernel
def metric_update(
    gradients: wp.array4d[float], stiffness: wp.array3d[float],
    rows: wp.array[int], cols: wp.array[int], timestep: float, ncable: int,
    mass: wp.array2d[float],
):
    world, entry = wp.tid()
    row, col = rows[entry], cols[entry]
    value = float(0.0)
    for cable in range(ncable):
        for region in range(3):
            value += (stiffness[world, cable, region]
                      * gradients[world, cable, region, row]
                      * gradients[world, cable, region, col])
    mass[world, entry] += timestep * timestep * value


@wp.kernel
def rhs_update(
    gradients: wp.array4d[float], stiffness: wp.array3d[float],
    velocity: wp.array2d[float], timestep: float, ncable: int, nv: int,
    passive: wp.array2d[float],
):
    world, dof = wp.tid()
    correction = float(0.0)
    for cable in range(ncable):
        for region in range(3):
            speed = float(0.0)
            for other in range(nv):
                speed += (gradients[world, cable, region, other]
                          * velocity[world, other])
            correction += (stiffness[world, cable, region]
                           * gradients[world, cable, region, dof] * speed)
    passive[world, dof] -= timestep * correction


class ImplicitElastic:
    """Optional elastic tangent, using MJWarp's existing sparse mass layout.

    The physical parameters and cable force/sensor diagnostics are unchanged.
    Only step() uses the tangent; forward() remains a physical force query.
    Cross-branch cable couplings outside the mass sparsity are rejected.
    """

    def __init__(self, backend):
        self.active = False
        self.enabled = np.array([
            c.config.get("integration_mode", "explicit") == "implicit_elastic"
            for c in backend.loaded.cables
        ], dtype=np.int32)
        self.any_enabled = bool(self.enabled.any())
        if not self.any_enabled:
            return
        if backend.cpu_model.opt.integrator not in (
            mujoco.mjtIntegrator.mjINT_EULER,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ):
            raise UnsupportedCableError(
                "implicit_elastic requires Euler or implicitfast"
            )
        model = backend.model
        required = ("M_rowadr", "M_rownnz", "M_colind", "ten_J_rowadr",
                    "ten_J_rownnz", "ten_J_colind")
        if any(not hasattr(model, key) for key in required):
            raise UnsupportedCableError(
                "implicit_elastic requires MJWarp CSR mass/tendon layouts"
            )
        cols = model.M_colind.numpy()
        rows = np.repeat(np.arange(backend.cpu_model.nv),
                         model.M_rownnz.numpy()).astype(np.int32)
        present = set(zip(rows.tolist(), cols.tolist()))
        body_dofs = backend.topology.body_dofs.numpy()
        for enabled, cable in zip(self.enabled, backend.loaded.cables):
            if not enabled:
                continue
            bodies = [int(backend.cpu_model.site_bodyid[s])
                      for s in cable.site_ids]
            bodies += [int(backend.cpu_model.geom_bodyid[g])
                       for g in cable.native_geom_ids]
            active = np.flatnonzero(np.any(body_dofs[bodies], axis=0))
            for i in active:
                for j in active:
                    if i >= j and (int(i), int(j)) not in present:
                        raise UnsupportedCableError(
                            "implicit_elastic cable crosses unsupported "
                            "mass-matrix branches or free/ball joint blocks"
                        )
        self.rows = wp.array(rows, dtype=int)
        self.cols = wp.array(cols, dtype=int)
        self.mask = wp.array(self.enabled, dtype=int)
        self.gradients = wp.zeros(
            (backend.nworld, backend.ncable, 3, backend.cpu_model.nv),
            dtype=float,
        )
        self.stiffness = wp.zeros((backend.nworld, backend.ncable, 3), dtype=float)

    def apply(self, backend, model, data):
        if not self.any_enabled or not self.active:
            return
        nv = backend.cpu_model.nv
        wp.launch(
            linearize, (backend.nworld, backend.ncable, 3, nv),
            inputs=[backend.topology, backend.state, self.mask,
                    data.subtree_com, data.cdof, data.ten_J,
                    model.ten_J_rowadr, model.ten_J_rownnz, model.ten_J_colind,
                    self.gradients, self.stiffness], device=backend.device,
        )
        wp.launch(
            metric_update, data.M.shape,
            inputs=[self.gradients, self.stiffness, self.rows, self.cols,
                    backend.timestep, backend.ncable, data.M],
            device=backend.device,
        )
        wp.launch(
            rhs_update, (backend.nworld, nv),
            inputs=[self.gradients, self.stiffness, data.qvel,
                    backend.timestep, backend.ncable, nv, data.qfrc_passive],
            device=backend.device,
        )
