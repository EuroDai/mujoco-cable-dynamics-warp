"""Finite-radius native routes and conservative, three-reservoir friction.

The constitutive return mapping follows the C++ stick_slip implementation.
Committed material history changes only in advance(), never in forward().
"""

import warp as wp

from .kernels import State, Topology, command, project_point_force


@wp.func
def packed_point(points: wp.array2d[wp.spatial_vector], world: int, index: int):
    pair = points[world, index // 2]
    return wp.vec3(pair[3 * (index % 2)], pair[3 * (index % 2) + 1], pair[3 * (index % 2) + 2])


@wp.kernel
def native_route(
    topology: Topology, state: State,
    site_xpos: wp.array2d[wp.vec3], geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33], ten_wrapadr: wp.array2d[int],
    ten_wrapnum: wp.array2d[int], wrap_obj: wp.array2d[wp.vec2i],
    wrap_xpos: wp.array2d[wp.spatial_vector],
):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    if p.native_surface == 0:
        return
    cursor = ten_wrapadr[world, p.tendon]
    end = cursor + ten_wrapnum[world, p.tendon]
    angles = wp.vec2(0.0)
    for element in range(5):
        if element % 2 == 0:
            site = topology.sites[cable, element // 2]
            node = 3 * (element // 2)
            if cursor >= end:
                state.values[world, cable, 8] = 2.0
                return
            if wrap_obj[world, cursor // 2][cursor % 2] != -1:
                state.values[world, cable, 8] = 2.0
                return
            state.points[world, cable, node] = packed_point(wrap_xpos, world, cursor)
            state.bodies[world, cable, node] = topology.site_body[site]
            cursor += 1
        else:
            contact = (element - 1) // 2
            geom = topology.native_geoms[cable, contact]
            node = 1 + 3 * contact
            entry = wp.vec3(0.0)
            exit = wp.vec3(0.0)
            if cursor < end and wrap_obj[world, cursor // 2][cursor % 2] == geom:
                if cursor + 1 >= end:
                    state.values[world, cable, 8] = 2.0
                    return
                if wrap_obj[world, (cursor + 1) // 2][(cursor + 1) % 2] != geom:
                    state.values[world, cable, 8] = 2.0
                    return
                entry = packed_point(wrap_xpos, world, cursor)
                exit = packed_point(wrap_xpos, world, cursor + 1)
                cursor += 2
                rotation = geom_xmat[world, geom]
                center = geom_xpos[world, geom]
                a = wp.transpose(rotation) * (entry - center)
                b = wp.transpose(rotation) * (exit - center)
                # Same planar adapter as C++; do not silently apply its material
                # partition to an out-of-plane helix.
                if wp.abs(a[2] - b[2]) > 2.0e-7:
                    state.values[world, cable, 8] = 2.0
                    return
                # acos(dot/r^2) amplifies float32 radius/position error near
                # lift-off into a fictitious finite arc and material transfer.
                # atan2 of cross and dot remains well conditioned as theta -> 0.
                ax, ay = wp.float64(a[0]), wp.float64(a[1])
                bx, by = wp.float64(b[0]), wp.float64(b[1])
                angles[contact] = float(wp.atan2(wp.abs(ax * by - ay * bx), ax * bx + ay * by))
            else:
                # Lift-off keeps the same material reservoirs. A virtual point
                # splits the straight span and transmits zero obstacle force.
                a = state.points[world, cable, node - 1]
                b = site_xpos[world, topology.sites[cable, contact + 1]]
                delta = b - a
                fraction = wp.clamp(wp.dot(geom_xpos[world, geom] - a, delta) / wp.max(wp.dot(delta, delta), 1.0e-20), 0.001, 0.999)
                entry = a + fraction * delta
                exit = entry
            state.points[world, cable, node] = entry
            state.points[world, cable, node + 1] = exit
            state.bodies[world, cable, node] = topology.geom_body[geom]
            state.bodies[world, cable, node + 1] = topology.geom_body[geom]
    if cursor != end:
        state.values[world, cable, 8] = 2.0
        return
    lengths = wp.vec3(0.0)
    for segment in range(6):
        delta = state.points[world, cable, segment + 1] - state.points[world, cable, segment]
        distance = wp.length(delta)
        state.directions[world, cable, segment] = wp.normalize(delta)
        if segment == 1 or segment == 4:
            contact = segment // 3
            distance = topology.geom_radius[topology.native_geoms[cable, contact]] * angles[contact]
            lengths[contact] += 0.5 * distance
            lengths[contact + 1] += 0.5 * distance
        elif segment == 0:
            lengths[0] += distance
        elif segment == 5:
            lengths[2] += distance
        else:
            lengths[1] += distance
    if wp.abs(lengths[0] + lengths[1] + lengths[2] - state.values[world, cable, 0]) > 2.0e-6:
        state.values[world, cable, 8] = 2.0
        return
    state.count[world, cable] = 7
    state.material_lengths[world, cable] = lengths
    state.contact_angles[world, cable] = angles
    if state.initialized[world, cable] == 0:
        state.home[world, cable] = p.home_length
        if p.home_length < 0.0:
            state.home[world, cable] = state.values[world, cable, 0]
        state.initialized[world, cable] = 1


@wp.kernel
def solve_material(
    topology: Topology, state: State, ctrl: wp.array2d[float],
    qpos: wp.array2d[float], timestep: float,
):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    state.pending_valid[world, cable] = 0
    if p.friction_direction == 3:
        alpha = float(1.0)
        if p.control_timeconstant > 0.0:
            alpha = 1.0 - wp.exp(-timestep / p.control_timeconstant)
        speed = alpha * (command(p, world, ctrl, qpos) - state.filtered[world, cable]) / timestep
        if p.max_contraction_rate > 0.0:
            speed = wp.clamp(speed, -p.max_contraction_rate, p.max_contraction_rate)
        state.contact_angles[world, cable] = wp.vec2(speed, 0.0)
    if p.friction_direction != 4 or state.values[world, cable, 8] == 2.0:
        return
    lengths = state.material_lengths[world, cable]
    angles = state.contact_angles[world, cable]
    if p.native_surface == 0:
        for i in range(3):
            lengths[i] = wp.length(state.points[world, cable, i + 1] - state.points[world, cable, i])
        for i in range(2):
            angles[i] = wp.acos(wp.clamp(wp.dot(state.directions[world, cable, i], state.directions[world, cable, i + 1]), -1.0, 1.0))
        state.material_lengths[world, cable] = lengths
        state.contact_angles[world, cable] = angles
    free_length = wp.float64(state.values[world, cable, 2]) + wp.float64(p.slack)
    if free_length <= wp.float64(0.0):
        state.values[world, cable, 8] = 2.0
        state.values[world, cable, 5] = 0.0
        state.values[world, cable, 6] = 0.0
        return
    ea = wp.float64(p.stiffness) * wp.float64(state.home[world, cable])
    k = wp.vec3d(0.0)
    rest = wp.vec3d(0.0)
    total_length = wp.float64(lengths[0]) + wp.float64(lengths[1]) + wp.float64(lengths[2])
    if total_length <= wp.float64(1.0e-12):
        state.values[world, cable, 8] = 2.0
        state.values[world, cable, 5] = 0.0
        state.values[world, cable, 6] = 0.0
        return
    for i in range(3):
        k[i] = ea / wp.max(wp.float64(lengths[i]), wp.float64(1.0e-5))
        if state.material_initialized[world, cable] != 0:
            rest[i] = state.material[world, cable][i]
        else:
            rest[i] = free_length * wp.float64(lengths[i]) / total_length
    if state.material_initialized[world, cable] == 0:
        rest[2] = free_length - rest[0] - rest[1]
    if state.material_initialized[world, cable] != 0:
        rest[2] += free_length - state.material[world, cable][3]
    ratios = wp.vec2d(0.0)
    for i in range(2):
        ratios[i] = wp.exp(wp.min(wp.float64(topology.contact_mu[cable, i]) * wp.float64(angles[i]), wp.float64(20.0)))
    for sweep in range(80):
        largest = wp.float64(0.0)
        for i in range(2):
            left = k[i] * (wp.float64(lengths[i]) - rest[i])
            right = k[i + 1] * (wp.float64(lengths[i + 1]) - rest[i + 1])
            tl = wp.max(wp.float64(0.0), left)
            tr = wp.max(wp.float64(0.0), right)
            delta = wp.float64(0.0)
            if tr > ratios[i] * tl + wp.float64(1.0e-10):
                delta = (right - ratios[i] * left) / (k[i + 1] + ratios[i] * k[i])
            elif tl > ratios[i] * tr + wp.float64(1.0e-10):
                delta = (ratios[i] * right - left) / (k[i] + ratios[i] * k[i + 1])
            rest[i] -= delta
            rest[i + 1] += delta
            largest = wp.max(largest, wp.abs(delta))
        if largest < wp.float64(1.0e-13):
            break
    tension = wp.vec3(0.0)
    for i in range(3):
        tension[i] = float(wp.max(wp.float64(0.0), k[i] * (wp.float64(lengths[i]) - rest[i])))
    state.pending_material[world, cable] = wp.vec4d(rest[0], rest[1], rest[2], free_length)
    state.pending_valid[world, cable] = 1
    state.values[world, cable, 5] = tension[0]
    state.values[world, cable, 6] = float(wp.max(tension[0], wp.max(tension[1], tension[2])) > 0.0)
    for i in range(2):
        log_ratio = float(0.0)
        if tension[i] > 1.0e-12 and tension[i + 1] > 1.0e-12:
            log_ratio = wp.log(tension[i + 1] / tension[i])
        state.capstan[world, cable, i] = wp.vec4(angles[i], tension[i], tension[i + 1], log_ratio)
    if p.native_surface != 0:
        state.tensions[world, cable, 0] = tension[0]
        state.tensions[world, cable, 1] = 0.0
        state.tensions[world, cable, 2] = tension[1]
        state.tensions[world, cable, 3] = tension[1]
        state.tensions[world, cable, 4] = 0.0
        state.tensions[world, cable, 5] = tension[2]
    else:
        for i in range(3):
            state.tensions[world, cable, i] = tension[i]


@wp.kernel
def apply_material_forces(
    topology: Topology, state: State, com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector], nv: int,
):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    if p.friction_direction != 4 or state.values[world, cable, 8] == 2.0:
        return
    count = state.count[world, cable]
    for node in range(count):
        force = wp.vec3(0.0)
        if node > 0:
            force -= state.tensions[world, cable, node - 1] * state.directions[world, cable, node - 1]
        if node + 1 < count:
            force += state.tensions[world, cable, node] * state.directions[world, cable, node]
        body = state.bodies[world, cable, node]
        state.body_forces[world, cable, body] += force
        project_point_force(topology, state, world, body, state.points[world, cable, node], force, com, cdof, nv)


@wp.kernel
def sensors(state: State, bindings: wp.array2d[int], sensordata: wp.array2d[float]):
    world, binding = wp.tid()
    cable = bindings[binding, 0]
    address = bindings[binding, 1]
    dimension = bindings[binding, 2]
    kind = bindings[binding, 3]
    for i in range(dimension):
        value = float(0.0)
        if kind == 0:
            value = state.values[world, cable, i]
        elif kind == 1:
            value = state.body_forces[world, cable, i // 3][i % 3]
        else:
            value = state.capstan[world, cable, i // 4][i % 4]
        sensordata[world, address + i] = value
