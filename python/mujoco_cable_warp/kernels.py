"""Device-side routing, unilateral constitutive law, friction and force mapping.

Kernels run unchanged on Warp CPU and CUDA devices. State is indexed by world
and cable; there is no host-side geometry or force evaluation in the step loop.
"""

import warp as wp


@wp.struct
class Parameters:
    surface: int
    native_surface: int
    analytic: int
    tendon: int
    actuator: int
    geom: int
    geom_body: int
    nsite: int
    mode: int
    joint_qpos: int
    joint_dof: int
    joint_zero: float
    reaction: int
    stiffness: float
    damping: float
    slack: float
    pretension_offset: float
    max_tension: float
    spool_radius: float
    spool_reserve_length: float
    reserve_sign: float
    control_timeconstant: float
    max_contraction_rate: float
    taut_transition: float
    taut_hysteresis: float
    capstan_mu: float
    guide_friction_mu: float
    capstan_velocity_scale: float
    friction_direction: int
    ctrl_low: float
    ctrl_high: float
    force_low: float
    force_high: float
    home_length: float
    radius: float
    half_length: float


@wp.struct
class Topology:
    parameters: wp.array[Parameters]
    sites: wp.array2d[int]
    roles: wp.array2d[int]
    site_body: wp.array[int]
    body_root: wp.array[int]
    dof_body: wp.array[int]
    body_dofs: wp.array2d[int]
    native_geoms: wp.array2d[int]
    geom_body: wp.array[int]
    geom_radius: wp.array[float]
    guide_contact: wp.array2d[int]
    contact_mu: wp.array2d[float]


@wp.struct
class State:
    values: wp.array3d[float]
    integrated: wp.array2d[float]
    filtered: wp.array2d[float]
    latch: wp.array2d[int]
    home: wp.array2d[float]
    initialized: wp.array2d[int]
    branch: wp.array2d[int]
    count: wp.array2d[int]
    points: wp.array3d[wp.vec3]
    bodies: wp.array3d[int]
    roles: wp.array3d[int]
    wrap: wp.array3d[int]
    directions: wp.array3d[wp.vec3]
    velocities: wp.array3d[wp.vec3]
    tensions: wp.array3d[float]
    force: wp.array2d[float]
    arc: wp.array2d[wp.vec4]
    body_forces: wp.array3d[wp.vec3]
    capstan: wp.array3d[wp.vec4]
    material: wp.array2d[wp.vec4d]
    pending_material: wp.array2d[wp.vec4d]
    material_initialized: wp.array2d[int]
    pending_valid: wp.array2d[int]
    material_lengths: wp.array2d[wp.vec3]
    contact_angles: wp.array2d[wp.vec2]


@wp.func
def positive_angle(value: float):
    tau = 6.283185307179586
    return value - wp.floor(value / tau) * tau


@wp.func
def point_velocity(
    world: int, body: int, point: wp.vec3, roots: wp.array[int],
    com: wp.array2d[wp.vec3], cvel: wp.array2d[wp.spatial_vector],
):
    velocity = wp.vec3(0.0)
    if body > 0:
        spatial = cvel[world, body]
        velocity = wp.spatial_bottom(spatial) + wp.cross(
            wp.spatial_top(spatial), point - com[world, roots[body]]
        )
    return velocity


@wp.kernel
def route(
    topology: Topology, state: State,
    site_xpos: wp.array2d[wp.vec3], geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33], com: wp.array2d[wp.vec3],
    cvel: wp.array2d[wp.spatial_vector],
    ten_length: wp.array2d[float], ten_velocity: wp.array2d[float],
):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    for i in range(12):
        state.values[world, cable, i] = 0.0
    state.count[world, cable] = 0
    if p.surface == 0 or p.native_surface != 0:
        state.values[world, cable, 0] = ten_length[world, p.tendon]
        state.values[world, cable, 1] = ten_velocity[world, p.tendon]
    else:
        if p.geom < 0:
            state.count[world, cable] = p.nsite
            for i in range(p.nsite):
                site = topology.sites[cable, i]
                state.points[world, cable, i] = site_xpos[world, site]
                state.bodies[world, cable, i] = topology.site_body[site]
                state.roles[world, cable, i] = topology.roles[cable, i]
                state.wrap[world, cable, i] = -1
        else:
            rotation = geom_xmat[world, p.geom]
            center = geom_xpos[world, p.geom]
            start_site = topology.sites[cable, 0]
            end_site = topology.sites[cable, 2]
            start = site_xpos[world, start_site]
            end = site_xpos[world, end_site]
            a = wp.transpose(rotation) * (start - center)
            b = wp.transpose(rotation) * (end - center)
            hint = wp.transpose(rotation) * (site_xpos[world, topology.sites[cable, 1]] - center)
            ra = wp.sqrt(a[0] * a[0] + a[1] * a[1])
            rb = wp.sqrt(b[0] * b[0] + b[1] * b[1])
            if ra <= p.radius + 1.0e-7 or rb <= p.radius + 1.0e-7:
                state.values[world, cable, 8] = 2.0
                return
            la = wp.sqrt(ra * ra - p.radius * p.radius)
            lb = wp.sqrt(rb * rb - p.radius * p.radius)
            phi_a = wp.atan2(a[1], a[0])
            phi_b = wp.atan2(b[1], b[0])
            seed = wp.atan2(hint[1], hint[0])
            best_score = float(1.0e30)
            entry = float(0.0)
            delta = float(0.0)
            sign_best = int(0)
            for candidate in range(2):
                sign = 2 * candidate - 1
                if state.branch[world, cable] != 0 and sign != state.branch[world, cable]:
                    continue
                angle_in = phi_a + float(sign) * wp.acos(p.radius / ra)
                angle_out = phi_b - float(sign) * wp.acos(p.radius / rb)
                sweep = positive_angle(float(sign) * (angle_out - angle_in))
                if sweep < 1.0e-6:
                    sweep = 6.283184307179586
                seed_offset = positive_angle(float(sign) * (seed - angle_in))
                seed_distance = float(0.0)
                if seed_offset > sweep:
                    seed_distance = wp.min(seed_offset - sweep, 6.283185307179586 - seed_offset)
                horizontal = la + p.radius * sweep + lb
                score = wp.sqrt(horizontal * horizontal + (b[2] - a[2]) * (b[2] - a[2]))
                if state.branch[world, cable] == 0:
                    score += 1000.0 * p.radius * seed_distance
                if score < best_score:
                    best_score = score
                    entry = angle_in
                    delta = float(sign) * sweep
                    sign_best = sign
            arc_xy = p.radius * wp.abs(delta)
            total_xy = la + arc_xy + lb
            z0 = a[2] + (b[2] - a[2]) * la / total_xy
            z1 = a[2] + (b[2] - a[2]) * (la + arc_xy) / total_xy
            # End-cap-constrained routing is deliberately not approximated.
            if wp.abs(z0) > p.half_length or wp.abs(z1) > p.half_length:
                state.values[world, cable, 8] = 2.0
                return
            state.branch[world, cable] = sign_best
            arc_length = wp.sqrt(arc_xy * arc_xy + (z1 - z0) * (z1 - z0))
            state.arc[world, cable] = wp.vec4(entry, delta, z0, z1)
            if p.analytic != 0:
                # Exact frictionless helix: only its endpoints transmit a net
                # wrench to each rigid body. No per-arc-point physics work.
                contact_in = center + rotation * wp.vec3(p.radius * wp.cos(entry), p.radius * wp.sin(entry), z0)
                contact_out = center + rotation * wp.vec3(p.radius * wp.cos(entry + delta), p.radius * wp.sin(entry + delta), z1)
                direction_in = wp.normalize(contact_in - start)
                direction_out = wp.normalize(end - contact_out)
                state.count[world, cable] = 4
                state.points[world, cable, 0] = start
                state.points[world, cable, 1] = contact_in
                state.points[world, cable, 2] = contact_out
                state.points[world, cable, 3] = end
                state.bodies[world, cable, 0] = topology.site_body[start_site]
                state.bodies[world, cable, 1] = p.geom_body
                state.bodies[world, cable, 2] = p.geom_body
                state.bodies[world, cable, 3] = topology.site_body[end_site]
                state.directions[world, cable, 0] = direction_in
                state.directions[world, cable, 2] = direction_out
                va = point_velocity(world, topology.site_body[start_site], start, topology.body_root, com, cvel)
                vb = point_velocity(world, topology.site_body[end_site], end, topology.body_root, com, cvel)
                vi = point_velocity(world, p.geom_body, contact_in, topology.body_root, com, cvel)
                vo = point_velocity(world, p.geom_body, contact_out, topology.body_root, com, cvel)
                state.values[world, cable, 0] = wp.sqrt(total_xy * total_xy + (b[2] - a[2]) * (b[2] - a[2]))
                state.values[world, cable, 1] = wp.dot(direction_in, vi - va) + wp.dot(direction_out, vb - vo)
                tangent_in = rotation * wp.normalize(wp.vec3(-p.radius * delta * wp.sin(entry), p.radius * delta * wp.cos(entry), z1 - z0))
                tangent_out = rotation * wp.normalize(wp.vec3(-p.radius * delta * wp.sin(entry + delta), p.radius * delta * wp.cos(entry + delta), z1 - z0))
                state.values[world, cable, 9] = wp.max(wp.length(direction_in - tangent_in), wp.length(direction_out - tangent_out))
                state.values[world, cable, 11] = 1.0
                if state.initialized[world, cable] == 0:
                    state.home[world, cable] = p.home_length
                    if p.home_length < 0.0:
                        state.home[world, cable] = state.values[world, cable, 0]
                    state.initialized[world, cable] = 1
                return
            samples = wp.clamp(int(wp.ceil(arc_length / 0.0015)) + 1, 3, 96)
            state.count[world, cable] = samples + 2
            state.points[world, cable, 0] = start
            state.bodies[world, cable, 0] = topology.site_body[start_site]
            state.roles[world, cable, 0] = 1
            state.wrap[world, cable, 0] = -1
            for i in range(samples):
                fraction = float(i) / float(samples - 1)
                angle = entry + fraction * delta
                local = wp.vec3(p.radius * wp.cos(angle), p.radius * wp.sin(angle), z0 + fraction * (z1 - z0))
                state.points[world, cable, i + 1] = center + rotation * local
                state.bodies[world, cable, i + 1] = p.geom_body
                state.roles[world, cable, i + 1] = 0
                state.wrap[world, cable, i + 1] = 0
            state.points[world, cable, samples + 1] = end
            state.bodies[world, cable, samples + 1] = topology.site_body[end_site]
            state.roles[world, cable, samples + 1] = 1
            state.wrap[world, cable, samples + 1] = -1
            tangent_in = rotation * wp.normalize(wp.vec3(-p.radius * delta * wp.sin(entry), p.radius * delta * wp.cos(entry), z1 - z0))
            tangent_out = rotation * wp.normalize(wp.vec3(-p.radius * delta * wp.sin(entry + delta), p.radius * delta * wp.cos(entry + delta), z1 - z0))
            residual = wp.max(
                wp.length(wp.normalize(state.points[world, cable, 1] - start) - tangent_in),
                wp.length(wp.normalize(end - state.points[world, cable, samples]) - tangent_out),
            )
            state.values[world, cable, 9] = residual
            state.values[world, cable, 11] = 1.0
        count = state.count[world, cable]
        for i in range(count):
            state.velocities[world, cable, i] = point_velocity(
                world, state.bodies[world, cable, i], state.points[world, cable, i],
                topology.body_root, com, cvel,
            )
        length = float(0.0)
        velocity = float(0.0)
        for i in range(count - 1):
            difference = state.points[world, cable, i + 1] - state.points[world, cable, i]
            direction = wp.normalize(difference)
            state.directions[world, cable, i] = direction
            length += wp.length(difference)
            velocity += wp.dot(direction, state.velocities[world, cable, i + 1] - state.velocities[world, cable, i])
        state.values[world, cable, 0] = length
        state.values[world, cable, 1] = velocity
    if state.initialized[world, cable] == 0 and p.native_surface == 0:
        state.home[world, cable] = p.home_length
        if p.home_length < 0.0:
            state.home[world, cable] = state.values[world, cable, 0]
        state.initialized[world, cable] = 1


@wp.func
def command(p: Parameters, world: int, ctrl: wp.array2d[float], qpos: wp.array2d[float]):
    value = float(0.0)
    if p.actuator >= 0:
        value = wp.clamp(ctrl[world, p.actuator], p.ctrl_low, p.ctrl_high)
    if p.mode == 4:
        value = qpos[world, p.joint_qpos] - p.joint_zero
    if p.mode == 1 or p.mode == 4:
        value = wp.abs(p.reserve_sign * p.spool_reserve_length + p.spool_radius * value) - p.spool_reserve_length
    return value


@wp.kernel
def evaluate(topology: Topology, state: State, ctrl: wp.array2d[float], qpos: wp.array2d[float]):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    if state.values[world, cable, 8] == 2.0:
        return
    contraction = command(p, world, ctrl, qpos)
    if p.mode == 2 or p.mode == 3:
        contraction = state.integrated[world, cable]
    elif p.control_timeconstant > 0.0 or p.max_contraction_rate > 0.0:
        contraction = state.filtered[world, cable]
    free_length = state.home[world, cable] - contraction - p.pretension_offset
    extension = state.values[world, cable, 0] - free_length - p.slack
    state.values[world, cable, 2] = free_length
    state.values[world, cable, 3] = contraction
    state.values[world, cable, 4] = extension
    engaged = extension > 0.0
    if p.taut_hysteresis > 0.0:
        engaged = state.latch[world, cable] != 0
        if engaged and extension <= 0.0:
            engaged = False
        elif not engaged and extension >= p.taut_hysteresis:
            engaged = True
        state.latch[world, cable] = int(engaged)
    tension = float(0.0)
    if engaged and extension > 0.0:
        elastic = extension
        damping_scale = float(1.0)
        if p.taut_transition > 0.0:
            if extension < p.taut_transition:
                elastic = extension * extension / (2.0 * p.taut_transition)
                damping_scale = extension / p.taut_transition
            else:
                elastic = extension - 0.5 * p.taut_transition
        tension = wp.max(0.0, p.stiffness * elastic + p.damping * damping_scale * state.values[world, cable, 1])
    if tension > p.max_tension:
        tension = p.max_tension
        state.values[world, cable, 7] = 1.0
    state.values[world, cable, 5] = tension
    state.values[world, cable, 6] = float(tension > 0.0)


@wp.func
def project_point_force(
    topology: Topology, state: State, world: int, body: int,
    point: wp.vec3, force: wp.vec3, com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector], nv: int,
):
    if body > 0:
        for dof in range(nv):
            if topology.body_dofs[body, dof] != 0:
                root = topology.body_root[topology.dof_body[dof]]
                motion = cdof[world, dof]
                jacobian = wp.spatial_bottom(motion) + wp.cross(wp.spatial_top(motion), point - com[world, root])
                wp.atomic_add(state.force, world, dof, wp.dot(jacobian, force))


@wp.kernel
def apply_forces(
    topology: Topology, state: State, qpos: wp.array2d[float],
    com: wp.array2d[wp.vec3], cdof: wp.array2d[wp.spatial_vector], nv: int,
):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    tension = state.values[world, cable, 5]
    if tension <= 0.0:
        return
    if p.reaction != 0 and p.mode == 4:
        angle = qpos[world, p.joint_qpos] - p.joint_zero
        winding = p.reserve_sign * p.spool_reserve_length + p.spool_radius * angle
        sign = p.reserve_sign
        if winding != 0.0:
            sign = wp.sign(winding)
        wp.atomic_add(state.force, world, p.joint_dof, -tension * sign * p.spool_radius)
    if p.surface == 0 or p.friction_direction == 4:
        return
    if p.analytic != 0:
        incoming = tension * state.directions[world, cable, 0]
        outgoing = tension * state.directions[world, cable, 2]
        for node in range(4):
            force = incoming
            if node == 1:
                force = -incoming
            elif node == 2:
                force = outgoing
            elif node == 3:
                force = -outgoing
            project_point_force(topology, state, world, state.bodies[world, cable, node],
                                state.points[world, cable, node], force, com, cdof, nv)
            state.body_forces[world, cable, state.bodies[world, cable, node]] += force
        return
    count = state.count[world, cable]
    material_speed = 0.5 * (
        wp.dot(state.velocities[world, cable, 0], state.directions[world, cable, 0])
        + wp.dot(state.velocities[world, cable, count - 1], state.directions[world, cable, count - 2])
    )
    current = tension
    for segment in range(count - 1):
        state.tensions[world, cable, segment] = current
        node = segment + 1
        if node + 1 >= count:
            continue
        mu = float(0.0)
        contact = int(-1)
        if state.wrap[world, cable, node] >= 0 and state.wrap[world, cable, node - 1] == state.wrap[world, cable, node] and state.wrap[world, cable, node + 1] == state.wrap[world, cable, node]:
            mu = p.capstan_mu
            contact = state.wrap[world, cable, node]
        elif state.roles[world, cable, node] == 3:
            mu = p.guide_friction_mu
            contact = topology.guide_contact[cable, node]
            if contact >= 0:
                mu = topology.contact_mu[cable, contact]
        if mu > 0.0 or contact >= 0:
            before = state.directions[world, cable, segment]
            after = state.directions[world, cable, segment + 1]
            exponent = mu * wp.acos(wp.clamp(wp.dot(before, after), -1.0, 1.0))
            if p.friction_direction == 2:
                tangent = wp.normalize(before + after)
                slip = material_speed - wp.dot(state.velocities[world, cable, node], tangent)
                exponent *= wp.tanh(slip / p.capstan_velocity_scale)
            elif p.friction_direction == 0:
                exponent = -exponent
            elif p.friction_direction == 3:
                exponent *= wp.tanh(state.contact_angles[world, cable][0] / p.capstan_velocity_scale)
            exponent = wp.clamp(exponent, -20.0, 20.0)
            before_tension = current
            current *= wp.exp(exponent)
            if contact >= 0:
                diagnostic = state.capstan[world, cable, contact]
                if diagnostic[0] == 0.0:
                    diagnostic[1] = before_tension
                diagnostic[0] += wp.acos(wp.clamp(wp.dot(before, after), -1.0, 1.0))
                diagnostic[2] = current
                diagnostic[3] += exponent
                state.capstan[world, cable, contact] = diagnostic
    for node in range(count):
        body = state.bodies[world, cable, node]
        force = wp.vec3(0.0)
        if node > 0:
            force -= state.tensions[world, cable, node - 1] * state.directions[world, cable, node - 1]
        if node + 1 < count:
            force += state.tensions[world, cable, node] * state.directions[world, cable, node]
        point = state.points[world, cable, node]
        state.body_forces[world, cable, body] += force
        project_point_force(topology, state, world, body, point, force, com, cdof, nv)


@wp.kernel
def sample_visual_routes(
    topology: Topology, state: State, geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33], world: int,
    points: wp.array2d[wp.vec3], counts: wp.array[int],
):
    # A thread per display point, for the requested world only. These buffers
    # are independent of physics state and never participate in step graphs.
    cable, node = wp.tid()
    p = topology.parameters[cable]
    count = state.count[world, cable]
    if p.analytic != 0 and count > 0:
        arc = state.arc[world, cable]
        arc_xy = p.radius * wp.abs(arc[1])
        arc_length = wp.sqrt(arc_xy * arc_xy + (arc[3] - arc[2]) * (arc[3] - arc[2]))
        samples = wp.clamp(int(wp.ceil(arc_length / 0.0015)) + 1, 3, 96)
        count = samples + 2
        if node == 0:
            points[cable, node] = state.points[world, cable, 0]
        elif node == count - 1:
            points[cable, node] = state.points[world, cable, 3]
        elif node < count:
            fraction = float(node - 1) / float(samples - 1)
            angle = arc[0] + fraction * arc[1]
            local = wp.vec3(p.radius * wp.cos(angle), p.radius * wp.sin(angle), arc[2] + fraction * (arc[3] - arc[2]))
            points[cable, node] = geom_xpos[world, p.geom] + geom_xmat[world, p.geom] * local
    elif node < count:
        points[cable, node] = state.points[world, cable, node]
    if node == 0:
        counts[cable] = count


@wp.kernel
def add_forces(force: wp.array2d[float], passive: wp.array2d[float]):
    world, dof = wp.tid()
    passive[world, dof] += force[world, dof]


@wp.kernel
def actuator_forces(topology: Topology, state: State, forces: wp.array2d[float]):
    world, cable = wp.tid()
    actuator = topology.parameters[cable].actuator
    if actuator >= 0:
        p = topology.parameters[cable]
        forces[world, actuator] = wp.clamp(-state.values[world, cable, 5], p.force_low, p.force_high)


@wp.kernel
def sensors(state: State, bindings: wp.array2d[int], sensordata: wp.array2d[float]):
    world, binding = wp.tid()
    cable = bindings[binding, 0]
    address = bindings[binding, 1]
    dimension = bindings[binding, 2]
    for i in range(dimension):
        sensordata[world, address + i] = state.values[world, cable, i]


@wp.kernel
def advance(topology: Topology, state: State, ctrl: wp.array2d[float], qpos: wp.array2d[float], timestep: float):
    world, cable = wp.tid()
    p = topology.parameters[cable]
    if state.pending_valid[world, cable] != 0:
        state.material[world, cable] = state.pending_material[world, cable]
        state.material_initialized[world, cable] = 1
        state.pending_valid[world, cable] = 0
    target = command(p, world, ctrl, qpos)
    if p.mode == 2 or p.mode == 3:
        speed = target
        if p.mode == 3:
            speed *= p.spool_radius
        if p.max_contraction_rate > 0.0:
            speed = wp.clamp(speed, -p.max_contraction_rate, p.max_contraction_rate)
        state.integrated[world, cable] = wp.max(0.0, state.integrated[world, cable] + timestep * speed)
    elif p.control_timeconstant > 0.0 or p.max_contraction_rate > 0.0:
        alpha = float(1.0)
        if p.control_timeconstant > 0.0:
            alpha = 1.0 - wp.exp(-timestep / p.control_timeconstant)
        change = alpha * (target - state.filtered[world, cable])
        if p.max_contraction_rate > 0.0:
            maximum = timestep * p.max_contraction_rate
            change = wp.clamp(change, -maximum, maximum)
        state.filtered[world, cable] += change


@wp.kernel
def reset_cables(state: State, mask: wp.array[bool]):
    world, cable = wp.tid()
    if mask[world]:
        state.integrated[world, cable] = 0.0
        state.filtered[world, cable] = 0.0
        state.latch[world, cable] = 0
        state.home[world, cable] = 0.0
        state.initialized[world, cable] = 0
        state.branch[world, cable] = 0
        state.count[world, cable] = 0
        state.material_initialized[world, cable] = 0
        state.pending_valid[world, cable] = 0
        state.material[world, cable] = wp.vec4d(0.0)
        state.pending_material[world, cable] = wp.vec4d(0.0)
        for i in range(12):
            state.values[world, cable, i] = 0.0
