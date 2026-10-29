# CPU capstan history and load reporting

The CPU `mujoco.cable.unilateral` plugin now supports two additional
`capstan_direction` values. Existing `forward`, `reverse`, and `velocity`
configurations retain their routing and direction semantics. These additions
do not implement or change the Warp backend.

## Direction modes

- `control`: a sliding approximation that chooses direction from the filtered
  take-up rate. Requires a surface route, one target-contraction actuator,
  positive velocity scale, and a control time constant or rate limit.
- `stick_slip`: three elastic material reservoirs and two capstan friction
  cones. At rest, unequal tensions can persist inside
  `exp(-mu*theta) <= T_out/T_in <= exp(mu*theta)`; exceeding a cone transfers
  material while conserving its total free length. Only the final pulling
  endpoint exchanges material with the spool. Reversing route order changes
  which endpoint is pulled.

`stick_slip` requires explicit cable integration, `target_contraction`, zero
cable damping, a non-RK4 integrator, and two guides. It supports either four
surface sites (the middle two having user role 3) or the native planar route
`site / cylinder / middle site / cylinder / site`. In the native route, the
two cylinders and the middle site must belong to the same rigid body.
`wrap_geoms` must be empty. The cable remains unilateral and elastic, with
segment stiffness derived from a common axial rigidity; this is not an
exactly inextensible constraint.

## Optional configuration and sensors

| Config | Meaning |
| --- | --- |
| `guide_friction_mus` | Space-separated nonnegative coefficients, one per named guide; otherwise use `guide_friction_mu`. |
| `capstan_sensor_guides` | Ordered role-3 site names identifying the two guides, from fixed endpoint toward pulling endpoint. For native cylinders these guide sites identify contacts on their common rigid body. |
| `body_force_sensor` | Name of a sensor attached to this plugin instance. Outputs `3*nbody` world-frame force components in body-ID order. |
| `capstan_sensor` | Distinct sensor name attached to this instance. Each wrap/guide reports angle, incoming tension, outgoing tension, and log tension ratio. |

Declare named sensors using `<plugin name="..." instance="..."/>` inside
`<sensor>`. Ordinary surface state sensors remain 12-dimensional. Ordinary
plugin state remains 5-dimensional; `stick_slip` uses 10 values including
material history. New configuration keys require the updated library; do
not replay saved plugin states using a different mode or state layout.

History is committed in `Advance`; repeated `mj_forward` calls do not advance
material history. Reset and `mj_copyData` retain their standard semantics.
Optional mjhinge integration consumes the body-force report through its
`body_force_sensors` configuration; cable does not link against mjhinge.

## Native finite-radius adapter

The planar two-cylinder adapter reads MuJoCo's native wrap tangent positions,
angles, total length, and velocity. It applies forces at actual tangent
contacts. Half of each short arc contributes to each adjacent elastic
reservoir, retaining three material histories. Lift-off uses zero wrap angle.
This adapter is for the planar, short-arc route; it is not a general spatial
multi-cylinder winding model. Use `gear="0"` for a surface plugin's geometry
transmission so native tendon transmission does not add duplicate forces.

## Build and test

Build against the exact MuJoCo library loaded by the host. CMake probes
experimental implicit-plugin and mesh-ray signatures instead of selecting
them from a version number alone. Unsupported `implicit_compliant` requests
are rejected rather than silently changing integration.

```bash
cmake -S . -B build/cpu -DMUJOCO_PYTHON_PACKAGE_DIR=/path/to/site-packages/mujoco
cmake --build build/cpu -j2
CABLE_PLUGIN_LIBRARY="$PWD/build/cpu/plugin/libcable_unilateral.so" \
  python tests/test_stick_slip_history.py -v
```

Set `MJHINGE_PLUGIN_LIBRARY` to a matching updated mjhinge library to enable
the optional cross-plugin regression. Tests cover material conservation,
both friction cones, static memory, zero-friction tension equality, force
balance, state copy/reset/forward behavior, native route geometry, and
frictionless force versus the native length gradient. Existing demo XMLs
using `contact vertcollide` require a compatible MuJoCo version; they cannot
be loaded unchanged by MuJoCo 3.9.
