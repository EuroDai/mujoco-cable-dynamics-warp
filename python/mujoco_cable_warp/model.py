"""Translate cable plugin MJCF into ordinary MJCF and explicit cable metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PLUGIN = "mujoco.cable.unilateral"
CONTROL_MODES = {
    "target_contraction": 0,
    "target_spool_angle": 1,
    "spool_velocity": 2,
    "spool_angular_velocity": 3,
    "joint_spool_angle": 4,
}
FLOAT_DEFAULTS = {
    "stiffness": 0.0,
    "damping": 0.0,
    "slack": 0.0,
    "pretension_offset": 0.0,
    "max_tension": 1.0e30,
    "spool_radius": 1.0,
    "spool_reserve_length": 0.0,
    "control_timeconstant": 0.0,
    "max_contraction_rate": 0.0,
    "taut_transition": 0.0,
    "taut_hysteresis": 0.0,
    "capstan_mu": 0.0,
    "guide_friction_mu": 0.0,
    "capstan_velocity_scale": 0.01,
    "visual_width": 2.0,
}
KNOWN_CONFIG = set(FLOAT_DEFAULTS) | {
    "home_length", "ctrl_mode", "spool_joint", "spool_reserve_direction",
    "spool_reaction_torque", "capstan_direction", "route_mode", "route_tendon",
    "wrap_geoms", "site_role_user_index", "integration_mode", "mesh_route_mode",
    "mesh_guide_axis", "mesh_guide_weight", "route_hysteresis",
    "composite_merge_distance", "visual_smoothing_timeconstant",
    "guide_friction_mus", "body_force_sensor", "capstan_sensor", "capstan_sensor_guides",
}


class UnsupportedCableError(ValueError):
    """The model requires a cable feature not implemented by this backend."""


@dataclass
class CableSpec:
    name: str
    config: dict[str, str]
    actuator_name: str | None = None
    tendon_name: str = ""
    actuator_id: int = -1
    tendon_id: int = -1
    site_ids: list[int] = field(default_factory=list)
    geom_id: int = -1
    sensor_ids: list[int] = field(default_factory=list)
    sensor_kinds: dict[int, int] = field(default_factory=dict)
    guide_site_ids: list[int] = field(default_factory=list)
    native_geom_ids: list[int] = field(default_factory=list)


@dataclass
class CableModel:
    model: mujoco.MjModel
    cables: list[CableSpec]
    source: Path
    hinges: list[dict[str, str]] = field(default_factory=list)


def _expand_includes(root: ET.Element, directory: Path, stack: tuple[Path, ...]) -> None:
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag != "include":
                continue
            path = (directory / child.attrib["file"]).resolve()
            if path in stack:
                raise ValueError(f"Recursive MJCF include: {path}")
            included = ET.parse(path).getroot()
            _expand_includes(included, path.parent, (*stack, path))
            position = list(parent).index(child)
            parent.remove(child)
            for offset, element in enumerate(included):
                parent.insert(position + offset, element)


def _validate_config(spec: CableSpec) -> None:
    c = spec.config
    unknown = set(c) - KNOWN_CONFIG
    if unknown:
        raise UnsupportedCableError(f"{spec.name}: unknown cable options: {sorted(unknown)}")
    if c.get("route_mode", "native") not in ("native", "surface"):
        raise ValueError(f"{spec.name}: invalid route_mode")
    if c.get("integration_mode", "explicit") not in ("explicit", "implicit_elastic"):
        raise UnsupportedCableError(
            f"{spec.name}: MJWarp supports explicit and implicit_elastic; "
            "local_implicit and implicit_compliant require separate solver integration."
        )
    if c.get("integration_mode") == "implicit_elastic":
        if float(c.get("damping", 0)) != 0:
            raise UnsupportedCableError(
                "implicit_elastic currently requires zero cable damping"
            )
        material = c.get("capstan_direction") == "stick_slip"
        if not material and c.get("route_mode", "native") != "native":
            raise UnsupportedCableError(
                "implicit_elastic supports native routing or stick_slip material"
            )
        if c.get("ctrl_mode", "target_contraction") == "joint_spool_angle":
            raise UnsupportedCableError("implicit_elastic excludes joint spools")
    if c.get("ctrl_mode", "target_contraction") not in CONTROL_MODES:
        raise ValueError(f"{spec.name}: invalid ctrl_mode")
    for name, default in FLOAT_DEFAULTS.items():
        value = float(c.get(name, default))
        if not math.isfinite(value):
            raise ValueError(f"{spec.name}: {name} must be finite")
        if name not in ("pretension_offset",) and value < 0:
            raise ValueError(f"{spec.name}: {name} must be nonnegative")
    if float(c.get("capstan_velocity_scale", 0.01)) <= 0:
        raise ValueError(f"{spec.name}: capstan_velocity_scale must be positive")
    if c.get("home_length", "auto_initial") != "auto_initial":
        if not math.isfinite(float(c["home_length"])) or float(c["home_length"]) < 0:
            raise ValueError(f"{spec.name}: invalid home_length")
    if c.get("spool_reserve_direction", "positive") not in ("positive", "negative"):
        raise ValueError(f"{spec.name}: invalid spool_reserve_direction")
    if c.get("spool_reaction_torque", "false") not in ("true", "false"):
        raise ValueError(f"{spec.name}: spool_reaction_torque must be true or false")
    if c.get("capstan_direction", "forward") not in ("forward", "reverse", "velocity", "control", "stick_slip"):
        raise ValueError(f"{spec.name}: invalid capstan_direction")
    guides = c.get("capstan_sensor_guides", "").split()
    mus = c.get("guide_friction_mus", "").split()
    if mus and (len(mus) != len(guides) or any(not math.isfinite(float(x)) or float(x) < 0 for x in mus)):
        raise ValueError(f"{spec.name}: guide_friction_mus requires one finite nonnegative value per capstan_sensor_guides site")
    if len(guides) != len(set(guides)):
        raise ValueError(f"{spec.name}: duplicate capstan_sensor_guides")
    if c.get("route_mode", "native") != "surface" and any(c.get(name) for name in ("body_force_sensor", "capstan_sensor", "capstan_sensor_guides", "guide_friction_mus")):
        raise UnsupportedCableError(f"{spec.name}: cable force and capstan sensors require a surface route")
    if c.get("capstan_direction") == "control" and (
        c.get("route_mode", "native") != "surface"
        or c.get("ctrl_mode", "target_contraction") != "target_contraction"
        or (float(c.get("control_timeconstant", 0)) <= 0 and float(c.get("max_contraction_rate", 0)) <= 0)
    ):
        raise ValueError(f"{spec.name}: control friction requires a surface cable with filtered target_contraction")
    if c.get("capstan_direction") == "stick_slip" and (
        c.get("route_mode", "native") != "surface"
        or c.get("ctrl_mode", "target_contraction") != "target_contraction"
        or float(c.get("damping", 0)) != 0
        or c.get("wrap_geoms", "").strip()
        or len(guides) != 2
    ):
        raise UnsupportedCableError(f"{spec.name}: stick_slip requires an elastic surface cable with two guides and no wrap_geoms")
    if c.get("body_force_sensor") and c.get("body_force_sensor") == c.get("capstan_sensor"):
        raise ValueError(f"{spec.name}: body_force_sensor and capstan_sensor must be distinct")
    if c.get("route_mode", "native") == "native" and (
        float(c.get("capstan_mu", 0)) or float(c.get("guide_friction_mu", 0))
    ):
        raise UnsupportedCableError(f"{spec.name}: friction is currently supported on surface routes only")
    for name in ("route_hysteresis", "composite_merge_distance", "visual_smoothing_timeconstant"):
        if float(c.get(name, 0)) != 0:
            raise UnsupportedCableError(f"{spec.name}: {name} is not implemented by the MJWarp backend")


def load_model(path: str | Path) -> CableModel:
    """Load without registering or compiling the C++ plugin.

    Cable actuators retain their names, order, controls and transmission. User
    gain callbacks supply their force. Surface actuators must have zero gear.
    Plugin sensors become user sensors with the same dimensions and order.
    """
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    _expand_includes(root, path.parent, (path,))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    assetdir = compiler.attrib.pop("assetdir", "")
    for name in ("meshdir", "texturedir"):
        compiler.set(name, str((path.parent / compiler.get(name, assetdir)).resolve()))

    specs: dict[str, CableSpec] = {}
    hinges = []
    for extension in root.findall("extension"):
        for plugin in extension:
            if plugin.get("plugin") == "mjhinge.hinge_friction":
                for instance in plugin:
                    hinges.append({"name": instance.attrib["name"], **{x.attrib["key"]: x.attrib["value"] for x in instance}})
                continue
            if plugin.get("plugin") != PLUGIN:
                raise UnsupportedCableError(f"Unsupported engine plugin: {plugin.get('plugin')}")
            for instance in plugin:
                name = instance.attrib["name"]
                if name in specs:
                    raise ValueError(f"Duplicate cable instance: {name}")
                spec = CableSpec(name, {x.attrib["key"]: x.attrib["value"] for x in instance})
                _validate_config(spec)
                specs[name] = spec
        root.remove(extension)

    for actuator in root.findall("./actuator/plugin"):
        instance = actuator.get("instance")
        if instance not in specs:
            raise UnsupportedCableError(f"Unknown cable actuator instance: {instance}")
        spec = specs[instance]
        if spec.actuator_name is not None:
            raise UnsupportedCableError(f"{instance}: use one actuator per cable instance")
        name = actuator.get("name", f"{instance}_warp_actuator")
        spec.actuator_name = name
        spec.tendon_name = actuator.attrib["tendon"]
        actuator.tag = "general"
        actuator.attrib.pop("instance")
        actuator.set("name", name)
        actuator.set("dyntype", "none")
        # Zero ordinary gain also permits CPU kinematics for the viewer without
        # installing process-global MuJoCo callbacks. MJWarp replaces the force.
        actuator.set("gaintype", "fixed")
        actuator.set("gainprm", "0")
        actuator.set("biastype", "none")
        if spec.config.get("route_mode", "native") == "surface":
            gear = np.fromstring(actuator.get("gear", "1"), sep=" ")
            if np.any(gear != 0):
                raise ValueError(f"{instance}: surface cable actuator requires gear=0")

    sensor_names: dict[str, str] = {}
    for index, sensor in enumerate(root.findall("./sensor/plugin")):
        instance = sensor.get("instance")
        if instance not in specs:
            raise UnsupportedCableError(f"Unknown cable sensor instance: {instance}")
        name = sensor.get("name", f"{instance}_warp_sensor_{index}")
        sensor_names[name] = instance
        sensor.attrib.pop("instance")
        sensor.tag = "user"
        sensor.set("name", name)
        c = specs[instance].config
        # nbody is known after compilation. Count bodies here so user sensors
        # have the correct dimensions when MuJoCo lays out sensordata.
        dimension = 12 if c.get("route_mode") == "surface" else 8
        if name == c.get("body_force_sensor"):
            dimension = 3 * (1 + len(root.findall(".//worldbody//body")))
        elif name == c.get("capstan_sensor"):
            dimension = 4 * (len(c.get("wrap_geoms", "").replace(",", " ").split()) + len(c.get("capstan_sensor_guides", "").split()))
            if dimension == 0:
                raise ValueError(f"{instance}: capstan_sensor requires wraps or guide sites")
        sensor.set("dim", str(dimension))
        sensor.set("needstage", "vel")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nplugin:
        raise UnsupportedCableError("Unconverted engine plugins remain in the model")
    if hasattr(mujoco.mjtDisableBit, "mjDSBL_SPRING"):
        passive_off = int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
        if model.opt.disableflags & passive_off == passive_off:
            raise UnsupportedCableError("Cable evaluation requires the MJWarp passive stage; do not disable both spring and damper")
    if getattr(model.opt, "disableactuator", 0):
        raise UnsupportedCableError("Actuator group disabling is not implemented by the cable callback")
    if np.any(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_USER) or np.any(model.actuator_biastype == mujoco.mjtBias.mjBIAS_USER) or np.any(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_USER):
        raise UnsupportedCableError("Additional user actuator callbacks must be integrated explicitly")
    if hasattr(mujoco.mjtEnableBit, "mjENBL_SLEEP") and model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
        raise UnsupportedCableError("Disable sleeping: custom cable connections are not registered with island wake-up")

    def lookup(kind, name):
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError(f"Missing {kind.name}: {name}")
        return index

    for spec in specs.values():
        c = spec.config
        if spec.actuator_name:
            spec.actuator_id = lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, spec.actuator_name)
        surface = c.get("route_mode", "native") == "surface"
        if surface:
            route_name = c.get("route_tendon", "")
            if spec.tendon_name and route_name != spec.tendon_name:
                raise ValueError(f"{spec.name}: route_tendon must match the actuator tendon")
            spec.tendon_name = route_name
        elif spec.actuator_id < 0:
            raise ValueError(f"{spec.name}: native route requires an actuator")
        spec.tendon_id = lookup(mujoco.mjtObj.mjOBJ_TENDON, spec.tendon_name)
        if c.get("ctrl_mode") == "joint_spool_angle":
            joint = lookup(mujoco.mjtObj.mjOBJ_JOINT, c.get("spool_joint", ""))
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"{spec.name}: spool_joint must be a hinge")
        if not surface:
            continue
        if model.tendon_stiffness[spec.tendon_id] or model.tendon_damping[spec.tendon_id] or model.tendon_limited[spec.tendon_id] or model.tendon_frictionloss[spec.tendon_id] or getattr(model, "tendon_armature", np.zeros(model.ntendon))[spec.tendon_id]:
            raise ValueError(f"{spec.name}: surface seed tendon must not apply native forces or limits")
        first = model.tendon_adr[spec.tendon_id]
        count = model.tendon_num[spec.tendon_id]
        types = model.wrap_type[first:first + count]
        objects = model.wrap_objid[first:first + count]
        native_cylinders = count == 5 and list(types) == [mujoco.mjtWrap.mjWRAP_SITE, mujoco.mjtWrap.mjWRAP_CYLINDER, mujoco.mjtWrap.mjWRAP_SITE, mujoco.mjtWrap.mjWRAP_CYLINDER, mujoco.mjtWrap.mjWRAP_SITE]
        if np.any(types != mujoco.mjtWrap.mjWRAP_SITE) and not native_cylinders:
            raise UnsupportedCableError(f"{spec.name}: surface seed requires sites or a native two-cylinder route")
        if native_cylinders:
            if c.get("capstan_direction") != "stick_slip" or c.get("wrap_geoms", "").strip():
                raise UnsupportedCableError(f"{spec.name}: native surface cylinders currently require stick_slip and no wrap_geoms")
            spec.native_geom_ids = objects[[1, 3]].tolist()
        spec.site_ids = objects[types == mujoco.mjtWrap.mjWRAP_SITE].tolist()
        role_index = int(c.get("site_role_user_index", 0))
        if not 0 <= role_index < model.nuser_site:
            raise ValueError(f"{spec.name}: invalid site_role_user_index")
        roles = np.rint(model.site_user[spec.site_ids, role_index]).astype(int).tolist()
        if len(roles) < 2 or roles[0] != 1 or roles[-1] != 1:
            raise ValueError(f"{spec.name}: surface endpoints must have role 1")
        spec.guide_site_ids = [lookup(mujoco.mjtObj.mjOBJ_SITE, name) for name in c.get("capstan_sensor_guides", "").split()]
        for i, site in enumerate(spec.guide_site_ids):
            if int(round(model.site_user[site, role_index])) != 3:
                raise ValueError(f"{spec.name}: capstan_sensor_guides must have role 3")
            if native_cylinders:
                if model.site_bodyid[site] != model.geom_bodyid[spec.native_geom_ids[i]]:
                    raise ValueError(f"{spec.name}: native guide must belong to its cylinder body")
            elif site not in spec.site_ids:
                raise ValueError(f"{spec.name}: capstan_sensor_guides must belong to the route")
        if c.get("capstan_direction") == "stick_slip" and not native_cylinders and (roles != [1, 3, 3, 1] or spec.guide_site_ids != spec.site_ids[1:3]):
            raise UnsupportedCableError(f"{spec.name}: stick_slip requires endpoint, two ordered guides, endpoint")
        wraps = c.get("wrap_geoms", "").replace(",", " ").split()
        if wraps:
            geom_ids = [lookup(mujoco.mjtObj.mjOBJ_GEOM, name) for name in wraps]
            if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER for g in geom_ids):
                raise UnsupportedCableError(f"{spec.name}: mesh surface routes are not implemented in MJWarp yet")
            if len(wraps) != 1 or roles != [1, 2, 1]:
                raise UnsupportedCableError(f"{spec.name}: cylinder routes currently require endpoint, one hint, endpoint; composite wraps are not implemented")
            spec.geom_id = geom_ids[0]
        elif not native_cylinders and any(role not in (1, 3) for role in roles):
            raise ValueError(f"{spec.name}: unwrapped surface routes may contain only endpoints and role-3 guides")
    for name, instance in sensor_names.items():
        spec = specs[instance]
        sensor = lookup(mujoco.mjtObj.mjOBJ_SENSOR, name)
        spec.sensor_ids.append(sensor)
        spec.sensor_kinds[sensor] = 1 if name == spec.config.get("body_force_sensor") else (2 if name == spec.config.get("capstan_sensor") else 0)
        if spec.sensor_kinds[sensor] == 1 and model.sensor_dim[sensor] != 3 * model.nbody:
            raise UnsupportedCableError(f"{instance}: body_force_sensor dimension changed during body fusion")
    for spec in specs.values():
        for option in ("body_force_sensor", "capstan_sensor"):
            if spec.config.get(option) and sensor_names.get(spec.config[option]) != spec.name:
                raise ValueError(f"{spec.name}: {option} must name a sensor attached to this surface cable")
    cable_sensors = {sensor for spec in specs.values() for sensor in spec.sensor_ids}
    if any(model.sensor_type[sensor] == mujoco.mjtSensor.mjSENS_USER and sensor not in cable_sensors for sensor in range(model.nsensor)):
        raise UnsupportedCableError("Additional user sensor callbacks must be integrated explicitly")
    if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise UnsupportedCableError("Cable stepping supports Euler and implicitfast; RK4 would advance route history in sub-stages")
    if hinges:
        from .hinge import validate_hinges

        validate_hinges(model, hinges, list(specs.values()))
    return CableModel(model, list(specs.values()), path, hinges)
