"""Interactive and headless entry point for the MJWarp cable backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import warp as wp

from .simulation import CableSimulation


ROOT = Path(__file__).resolve().parents[2]


@wp.kernel
def ramp_control(time_in: wp.array[float], controls: wp.array2d[float], actuator: int, start: float, end: float, final: float):
    world = wp.tid()
    controls[world, actuator] = final * wp.clamp((time_in[world] - start) / (end - start), 0.0, 1.0)


def resolve_model(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    matches = list((ROOT / "cable_plugin_demos").glob(f"{value.zfill(2)}_*.xml"))
    if len(matches) != 1:
        raise ValueError(f"Expected a model path or an unambiguous demo number: {value}")
    return matches[0]


def draw_cables(sim: CableSimulation, scene, world: int):
    points, counts = sim.get_route_points(world)
    values = sim.state.values.numpy()[world]
    for cable, spec in enumerate(sim.loaded.cables):
        if spec.config.get("route_mode", "native") != "surface":
            continue
        rgba = sim.cpu_model.tendon_rgba[spec.tendon_id].copy()
        if values[cable, 6] == 0:
            rgba = np.array([0.48, 0.51, 0.55, 0.6], dtype=np.float32)
        if values[cable, 8] == 2:
            continue
        for i in range(int(counts[cable]) - 1):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, float(spec.config.get("visual_width", 2)), points[cable, i], points[cable, i + 1])
            scene.ngeom += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MuJoCable with MJWarp (no C++ plugin required)")
    parser.add_argument("model", nargs="?", default="15", help="Demo number or MJCF path")
    parser.add_argument("--device", default="cuda:0", help="Execution device (default: cuda:0; use cpu explicitly for CPU validation)")
    parser.add_argument("--nworld", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="Headless step count")
    parser.add_argument("--duration", type=float, default=120.0, help="Viewer simulation duration before exit")
    parser.add_argument("--world", type=int, default=0, help="World shown in the viewer")
    parser.add_argument("--ctrl-final", type=float, default=0.05)
    parser.add_argument("--ctrl-ramp-start", type=float, default=0.5)
    parser.add_argument("--ctrl-ramp-end", type=float, default=3.0)
    parser.add_argument("--auto-actuator", help="Actuator to ramp; default is the first actuator")
    parser.add_argument("--no-auto-control", action="store_true", help="Use zero controls headlessly, or viewer sliders interactively")
    parser.add_argument("--show-route-debug", action="store_true", help="Reveal endpoint and route-hint sites in the viewer")
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--graph", action="store_true", help="Capture physics step as a CUDA graph")
    parser.add_argument("--output", type=Path, help="Also write the JSON summary here")
    args = parser.parse_args(argv)
    if args.steps < 0 or args.duration <= 0 or args.ctrl_ramp_end <= args.ctrl_ramp_start:
        parser.error("steps must be nonnegative, duration positive, and ramp end greater than start")
    if not 0 <= args.world < args.nworld:
        parser.error("--world must be in [0, nworld)")
    model_path = resolve_model(args.model)
    if model_path.name == "30_cpho_2018_problem3_massive_rope.xml":
        parser.error("Demo 30 requires its dedicated massive-rope continuum controller and has not been ported to MJWarp")
    sim = CableSimulation(model_path, nworld=args.nworld, device=args.device, nconmax=args.nconmax, njmax=args.njmax)
    actuator = 0
    if args.auto_actuator:
        actuator = mujoco.mj_name2id(sim.cpu_model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.auto_actuator)
        if actuator < 0:
            parser.error(f"Unknown actuator: {args.auto_actuator}")
    # Warm up JIT compilation before timing, then restore the reference state.
    sim.step()
    sim.reset()
    if sim.cpu_model.nu and not args.no_auto_control:
        wp.launch(ramp_control, sim.nworld, inputs=[sim.data.time, sim.data.ctrl, actuator, args.ctrl_ramp_start, args.ctrl_ramp_end, args.ctrl_final], device=sim.device)
    if args.graph:
        sim.capture_step()

    def step():
        if sim.cpu_model.nu and not args.no_auto_control:
            wp.launch(ramp_control, sim.nworld, inputs=[sim.data.time, sim.data.ctrl, actuator, args.ctrl_ramp_start, args.ctrl_ramp_end, args.ctrl_final], device=sim.device)
        sim.step()

    # CUDA launches are asynchronous: finish warm-up/reset before timing.
    wp.synchronize_device(sim.device)
    started = time.perf_counter()
    steps = 0
    if args.headless:
        for steps in range(1, args.steps + 1):
            step()
    else:
        from mujoco import viewer as mj_viewer

        host = sim.get_data(args.world)
        if args.show_route_debug:
            sim.cpu_model.site_rgba[:, 3] = 1.0
            sim.cpu_model.site_size[:, 0] = np.maximum(sim.cpu_model.site_size[:, 0], 0.002)
        with mj_viewer.launch_passive(sim.cpu_model, host) as viewer:
            overview = mujoco.mj_name2id(sim.cpu_model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
            if overview >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = overview
            frame_steps = max(1, round(1.0 / (60.0 * sim.timestep)))
            total_steps = int(args.duration / sim.timestep)
            while viewer.is_running() and steps < total_steps:
                frame_start = time.perf_counter()
                if args.no_auto_control:
                    # Sliders drive all worlds equally; programmatic callers can
                    # instead write distinct controls into each data.ctrl row.
                    sim.set_control(host.ctrl.copy())
                for _ in range(min(frame_steps, total_steps - steps)):
                    step()
                    steps += 1
                sim.check()
                sim.get_data(args.world, host)
                viewer.user_scn.ngeom = 0
                draw_cables(sim, viewer.user_scn, args.world)
                viewer.sync()
                time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_start)))
    wp.synchronize_device(sim.device)
    elapsed = time.perf_counter() - started
    sim.check()
    values = sim.state.values.numpy()
    summary = {
        "backend": "mjwarp", "device": str(sim.device), "model": str(sim.loaded.source),
        "nworld": sim.nworld, "steps_per_world": steps, "simulation_seconds": steps * sim.timestep,
        "wall_seconds": elapsed, "world_steps_per_second": steps * sim.nworld / max(elapsed, 1e-12),
        "cables": [spec.name for spec in sim.loaded.cables],
        "max_final_tension": float(values[:, :, 5].max()) if sim.ncable else 0.0,
        "world0_cable_state": values[0].tolist(), "world0_qpos": sim.data.qpos.numpy()[0].tolist(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
