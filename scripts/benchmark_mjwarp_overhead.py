#!/usr/bin/env python3
"""Measure cable compute overhead at identical physical states on CUDA.

The baseline replays precomputed cable forces and sensors. Both variants restore
the same input state before each step, so contacts/solver work stay comparable.
This is a fixed-state microbenchmark, not a free rollout or an RL benchmark.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mujoco_cable_warp import CableSimulation
from mujoco_cable_warp import kernels
from mujoco_cable_warp.cli import resolve_model


def benchmark(args):
    sim = CableSimulation(resolve_model(args.model), device=args.device,
                          nworld=args.nworld, nconmax=args.nconmax,
                          njmax=args.njmax)
    if not sim.device.is_cuda:
        raise ValueError("This benchmark requires CUDA")
    sim.set_control(np.full(sim.cpu_model.nu, args.control, dtype=np.float32))
    sim.step()
    sim.capture_step()
    sim.step(args.settle_steps)
    sim.forward()
    sim.check()

    with wp.ScopedDevice(sim.device):
        saved = [(a, wp.clone(a)) for a in (
            sim.data.qpos, sim.data.qvel, sim.data.act, sim.data.time,
            sim.data.qacc_warmstart, sim.state.integrated,
            sim.state.filtered, sim.state.latch,
        ) if a.size]
        force = wp.clone(sim.state.force)
        actuator = wp.clone(sim.data.actuator_force)
        sensors = wp.clone(sim.data.sensordata)

        def restore():
            for target, snapshot in saved:
                wp.copy(target, snapshot)

        def full_step():
            restore()
            sim._step()

        def replay_passive(model, data):
            wp.launch(kernels.add_forces, (sim.nworld, sim.cpu_model.nv),
                      inputs=[force, data.qfrc_passive], device=sim.device)

        def replay_actuator(model, data):
            wp.copy(data.actuator_force, actuator)

        def replay_sensor(model, data, stage):
            if int(stage) == int(mujoco.mjtStage.mjSTAGE_VEL):
                wp.copy(data.sensordata, sensors)

        def baseline_step():
            restore()
            mjw.step(sim.model, sim.data)

        full_step()
        with wp.ScopedCapture() as captured:
            full_step()
        full_graph = captured.graph
        wp.capture_launch(full_graph)
        expected = {name: getattr(sim.data, name).numpy().copy() for name in (
            "qpos", "qvel", "qacc", "qfrc_actuator", "qfrc_passive", "sensordata",
        )}
        # GPU contact ordering/reductions can vary even for the same graph.
        repeat_noise = {name: 0.0 for name in expected}
        for _ in range(5):
            wp.capture_launch(full_graph)
            for name, value in expected.items():
                if value.size:
                    repeat_noise[name] = max(repeat_noise[name], float(np.max(
                        np.abs(getattr(sim.data, name).numpy() - value))))
        sim.model.callback.passive = replay_passive
        sim.model.callback.act_gain = replay_actuator
        sim.model.callback.sensor = replay_sensor
        baseline_step()
        with wp.ScopedCapture() as captured:
            baseline_step()
        baseline_graph = captured.graph
        wp.capture_launch(baseline_graph)
        differences = {}
        for name, value in expected.items():
            actual = getattr(sim.data, name).numpy()
            differences[name] = float(np.max(np.abs(value - actual))) if value.size else 0.0
            # Require equal cable inputs/forces; allow only measured solver
            # repeat noise for acceleration (do not loosen force tolerances).
            atol = 2e-5 + (2 * repeat_noise[name] if name == "qacc" else 0)
            np.testing.assert_allclose(actual, value, atol=atol, rtol=2e-5,
                                       err_msg=f"Baseline physics differs: {name}")

        graphs = {"precomputed_cable": baseline_graph, "online_cable": full_graph}
        samples = {name: [] for name in graphs}
        for graph in graphs.values():
            for _ in range(100):
                wp.capture_launch(graph)
        wp.synchronize_device(sim.device)
        for repeat in range(args.repeats):
            # Alternate order to reduce temperature/clock drift bias.
            order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
            for name in order:
                wp.synchronize_device(sim.device)
                started = time.perf_counter()
                for _ in range(args.steps):
                    wp.capture_launch(graphs[name])
                wp.synchronize_device(sim.device)
                samples[name].append((time.perf_counter() - started) / args.steps * 1000)

    medians = {name: statistics.median(values) for name, values in samples.items()}
    baseline = medians["precomputed_cable"]
    online = medians["online_cable"]
    return {
        "method": "fixed_state_precomputed_cable_replay_vs_online_cable",
        "notes": "Both include identical input-state restore copies; excludes policy, rewards, resets, rendering and JIT. Not a free-rollout benchmark. Baseline retains cable arrays and tendon topology.",
        "device": str(sim.device), "gpu": sim.device.name,
        "versions": {name: importlib.metadata.version(name) for name in (
            "mujoco", "mujoco-warp", "warp-lang")},
        "model": str(sim.loaded.source), "nworld": args.nworld,
        "cables_per_world": sim.ncable, "nv": sim.cpu_model.nv,
        "timestep": sim.timestep, "control": args.control,
        "settle_steps": args.settle_steps, "steps_per_repeat": args.steps,
        "repeats": args.repeats, "nconmax": args.nconmax, "njmax": args.njmax,
        "same_state_max_absolute_difference": differences,
        "same_graph_repeat_max_absolute_difference": repeat_noise,
        "samples_ms_per_batch_step": samples,
        "median_ms_per_batch_step": medians,
        "extra_ms_per_batch_step": online - baseline,
        "step_time_increase_percent": (online / baseline - 1) * 100,
        "throughput_drop_percent": (1 - baseline / online) * 100,
        "world_steps_per_second": {
            name: args.nworld * 1000 / value for name, value in medians.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="15")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nworld", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--control", type=float, default=0.0)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.settle_steps < 0:
        parser.error("steps/repeats must be positive and settle-steps nonnegative")
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
