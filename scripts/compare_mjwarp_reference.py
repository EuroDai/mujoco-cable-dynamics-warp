#!/usr/bin/env python3
"""Compare MJWarp to the C++ plugin in separate, ABI-compatible environments.

Run this script with the MJWarp interpreter. --reference-python must provide
the MuJoCo runtime against which --plugin was compiled (normally MuJoCo 3.4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def record_reference(args):
    import mujoco

    mujoco.mj_loadPluginLibrary(str(args.plugin.resolve()))
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    if model.nu:
        data.ctrl[args.actuator] = args.control
    mujoco.mj_forward(model, data)
    sensors = [i for i in range(model.nsensor) if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_PLUGIN]
    qpos, qvel, cables, force = [], [], [], []
    for step in range(args.steps + 1):
        if step:
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        cables.append([data.sensordata[model.sensor_adr[i]:model.sensor_adr[i] + 8].copy() for i in sensors])
        force.append((data.qfrc_passive + data.qfrc_actuator).copy())
    np.savez(args.reference_output, qpos=qpos, qvel=qvel, cables=cables, force=force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "cable_plugin_demos/15_cpp_plugin_surface_single_pulley.xml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--control", type=float, default=0.01)
    parser.add_argument("--actuator", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("trajectory", "forces"), default="trajectory", help="Compare free trajectories, or forces at the same reference poses/velocities")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("steps must be nonnegative")
    if args.reference_output:
        record_reference(args)
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    from mujoco_cable_warp import CableSimulation

    with tempfile.TemporaryDirectory(prefix="mujocable-reference-") as directory:
        reference = Path(directory) / "reference.npz"
        subprocess.run([
            # Preserve the virtualenv executable symlink: resolving it would
            # invoke the base interpreter and lose the reference environment.
            str(args.reference_python.absolute()), str(Path(__file__).resolve()),
            "--plugin", str(args.plugin.resolve()), "--model", str(args.model.resolve()),
            "--steps", str(args.steps), "--control", str(args.control), "--actuator", str(args.actuator),
            "--reference-output", str(reference),
        ], check=True)
        with np.load(reference) as cpu:
            reference_data = {name: cpu[name].copy() for name in cpu.files}
    sim = CableSimulation(args.model, device=args.device)
    controls = np.zeros(sim.cpu_model.nu, dtype=np.float32)
    if sim.cpu_model.nu:
        controls[args.actuator] = args.control
    sim.set_control(controls)
    sim.forward()
    qpos, qvel, cables, force = [], [], [], []
    # Match plugin sensor order, including models with more than one sensor
    # attached to a single cable instance.
    sensor_order = sorted((sensor, cable) for cable, spec in enumerate(sim.loaded.cables) for sensor in spec.sensor_ids)
    if not sensor_order:
        parser.error("comparison requires at least one cable sensor")
    for step in range(args.steps + 1):
        if args.mode == "forces":
            # mj_step diagnostics describe its input state; controls/history
            # still advance normally, but integration drift is removed.
            sample = max(0, step - 1)
            sim.data.qpos.assign(reference_data["qpos"][sample:sample + 1].astype(np.float32))
            sim.data.qvel.assign(reference_data["qvel"][sample:sample + 1].astype(np.float32))
        if step:
            sim.step()
        else:
            sim.forward()
        qpos.append(sim.data.qpos.numpy()[0].copy())
        qvel.append(sim.data.qvel.numpy()[0].copy())
        values = sim.state.values.numpy()[0]
        cables.append([values[cable, :8].copy() for _, cable in sensor_order])
        force.append(sim.data.qfrc_passive.numpy()[0] + sim.data.qfrc_actuator.numpy()[0])
    sim.check()
    cpu = reference_data
    if args.mode == "trajectory":
        errors = {
            "qpos": float(np.max(np.abs(np.asarray(qpos) - cpu["qpos"]))),
            "qvel": float(np.max(np.abs(np.asarray(qvel) - cpu["qvel"]))),
            "length": float(np.max(np.abs(np.asarray(cables)[:, :, 0] - cpu["cables"][:, :, 0]))),
            "tension": float(np.max(np.abs(np.asarray(cables)[:, :, 5] - cpu["cables"][:, :, 5]))),
        }
        tolerances = {"qpos": 5e-5, "qvel": 5e-4, "length": 2e-5, "tension": 5e-3}
    else:
        errors = {
            "length": float(np.max(np.abs(np.asarray(cables)[:, :, 0] - cpu["cables"][:, :, 0]))),
            "tension": float(np.max(np.abs(np.asarray(cables)[:, :, 5] - cpu["cables"][:, :, 5]))),
            "generalized_force": float(np.max(np.abs(np.asarray(force) - cpu["force"]))),
        }
        tolerances = {"length": 2e-5, "tension": 5e-3, "generalized_force": 5e-3}
    report = {
        "model": str(args.model.resolve()), "steps": args.steps, "device": str(sim.device), "mode": args.mode,
        "maximum_absolute_error": errors, "absolute_tolerance": tolerances,
        "pass": all(np.isfinite(error) and error <= tolerances[name] for name, error in errors.items()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
