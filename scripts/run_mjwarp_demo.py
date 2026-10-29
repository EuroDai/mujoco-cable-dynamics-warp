#!/usr/bin/env python3
"""Run a cable scene on MJWarp from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

if __name__ == "__main__":
    from mujoco_cable_warp.cli import main

    raise SystemExit(main())
