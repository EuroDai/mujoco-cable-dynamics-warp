"""Batched MuJoCable simulation using MuJoCo Warp.

Importing this package does not initialize Warp or require a CUDA device.
"""


def __getattr__(name):
    if name == "CableSimulation":
        from .simulation import CableSimulation

        return CableSimulation
    raise AttributeError(name)


__all__ = ["CableSimulation"]
