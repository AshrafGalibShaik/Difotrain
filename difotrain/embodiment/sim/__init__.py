"""Simulation backends. The dependency-free :class:`PlanarArm` is the reference
embodiment used by tests and examples; heavier backends (PyBullet / MuJoCo /
Isaac) plug in here behind the same ``Robot`` API.
"""

from .planar_arm import PlanarArm, NAMED_TARGETS

__all__ = ["PlanarArm", "NAMED_TARGETS"]
