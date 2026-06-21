"""Simulation backends. The dependency-free :class:`PlanarArm` is the reference
embodiment used by tests and examples; heavier backends (PyBullet / MuJoCo /
Isaac) plug in here behind the same ``Robot`` API.
"""

from .planar_arm import PlanarArm, NAMED_TARGETS

# PyBulletArm registers itself on import; pybullet itself is imported lazily
# inside its constructor, so this import is safe without the optional dependency.
from .pybullet_arm import PyBulletArm

__all__ = ["PlanarArm", "NAMED_TARGETS", "PyBulletArm"]
