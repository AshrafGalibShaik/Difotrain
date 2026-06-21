"""Embodiment layer: one ``Robot`` API shared by simulation and real hardware."""

from .base import Robot, RobotSpec
from .sim.planar_arm import PlanarArm, NAMED_TARGETS

__all__ = ["Robot", "RobotSpec", "PlanarArm", "NAMED_TARGETS"]
