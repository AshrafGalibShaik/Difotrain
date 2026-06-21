"""Retargeting: map human kinematics (e.g. MediaPipe pose) into a robot's action
or joint space. This is what turns cheap human-video demonstrations into
trainable data for a given embodiment.
"""

from .base import Retargeter, LinearRetargeter

__all__ = ["Retargeter", "LinearRetargeter"]
