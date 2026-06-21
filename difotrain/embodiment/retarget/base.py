"""Retargeter interface plus a simple linear/affine implementation.

A retargeter consumes a human feature vector (selected joint angles/positions)
and produces a vector in the *target robot's* action or joint space. Real
deployments will need calibrated, often learned, retargeters; the linear one is
a working baseline and a reference for the interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..base import RobotSpec


class Retargeter(ABC):
    @abstractmethod
    def retarget(self, human_features: np.ndarray) -> np.ndarray:
        """Map a human feature vector to the robot space."""


class LinearRetargeter(Retargeter):
    """``y = clip(W @ x + b)`` into the robot action space.

    With ``W=None`` it falls back to identity / truncation / zero-padding so it
    works out of the box for quick experiments.
    """

    def __init__(
        self,
        spec: RobotSpec,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ):
        self.spec = spec
        self.weight = weight
        self.bias = bias
        self.scale = scale

    def retarget(self, human_features: np.ndarray) -> np.ndarray:
        x = np.asarray(human_features, dtype=np.float32).flatten()
        out_dim = self.spec.action_space.dim
        if self.weight is not None:
            y = self.weight @ x
            if self.bias is not None:
                y = y + self.bias
        else:
            y = np.zeros(out_dim, dtype=np.float32)
            n = min(out_dim, x.shape[0])
            y[:n] = x[:n]
        y = y * self.scale
        return self.spec.action_space.clip(y)
