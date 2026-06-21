"""Safety layer applied between policy output and the robot.

On real hardware this is non-negotiable. It clips actions into the robot's
action space and optionally bounds per-step change (rate limiting). Extend with
workspace limits / e-stop hooks for a specific robot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..embodiment.base import RobotSpec


class SafetyLayer:
    def __init__(self, spec: RobotSpec, max_delta: Optional[float] = None):
        self.spec = spec
        self.max_delta = max_delta
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def filter(self, action: np.ndarray) -> np.ndarray:
        action = self.spec.action_space.clip(action)
        if self.max_delta is not None and self._prev is not None:
            delta = np.clip(action - self._prev, -self.max_delta, self.max_delta)
            action = (self._prev + delta).astype(np.float32)
            action = self.spec.action_space.clip(action)
        self._prev = action
        return action
