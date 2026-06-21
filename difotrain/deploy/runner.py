"""Deploy runner: the real-time inference loop.

Pulls observations from a robot, asks the policy for an action, passes it
through the :class:`SafetyLayer`, and applies it. The same loop drives sim or
real hardware (both are just a ``Robot``). Rollouts can be logged back as
:class:`Episode` objects to close the real-world feedback loop (DAgger-style).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.episode import Episode, EpisodeMeta
from ..embodiment.base import Robot
from ..policy.base import Policy
from .safety import SafetyLayer


@dataclass
class RolloutLog:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)

    def to_episode(self, instruction: str, robot_name: str) -> Episode:
        return Episode(
            np.array(self.observations),
            np.array(self.actions),
            EpisodeMeta(instruction=instruction, source="deploy", robot=robot_name),
        )


class DeployRunner:
    def __init__(
        self,
        robot: Robot,
        policy: Policy,
        safety: Optional[SafetyLayer] = None,
        realtime: bool = False,
    ):
        self.robot = robot
        self.policy = policy
        self.safety = safety or SafetyLayer(robot.spec)
        self.realtime = realtime

    def run(self, instruction: str, max_steps: int = 60, seed: Optional[int] = None) -> RolloutLog:
        obs = self.robot.reset(instruction=instruction, seed=seed)
        self.policy.reset()
        self.safety.reset()
        log = RolloutLog()
        dt = self.robot.spec.dt

        for _ in range(max_steps):
            action = self.policy.predict(obs, instruction)
            action = self.safety.filter(action)
            log.observations.append(np.asarray(obs, dtype=np.float32))
            log.actions.append(np.asarray(action, dtype=np.float32))
            obs = self.robot.apply_action(action)
            if self.realtime:  # pragma: no cover - timing only
                time.sleep(dt)

        return log
