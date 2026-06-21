"""Scripted-teleop source.

Stands in for a human teleoperating the robot: a scripted expert drives the
:class:`PlanarArm` toward a randomly chosen named target and we record the
proprioception/action stream as a demonstration. This is the high-quality
teleop branch of the hybrid data strategy; a real teleop source would swap the
expert for live joystick/VR/leader-arm input behind the same interface.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from ...core.episode import Episode, EpisodeMeta
from ...core.registry import register_source
from ...embodiment.sim.planar_arm import (
    NAMED_TARGETS,
    PlanarArm,
    instruction_for,
    target_xy,
)
from .base import DataSource


@register_source("scripted_teleop")
class ScriptedTeleopSource(DataSource):
    name = "scripted_teleop"

    def __init__(
        self,
        robot: Optional[PlanarArm] = None,
        max_steps: int = 40,
        targets: Optional[List[str]] = None,
        success_tol: float = 0.15,
        seed: int = 0,
    ):
        self.robot = robot or PlanarArm()
        self.max_steps = max_steps
        self.targets = targets or list(NAMED_TARGETS)
        self.success_tol = success_tol
        self._rng = np.random.default_rng(seed)

    def collect(self, num_episodes: int) -> Iterator[Episode]:
        for i in range(num_episodes):
            target = self.targets[int(self._rng.integers(len(self.targets)))]
            yield self._rollout(target, seed=int(self._rng.integers(1 << 31)))

    def _rollout(self, target: str, seed: int) -> Episode:
        obs = self.robot.reset(instruction=instruction_for(target), seed=seed)
        observations, actions = [], []
        for _ in range(self.max_steps):
            action = self.robot.expert_action(target)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs = self.robot.apply_action(action)

        ee = self.robot.end_effector()
        tx, ty = target_xy(target)
        success = bool(np.linalg.norm(ee - np.array([tx, ty])) < self.success_tol)
        meta = EpisodeMeta(
            instruction=instruction_for(target),
            source=self.name,
            robot=self.robot.spec.name,
            success=success,
            extra={"target": target},
        )
        return Episode(np.array(observations), np.array(actions), meta)
