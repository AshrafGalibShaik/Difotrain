"""The standard unit of data exchanged across the framework: an ``Episode``.

An episode is a single demonstration / rollout: a language instruction plus a
time-aligned sequence of observations and the actions taken. This mirrors the
conventions used by RLDS / LeRobotDataset so data stays interoperable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EpisodeMeta:
    instruction: str = ""
    source: str = "unknown"
    robot: str = "unknown"
    success: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "source": self.source,
            "robot": self.robot,
            "success": self.success,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeMeta":
        return cls(
            instruction=d.get("instruction", ""),
            source=d.get("source", "unknown"),
            robot=d.get("robot", "unknown"),
            success=d.get("success"),
            extra=d.get("extra", {}),
        )


@dataclass
class Episode:
    """A demonstration.

    ``observations`` has shape ``[T, obs_dim]`` and ``actions`` has shape
    ``[T, act_dim]`` where ``actions[t]`` is the action taken from
    ``observations[t]``.
    """

    observations: np.ndarray
    actions: np.ndarray
    meta: EpisodeMeta = field(default_factory=EpisodeMeta)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.observations.ndim != 2:
            raise ValueError(
                f"observations must be 2-D [T, obs_dim], got {self.observations.shape}"
            )
        if self.actions.ndim != 2:
            raise ValueError(
                f"actions must be 2-D [T, act_dim], got {self.actions.shape}"
            )
        if len(self.observations) != len(self.actions):
            raise ValueError(
                f"length mismatch: {len(self.observations)} obs vs "
                f"{len(self.actions)} actions"
            )

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def instruction(self) -> str:
        return self.meta.instruction

    @property
    def obs_dim(self) -> int:
        return self.observations.shape[1]

    @property
    def act_dim(self) -> int:
        return self.actions.shape[1]
