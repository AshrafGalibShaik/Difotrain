"""Human-video source: learn from RGB video of a person.

This is the cheap, scalable branch of the hybrid data strategy. Human pose is
estimated with MediaPipe, reduced to a feature vector (selected joint angles),
and a :class:`Retargeter` maps those features into the target robot's action
space, producing trainable demonstrations without any robot or teleop rig.

MediaPipe / OpenCV are imported lazily so the framework (and its tests) work
without them installed. An offline mode reads an already-recorded trajectory
JSON, which keeps this source fully testable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from ...core.episode import Episode, EpisodeMeta
from ...core.registry import register_source
from ...embodiment.base import RobotSpec
from ...embodiment.retarget.base import LinearRetargeter, Retargeter
from .base import DataSource

# Joints whose angles we turn into the human feature vector.
_ANGLE_TRIPLETS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
}


def _angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-8:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return float(math.acos(cos))


def pose_to_features(joints: dict) -> np.ndarray:
    feats: List[float] = []
    for _, (a, b, c) in _ANGLE_TRIPLETS.items():
        if a in joints and b in joints and c in joints:
            feats.append(_angle(joints[a], joints[b], joints[c]))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


@register_source("human_video")
class HumanVideoSource(DataSource):
    name = "human_video"

    def __init__(
        self,
        robot_spec: RobotSpec,
        retargeter: Optional[Retargeter] = None,
        instruction: str = "imitate the human",
        trajectory_json: Optional[str] = None,
    ):
        self.robot_spec = robot_spec
        self.retargeter = retargeter or LinearRetargeter(robot_spec)
        self.instruction = instruction
        self.trajectory_json = trajectory_json

    # Offline: build an episode from a recorded trajectory JSON ----------
    def episode_from_trajectory(self, raw: List[dict]) -> Episode:
        observations, actions = [], []
        for frame in raw:
            feats = pose_to_features(frame.get("joints", {}))
            action = self.retargeter.retarget(feats)
            observations.append(feats)
            actions.append(action)
        meta = EpisodeMeta(
            instruction=self.instruction,
            source=self.name,
            robot=self.robot_spec.name,
        )
        return Episode(np.array(observations), np.array(actions), meta)

    def collect(self, num_episodes: int) -> Iterator[Episode]:
        if self.trajectory_json:
            raw = json.loads(Path(self.trajectory_json).read_text(encoding="utf-8"))
            for _ in range(num_episodes):
                yield self.episode_from_trajectory(raw)
        else:  # pragma: no cover - requires a webcam
            for _ in range(num_episodes):
                yield self._capture_live()

    def _capture_live(self) -> Episode:  # pragma: no cover - requires a webcam
        from ...capture.record_pose import capture_trajectory

        raw = capture_trajectory()
        return self.episode_from_trajectory(raw)
