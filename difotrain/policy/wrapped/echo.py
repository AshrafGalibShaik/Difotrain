"""Reference wrapper for an external VLA.

A real adapter would, in ``predict``, render the observation into the format the
backbone expects (images + text), call the model, and decode its action tokens
back into the robot's action space. ``EchoVLAPolicy`` skips the heavy model and
returns a fixed/zero action, but demonstrates exactly where a wrapped OpenVLA /
Octo / π0 / ACT model slots in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..base import Policy


class EchoVLAPolicy(Policy):
    def __init__(self, act_dim: int, gain: float = 0.0):
        self.act_dim = act_dim
        self.gain = gain

    def predict(self, observation: np.ndarray, instruction: str = "") -> np.ndarray:
        # A genuine wrapper would call the backbone here.
        return np.zeros(self.act_dim, dtype=np.float32) + self.gain

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"act_dim": self.act_dim, "gain": self.gain}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "EchoVLAPolicy":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(act_dim=d["act_dim"], gain=d.get("gain", 0.0))
