"""Per-dimension normalization stats.

Action/observation normalization is one of the silent make-or-break details of
VLA training, so it is a first-class, serializable object that travels with the
checkpoint.
"""
from __future__ import annotations

import numpy as np


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normalizer":
        data = np.asarray(data, dtype=np.float32)
        if data.size == 0:
            return cls(np.zeros(1), np.ones(1))
        return cls(data.mean(axis=0), data.std(axis=0))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["mean"]), np.array(d["std"]))
