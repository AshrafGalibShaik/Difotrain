"""Minimal observation / action space definitions.

We deliberately avoid a hard dependency on ``gymnasium`` so the framework stays
lightweight; ``Box`` mirrors the subset of the Gym ``Box`` API we rely on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class Space:
    """Base class for all spaces."""

    shape: Tuple[int, ...]

    def contains(self, x: np.ndarray) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Box(Space):
    """A bounded, continuous box space ``[low, high]``."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        if self.low.shape != self.high.shape:
            raise ValueError(
                f"low/high shape mismatch: {self.low.shape} vs {self.high.shape}"
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.low.shape

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return (
            x.shape == self.shape
            and bool(np.all(x >= self.low - 1e-6))
            and bool(np.all(x <= self.high + 1e-6))
        )

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float32), self.low, self.high)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(np.float32)

    def __repr__(self) -> str:
        return f"Box(shape={self.shape}, low={self.low.min():.3g}, high={self.high.max():.3g})"
