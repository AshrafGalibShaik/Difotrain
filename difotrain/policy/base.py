"""The ``Policy`` interface.

A policy maps an observation plus a language instruction to an action. Native
models (implemented here) and wrapped third-party VLAs (OpenVLA / Octo / π0 /
ACT adapters) both implement this, so the trainer, evaluator and deploy runner
are model-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Policy(ABC):
    @abstractmethod
    def predict(self, observation: np.ndarray, instruction: str = "") -> np.ndarray:
        """Return an action for a single observation + instruction."""

    def reset(self) -> None:  # pragma: no cover - optional for stateful policies
        """Reset any internal state between episodes."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Policy":
        ...
