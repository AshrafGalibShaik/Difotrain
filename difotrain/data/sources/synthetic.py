"""Synthetic source: random episodes with a known linear obs->action rule.

Useful for unit tests and for sanity-checking that training can fit a known
function, independent of any robot.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from ...core.episode import Episode, EpisodeMeta
from ...core.registry import register_source
from .base import DataSource


@register_source("synthetic")
class SyntheticSource(DataSource):
    name = "synthetic"

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, length: int = 20, seed: int = 0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.length = length
        self._rng = np.random.default_rng(seed)
        self._w = self._rng.normal(size=(act_dim, obs_dim)).astype(np.float32)

    def collect(self, num_episodes: int) -> Iterator[Episode]:
        for _ in range(num_episodes):
            obs = self._rng.normal(size=(self.length, self.obs_dim)).astype(np.float32)
            act = (obs @ self._w.T).astype(np.float32)
            meta = EpisodeMeta(instruction="synthetic", source=self.name)
            yield Episode(obs, act, meta)
