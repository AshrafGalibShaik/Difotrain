"""``DataSource`` interface.

Teleop, human-video and sim-rollout sources all implement ``collect`` and yield
standardized :class:`~difotrain.core.episode.Episode` objects, so downstream
storage/training never cares where data came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ...core.episode import Episode


class DataSource(ABC):
    name: str = "data_source"

    @abstractmethod
    def collect(self, num_episodes: int) -> Iterator[Episode]:
        """Yield ``num_episodes`` demonstrations."""
