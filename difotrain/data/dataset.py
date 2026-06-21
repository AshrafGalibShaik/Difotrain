"""On-disk episode dataset.

Layout (directory based, à la LeRobotDataset):

    <root>/
        meta.json            # dataset-level info + per-episode index
        episodes/
            ep_000000.npz    # observations, actions
            ...

``.npz`` keeps numeric arrays compact; ``meta.json`` keeps the human-readable
index (instruction, length, success, source). Large image observations would be
stored as encoded video alongside; the interface is the same.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np

from ..core.episode import Episode, EpisodeMeta


class EpisodeDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self._index: List[dict] = []
        if (self.root / "meta.json").exists():
            self._load_index()

    # Index -------------------------------------------------------------
    def _load_index(self) -> None:
        with open(self.root / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._index = meta.get("episodes", [])

    def _save_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": "difotrain-episode-dataset-v1",
            "num_episodes": len(self._index),
            "episodes": self._index,
        }
        with open(self.root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Writing -----------------------------------------------------------
    def add(self, episode: Episode) -> int:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        ep_id = len(self._index)
        fname = f"ep_{ep_id:06d}.npz"
        np.savez_compressed(
            self.episodes_dir / fname,
            observations=episode.observations,
            actions=episode.actions,
        )
        self._index.append(
            {
                "id": ep_id,
                "file": f"episodes/{fname}",
                "length": len(episode),
                "meta": episode.meta.to_dict(),
            }
        )
        self._save_index()
        return ep_id

    def extend(self, episodes: Iterable[Episode]) -> None:
        for ep in episodes:
            self.add(ep)

    # Reading -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Episode:
        entry = self._index[idx]
        data = np.load(self.root / entry["file"])
        return Episode(
            observations=data["observations"],
            actions=data["actions"],
            meta=EpisodeMeta.from_dict(entry["meta"]),
        )

    def __iter__(self) -> Iterator[Episode]:
        for i in range(len(self)):
            yield self[i]

    @property
    def instructions(self) -> List[str]:
        return [e["meta"].get("instruction", "") for e in self._index]

    def stacked(self) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """Return all transitions flattened: (obs[N,D], act[N,A], instr[N])."""
        obs, act, instr = [], [], []
        for ep in self:
            obs.append(ep.observations)
            act.append(ep.actions)
            instr.extend([ep.instruction] * len(ep))
        if not obs:
            return np.zeros((0, 0)), np.zeros((0, 0)), []
        return np.concatenate(obs), np.concatenate(act), instr
