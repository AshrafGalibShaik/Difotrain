"""Behavior-cloning trainer.

Fits an :class:`MLPVLAPolicy` to map (observation, instruction) -> action by
minimizing MSE against demonstrated actions. Handles language-vocabulary fitting
and observation/action normalization, then returns a ready-to-deploy policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..core.language import LanguageEncoder
from ..data.dataset import EpisodeDataset
from ..data.normalize import Normalizer
from ..policy.native.mlp_vla import MLPVLAPolicy


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    hidden: int = 128
    normalize: bool = True
    seed: int = 0
    device: str = "cpu"


class BCTrainer:
    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()

    def fit(
        self,
        dataset: EpisodeDataset,
        on_epoch: Optional[Callable[[int, float], None]] = None,
    ) -> MLPVLAPolicy:
        cfg = self.config
        torch.manual_seed(cfg.seed)

        obs, act, instr = dataset.stacked()
        if len(obs) == 0:
            raise ValueError("dataset is empty; collect demonstrations first")

        lang = LanguageEncoder.fit(instr)
        obs_norm = Normalizer.fit(obs) if cfg.normalize else None
        act_norm = Normalizer.fit(act) if cfg.normalize else None

        policy = MLPVLAPolicy(
            obs_dim=obs.shape[1],
            act_dim=act.shape[1],
            language_encoder=lang,
            hidden=cfg.hidden,
            obs_norm=obs_norm,
            act_norm=act_norm,
            device=cfg.device,
        )

        # Pre-encode features once (small toy datasets).
        obs_in = obs_norm.normalize(obs) if obs_norm else obs
        lang_feats = np.stack([lang.encode(t) for t in instr])
        x = np.concatenate([obs_in, lang_feats], axis=1).astype(np.float32)
        y = act_norm.normalize(act) if act_norm else act
        y = y.astype(np.float32)

        device = torch.device(cfg.device)
        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)

        optimizer = torch.optim.Adam(policy.model.parameters(), lr=cfg.lr)
        criterion = nn.MSELoss()
        n = len(x_t)
        rng = np.random.default_rng(cfg.seed)

        policy.model.train()
        last_loss = float("nan")
        for epoch in range(cfg.epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                bi = torch.from_numpy(idx).to(device)
                optimizer.zero_grad()
                pred = policy.model(x_t[bi])
                loss = criterion(pred, y_t[bi])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(idx)
            last_loss = epoch_loss / n
            if on_epoch is not None:
                on_epoch(epoch + 1, last_loss)

        policy._final_loss = last_loss  # type: ignore[attr-defined]
        return policy


def train_policy(
    dataset: EpisodeDataset,
    config: TrainConfig | None = None,
    verbose: bool = False,
) -> MLPVLAPolicy:
    def _log(epoch: int, loss: float) -> None:
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:4d}  loss {loss:.6f}")

    return BCTrainer(config).fit(dataset, on_epoch=_log)
