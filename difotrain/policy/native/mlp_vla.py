"""A small, self-contained VLA-style policy.

It conditions on *both* the observation and a language-instruction embedding,
concatenates them, and regresses an action with an MLP. It is intentionally a
real (if tiny) vision-language-action model in miniature: language is a genuine
input the policy must use to pick the right behavior. Swap the
:class:`LanguageEncoder` for a transformer text encoder and add an image encoder
to scale this up; the :class:`Policy` contract is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ...core.language import LanguageEncoder
from ...data.normalize import Normalizer
from ..base import Policy


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPVLAPolicy(Policy):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        language_encoder: LanguageEncoder,
        hidden: int = 128,
        obs_norm: Optional[Normalizer] = None,
        act_norm: Optional[Normalizer] = None,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.language_encoder = language_encoder
        self.hidden = hidden
        self.lang_dim = language_encoder.dim
        self.device = torch.device(device)
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.model = _MLP(obs_dim + self.lang_dim, act_dim, hidden).to(self.device)

    # Internals ---------------------------------------------------------
    def _featurize(self, observation: np.ndarray, instruction: str) -> torch.Tensor:
        obs = np.asarray(observation, dtype=np.float32)
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize(obs)
        lang = self.language_encoder.encode(instruction)
        vec = np.concatenate([obs, lang]).astype(np.float32)
        return torch.from_numpy(vec).to(self.device)

    def forward_batch(self, obs: np.ndarray, instructions) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize(obs)
        lang = np.stack([self.language_encoder.encode(t) for t in instructions])
        x = np.concatenate([obs, lang], axis=1).astype(np.float32)
        return self.model(torch.from_numpy(x).to(self.device))

    # Policy API --------------------------------------------------------
    def predict(self, observation: np.ndarray, instruction: str = "") -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._featurize(observation, instruction).unsqueeze(0))
        action = out.squeeze(0).cpu().numpy()
        if self.act_norm is not None:
            action = self.act_norm.denormalize(action)
        return action.astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": self.hidden,
            "language": self.language_encoder.to_dict(),
            "obs_norm": self.obs_norm.to_dict() if self.obs_norm else None,
            "act_norm": self.act_norm.to_dict() if self.act_norm else None,
        }
        torch.save(
            {"state_dict": self.model.state_dict(), "config": config}, path
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MLPVLAPolicy":
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        cfg = ckpt["config"]
        policy = cls(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            language_encoder=LanguageEncoder.from_dict(cfg["language"]),
            hidden=cfg["hidden"],
            obs_norm=Normalizer.from_dict(cfg["obs_norm"]) if cfg["obs_norm"] else None,
            act_norm=Normalizer.from_dict(cfg["act_norm"]) if cfg["act_norm"] else None,
            device=device,
        )
        policy.model.load_state_dict(ckpt["state_dict"])
        return policy
