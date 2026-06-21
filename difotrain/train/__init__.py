"""Training: behavior cloning of a :class:`Policy` from an ``EpisodeDataset``."""

from .trainer import BCTrainer, TrainConfig, train_policy

__all__ = ["BCTrainer", "TrainConfig", "train_policy"]
