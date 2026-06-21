"""The ``Robot`` interface.

Both simulated and real robots implement this single ABC, so policies, the
evaluator and the deploy runner are completely agnostic to what they are
driving. A "custom robot" is just a subclass that fills in ``spec`` and the
four lifecycle methods, registered via :func:`difotrain.core.register_robot`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..core.spaces import Box


@dataclass
class RobotSpec:
    """Static description of an embodiment."""

    name: str
    dof: int
    observation_space: Box
    action_space: Box
    control_hz: float = 20.0
    urdf: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz


class Robot(ABC):
    """Common interface for every embodiment."""

    spec: RobotSpec

    @abstractmethod
    def reset(self, *, instruction: str = "", seed: Optional[int] = None) -> np.ndarray:
        """Reset to an initial state and return the first observation."""

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the current observation vector."""

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """Apply one action and return the resulting observation."""

    def close(self) -> None:  # pragma: no cover - optional
        """Release any hardware / simulator resources."""

    # Convenience -------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return self.spec.observation_space.dim

    @property
    def act_dim(self) -> int:
        return self.spec.action_space.dim
