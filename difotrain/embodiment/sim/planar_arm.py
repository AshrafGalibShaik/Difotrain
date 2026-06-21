"""A dependency-free 2-link planar arm.

This is the reference embodiment: it needs no GPU, no webcam and no physics
engine, so the whole collect -> train -> eval -> deploy loop is runnable in CI in
under a second. It is deliberately a real (if tiny) control problem:

* state            : joint angles ``q = (q1, q2)``
* observation      : ``[q1, q2, ee_x, ee_y]`` (proprioception only)
* action           : joint angular velocities, integrated each step
* task             : reach a *named* target. The target is conveyed only through
                     the language instruction, so a policy must ground language
                     into action -- exactly the VLA learning signal we want.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ...core.registry import register_robot
from ...core.spaces import Box
from ..base import Robot, RobotSpec

L1 = 1.0
L2 = 1.0
_TARGET_RADIUS = 1.2

# Named reach targets. The instruction names one of these; nothing about the
# target appears in the observation, so language fully determines the goal.
NAMED_TARGETS: Dict[str, float] = {
    "right": 0.0,
    "upper right": 45.0,
    "up": 90.0,
    "upper left": 135.0,
    "left": 180.0,
    "lower left": 225.0,
    "down": 270.0,
    "lower right": 315.0,
}


def target_xy(name: str) -> Tuple[float, float]:
    deg = NAMED_TARGETS[name]
    rad = math.radians(deg)
    return _TARGET_RADIUS * math.cos(rad), _TARGET_RADIUS * math.sin(rad)


def instruction_for(name: str) -> str:
    return f"reach to the {name}"


def parse_instruction(instruction: str) -> Optional[str]:
    """Best-effort map an instruction string back to a known target name."""
    text = instruction.lower()
    # Longest names first so "upper right" wins over "right".
    for name in sorted(NAMED_TARGETS, key=len, reverse=True):
        if name in text:
            return name
    return None


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    q1, q2 = float(q[0]), float(q[1])
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    return np.array([x, y], dtype=np.float32)


def inverse_kinematics(x: float, y: float, elbow_up: bool = True) -> np.ndarray:
    """Analytic 2-link IK. Returns joint angles reaching ``(x, y)``."""
    r2 = x * x + y * y
    cos_q2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = math.acos(cos_q2)
    if not elbow_up:
        q2 = -q2
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([q1, q2], dtype=np.float32)


@register_robot("planar_arm")
class PlanarArm(Robot):
    def __init__(self, control_hz: float = 20.0, max_vel: float = 2.5):
        joint_limit = math.pi
        self.spec = RobotSpec(
            name="planar_arm",
            dof=2,
            observation_space=Box(
                low=np.array([-joint_limit, -joint_limit, -2.0, -2.0]),
                high=np.array([joint_limit, joint_limit, 2.0, 2.0]),
            ),
            action_space=Box(
                low=np.array([-max_vel, -max_vel]),
                high=np.array([max_vel, max_vel]),
            ),
            control_hz=control_hz,
            metadata={"links": [L1, L2], "targets": list(NAMED_TARGETS)},
        )
        self.max_vel = max_vel
        self.q = np.zeros(2, dtype=np.float32)
        self._rng = np.random.default_rng()

    # Robot API ---------------------------------------------------------
    def reset(self, *, instruction: str = "", seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Start from a small random pose so demos are not all identical.
        self.q = self._rng.uniform(-0.3, 0.3, size=2).astype(np.float32)
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        ee = forward_kinematics(self.q)
        return np.concatenate([self.q, ee]).astype(np.float32)

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        action = self.spec.action_space.clip(action)
        self.q = self.q + action * self.spec.dt
        limit = self.spec.observation_space.high[:2]
        self.q = np.clip(self.q, -limit, limit).astype(np.float32)
        return self.get_observation()

    # Helpers used by the expert / evaluator ----------------------------
    def end_effector(self) -> np.ndarray:
        return forward_kinematics(self.q)

    def expert_action(self, target_name: str, gain: float = 4.0) -> np.ndarray:
        """A proportional expert that drives joints toward the IK solution."""
        tx, ty = target_xy(target_name)
        q_goal = inverse_kinematics(tx, ty)
        action = gain * (q_goal - self.q)
        return self.spec.action_space.clip(action)
