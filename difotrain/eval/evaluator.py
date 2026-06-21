"""Sim evaluation harness for the planar-arm reaching task.

For each named target it issues the language instruction, rolls the policy out
on the robot, and checks whether the end effector reaches the commanded target.
Because the target is conveyed *only* through language, success measures genuine
language grounding, not memorized motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..embodiment.sim.planar_arm import (
    NAMED_TARGETS,
    PlanarArm,
    instruction_for,
    target_xy,
)
from ..policy.base import Policy


@dataclass
class EvalResult:
    success_rate: float
    mean_final_error: float
    per_target: Dict[str, dict] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"success_rate     : {self.success_rate:.2%}",
            f"mean_final_error : {self.mean_final_error:.4f}",
        ]
        for name, r in self.per_target.items():
            ok = "OK " if r["success"] else "   "
            lines.append(f"  [{ok}] {name:<12} err={r['error']:.4f}")
        return "\n".join(lines)


def evaluate_reaching(
    policy: Policy,
    robot: Optional[PlanarArm] = None,
    targets: Optional[List[str]] = None,
    max_steps: int = 60,
    success_tol: float = 0.15,
    seed: int = 123,
) -> EvalResult:
    robot = robot or PlanarArm()
    targets = targets or list(NAMED_TARGETS)

    errors: List[float] = []
    successes: List[bool] = []
    per_target: Dict[str, dict] = {}

    for i, name in enumerate(targets):
        instruction = instruction_for(name)
        obs = robot.reset(instruction=instruction, seed=seed + i)
        policy.reset()
        for _ in range(max_steps):
            action = policy.predict(obs, instruction)
            obs = robot.apply_action(action)

        ee = robot.end_effector()
        tx, ty = target_xy(name)
        err = float(np.linalg.norm(ee - np.array([tx, ty])))
        ok = err < success_tol
        errors.append(err)
        successes.append(ok)
        per_target[name] = {"error": err, "success": ok}

    return EvalResult(
        success_rate=float(np.mean(successes)) if successes else 0.0,
        mean_final_error=float(np.mean(errors)) if errors else float("nan"),
        per_target=per_target,
    )
