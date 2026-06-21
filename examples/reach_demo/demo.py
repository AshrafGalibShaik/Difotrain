"""DifoTrain demo: teach a robot arm to reach where you tell it.

Runs the full DifoTrain lifecycle end to end and *visualizes* the result:

    collect demonstrations -> train a language-conditioned policy ->
    evaluate -> deploy for each instruction and draw the arm in the terminal.

Run it:

    python examples/reach_demo/demo.py

No GPU, webcam, or robot required -- everything uses DifoTrain's built-in
simulated 2-link arm.
"""
from __future__ import annotations

import numpy as np

from difotrain.data.dataset import EpisodeDataset
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.deploy.runner import DeployRunner
from difotrain.deploy.safety import SafetyLayer
from difotrain.embodiment.sim.planar_arm import (
    NAMED_TARGETS,
    PlanarArm,
    forward_kinematics,
    instruction_for,
    target_xy,
)
from difotrain.eval.evaluator import evaluate_reaching
from difotrain.train.trainer import TrainConfig, train_policy

GRID = 17          # ASCII canvas size (cells per side)
SCALE = 2.2        # world coordinate range mapped onto the grid


def _to_cell(x: float, y: float) -> tuple[int, int]:
    """Map world coords (-SCALE..SCALE) to grid row/col (y is up)."""
    col = int(round((x + SCALE) / (2 * SCALE) * (GRID - 1)))
    row = int(round((SCALE - y) / (2 * SCALE) * (GRID - 1)))
    return max(0, min(GRID - 1, row)), max(0, min(GRID - 1, col))


def render(q: np.ndarray, target: tuple[float, float], instruction: str) -> str:
    """Draw the arm: B=base, +=elbow, O=end effector, X=goal."""
    canvas = [["." for _ in range(GRID)] for _ in range(GRID)]

    base = (0.0, 0.0)
    elbow = (np.cos(q[0]), np.sin(q[0]))            # L1 = 1
    ee = forward_kinematics(q)

    # Draw the two links as straight lines of '*'.
    for a, b in [(base, elbow), (elbow, (ee[0], ee[1]))]:
        for t in np.linspace(0, 1, 24):
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            r, c = _to_cell(x, y)
            if canvas[r][c] == ".":
                canvas[r][c] = "*"

    for (x, y), ch in [(target, "X"), (base, "B"), (tuple(elbow), "+"), (tuple(ee), "O")]:
        r, c = _to_cell(x, y)
        canvas[r][c] = ch

    err = float(np.linalg.norm(np.array(ee) - np.array(target)))
    hit = "REACHED" if err < 0.15 else "missed "
    lines = ["  " + " ".join(row) for row in canvas]
    header = f'  "{instruction}"   [{hit}]  error={err:.3f}'
    legend = "  B=base  +=elbow  O=hand  X=goal"
    return "\n".join([header, *lines, legend])


def main() -> None:
    print("=" * 56)
    print("  DifoTrain demo - language-conditioned robot reaching")
    print("=" * 56)

    # 1. COLLECT --------------------------------------------------------
    print("\n[1/4] Collecting 200 demonstrations from the expert...")
    ds = EpisodeDataset("examples/reach_demo/_data")
    if len(ds) == 0:
        ds.extend(ScriptedTeleopSource(seed=0).collect(200))
    print(f"      {len(ds)} episodes ready.")

    # 2. TRAIN ----------------------------------------------------------
    print("\n[2/4] Training the policy (behavior cloning)...")
    policy = train_policy(ds, TrainConfig(epochs=120, seed=0), verbose=True)

    # 3. EVAL -----------------------------------------------------------
    print("\n[3/4] Evaluating on all targets:")
    result = evaluate_reaching(policy)
    print(f"      success rate     : {result.success_rate:.0%}")
    print(f"      mean final error : {result.mean_final_error:.3f}")

    # 4. DEPLOY + VISUALIZE --------------------------------------------
    print("\n[4/4] Deploying the policy for each instruction:\n")
    robot = PlanarArm()
    runner = DeployRunner(robot, policy, SafetyLayer(robot.spec))
    for name in ["up", "down", "left", "right", "upper right", "lower left"]:
        instruction = instruction_for(name)
        runner.run(instruction, max_steps=60, seed=7)
        print(render(robot.q, target_xy(name), instruction))
        print()

    print("Done. Try editing the instruction list above and re-running!")


if __name__ == "__main__":
    main()
