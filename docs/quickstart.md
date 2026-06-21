# Quickstart

This walkthrough trains a language-conditioned policy and runs it — end to end —
in about five minutes on a laptop CPU.

## The reference task

A 2-link robot arm must **reach a target that is named only in the language
instruction** (e.g. `"reach to the up"`). The target is *not* part of what the
robot observes, so the policy can only succeed by grounding the instruction into
motion. That makes it a genuine, if miniature, Vision-Language-Action problem.

There are eight named targets: `right`, `upper right`, `up`, `upper left`,
`left`, `lower left`, `down`, `lower right`.

## Option A — the command line

```bash
# 1. Collect 200 demonstrations from the scripted expert.
difotrain collect --out data/reach --episodes 200

# 2. Train a policy on them (behavior cloning).
difotrain train --data data/reach --out runs/policy.pt --epochs 150

# 3. Evaluate task success in simulation.
difotrain eval --policy runs/policy.pt

# 4. Run the policy for a single instruction.
difotrain deploy --policy runs/policy.pt --instruction "reach to the lower left"
```

Typical evaluation output:

```
success_rate     : 87.50%
mean_final_error : 0.0259
  [OK ] right        err=0.0055
  [OK ] up           err=0.0180
  [OK ] down         err=0.0037
  ...
```

## Option B — Python

```python
from difotrain.data.dataset import EpisodeDataset
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.train.trainer import train_policy, TrainConfig
from difotrain.eval.evaluator import evaluate_reaching
from difotrain.deploy.runner import DeployRunner
from difotrain.deploy.safety import SafetyLayer
from difotrain.embodiment.sim.planar_arm import PlanarArm

# 1. Collect
ds = EpisodeDataset("data/reach")
ds.extend(ScriptedTeleopSource(seed=0).collect(200))

# 2. Train
policy = train_policy(ds, TrainConfig(epochs=150))
policy.save("runs/policy.pt")

# 3. Evaluate
print(evaluate_reaching(policy))

# 4. Deploy
robot = PlanarArm()
runner = DeployRunner(robot, policy, SafetyLayer(robot.spec))
runner.run("reach to the up", max_steps=60)
print("final end-effector:", robot.end_effector())
```

## What just happened

| Step | What it did |
|------|-------------|
| Collect | A scripted expert drove the arm to random named targets; each run was saved as an `Episode` (observations + actions + instruction). |
| Train | All transitions were flattened; a language encoder + normalizers were fit; an MLP learned `(obs, instruction) → action` by minimizing MSE. |
| Eval | For every target, the instruction was issued and the policy rolled out; success = end effector reached the commanded point. |
| Deploy | The trained policy drove the robot in a safe inference loop (actions clipped to limits). |

## Next steps

- Understand the design: [Concepts](concepts.md).
- Plug in your own robot or model: [Extending DifoTrain](extending.md).
- Train from webcam video of a human: [Learning from human video](human-video.md).
