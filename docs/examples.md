# Examples

Runnable snippets covering common workflows. Each is self-contained.

## 1. Full pipeline in one script

```python
from difotrain.data.dataset import EpisodeDataset
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.train.trainer import train_policy, TrainConfig
from difotrain.eval.evaluator import evaluate_reaching

ds = EpisodeDataset("data/reach")
ds.extend(ScriptedTeleopSource(seed=0).collect(200))

policy = train_policy(ds, TrainConfig(epochs=150), verbose=True)
policy.save("runs/policy.pt")

print(evaluate_reaching(policy))
```

## 2. Deploy and inspect a rollout

```python
from difotrain.policy.native.mlp_vla import MLPVLAPolicy
from difotrain.embodiment.sim.planar_arm import PlanarArm, target_xy
from difotrain.deploy.runner import DeployRunner
from difotrain.deploy.safety import SafetyLayer
import numpy as np

policy = MLPVLAPolicy.load("runs/policy.pt")
robot = PlanarArm()
runner = DeployRunner(robot, policy, SafetyLayer(robot.spec))

for target in ["up", "down", "left", "right"]:
    runner.run(f"reach to the {target}", max_steps=60, seed=1)
    err = np.linalg.norm(robot.end_effector() - np.array(target_xy(target)))
    print(f"{target:>5}: final error = {err:.3f}")
```

## 3. Close the feedback loop (DAgger-style)

```python
# Log deployment rollouts back into the dataset, then retrain on the larger set.
log = runner.run("reach to the up", max_steps=60)
ds.add(log.to_episode(instruction="reach to the up", robot_name=robot.spec.name))
policy = train_policy(ds, TrainConfig(epochs=150))
```

## 4. Train on a GPU

```python
from difotrain.train.trainer import BCTrainer, TrainConfig

policy = BCTrainer(TrainConfig(epochs=300, device="cuda")).fit(ds)
policy = policy  # already on cuda; predict() handles device internally
```

## 5. Custom training loop with logging

```python
from difotrain.train.trainer import BCTrainer, TrainConfig

losses = []
def log(epoch, loss):
    losses.append(loss)
    if epoch % 25 == 0:
        print(f"epoch {epoch}: {loss:.5f}")

policy = BCTrainer(TrainConfig(epochs=200)).fit(ds, on_epoch=log)
```

## 6. Inspect a dataset

```python
from difotrain.data.dataset import EpisodeDataset

ds = EpisodeDataset("data/reach")
print("episodes:", len(ds))
print("instructions:", set(ds.instructions))

ep = ds[0]
print("length:", len(ep), "obs_dim:", ep.obs_dim, "act_dim:", ep.act_dim)
print("success:", ep.meta.success, "target:", ep.meta.extra.get("target"))
```

## 7. Use a wrapped (external) policy stub

```python
from difotrain.policy.wrapped.echo import EchoVLAPolicy
from difotrain.eval.evaluator import evaluate_reaching

# Reference wrapper -- replace with a real OpenVLA/Octo/pi0/ACT adapter.
policy = EchoVLAPolicy(act_dim=2)
print(evaluate_reaching(policy))   # baseline (does nothing useful yet)
```

## 8. Mix human-video and teleop data

```python
from difotrain.embodiment.sim.planar_arm import PlanarArm
from difotrain.data.sources.human_video import HumanVideoSource
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.data.dataset import EpisodeDataset

arm = PlanarArm()
ds = EpisodeDataset("data/mixed")
ds.extend(HumanVideoSource(arm.spec, trajectory_json="storage/human_trajectory.json").collect(5))
ds.extend(ScriptedTeleopSource(seed=0).collect(200))
```
