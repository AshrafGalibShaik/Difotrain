# Python API

The CLI is a thin wrapper over a plain Python API. There is no custom
DSL — you write ordinary Python against a small, consistent set of classes.

## Imports at a glance

```python
from difotrain.core.episode import Episode, EpisodeMeta
from difotrain.core.spaces import Box
from difotrain.core.language import LanguageEncoder
from difotrain.core.registry import registry

from difotrain.embodiment.base import Robot, RobotSpec
from difotrain.embodiment.sim.planar_arm import PlanarArm
from difotrain.embodiment.retarget.base import LinearRetargeter

from difotrain.data.dataset import EpisodeDataset
from difotrain.data.normalize import Normalizer
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.data.sources.synthetic import SyntheticSource
from difotrain.data.sources.human_video import HumanVideoSource

from difotrain.policy.base import Policy
from difotrain.policy.native.mlp_vla import MLPVLAPolicy

from difotrain.train.trainer import BCTrainer, TrainConfig, train_policy
from difotrain.eval.evaluator import evaluate_reaching, EvalResult
from difotrain.deploy.runner import DeployRunner, RolloutLog
from difotrain.deploy.safety import SafetyLayer
```

## Datasets

```python
ds = EpisodeDataset("data/reach")        # create or open a dataset directory
ds.add(episode)                          # append one Episode
ds.extend(list_of_episodes)              # append many
len(ds)                                  # number of episodes
ds[0]                                    # load episode 0
for ep in ds: ...                        # iterate
obs, act, instr = ds.stacked()           # flatten to transitions for training
```

## Data sources

Every source yields `Episode`s:

```python
source = ScriptedTeleopSource(max_steps=40, seed=0)
for ep in source.collect(200):
    ds.add(ep)
```

## Training

The high-level helper:

```python
from difotrain.train.trainer import train_policy, TrainConfig

policy = train_policy(
    ds,
    TrainConfig(epochs=150, batch_size=64, lr=1e-3, hidden=128,
                normalize=True, seed=0, device="cpu"),
    verbose=True,
)
```

Or the trainer object directly, with a callback per epoch:

```python
from difotrain.train.trainer import BCTrainer, TrainConfig

trainer = BCTrainer(TrainConfig(epochs=200, device="cuda"))
policy = trainer.fit(ds, on_epoch=lambda e, loss: print(e, loss))
```

`TrainConfig` fields: `epochs`, `batch_size`, `lr`, `hidden`, `normalize`,
`seed`, `device`.

## Policies

```python
action = policy.predict(observation, "reach to the left")   # core contract
policy.save("runs/policy.pt")
policy = MLPVLAPolicy.load("runs/policy.pt")                 # or device="cuda"
```

Constructing one manually (the trainer normally does this for you):

```python
from difotrain.core.language import LanguageEncoder
from difotrain.policy.native.mlp_vla import MLPVLAPolicy

enc = LanguageEncoder.fit(["reach to the up", "reach to the down"])
policy = MLPVLAPolicy(obs_dim=4, act_dim=2, language_encoder=enc, hidden=128)
```

## Evaluation

```python
result = evaluate_reaching(policy, success_tol=0.15)
print(result.success_rate, result.mean_final_error)
print(result)               # pretty per-target breakdown
```

## Deployment

```python
robot = PlanarArm()
runner = DeployRunner(robot, policy, SafetyLayer(robot.spec, max_delta=0.5))
log = runner.run("reach to the up", max_steps=60)

# Close the feedback loop: log the rollout back into the dataset.
ds.add(log.to_episode(instruction="reach to the up", robot_name=robot.spec.name))
```

## Robots directly

```python
robot = PlanarArm()
obs = robot.reset(instruction="reach to the up", seed=0)
obs = robot.apply_action([0.5, -0.2])     # actions clipped to the action space
ee = robot.end_effector()                  # PlanarArm helper
```

## The registry

```python
from difotrain.core.registry import registry

registry.robots      # {'planar_arm': <factory>}
registry.sources     # {'scripted_teleop': ..., 'synthetic': ...}
robot  = registry.make_robot("planar_arm")
source = registry.make_source("scripted_teleop", max_steps=40)
```

See [Extending DifoTrain](extending.md) to register your own components.
