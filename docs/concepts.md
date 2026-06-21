# Concepts & architecture

DifoTrain exists to make one loop easy and reusable:

> Collect demonstrations → train a policy to imitate them → evaluate it →
> deploy it on a robot → feed real rollouts back into the data.

This document explains the design that makes every stage swappable.

## The central object: `Episode`

Everything the framework passes around is an **`Episode`** — one demonstration or
rollout:

```python
Episode(
    observations,   # np.ndarray [T, obs_dim]
    actions,        # np.ndarray [T, act_dim]   actions[t] taken from observations[t]
    meta,           # EpisodeMeta(instruction, source, robot, success, extra)
)
```

The shape and meaning are deliberately aligned with RLDS / LeRobotDataset
conventions so data stays interoperable with the wider VLA ecosystem.

## The four interfaces

A component is "first-class" in DifoTrain if it implements one of these abstract
base classes. The trainer, evaluator, and deploy runner are written **only**
against these interfaces, so any conforming implementation works unchanged.

### 1. `Robot` — the embodiment

`difotrain/embodiment/base.py`

```python
class Robot:
    spec: RobotSpec                       # dof, obs/action spaces, control_hz, urdf
    def reset(self, *, instruction="", seed=None) -> obs
    def get_observation(self) -> obs
    def apply_action(self, action) -> obs
    def close(self) -> None               # optional
```

The same interface is implemented by simulated robots and real hardware. A
"custom robot" is just a subclass plus a `RobotSpec`. The reference
implementation, `PlanarArm`, is a dependency-free 2-link arm used by all the
examples and tests.

### 2. `DataSource` — where demonstrations come from

`difotrain/data/sources/base.py`

```python
class DataSource:
    def collect(self, num_episodes) -> Iterator[Episode]
```

Shipped sources:
- `ScriptedTeleopSource` — a scripted expert drives the sim robot (stands in for
  teleoperation; swap the expert for a joystick/VR/leader-arm).
- `HumanVideoSource` — human pose (MediaPipe) retargeted into robot actions.
- `SyntheticSource` — random data with a known rule, for tests.

### 3. `Policy` — the model

`difotrain/policy/base.py`

```python
class Policy:
    def predict(self, observation, instruction="") -> action
    def reset(self) -> None               # optional, for stateful policies
    def save(self, path); load(cls, path)
```

`predict` is the entire VLA contract. Shipped:
- `MLPVLAPolicy` — native model: concatenates the observation with a
  language-instruction embedding and regresses an action.
- `EchoVLAPolicy` — a reference wrapper showing where an external VLA
  (OpenVLA / Octo / π0 / ACT) plugs in.

### 4. `EpisodeDataset` — storage

`difotrain/data/dataset.py`

A directory-based dataset (numeric arrays in `.npz`, an index in `meta.json`).
This is the durable, interoperable boundary between collection and training.

## The plugin registry

`difotrain/core/registry.py`

Robots, data sources, and policies register themselves by name via decorators:

```python
@register_robot("planar_arm")
class PlanarArm(Robot): ...
```

Registered components can then be created by string — which is what lets the CLI
(and future config files) instantiate them without importing your code directly:

```python
from difotrain.core.registry import registry
robot = registry.make_robot("planar_arm")
```

## Language conditioning

`difotrain/core/language.py`

A `LanguageEncoder` turns an instruction string into a fixed-length vector. The
default is a deterministic bag-of-words encoder (zero heavy dependencies), which
is enough for the reference task while still forcing the policy to *use* the
instruction. It implements `encode(text) -> np.ndarray`, so swapping in a
transformer text encoder later is a drop-in change.

## Normalization

`difotrain/data/normalize.py`

Per-dimension mean/std normalization of observations and actions. This is one of
the silent make-or-break details of VLA training, so the stats are a serializable
object that travels inside the saved policy checkpoint.

## Training: behavior cloning

`difotrain/train/trainer.py`

`BCTrainer` flattens the dataset into `(obs, action, instruction)` transitions,
fits the language vocabulary and normalizers, then minimizes MSE between the
policy's predicted actions and the demonstrated actions. The result is a
self-contained `MLPVLAPolicy` (it carries its own encoder + normalizers).

## Evaluation

`difotrain/eval/evaluator.py`

`evaluate_reaching` issues each instruction, rolls the policy out on a fresh
robot, and measures whether the end effector reached the commanded target.
Because the target is conveyed only through language, the success rate measures
genuine language grounding rather than memorized motion.

## Deployment & safety

`difotrain/deploy/`

`DeployRunner` is the real-time inference loop: observe → `policy.predict` →
`SafetyLayer.filter` (clip to limits, optional rate-limit) → `apply_action`. The
same loop drives sim or real hardware. Rollouts can be logged back as `Episode`s
to close the real-world feedback loop (DAgger-style correction).

## Why this matters

| To change... | You implement... | And nothing else changes |
|--------------|------------------|--------------------------|
| The robot (sim → real, new arm) | `Robot` | trainer, eval, deploy |
| How data is gathered | `DataSource` | dataset, training |
| The model (native → OpenVLA) | `Policy` | collect, eval, deploy |

That separation is what makes DifoTrain a framework rather than a script.
