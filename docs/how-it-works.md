# How DifoTrain works — complete architecture

This is the deep-dive reference for DifoTrain's internals: every layer, how data
flows through them, the exact transformations applied at each step, and the
design decisions behind them. If you want the short version, read
[Concepts](concepts.md) first; this document assumes you want the whole picture.

---

## Table of contents

1. [Design philosophy](#1-design-philosophy)
2. [The big picture](#2-the-big-picture)
3. [The data model: `Episode`](#3-the-data-model-episode)
4. [Spaces and the observation/action contract](#4-spaces)
5. [The four interfaces](#5-the-four-interfaces)
6. [The embodiment layer](#6-the-embodiment-layer)
7. [The data layer](#7-the-data-layer)
8. [Language conditioning](#8-language-conditioning)
9. [The policy layer](#9-the-policy-layer)
10. [Training internals](#10-training-internals)
11. [Evaluation internals](#11-evaluation-internals)
12. [Deployment and safety](#12-deployment-and-safety)
13. [The plugin registry](#13-the-plugin-registry)
14. [End-to-end annotated trace](#14-end-to-end-annotated-trace)
15. [Directory map](#15-directory-map)
16. [Extension points summary](#16-extension-points-summary)

---

## 1. Design philosophy

DifoTrain is built on one bet: **the hard part of real-world robot learning is
not the model, it is the plumbing around it** — collecting time-aligned data,
standardizing it, conditioning on language, normalizing actions, evaluating task
success, and deploying safely on hardware. So the framework fixes that plumbing
behind stable interfaces and makes the model, the robot, and the data source
swappable.

Three principles follow:

- **One data unit everywhere.** Every stage consumes and/or produces an
  `Episode`. Nothing passes ad-hoc tuples around.
- **Four interfaces, everything else is a plugin.** `Robot`, `DataSource`,
  `Policy`, `EpisodeDataset`. The trainer/evaluator/deploy runner depend only on
  these abstractions, never on a concrete robot or model.
- **Light core, heavy things optional.** The core needs only NumPy + PyTorch and
  runs end to end on CPU. MediaPipe, OpenCV, and PyBullet are optional extras.

---

## 2. The big picture

```
        ┌─────────────┐      ┌──────────────────┐      ┌────────────┐
        │  DataSource  │────▶│  EpisodeDataset   │────▶│  BCTrainer  │
        │ teleop /     │ Ep   │ .npz + meta.json  │ Ep   │ behavior   │
        │ human-video  │      │ (on disk)         │      │ cloning    │
        └─────────────┘      └──────────────────┘      └─────┬──────┘
                                                              │ Policy
                       ┌──────────────────────────────────────┤
                       ▼                                        ▼
                ┌────────────┐                          ┌──────────────┐
                │ Evaluator   │◀── Robot ──▶ ... ◀─────│ DeployRunner  │
                │ success rate│                          │ + SafetyLayer │
                └────────────┘                          └──────┬───────┘
                                                               │ rollout
                                                               ▼
                                              log Episode back to EpisodeDataset
                                                  (real-world feedback loop)
```

The arrows labeled `Ep` carry `Episode` objects; the arrows labeled `Policy`
carry a trained model object; the `Robot` is shared by both the evaluator and the
deploy runner. The feedback loop closes the cycle: deployment rollouts become new
training data.

---

## 3. The data model: `Episode`

Source: `difotrain/core/episode.py`

An `Episode` is one demonstration or rollout. It is intentionally minimal and
matches RLDS / LeRobotDataset conventions so data stays interoperable.

```python
@dataclass
class Episode:
    observations: np.ndarray   # shape [T, obs_dim], float32
    actions:      np.ndarray   # shape [T, act_dim], float32
    meta:         EpisodeMeta
```

- `actions[t]` is the action taken **from** `observations[t]`. This alignment is
  the contract the trainer relies on.
- `__post_init__` enforces both arrays are 2-D and equal length, casting to
  float32. Bad data fails loudly at construction, not deep inside training.

`EpisodeMeta` carries the non-numeric context:

```python
@dataclass
class EpisodeMeta:
    instruction: str         # the language command for this episode
    source:      str         # which DataSource produced it
    robot:       str         # which embodiment
    success:     Optional[bool]
    extra:       dict        # free-form, e.g. {"target": "up"}
```

`to_dict` / `from_dict` make it JSON-serializable for the dataset index.

---

## 4. Spaces

Source: `difotrain/core/spaces.py`

To avoid a hard dependency on `gymnasium`, DifoTrain ships a tiny `Box` space —
a bounded continuous region `[low, high]`. It provides exactly what the rest of
the framework needs:

- `dim` / `shape` — sizing networks and buffers.
- `contains(x)` — validation (used heavily in tests).
- `clip(x)` — squashing actions/observations into legal bounds. This is the
  single most important method: the safety layer, every robot's `apply_action`,
  and retargeters all call it to guarantee values stay in range.
- `sample(rng)` — random valid points (used by the PyBullet demo).

Every `Robot` declares an `observation_space` and an `action_space` as `Box`es in
its `RobotSpec`. That declaration is what lets generic code size a network or
clip an action without knowing the robot.

---

## 5. The four interfaces

| Interface | File | Core method(s) | Implemented by |
|-----------|------|----------------|----------------|
| `Robot` | `embodiment/base.py` | `reset`, `get_observation`, `apply_action`, `close` | `PlanarArm`, `PyBulletArm`, your hardware |
| `DataSource` | `data/sources/base.py` | `collect(n) -> Iterator[Episode]` | `ScriptedTeleopSource`, `HumanVideoSource`, `SyntheticSource` |
| `Policy` | `policy/base.py` | `predict(obs, instruction) -> action`, `save`, `load`, `reset` | `MLPVLAPolicy`, `EchoVLAPolicy`, wrapped VLAs |
| `EpisodeDataset` | `data/dataset.py` | `add`, `extend`, `__getitem__`, `stacked` | the shipped directory dataset |

The trainer, evaluator, and deploy runner import **only** these abstractions.
That is the whole reason a new robot or model "just works" — the surrounding code
literally cannot tell the difference.

---

## 6. The embodiment layer

Source: `difotrain/embodiment/`

### 6.1 `RobotSpec`

A static description of an embodiment:

```python
@dataclass
class RobotSpec:
    name: str
    dof: int
    observation_space: Box
    action_space: Box
    control_hz: float = 20.0     # -> dt = 1/control_hz
    urdf: Optional[str] = None
    metadata: dict = {}
```

`dt` (derived from `control_hz`) is used to integrate velocity actions and to
pace the real-time deploy loop.

### 6.2 The `Robot` lifecycle

```python
obs = robot.reset(instruction=..., seed=...)   # -> first observation
obs = robot.get_observation()                   # -> current observation
obs = robot.apply_action(action)                # apply one step -> new observation
robot.close()                                   # release resources
```

`reset` takes the instruction because some embodiments need to set up the scene
for the commanded task; the planar arm ignores it (the task lives entirely in the
policy), the convention is there for richer robots.

### 6.3 `PlanarArm` — the reference embodiment

Source: `embodiment/sim/planar_arm.py`. A dependency-free 2-link arm; this is
what every test and example uses because it runs in microseconds.

- **State**: two joint angles `q = (q1, q2)`.
- **Observation** (`get_observation`): `[q1, q2, ee_x, ee_y]` — joint angles plus
  the end-effector position from forward kinematics. **Note what is *not* here:
  the target.** The goal is never observed, so the policy can only get it from
  language.
- **Action** (`apply_action`): joint angular velocities. Integrated as
  `q ← clip(q + action·dt)`, with the action first clipped to the action space.
- **Forward kinematics**: `ee = (L1·cosq1 + L2·cos(q1+q2), L1·sinq1 + L2·sin(q1+q2))`
  with `L1 = L2 = 1`.
- **Inverse kinematics**: analytic 2-link solution (`inverse_kinematics`), used by
  the scripted expert.
- **Named targets**: `NAMED_TARGETS` maps eight compass directions to angles on a
  circle of radius 1.2. `target_xy(name)` gives the goal coordinates;
  `instruction_for(name)` gives the sentence ("reach to the up").
- **Expert** (`expert_action`): a proportional controller toward the IK solution
  for the named target — `action = clip(gain·(q_goal − q))`. This is the teacher
  the policy imitates.

### 6.4 `PyBulletArm` — the 3D physics backend

Source: `embodiment/sim/pybullet_arm.py`. Optional (`difotrain[sim]`); pybullet
is imported lazily inside the constructor so importing the module is free.

- Loads a URDF (Kuka iiwa by default) into a headless or GUI PyBullet client.
- Discovers all non-fixed joints and reads their limits to build the
  observation/action `Box`es automatically.
- **Observation**: `[joint_positions (n), end_effector_xyz (3)]`.
- **Action**: per-joint position deltas, scaled by `max_delta`, applied through
  PyBullet's `POSITION_CONTROL`, then `stepSimulation`.
- Proves the interface generalizes: the same `DeployRunner`, `SafetyLayer`, and
  `Policy` drive it unchanged.

### 6.5 Retargeting

Source: `embodiment/retarget/`. A `Retargeter` maps a human feature vector into a
robot's action space — the bridge that turns human video into robot
demonstrations. `LinearRetargeter` is `y = clip(W·x + b)`, falling back to
identity/truncation when no weights are given.

---

## 7. The data layer

Source: `difotrain/data/`

### 7.1 `DataSource`

```python
class DataSource:
    def collect(self, num_episodes) -> Iterator[Episode]: ...
```

Three implementations ship:

- **`ScriptedTeleopSource`** — the teleop branch. For each episode it picks a
  random named target, resets the arm, and for `max_steps` records
  `(observation, expert_action)` while stepping. It tags each episode with the
  instruction and a `success` flag (did the end effector land within tolerance of
  the target). Swap the expert for live joystick/VR input and this becomes real
  teleop with no other changes.
- **`HumanVideoSource`** — the human-video branch. Runs MediaPipe pose
  estimation (or reads a recorded trajectory JSON in offline mode), reduces each
  frame to a feature vector via `pose_to_features` (selected joint angles), and
  maps it through a `Retargeter` to produce actions. Emits `Episode`s identical
  in type to the teleop ones.
- **`SyntheticSource`** — random observations with a known linear obs→action
  rule. Used to verify training can fit a known function, independent of any
  robot.

### 7.2 `EpisodeDataset`

A directory-based store (LeRobot-style):

```
<root>/
  meta.json              # {format, num_episodes, episodes: [{id, file, length, meta}]}
  episodes/
    ep_000000.npz        # observations, actions (compressed)
    ep_000001.npz
    ...
```

- `add(episode)` writes one `.npz` and appends to the JSON index (re-saved each
  time, so the dataset is consistent on disk after every add).
- `__getitem__(i)` lazily loads episode `i` from disk.
- `stacked()` is the bridge to training: it concatenates every episode into flat
  arrays `obs [N, obs_dim]`, `act [N, act_dim]`, and a parallel list of `N`
  instruction strings (each episode's instruction repeated per timestep). `N` is
  the total number of transitions across all episodes.

Numeric arrays live in `.npz` (compact); human-readable metadata lives in JSON.
Large image observations would be stored as encoded video alongside — the
interface is unchanged.

### 7.3 `Normalizer`

Per-dimension mean/std, fit from data:

```python
normalize(x)   = (x - mean) / std
denormalize(x) =  x * std + mean
```

Zero-variance dimensions get `std = 1` to avoid division by zero. Normalization
of **actions** especially is a known make-or-break factor for VLA training, so
these stats are serialized inside the policy checkpoint and travel with it.

---

## 8. Language conditioning

Source: `difotrain/core/language.py`

A `LanguageEncoder` turns an instruction string into a fixed-length vector via
`encode(text) -> np.ndarray`. The default is a deterministic **bag-of-words**
encoder:

- `fit(instructions)` builds a sorted vocabulary from all words seen.
- `encode(text)` returns a multi-hot vector of length `dim` (1.0 for each
  vocabulary word present).

It is intentionally simple (zero heavy dependencies) but it makes language a
**real, used input**: in the reaching task the target is not observed, so the
only signal distinguishing "reach to the up" from "reach to the down" is this
vector. Because the contract is just `encode(text) -> vector`, swapping in a
transformer text encoder later changes nothing else.

---

## 9. The policy layer

Source: `difotrain/policy/`

### 9.1 The contract

```python
class Policy:
    def predict(self, observation, instruction="") -> action
    def reset(self) -> None         # optional, for stateful policies
    def save(self, path); load(cls, path)
```

`predict` is the entire VLA interface — observation + instruction in, action out.

### 9.2 `MLPVLAPolicy` — the native model

Source: `policy/native/mlp_vla.py`. A miniature but genuine vision-language-action
model (proprioception stands in for vision in the reference task).

**Architecture**: an MLP over the concatenation of the (normalized) observation
and the language embedding:

```
input  = [ normalize(obs)  ⊕  language_encoder.encode(instruction) ]   # dim = obs_dim + lang_dim
hidden = Linear(in, 128) → ReLU → Linear(128, 128) → ReLU
output = Linear(128, act_dim)                                          # predicted (normalized) action
```

**`predict` path** (inference):
1. Normalize the observation (if a normalizer is attached).
2. Encode the instruction to a vector.
3. Concatenate, run the MLP under `torch.no_grad()`.
4. Denormalize the output back into real action units.
5. Return a float32 NumPy array.

**Self-containment**: the policy stores its own `LanguageEncoder` and both
`Normalizer`s. `save` writes `{state_dict, config}` where config includes the
vocabulary and normalization stats; `load` reconstructs everything. A checkpoint
is therefore fully portable — no external vocab/stats files.

### 9.3 `EchoVLAPolicy` — the wrapper reference

Source: `policy/wrapped/echo.py`. A do-nothing policy that shows exactly where a
real OpenVLA / Octo / π0 / ACT model plugs in: in `predict`, render the
observation into the backbone's expected format (images + text), call the model,
and decode its action tokens back into the robot's action space.

---

## 10. Training internals

Source: `difotrain/train/trainer.py`. The algorithm is **behavior cloning** —
supervised regression from `(observation, instruction)` to the demonstrated
action.

`BCTrainer.fit(dataset)` does, in order:

1. **Flatten** the dataset: `obs, act, instr = dataset.stacked()`. Raises if
   empty.
2. **Fit the language vocabulary** from all instructions:
   `lang = LanguageEncoder.fit(instr)`.
3. **Fit normalizers** for observations and actions (if `normalize=True`).
4. **Build** an `MLPVLAPolicy` sized to `obs_dim`, `act_dim`, and the language
   dimension, carrying the encoder + normalizers.
5. **Pre-encode the full training tensor once** (the toy datasets are small):
   - `obs_in = normalize(obs)`
   - `lang_feats = stack(encode(t) for t in instr)`
   - `x = concat([obs_in, lang_feats], axis=1)`  → model input
   - `y = normalize(act)`                          → regression target
6. **Optimize**: Adam (`lr` default 1e-3) on MSE loss. Each epoch shuffles a
   permutation of the `N` transitions and steps over mini-batches
   (`batch_size` default 64):

   ```
   for epoch in range(epochs):
       for batch in shuffled(transitions):
           pred = model(x[batch])
           loss = MSE(pred, y[batch])
           loss.backward(); optimizer.step()
   ```

7. **Report**: per-epoch mean loss is passed to an optional `on_epoch` callback;
   the final loss is stashed on `policy._final_loss`. Returns the trained policy.

`train_policy(dataset, config, verbose)` is the convenience wrapper used by the
CLI; it just prints loss every 10 epochs.

Why behavior cloning: it is the simplest method that demonstrates the whole
pipeline learns a language-conditioned mapping. The interfaces do not assume BC —
a different `Policy` and trainer (e.g. diffusion policy, or fine-tuning a wrapped
VLA) slot into the same surrounding code.

---

## 11. Evaluation internals

Source: `difotrain/eval/evaluator.py`. `evaluate_reaching` measures real task
success, not training loss:

```
for each named target:
    instruction = instruction_for(target)
    obs = robot.reset(instruction, seed)
    repeat max_steps:
        obs = robot.apply_action( policy.predict(obs, instruction) )
    error   = ‖ end_effector − target_xy(target) ‖
    success = error < success_tol            # default 0.15
```

It returns an `EvalResult` with the overall `success_rate`, `mean_final_error`,
and a per-target breakdown (the pretty `__str__` is what the CLI prints).

Because the target is conveyed **only** through the instruction, a high success
rate is direct evidence the policy grounded language into action — it could not
have memorized a single motion.

---

## 12. Deployment and safety

Source: `difotrain/deploy/`

### 12.1 `SafetyLayer`

Sits between policy output and the robot. Non-negotiable on real hardware.

- `filter(action)` first clips the action into the robot's action space.
- If `max_delta` is set, it rate-limits: the change from the previous action is
  bounded, then re-clipped. This prevents jerky, unsafe commands.
- `reset()` clears the previous-action memory between episodes.

### 12.2 `DeployRunner`

The real-time inference loop, identical for sim and real robots:

```python
obs = robot.reset(instruction, seed)
policy.reset(); safety.reset()
for _ in range(max_steps):
    action = policy.predict(obs, instruction)
    action = safety.filter(action)
    log.observations.append(obs); log.actions.append(action)
    obs = robot.apply_action(action)
    if realtime: sleep(dt)            # pace to the robot's control_hz
return log    # a RolloutLog
```

`RolloutLog.to_episode(instruction, robot_name)` converts a rollout into an
`Episode`, which can be added straight back into an `EpisodeDataset` — this is the
**real-world feedback loop** (DAgger-style correction): deploy, log, retrain.

---

## 13. The plugin registry

Source: `difotrain/core/registry.py`. A small name → factory table for robots,
data sources, and policies. Components register via decorators:

```python
@register_robot("planar_arm")
class PlanarArm(Robot): ...
```

and are created by name:

```python
registry.make_robot("planar_arm")
registry.make_source("scripted_teleop", max_steps=40)
```

This indirection is what lets the CLI (and future config files) build components
from strings without importing user code. `difotrain info` simply prints the
registry contents. (Registration runs when a module is imported, so the CLI
imports the relevant packages before listing them.)

---

## 14. End-to-end annotated trace

What actually happens when you run the four commands, with the data shapes at
each boundary (numbers from the reference task: `obs_dim=4`, `act_dim=2`, 200
episodes × 40 steps = 8000 transitions).

```
difotrain collect --out data/reach --episodes 200
└─ ScriptedTeleopSource.collect(200)
   └─ per episode: pick target → reset arm → 40×(record obs[4], expert_action[2])
   └─ EpisodeDataset.add(ep)  →  data/reach/episodes/ep_000000.npz + meta.json
   Result on disk: 200 .npz files, one JSON index.

difotrain train --data data/reach --out runs/policy.pt --epochs 150
└─ EpisodeDataset.stacked()  →  obs[8000,4], act[8000,2], instr[8000]
└─ LanguageEncoder.fit(instr)        →  vocab (~6 words), lang_dim ≈ 6
└─ Normalizer.fit(obs), Normalizer.fit(act)
└─ x = concat[ norm(obs)[8000,4], lang[8000,6] ] = [8000,10]
   y = norm(act) = [8000,2]
└─ MLP(10 → 128 → 128 → 2), Adam+MSE, 150 epochs
└─ policy.save("runs/policy.pt")  →  {state_dict, config{vocab, norm stats}}

difotrain eval --policy runs/policy.pt
└─ for each of 8 targets: reset → 60 steps of predict→apply → measure error
└─ prints success_rate (~87%), mean_final_error, per-target table

difotrain deploy --policy runs/policy.pt --instruction "reach to the up"
└─ DeployRunner: 60×( predict[2] → SafetyLayer.filter → apply_action )
└─ prints final end-effector position (≈ (0, 1.2) for "up")
```

---

## 15. Directory map

```
difotrain/
  __init__.py            # version, package docstring
  __main__.py            # `python -m difotrain`; also the MediaPipe downloader
  cli.py                 # argparse CLI: collect/train/eval/deploy/info/setup/record

  core/
    episode.py           # Episode, EpisodeMeta
    spaces.py            # Box
    language.py          # LanguageEncoder (bag-of-words)
    registry.py          # plugin registry + decorators

  embodiment/
    base.py              # Robot ABC, RobotSpec
    sim/
      planar_arm.py      # PlanarArm (reference), FK/IK, named targets, expert
      pybullet_arm.py    # PyBulletArm (optional 3D physics)
    retarget/
      base.py            # Retargeter, LinearRetargeter

  data/
    dataset.py           # EpisodeDataset (directory store)
    normalize.py         # Normalizer
    sources/
      base.py            # DataSource ABC
      scripted_teleop.py # expert-driven teleop source
      human_video.py     # MediaPipe pose → retarget source
      synthetic.py       # random known-rule source

  policy/
    base.py              # Policy ABC
    native/mlp_vla.py    # MLPVLAPolicy (the native model)
    wrapped/echo.py      # EchoVLAPolicy (wrapper reference)

  train/trainer.py       # BCTrainer, TrainConfig, train_policy
  eval/evaluator.py      # evaluate_reaching, EvalResult
  deploy/
    runner.py            # DeployRunner, RolloutLog
    safety.py            # SafetyLayer
  capture/record_pose.py # webcam pose capture (MediaPipe)
```

---

## 16. Extension points summary

| You want to… | Implement / swap | Everything else unchanged |
|--------------|------------------|---------------------------|
| Add a robot (sim or real) | subclass `Robot` (+ `RobotSpec`), `@register_robot` | trainer, eval, deploy, safety |
| Add a way to collect data | subclass `DataSource`, `@register_source` | dataset, training |
| Use a different model / wrap a VLA | subclass `Policy`, `@register_policy` | collect, eval, deploy |
| Learn from human video | implement `Retargeter` | `HumanVideoSource` |
| Use real language understanding | implement `encode(text)->vec` | the policy |
| Change the learning algorithm | new trainer producing a `Policy` | eval, deploy |

The throughline: **depend on the four interfaces, and every other layer keeps
working when you replace a piece.** That is the entire architecture in one
sentence.

See also: [Concepts](concepts.md) · [Python API](python-api.md) ·
[Extending DifoTrain](extending.md).
