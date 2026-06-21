<div align="center">
<img width="1200" height="626" alt="difotrain" src="https://github.com/user-attachments/assets/cefc993b-c790-4e33-8b3a-51f13135f9dc" />

# DifoTrain

### An embodiment-agnostic framework for training Vision-Language-Action (VLA) models from the real world

[![CI](https://github.com/AshrafGalibShaik/Difotrain/actions/workflows/ci.yml/badge.svg)](https://github.com/AshrafGalibShaik/Difotrain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/difotrain.svg)](https://pypi.org/project/difotrain/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Overview

DifoTrain is a framework for collecting demonstrations, training **Vision-Language-Action (VLA)** policies on them, and deploying those policies to robots — in simulation or on real hardware, on any embodiment. It learns from both **teleoperation** and **human video**, and it can either **wrap an existing VLA** (OpenVLA / Octo / π0 / ACT) or train a **native** model.

The whole `collect → train → eval → deploy` loop runs out of the box on CPU using a built-in, dependency-free simulated robot — no webcam, GPU, or hardware required.

## Architecture

The framework hangs off four pluggable interfaces. A custom robot, data source, or model is just an implementation of one of these, registered into the framework.

| Interface | Module | Implementations shipped |
|-----------|--------|-------------------------|
| `Robot` | `difotrain.embodiment` | `PlanarArm` (sim, dependency-free); PyBullet/MuJoCo/real plug in here |
| `DataSource` | `difotrain.data.sources` | `ScriptedTeleopSource`, `HumanVideoSource`, `SyntheticSource` |
| `Policy` | `difotrain.policy` | `MLPVLAPolicy` (native); `EchoVLAPolicy` (wrapper reference) |
| `EpisodeDataset` | `difotrain.data` | LeRobot/RLDS-style directory dataset |

```
Collect ─▶ Standardize ─▶ Train/Finetune ─▶ Evaluate ─▶ Deploy ─▶ (feedback loop)
 teleop      Episode         BC trainer        sim/real    safety     log rollouts
 human-vid   dataset         + normalize       success     layer      back to dataset
```

## Install

```bash
pip install difotrain                 # core (numpy + torch)
pip install "difotrain[capture]"      # + MediaPipe/OpenCV for webcam human-video
pip install "difotrain[sim]"          # + PyBullet for heavier sim backends
pip install "difotrain[dev]"          # + pytest
```

## Quickstart

The reference task: a 2-link arm must **reach a named target conveyed only through language** — so success measures genuine language grounding, not memorized motion.

```bash
difotrain info                                   # list registered robots/sources
difotrain collect --out data/reach --episodes 200
difotrain train   --data data/reach --out runs/policy.pt --epochs 150
difotrain eval    --policy runs/policy.pt
difotrain deploy  --policy runs/policy.pt --instruction "reach to the lower left"
```

Typical `eval` output after training:

```
success_rate     : 87.50%
mean_final_error : 0.0259
  [OK ] up           err=0.0180
  [OK ] down         err=0.0037
  ...
```

### Python API

```python
from difotrain.data.dataset import EpisodeDataset
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.train.trainer import train_policy, TrainConfig
from difotrain.eval.evaluator import evaluate_reaching

ds = EpisodeDataset("data/reach")
ds.extend(ScriptedTeleopSource(seed=0).collect(200))     # collect demos
policy = train_policy(ds, TrainConfig(epochs=150))       # behavior cloning
print(evaluate_reaching(policy))                          # score in sim
```

## Adding your own robot

```python
from difotrain.core import register_robot
from difotrain.embodiment.base import Robot, RobotSpec

@register_robot("my_arm")
class MyArm(Robot):
    def __init__(self):
        self.spec = RobotSpec(name="my_arm", dof=..., observation_space=..., action_space=...)
    def reset(self, *, instruction="", seed=None): ...
    def get_observation(self): ...
    def apply_action(self, action): ...
```

Sim and real robots implement the *same* `Robot` API, so policies, the evaluator, and the deploy runner drive them unchanged.

## Learning from human video

`HumanVideoSource` estimates human pose (MediaPipe) and uses a `Retargeter` to map it into a robot's action space, turning cheap video into trainable demonstrations. It also runs offline from a recorded trajectory JSON:

```bash
difotrain setup     # download the MediaPipe pose model
difotrain record    # capture human motion from a webcam
```

## Documentation

Full docs live in [`docs/`](docs/README.md):

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Concepts & architecture](docs/concepts.md)
- [CLI reference](docs/cli.md)
- [Python API](docs/python-api.md)
- [Extending DifoTrain](docs/extending.md) — add your own robot, data source, or model
- [Learning from human video](docs/human-video.md)
- [Examples](docs/examples.md)
- [FAQ & troubleshooting](docs/faq.md)

## Roadmap

- [x] Core interfaces, sim robot, dataset, native VLA, BC training, eval, deploy + safety
- [x] Scripted-teleop and human-video data sources, retargeting
- [x] PyBullet backend behind the `Robot` API (`difotrain[sim]`); MuJoCo next
- [ ] Wrapped OpenVLA / Octo / π0 / ACT policies
- [ ] Image observations + transformer language encoder
- [ ] Real-hardware drivers and online DAgger feedback loop

## Development

```bash
uv sync               # or: pip install -e ".[dev,capture]"
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
