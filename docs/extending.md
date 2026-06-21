# Extending DifoTrain

DifoTrain is designed to be extended by implementing one of its four interfaces
and registering it. Your component then works with the existing trainer,
evaluator, and deploy runner unchanged.

## Add a custom robot

Subclass `Robot`, fill in a `RobotSpec`, and register it. Sim and real hardware
use the *same* interface — for a real robot, the four methods talk to your driver
instead of a simulator.

```python
import numpy as np
from difotrain.core import register_robot
from difotrain.core.spaces import Box
from difotrain.embodiment.base import Robot, RobotSpec


@register_robot("my_arm")
class MyArm(Robot):
    def __init__(self, control_hz=30.0):
        self.spec = RobotSpec(
            name="my_arm",
            dof=3,
            observation_space=Box(low=[-3.14]*3 + [-1]*3, high=[3.14]*3 + [1]*3),
            action_space=Box(low=[-1]*3, high=[1]*3),
            control_hz=control_hz,
            urdf="path/to/my_arm.urdf",   # optional
        )
        self._state = np.zeros(3, dtype=np.float32)

    def reset(self, *, instruction="", seed=None):
        self._state[:] = 0.0
        return self.get_observation()

    def get_observation(self):
        # proprioception (+ camera features, etc.)
        return np.concatenate([self._state, np.zeros(3)]).astype(np.float32)

    def apply_action(self, action):
        action = self.spec.action_space.clip(action)
        self._state += action * self.spec.dt    # or: send to hardware
        return self.get_observation()

    def close(self):
        pass    # release hardware / simulator
```

Use it:

```python
from difotrain.core.registry import registry
robot = registry.make_robot("my_arm")
```

> **Note:** the bundled `evaluate_reaching` is specific to the planar-arm reach
> task. For a different robot/task, write a small evaluation loop (see
> `difotrain/eval/evaluator.py` as a template) or just deploy and measure your
> own success criterion.

## Add a custom data source

Yield standardized `Episode`s from anywhere — a teleop rig, a log file, another
simulator.

```python
import numpy as np
from difotrain.core import register_source
from difotrain.core.episode import Episode, EpisodeMeta
from difotrain.data.sources.base import DataSource


@register_source("joystick_teleop")
class JoystickTeleopSource(DataSource):
    name = "joystick_teleop"

    def __init__(self, robot):
        self.robot = robot

    def collect(self, num_episodes):
        for _ in range(num_episodes):
            obs = self.robot.reset()
            observations, actions = [], []
            for _ in range(100):
                action = self._read_joystick()        # your input device
                observations.append(obs)
                actions.append(action)
                obs = self.robot.apply_action(action)
            yield Episode(
                np.array(observations), np.array(actions),
                EpisodeMeta(instruction="...", source=self.name,
                            robot=self.robot.spec.name),
            )

    def _read_joystick(self):
        ...
```

## Add / wrap a custom policy

Implement the `Policy` contract. This is how you wrap an external VLA such as
OpenVLA, Octo, π0, or ACT.

```python
import numpy as np
from difotrain.policy.base import Policy


class OpenVLAPolicy(Policy):
    def __init__(self, checkpoint):
        self.model = load_openvla(checkpoint)        # your backbone

    def predict(self, observation, instruction=""):
        # 1. render observation into the model's expected inputs (images + text)
        # 2. call the backbone
        # 3. decode action tokens back into the robot's action space
        return self.model.act(observation, instruction)

    def reset(self):
        pass                                          # clear any history

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

Because the trainer/evaluator/deploy runner only call `predict`, `reset`,
`save`, and `load`, a wrapped policy is a drop-in replacement for the native one.

## Add a custom retargeter (for human-video learning)

A retargeter maps a human feature vector into a robot's action space. Subclass
`Retargeter`:

```python
import numpy as np
from difotrain.embodiment.retarget.base import Retargeter


class MyRetargeter(Retargeter):
    def __init__(self, spec):
        self.spec = spec

    def retarget(self, human_features):
        action = my_mapping(human_features)
        return self.spec.action_space.clip(action)
```

Pass it to `HumanVideoSource(robot_spec, retargeter=MyRetargeter(spec))`. See
[Learning from human video](human-video.md).

## Swap the language encoder

`LanguageEncoder` implements `encode(text) -> np.ndarray`. To use a transformer
text encoder, implement the same method and pass your encoder when constructing
`MLPVLAPolicy`. Nothing else changes.

## Checklist for a new component

1. Subclass the right interface (`Robot` / `DataSource` / `Policy` / `Retargeter`).
2. Implement every abstract method.
3. Keep arrays inside the declared `Box` spaces (use `space.clip`).
4. Register it with the matching decorator if you want it available by name.
5. Add a test mirroring the ones in `tests/`.
