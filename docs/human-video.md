# Learning from human video

Teleoperation produces high-quality data but needs a robot and a human in the
loop. Human video is the cheap, scalable alternative: record a person performing
a task and turn that into trainable demonstrations. DifoTrain supports both and
treats them identically downstream (both yield `Episode`s).

This path uses MediaPipe for pose estimation, so install the capture extra:

```bash
pip install "difotrain[capture]"
```

## How it works

```
webcam ─▶ MediaPipe pose ─▶ human features ─▶ Retargeter ─▶ robot actions ─▶ Episode
          (33 landmarks)    (joint angles)    (your map)    (action space)
```

1. **Pose estimation** — `difotrain/capture/record_pose.py` runs the MediaPipe
   Pose Landmarker on each frame and records selected joint positions.
2. **Feature extraction** — `pose_to_features` reduces a pose to a compact
   vector (e.g. left/right elbow angles).
3. **Retargeting** — a `Retargeter` maps human features into the target robot's
   action space. The default `LinearRetargeter` is identity/affine; supply your
   own for a calibrated mapping (see [Extending](extending.md)).
4. **Episodes** — `HumanVideoSource` packages the result as standard `Episode`s.

## Step 1 — download the pose model

```bash
difotrain setup
```

This downloads `pose_landmarker_lite.task` into the current directory.

## Step 2 — record from a webcam

```bash
difotrain record      # press 'q' to stop
```

Saves a trajectory to `storage/human_trajectory.json`.

## Step 3 — turn it into demonstrations

```python
from difotrain.embodiment.sim.planar_arm import PlanarArm
from difotrain.data.sources.human_video import HumanVideoSource
from difotrain.data.dataset import EpisodeDataset

arm = PlanarArm()
source = HumanVideoSource(
    robot_spec=arm.spec,
    instruction="imitate the human",
    trajectory_json="storage/human_trajectory.json",   # offline mode
)

ds = EpisodeDataset("data/human")
ds.extend(source.collect(1))
```

You can then train on `ds` exactly like any other dataset (see
[Quickstart](quickstart.md)).

## Offline vs. live

`HumanVideoSource` runs in two modes:

- **Offline** (`trajectory_json=...`) — reads an already-recorded JSON. No camera
  required; fully reproducible; used in the test suite.
- **Live** (no `trajectory_json`) — captures from the webcam on each `collect`.
  Requires the `capture` extra and a camera.

## Custom retargeting

The quality of human-video learning depends heavily on the retargeter. The
default is a placeholder; for a real robot you will want a calibrated mapping
from human kinematics to your robot's joints. Implement `Retargeter` and pass it
in:

```python
source = HumanVideoSource(arm.spec, retargeter=MyRetargeter(arm.spec),
                          trajectory_json="storage/human_trajectory.json")
```

## Hybrid data

A common recipe is to combine both sources — human video for breadth, teleop for
precision — into one dataset:

```python
ds.extend(HumanVideoSource(arm.spec, trajectory_json="...").collect(10))
ds.extend(ScriptedTeleopSource(seed=0).collect(200))
```

Since both emit `Episode`s, the trainer consumes the mixed dataset directly.
