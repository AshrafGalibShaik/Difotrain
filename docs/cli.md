# CLI reference

Every command is available as `difotrain <command>` (after install) or
`python -m difotrain <command>` (from source).

```
difotrain <command> [options]
```

Run `difotrain <command> --help` for the flags of any single command.

---

## `difotrain info`

Print the version and all registered robots, data sources, and the reference
task's named targets. Useful to confirm the install and to see what plugins are
available.

```bash
difotrain info
```

---

## `difotrain collect`

Collect demonstrations from the scripted-teleop expert into a dataset.

```bash
difotrain collect --out data/reach --episodes 200 --steps 40 --seed 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `data/reach` | Output dataset directory. |
| `--episodes` | `200` | Number of demonstrations to collect. |
| `--steps` | `40` | Maximum steps per demonstration. |
| `--seed` | `0` | Random seed (target choice + start pose). |

Prints how many episodes were written and the expert's success rate.

---

## `difotrain train`

Train a policy on a dataset via behavior cloning.

```bash
difotrain train --data data/reach --out runs/policy.pt --epochs 150 --batch-size 64 --lr 0.001
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/reach` | Dataset directory to train on. |
| `--out` | `runs/policy.pt` | Path to write the trained policy checkpoint. |
| `--epochs` | `150` | Training epochs. |
| `--batch-size` | `64` | Mini-batch size. |
| `--lr` | `0.001` | Adam learning rate. |

Prints per-epoch loss and the final loss. The checkpoint is self-contained
(model weights + language vocabulary + normalization stats).

---

## `difotrain eval`

Evaluate a trained policy on every named target in simulation.

```bash
difotrain eval --policy runs/policy.pt --tol 0.15
```

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | `runs/policy.pt` | Policy checkpoint to evaluate. |
| `--tol` | `0.15` | Distance tolerance counted as success. |

Prints overall success rate, mean final error, and a per-target breakdown.

---

## `difotrain deploy`

Run a policy for a single instruction on the simulated robot.

```bash
difotrain deploy --policy runs/policy.pt --instruction "reach to the up" --steps 60
```

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | `runs/policy.pt` | Policy checkpoint to run. |
| `--instruction` | `"reach to the up"` | Natural-language command. |
| `--steps` | `60` | Number of control steps to run. |

Prints the final end-effector position. Actions pass through the safety layer.

---

## `difotrain setup`

Download the MediaPipe pose model (`pose_landmarker_lite.task`) into the current
directory. Required only for webcam capture.

```bash
difotrain setup
```

---

## `difotrain record`

Capture human motion from a webcam and save a trajectory to
`storage/human_trajectory.json`. Requires the `capture` extra and a camera.

```bash
difotrain record   # press 'q' to stop
```

See [Learning from human video](human-video.md) for the full workflow.
