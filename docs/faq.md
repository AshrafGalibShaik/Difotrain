# FAQ & troubleshooting

## Do I need a robot, GPU, or webcam to try DifoTrain?

No. The core ships a dependency-free simulated robot (`PlanarArm`) and trains on
CPU in seconds. A webcam (with the `capture` extra) is only needed for the
human-video path; a GPU only helps for larger models.

## `difotrain: command not found`

The console script directory may not be on your `PATH`. Either add it, or run the
module form:

```bash
python -m difotrain info
```

## Is this a full VLA like OpenVLA / Octo / π0?

Not yet — it is a *framework* around the VLA lifecycle. The shipped native model
(`MLPVLAPolicy`) is a small proprioception + language model used to validate the
end-to-end pipeline. The intended use is to wrap a real VLA behind the `Policy`
interface (see [Extending](extending.md)) or to grow the native model with image
and transformer-text encoders.

## Where do images / vision come in?

The reference task uses proprioception only. The interfaces are vision-ready:
`Robot.get_observation` can return image features, and a `Policy` can consume
them. Adding an image encoder to the native model is on the [roadmap](../README.md#roadmap).

## Why is the success rate not 100%?

The reference task is a real control problem and the default training is small
(few epochs, tiny MLP). Typical results are ~85–90%. Increase `--episodes` and
`--epochs`, or raise `--tol`, to push it higher.

## The tests are slow.

Two integration tests train a model end to end (~90s total). For fast iteration,
run a single fast module:

```bash
python -m unittest tests.test_core -v
```

## How do I save and reload a policy?

```python
policy.save("runs/policy.pt")
from difotrain.policy.native.mlp_vla import MLPVLAPolicy
policy = MLPVLAPolicy.load("runs/policy.pt")   # carries weights + vocab + norm stats
```

## How do I add my own robot / model / data source?

See [Extending DifoTrain](extending.md). In short: subclass the relevant
interface, implement its methods, and (optionally) register it with a decorator.

## Which Python versions are supported?

Python 3.10 and newer.

## How do I report a bug or request a feature?

Open an issue on the GitHub repository.
