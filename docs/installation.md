# Installation

DifoTrain runs on **Python 3.10+**. The core only needs NumPy and PyTorch, both
of which are pulled in automatically.

## Standard install

```bash
pip install difotrain
```

This gives you the full `collect → train → eval → deploy` pipeline using the
built-in simulated robot. No GPU, webcam, or physical robot is required.

## Optional extras

Some features need heavier dependencies, kept optional so the core stays light:

| Extra | Command | Adds |
|-------|---------|------|
| Webcam human-video capture | `pip install "difotrain[capture]"` | MediaPipe, OpenCV |
| Heavier sim backends | `pip install "difotrain[sim]"` | PyBullet |
| Development / tests | `pip install "difotrain[dev]"` | pytest |

You can combine them: `pip install "difotrain[capture,dev]"`.

## Verify the install

```bash
difotrain info
```

Expected output:

```
DifoTrain vX.Y.Z
  robots   : ['planar_arm']
  sources  : ['scripted_teleop', 'synthetic']
  targets  : ['right', 'upper right', 'up', 'upper left', 'left', 'lower left', 'down', 'lower right']
```

If the `difotrain` command is not found, your Python scripts directory may not be
on `PATH`. You can always invoke it as a module instead:

```bash
python -m difotrain info
```

## Install from source (development)

```bash
git clone https://github.com/AshrafGalibShaik/difotrain.git
cd difotrain

# Using uv (recommended)
uv sync
uv run difotrain info

# Or using pip
pip install -e ".[dev,capture]"
difotrain info
```

## GPU / CPU

The framework defaults to CPU and the reference task trains in seconds there. To
use a GPU, pass a device through the training config (see
[Python API](python-api.md)); a standard CUDA-enabled PyTorch install is all
that is required.

## Uninstall

```bash
pip uninstall difotrain
```
