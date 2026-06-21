# Contributing to DifoTrain

Thanks for your interest in improving DifoTrain.

## Development setup

```bash
git clone https://github.com/AshrafGalibShaik/difotrain.git
cd difotrain
uv sync                       # or: pip install -e ".[dev,capture]"
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

Two integration tests train a model end to end and take ~90s; the rest are fast.
Run a single module while iterating, e.g. `python -m unittest tests.test_core`.

## Project layout

```
difotrain/
  core/          # Episode, spaces, language encoder, registry
  embodiment/    # Robot interface, sim robots, retargeting
  data/          # EpisodeDataset, normalization, data sources
  policy/        # Policy interface, native + wrapped models
  train/         # behavior-cloning trainer
  eval/          # evaluation harness
  deploy/        # inference runner + safety layer
  capture/       # webcam pose capture (MediaPipe)
  cli.py         # command line interface
docs/            # documentation
tests/           # unittest suite
```

## Adding a component

New robots, data sources, and policies should implement the relevant interface
and register themselves. See [docs/extending.md](docs/extending.md). Please add a
test mirroring the existing ones in `tests/`.

## Code style

- Keep the core dependency-light (NumPy + PyTorch); put heavier deps behind the
  optional extras in `pyproject.toml`.
- Match the style of surrounding code; keep arrays inside their declared `Box`
  spaces.
- Add or update docs for user-facing changes.

## Pull requests

1. Branch from `main`.
2. Make sure the test suite passes.
3. Describe the change and link any related issue.
