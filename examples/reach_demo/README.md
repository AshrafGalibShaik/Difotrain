# Reach Demo

A self-contained demo of the full DifoTrain lifecycle, with a terminal
visualization so you can *see* the trained policy work.

It runs:

1. **Collect** — 200 demonstrations from the built-in scripted expert.
2. **Train** — a language-conditioned policy via behavior cloning.
3. **Eval** — success rate across all targets.
4. **Deploy** — runs the policy for several instructions and draws the arm
   reaching each one in ASCII.
5. **Bonus (optional)** — if the `[sim]` extra is installed, drives a full 3D
   physics arm (PyBullet) through the *same* interface, to show the framework is
   embodiment-agnostic. Install it with `pip install "difotrain[sim]"`; the demo
   skips this step gracefully otherwise.

## Run it

```bash
pip install difotrain        # if you haven't already
python examples/reach_demo/demo.py
```

> Running from a clone instead of a pip install? Install the package first so the
> script imports the right version: `pip install -e .` (from the repo root).

## What you'll see

For each instruction the arm is drawn on a grid:

```
  "reach to the up"   [REACHED]  error=0.006
  . . . . O . . . .
  . . . . * . . . .
  . . . . + . . . .
  . . . . B . . . .
  B=base  +=elbow  O=hand  X=goal
```

`B` is the fixed base, `+` the elbow joint, `O` the hand (end effector), and `X`
the goal. The target is conveyed **only** through the language instruction — the
robot never sees the goal coordinates — so a reach means the policy grounded the
sentence into motion.

## Make it yours

Edit the instruction list near the bottom of `demo.py`:

```python
for name in ["up", "down", "left", "right", "upper right", "lower left"]:
```

Valid targets: `right`, `upper right`, `up`, `upper left`, `left`, `lower left`,
`down`, `lower right`.

Then re-run. Delete `examples/reach_demo/_data/` to recollect fresh data.
