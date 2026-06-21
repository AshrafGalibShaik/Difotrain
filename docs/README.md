# DifoTrain Documentation

Welcome to the DifoTrain documentation. DifoTrain is an **embodiment-agnostic
framework for training Vision-Language-Action (VLA) models** from real-world and
simulated demonstrations.

If you can install Python and run one command, you can train and run your first
language-conditioned policy in a couple of minutes — no webcam, GPU, or robot
required (a dependency-free simulated robot ships with the framework).

## Table of contents

1. [Installation](installation.md) — install options and verification.
2. [Quickstart](quickstart.md) — train and run your first policy in 5 minutes.
3. [Concepts](concepts.md) — how the framework is designed and why.
4. [How it works (full architecture)](how-it-works.md) — the complete deep-dive: every layer, data flow, and internals.
5. [CLI reference](cli.md) — every command and flag.
6. [Python API](python-api.md) — using the framework from code.
7. [Extending DifoTrain](extending.md) — add your own robot, data source, or model.
8. [Learning from human video](human-video.md) — the MediaPipe capture path.
9. [Examples](examples.md) — runnable end-to-end snippets.
10. [FAQ & troubleshooting](faq.md).

## The 30-second version

```bash
pip install difotrain
difotrain collect --out data/reach --episodes 200   # gather demonstrations
difotrain train   --data data/reach --out runs/policy.pt --epochs 150
difotrain eval    --policy runs/policy.pt            # ~87% task success
difotrain deploy  --policy runs/policy.pt --instruction "reach to the up"
```

## How it fits together

```
COLLECT ──▶ STORE ──▶ TRAIN ──▶ EVAL ──▶ DEPLOY ──┐
 DataSource  Episode-  BC        success  Robot +  │
            Dataset   Trainer    rate     Safety   │
   ▲                                                │
   └──────────── feedback loop (log rollouts) ◀─────┘
```

Everything is built around four pluggable interfaces — `Robot`, `DataSource`,
`Policy`, `EpisodeDataset` — so you can swap simulation for real hardware, teleop
for human video, or a native model for a wrapped OpenVLA without changing the
rest of the pipeline. See [Concepts](concepts.md).
