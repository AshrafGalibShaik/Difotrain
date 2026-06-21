"""DifoTrain command line interface.

Subcommands cover the full lifecycle:

    difotrain collect  --out data/reach --episodes 200
    difotrain train    --data data/reach --out runs/policy.pt --epochs 150
    difotrain eval     --policy runs/policy.pt
    difotrain deploy   --policy runs/policy.pt --instruction "reach to the up"
    difotrain info

Plus the legacy capture helpers:

    difotrain setup     # download the MediaPipe model
    difotrain record    # capture human motion from a webcam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_collect(args: argparse.Namespace) -> int:
    from .data.dataset import EpisodeDataset
    from .data.sources.scripted_teleop import ScriptedTeleopSource

    source = ScriptedTeleopSource(max_steps=args.steps, seed=args.seed)
    dataset = EpisodeDataset(args.out)
    n_success = 0
    for ep in source.collect(args.episodes):
        dataset.add(ep)
        n_success += int(bool(ep.meta.success))
    print(f"Collected {len(dataset)} episodes into {args.out}")
    print(f"Expert success rate: {n_success}/{len(dataset)}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .data.dataset import EpisodeDataset
    from .train.trainer import TrainConfig, train_policy

    dataset = EpisodeDataset(args.data)
    if len(dataset) == 0:
        print(f"No episodes found in {args.data}. Run 'difotrain collect' first.")
        return 1
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    policy = train_policy(dataset, cfg, verbose=True)
    policy.save(args.out)
    print(f"Saved policy to {args.out} (final loss {policy._final_loss:.6f})")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval.evaluator import evaluate_reaching
    from .policy.native.mlp_vla import MLPVLAPolicy

    policy = MLPVLAPolicy.load(args.policy)
    result = evaluate_reaching(policy, success_tol=args.tol)
    print(result)
    return 0


def _cmd_deploy(args: argparse.Namespace) -> int:
    from .deploy.runner import DeployRunner
    from .deploy.safety import SafetyLayer
    from .embodiment.sim.planar_arm import PlanarArm
    from .policy.native.mlp_vla import MLPVLAPolicy

    policy = MLPVLAPolicy.load(args.policy)
    robot = PlanarArm()
    runner = DeployRunner(robot, policy, SafetyLayer(robot.spec))
    runner.run(args.instruction, max_steps=args.steps)
    ee = robot.end_effector()
    print(f"Instruction: {args.instruction!r}")
    print(f"Final end-effector position: ({ee[0]:.3f}, {ee[1]:.3f})")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from . import __version__
    from .core.registry import registry
    from .embodiment.sim.planar_arm import NAMED_TARGETS

    # Importing the sources/policies/robots registers them.
    from .data import sources  # noqa: F401
    from .policy.native import mlp_vla  # noqa: F401
    from .embodiment import sim  # noqa: F401

    print(f"DifoTrain v{__version__}")
    print(f"  robots   : {sorted(registry.robots)}")
    print(f"  sources  : {sorted(registry.sources)}")
    print(f"  targets  : {list(NAMED_TARGETS)}")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from .__main__ import setup_mediapipe

    setup_mediapipe()
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    from .capture.record_pose import main as record_main

    record_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="difotrain",
        description="Embodiment-agnostic VLA training framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="Collect demonstrations into a dataset")
    p.add_argument("--out", default="data/reach")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("train", help="Train a policy on a dataset")
    p.add_argument("--data", default="data/reach")
    p.add_argument("--out", default="runs/policy.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="Evaluate a policy in simulation")
    p.add_argument("--policy", default="runs/policy.pt")
    p.add_argument("--tol", type=float, default=0.15)
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("deploy", help="Run a policy on a robot")
    p.add_argument("--policy", default="runs/policy.pt")
    p.add_argument("--instruction", default="reach to the up")
    p.add_argument("--steps", type=int, default=60)
    p.set_defaults(func=_cmd_deploy)

    p = sub.add_parser("info", help="Show version and registered plugins")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("setup", help="Download required MediaPipe models")
    p.set_defaults(func=_cmd_setup)

    p = sub.add_parser("record", help="Record human motion from a webcam")
    p.set_defaults(func=_cmd_record)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
