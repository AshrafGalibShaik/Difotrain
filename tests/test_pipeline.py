"""End-to-end: collect -> train -> eval -> deploy on the planar-arm reach task.

This is the integration test that proves the framework actually learns language
-conditioned behavior from demonstrations.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from difotrain.data.dataset import EpisodeDataset
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.deploy.runner import DeployRunner
from difotrain.deploy.safety import SafetyLayer
from difotrain.embodiment.sim.planar_arm import PlanarArm, target_xy
from difotrain.eval.evaluator import evaluate_reaching
from difotrain.policy.native.mlp_vla import MLPVLAPolicy
from difotrain.train.trainer import TrainConfig, train_policy


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_learns(self):
        with tempfile.TemporaryDirectory() as d:
            # 1. Collect
            ds = EpisodeDataset(Path(d) / "data")
            ds.extend(ScriptedTeleopSource(max_steps=40, seed=0).collect(200))
            self.assertGreaterEqual(len(ds), 200)

            # 2. Train
            policy = train_policy(ds, TrainConfig(epochs=120, seed=0))

            # 3. Eval: a trained policy should ground language and reach targets.
            result = evaluate_reaching(policy)
            self.assertGreater(
                result.success_rate, 0.7, f"low success:\n{result}"
            )

            # 4. Save / load roundtrip preserves behavior
            ckpt = Path(d) / "policy.pt"
            policy.save(ckpt)
            loaded = MLPVLAPolicy.load(ckpt)
            obs = np.zeros(4, dtype=np.float32)
            np.testing.assert_allclose(
                policy.predict(obs, "reach to the up"),
                loaded.predict(obs, "reach to the up"),
                atol=1e-5,
            )

    def test_language_changes_behavior(self):
        """Different instructions must produce different actions / outcomes."""
        with tempfile.TemporaryDirectory() as d:
            ds = EpisodeDataset(Path(d) / "data")
            ds.extend(ScriptedTeleopSource(max_steps=40, seed=1).collect(200))
            policy = train_policy(ds, TrainConfig(epochs=120, seed=0))

            robot = PlanarArm()
            runner = DeployRunner(robot, policy, SafetyLayer(robot.spec))

            runner.run("reach to the up", max_steps=60, seed=5)
            ee_up = robot.end_effector().copy()
            runner.run("reach to the down", max_steps=60, seed=5)
            ee_down = robot.end_effector().copy()

            # Should end up near the respective commanded targets.
            self.assertLess(np.linalg.norm(ee_up - np.array(target_xy("up"))), 0.25)
            self.assertLess(np.linalg.norm(ee_down - np.array(target_xy("down"))), 0.25)


if __name__ == "__main__":
    unittest.main()
