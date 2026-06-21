import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from difotrain.data.dataset import EpisodeDataset
from difotrain.data.normalize import Normalizer
from difotrain.data.sources.human_video import HumanVideoSource, pose_to_features
from difotrain.data.sources.scripted_teleop import ScriptedTeleopSource
from difotrain.data.sources.synthetic import SyntheticSource
from difotrain.embodiment.sim.planar_arm import PlanarArm


class TestDataset(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            ds = EpisodeDataset(d)
            for ep in SyntheticSource(seed=0).collect(3):
                ds.add(ep)
            self.assertEqual(len(ds), 3)
            # Reload from disk
            ds2 = EpisodeDataset(d)
            self.assertEqual(len(ds2), 3)
            obs, act, instr = ds2.stacked()
            self.assertEqual(obs.shape[0], act.shape[0])
            self.assertEqual(len(instr), obs.shape[0])
            self.assertTrue((Path(d) / "meta.json").exists())


class TestNormalizer(unittest.TestCase):
    def test_normalize_denormalize(self):
        data = np.random.randn(100, 3).astype(np.float32) * 5 + 2
        norm = Normalizer.fit(data)
        z = norm.normalize(data)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=4)
        np.testing.assert_allclose(norm.denormalize(z), data, atol=1e-4)

    def test_zero_std_safe(self):
        data = np.ones((10, 2), dtype=np.float32)
        norm = Normalizer.fit(data)
        self.assertFalse(np.any(np.isnan(norm.normalize(data))))


class TestScriptedTeleop(unittest.TestCase):
    def test_collect_success(self):
        src = ScriptedTeleopSource(max_steps=60, seed=0)
        eps = list(src.collect(8))
        self.assertEqual(len(eps), 8)
        # The scripted expert should mostly succeed.
        rate = np.mean([e.meta.success for e in eps])
        self.assertGreater(rate, 0.7)
        self.assertEqual(eps[0].obs_dim, 4)
        self.assertEqual(eps[0].act_dim, 2)


class TestHumanVideo(unittest.TestCase):
    def test_offline_trajectory(self):
        joints = {
            "left_shoulder": [0, 0, 0],
            "left_elbow": [1, 0, 0],
            "left_wrist": [1, 1, 0],
            "right_shoulder": [0, 0, 0],
            "right_elbow": [1, 0, 0],
            "right_wrist": [1, 1, 0],
        }
        raw = [{"timestamp": t * 0.1, "joints": joints} for t in range(5)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "traj.json"
            p.write_text(json.dumps(raw), encoding="utf-8")
            arm = PlanarArm()
            src = HumanVideoSource(arm.spec, trajectory_json=str(p))
            eps = list(src.collect(1))
            self.assertEqual(len(eps), 1)
            self.assertEqual(len(eps[0]), 5)
            self.assertEqual(eps[0].act_dim, 2)

    def test_pose_to_features(self):
        feats = pose_to_features(
            {
                "left_shoulder": [0, 0, 0],
                "left_elbow": [1, 0, 0],
                "left_wrist": [1, 1, 0],
            }
        )
        self.assertEqual(feats.shape[0], 2)  # two elbow angles
        self.assertAlmostEqual(float(feats[0]), np.pi / 2, places=4)


if __name__ == "__main__":
    unittest.main()
