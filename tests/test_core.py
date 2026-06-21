import unittest

import numpy as np

from difotrain.core.episode import Episode, EpisodeMeta
from difotrain.core.language import LanguageEncoder
from difotrain.core.registry import registry
from difotrain.core.spaces import Box


class TestSpaces(unittest.TestCase):
    def test_box_clip_and_contains(self):
        box = Box(low=[-1, -1], high=[1, 1])
        self.assertEqual(box.dim, 2)
        self.assertTrue(box.contains(np.array([0.5, -0.5])))
        self.assertFalse(box.contains(np.array([2.0, 0.0])))
        np.testing.assert_allclose(box.clip(np.array([5.0, -5.0])), [1.0, -1.0])


class TestEpisode(unittest.TestCase):
    def test_shapes_and_meta(self):
        ep = Episode(
            observations=np.zeros((5, 4)),
            actions=np.zeros((5, 2)),
            meta=EpisodeMeta(instruction="go", success=True),
        )
        self.assertEqual(len(ep), 5)
        self.assertEqual(ep.obs_dim, 4)
        self.assertEqual(ep.act_dim, 2)
        self.assertEqual(ep.instruction, "go")

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            Episode(np.zeros((5, 4)), np.zeros((4, 2)))


class TestLanguage(unittest.TestCase):
    def test_fit_and_encode(self):
        enc = LanguageEncoder.fit(["reach to the left", "reach to the right"])
        self.assertIn("left", enc.vocab)
        self.assertIn("right", enc.vocab)
        left = enc.encode("reach to the left")
        right = enc.encode("reach to the right")
        self.assertEqual(left.shape[0], enc.dim)
        self.assertFalse(np.allclose(left, right))  # language must differ

    def test_roundtrip(self):
        enc = LanguageEncoder.fit(["up down"])
        enc2 = LanguageEncoder.from_dict(enc.to_dict())
        np.testing.assert_array_equal(enc.encode("up"), enc2.encode("up"))


class TestRegistry(unittest.TestCase):
    def test_planar_arm_registered(self):
        import difotrain.embodiment  # noqa: F401  (triggers registration)

        robot = registry.make_robot("planar_arm")
        self.assertEqual(robot.spec.name, "planar_arm")

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            registry.make_robot("does_not_exist")


if __name__ == "__main__":
    unittest.main()
