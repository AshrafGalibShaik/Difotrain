"""Tests for the optional PyBullet backend.

Skipped automatically if pybullet is not installed, so the suite stays green on
a core-only install.
"""
import unittest

import numpy as np

try:
    import pybullet  # noqa: F401
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


@unittest.skipUnless(HAS_PYBULLET, "pybullet not installed")
class TestPyBulletArm(unittest.TestCase):
    def setUp(self):
        from difotrain.embodiment.sim.pybullet_arm import PyBulletArm

        self.robot = PyBulletArm(gui=False)

    def tearDown(self):
        self.robot.close()

    def test_spec_and_spaces(self):
        self.assertEqual(self.robot.spec.name, "pybullet_arm")
        self.assertGreater(self.robot.spec.dof, 0)
        # observation = joint positions + end-effector xyz
        self.assertEqual(self.robot.obs_dim, self.robot.spec.dof + 3)

    def test_reset_returns_valid_observation(self):
        obs = self.robot.reset(seed=0)
        self.assertEqual(obs.shape[0], self.robot.obs_dim)
        self.assertTrue(self.robot.spec.observation_space.contains(obs))

    def test_apply_action_moves_and_stays_in_bounds(self):
        self.robot.reset(seed=0)
        before = self.robot.end_effector().copy()
        action = np.ones(self.robot.act_dim, dtype=np.float32)
        obs = None
        for _ in range(20):
            obs = self.robot.apply_action(action)
        self.assertTrue(self.robot.spec.observation_space.contains(obs))
        after = self.robot.end_effector()
        # Commanding all joints should move the end effector.
        self.assertGreater(float(np.linalg.norm(after - before)), 1e-3)

    def test_registered(self):
        from difotrain.core.registry import registry

        robot = registry.make_robot("pybullet_arm")
        try:
            self.assertEqual(robot.spec.name, "pybullet_arm")
        finally:
            robot.close()


if __name__ == "__main__":
    unittest.main()
