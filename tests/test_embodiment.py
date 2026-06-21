import math
import unittest

import numpy as np

from difotrain.embodiment.retarget.base import LinearRetargeter
from difotrain.embodiment.sim.planar_arm import (
    NAMED_TARGETS,
    PlanarArm,
    forward_kinematics,
    instruction_for,
    inverse_kinematics,
    parse_instruction,
    target_xy,
)


class TestPlanarArm(unittest.TestCase):
    def test_spec(self):
        arm = PlanarArm()
        self.assertEqual(arm.spec.dof, 2)
        self.assertEqual(arm.obs_dim, 4)
        self.assertEqual(arm.act_dim, 2)

    def test_reset_deterministic(self):
        a = PlanarArm().reset(seed=1)
        b = PlanarArm().reset(seed=1)
        np.testing.assert_array_equal(a, b)

    def test_ik_fk_consistency(self):
        for name in NAMED_TARGETS:
            tx, ty = target_xy(name)
            q = inverse_kinematics(tx, ty)
            ee = forward_kinematics(q)
            self.assertAlmostEqual(ee[0], tx, places=4)
            self.assertAlmostEqual(ee[1], ty, places=4)

    def test_expert_reaches_target(self):
        arm = PlanarArm()
        arm.reset(seed=3)
        for _ in range(60):
            arm.apply_action(arm.expert_action("up"))
        ee = arm.end_effector()
        tx, ty = target_xy("up")
        self.assertLess(np.linalg.norm(ee - np.array([tx, ty])), 0.15)

    def test_action_clipping(self):
        arm = PlanarArm()
        arm.reset(seed=0)
        obs = arm.apply_action(np.array([100.0, -100.0]))  # huge action
        self.assertTrue(arm.spec.observation_space.contains(obs))

    def test_parse_instruction_prefers_longest(self):
        self.assertEqual(parse_instruction(instruction_for("upper right")), "upper right")
        self.assertEqual(parse_instruction(instruction_for("right")), "right")


class TestRetarget(unittest.TestCase):
    def test_identity_truncation(self):
        arm = PlanarArm()
        rt = LinearRetargeter(arm.spec)
        out = rt.retarget(np.array([0.5, -0.3, 9.9]))  # 3 -> 2 dims
        self.assertEqual(out.shape[0], 2)
        self.assertTrue(arm.spec.action_space.contains(out))


if __name__ == "__main__":
    unittest.main()
