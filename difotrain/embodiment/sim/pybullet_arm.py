"""PyBullet robot backend.

Drives a URDF robot arm in the PyBullet physics simulator behind the standard
:class:`Robot` interface, so the same policies / trainer / deploy runner that
work with :class:`PlanarArm` also work in full 3D physics.

PyBullet is an optional dependency (``pip install "difotrain[sim]"``) and is
imported lazily, so the rest of the framework works without it. By default it
loads the Kuka iiwa arm shipped with ``pybullet_data``; pass a ``urdf`` path to
use any other robot.

Observation : ``[joint_positions (n_dof), end_effector_xyz (3)]``
Action      : per-joint position deltas, integrated and clipped each step.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...core.registry import register_robot
from ...core.spaces import Box
from ..base import Robot, RobotSpec


@register_robot("pybullet_arm")
class PyBulletArm(Robot):
    def __init__(
        self,
        urdf: Optional[str] = None,
        gui: bool = False,
        control_hz: float = 20.0,
        max_delta: float = 0.1,
        end_effector_link: Optional[int] = None,
    ):
        try:
            import pybullet as p
            import pybullet_data
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "PyBulletArm requires pybullet. Install with: "
                'pip install "difotrain[sim]"'
            ) from e

        self._p = p
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        urdf = urdf or "kuka_iiwa/model.urdf"
        self._body = p.loadURDF(urdf, useFixedBase=True, physicsClientId=self._client)

        # Discover controllable (non-fixed) joints.
        self._joints: List[int] = []
        lowers, uppers = [], []
        for j in range(p.getNumJoints(self._body, physicsClientId=self._client)):
            info = p.getJointInfo(self._body, j, physicsClientId=self._client)
            if info[2] != p.JOINT_FIXED:
                self._joints.append(j)
                lo, hi = info[8], info[9]
                if lo >= hi:  # unlimited joint
                    lo, hi = -np.pi, np.pi
                lowers.append(lo)
                uppers.append(hi)

        self._lowers = np.array(lowers, dtype=np.float32)
        self._uppers = np.array(uppers, dtype=np.float32)
        n = len(self._joints)
        self._ee_link = end_effector_link if end_effector_link is not None else self._joints[-1]
        self.max_delta = max_delta

        obs_low = np.concatenate([self._lowers, np.full(3, -5.0, dtype=np.float32)])
        obs_high = np.concatenate([self._uppers, np.full(3, 5.0, dtype=np.float32)])
        self.spec = RobotSpec(
            name="pybullet_arm",
            dof=n,
            observation_space=Box(low=obs_low, high=obs_high),
            action_space=Box(low=np.full(n, -1.0), high=np.full(n, 1.0)),
            control_hz=control_hz,
            urdf=urdf,
            metadata={"end_effector_link": self._ee_link},
        )
        self._rng = np.random.default_rng()
        self.reset()

    # Robot API ---------------------------------------------------------
    def reset(self, *, instruction: str = "", seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mid = (self._lowers + self._uppers) / 2.0
        for j, q in zip(self._joints, mid):
            self._p.resetJointState(self._body, j, float(q), physicsClientId=self._client)
        return self.get_observation()

    def _joint_positions(self) -> np.ndarray:
        states = self._p.getJointStates(self._body, self._joints, physicsClientId=self._client)
        return np.array([s[0] for s in states], dtype=np.float32)

    def end_effector(self) -> np.ndarray:
        state = self._p.getLinkState(self._body, self._ee_link, physicsClientId=self._client)
        return np.array(state[0], dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        return np.concatenate([self._joint_positions(), self.end_effector()]).astype(np.float32)

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        action = self.spec.action_space.clip(action)
        target = self._joint_positions() + action * self.max_delta
        target = np.clip(target, self._lowers, self._uppers)
        self._p.setJointMotorControlArray(
            self._body,
            self._joints,
            self._p.POSITION_CONTROL,
            targetPositions=target.tolist(),
            physicsClientId=self._client,
        )
        self._p.stepSimulation(physicsClientId=self._client)
        return self.get_observation()

    def close(self) -> None:
        try:
            self._p.disconnect(physicsClientId=self._client)
        except Exception:  # pragma: no cover
            pass
