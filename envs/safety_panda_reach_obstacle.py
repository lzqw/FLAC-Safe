from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:  # pragma: no cover - exercised by integration tests when pybullet exists.
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover
    p = None
    pybullet_data = None
    _PYBULLET_IMPORT_ERROR = exc
else:
    _PYBULLET_IMPORT_ERROR = None


ENV_ID = "SafetyPandaReachObstacle-v0"


@dataclass(frozen=True)
class PandaTaskConfig:
    action_scale: float = 0.03
    max_episode_steps: int = 100
    success_radius: float = 0.05
    obstacle_radius: float = 0.07
    safe_margin: float = 0.13
    success_bonus: float = 1.0
    action_penalty: float = 0.01
    start_center: tuple[float, float, float] = (0.45, -0.18, 0.25)
    goal_center: tuple[float, float, float] = (0.55, 0.18, 0.25)
    obstacle_center: tuple[float, float, float] = (0.50, 0.00, 0.25)
    start_noise: float = 0.02
    goal_noise: float = 0.04
    obstacle_noise: float = 0.03
    workspace_low: tuple[float, float, float] = (0.35, -0.25, 0.10)
    workspace_high: tuple[float, float, float] = (0.65, 0.25, 0.45)
    control_substeps: int = 10


class SafetyPandaReachObstacleEnv(gym.Env):
    """Headless PyBullet Panda reaching task with an explicit keep-out cost.

    The policy controls Cartesian end-effector deltas.  Reward is dense reaching
    progress; obstacle avoidance is exposed as a safety cost in ``info["cost"]``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    PANDA_JOINTS = tuple(range(7))
    EE_LINK = 8

    def __init__(
        self,
        render_mode: str | None = None,
        config: PandaTaskConfig | None = None,
        deterministic_resets: bool = False,
    ) -> None:
        if p is None:
            raise ImportError(f"pybullet is required for {ENV_ID}: {_PYBULLET_IMPORT_ERROR!r}")
        self.render_mode = render_mode
        self.cfg = config or PandaTaskConfig()
        self.deterministic_resets = bool(deterministic_resets)
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + 3 + 3 + 2 + 7 + 7
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._rng = np.random.default_rng()
        self.step_count = 0
        self.robot_id: int | None = None
        self.obstacle_id: int | None = None
        self.goal_pos = np.asarray(self.cfg.goal_center, dtype=np.float32)
        self.obstacle_pos = np.asarray(self.cfg.obstacle_center, dtype=np.float32)
        self.start_pos = np.asarray(self.cfg.start_center, dtype=np.float32)
        self.prev_ee_pos = self.start_pos.copy()
        self.trajectory: list[np.ndarray] = []
        self._last_path_increment = 0.0
        self._load_world()

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng = np.random.default_rng(seed)
        self.action_space.seed(seed)
        return [seed]

    def _load_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=(0.0, 0.0, 0.0),
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        for joint in self.PANDA_JOINTS:
            p.changeDynamics(self.robot_id, joint, linearDamping=0.04, angularDamping=0.04, physicsClientId=self.client_id)
        self._make_obstacle()

    def _make_obstacle(self) -> None:
        if self.obstacle_id is not None:
            try:
                p.removeBody(self.obstacle_id, physicsClientId=self.client_id)
            except Exception:
                pass
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.cfg.obstacle_radius,
            rgbaColor=(0.85, 0.12, 0.10, 0.85),
            physicsClientId=self.client_id,
        )
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.cfg.obstacle_radius,
            physicsClientId=self.client_id,
        )
        self.obstacle_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self.obstacle_pos.tolist(),
            physicsClientId=self.client_id,
        )

    def _sample_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        start = np.asarray(cfg.start_center, dtype=np.float32)
        goal = np.asarray(cfg.goal_center, dtype=np.float32)
        obstacle = np.asarray(cfg.obstacle_center, dtype=np.float32)
        if self.deterministic_resets:
            return start, goal, obstacle
        low = np.asarray(cfg.workspace_low, dtype=np.float32)
        high = np.asarray(cfg.workspace_high, dtype=np.float32)
        for _ in range(200):
            s = start + self._rng.uniform(-cfg.start_noise, cfg.start_noise, size=3)
            g = goal + self._rng.uniform(-cfg.goal_noise, cfg.goal_noise, size=3)
            o = obstacle + self._rng.uniform(-cfg.obstacle_noise, cfg.obstacle_noise, size=3)
            s = np.clip(s, low, high)
            g = np.clip(g, low, high)
            o = np.clip(o, low, high)
            if np.linalg.norm(s - o) > cfg.safe_margin + 0.02 and np.linalg.norm(g - o) > cfg.safe_margin + 0.02:
                return s.astype(np.float32), g.astype(np.float32), o.astype(np.float32)
        return start, goal, obstacle

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self.robot_id, self.PANDA_JOINTS, physicsClientId=self.client_id)
        q = np.asarray([s[0] for s in states], dtype=np.float32)
        dq = np.asarray([s[1] for s in states], dtype=np.float32)
        return q, dq

    def _ee_position(self) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id,
            self.EE_LINK,
            computeLinkVelocity=1,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[4], dtype=np.float32)

    def _ee_velocity(self) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id,
            self.EE_LINK,
            computeLinkVelocity=1,
            physicsClientId=self.client_id,
        )
        if state[6] is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(state[6], dtype=np.float32)

    def _set_ee_target(self, target: np.ndarray, settle_steps: int | None = None) -> None:
        target = np.asarray(target, dtype=np.float32)
        orn = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))
        joints = p.calculateInverseKinematics(
            self.robot_id,
            self.EE_LINK,
            target.tolist(),
            targetOrientation=orn,
            maxNumIterations=80,
            residualThreshold=1e-4,
            physicsClientId=self.client_id,
        )
        for joint, value in zip(self.PANDA_JOINTS, joints[:7]):
            p.resetJointState(self.robot_id, joint, float(value), targetVelocity=0.0, physicsClientId=self.client_id)
        for _ in range(settle_steps if settle_steps is not None else 1):
            p.stepSimulation(physicsClientId=self.client_id)

    def _reset_robot_to(self, target: np.ndarray) -> None:
        orn = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))
        joints = p.calculateInverseKinematics(
            self.robot_id,
            self.EE_LINK,
            np.asarray(target, dtype=np.float32).tolist(),
            targetOrientation=orn,
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )
        for joint, value in zip(self.PANDA_JOINTS, joints[:7]):
            p.resetJointState(self.robot_id, joint, float(value), targetVelocity=0.0, physicsClientId=self.client_id)
        p.stepSimulation(physicsClientId=self.client_id)

    def _observation(self) -> np.ndarray:
        ee = self._ee_position()
        vel = self._ee_velocity()
        q, dq = self._joint_state()
        goal_vec = self.goal_pos - ee
        obs_vec = self.obstacle_pos - ee
        distances = np.asarray([np.linalg.norm(goal_vec), np.linalg.norm(obs_vec)], dtype=np.float32)
        obs = np.concatenate([ee, vel, self.goal_pos, self.obstacle_pos, goal_vec, obs_vec, distances, q, dq])
        return obs.astype(np.float32)

    def _safety_info(self) -> dict[str, float]:
        ee = self._ee_position()
        distance_to_goal = float(np.linalg.norm(self.goal_pos - ee))
        distance_to_obstacle = float(np.linalg.norm(self.obstacle_pos - ee))
        contacts = p.getContactPoints(self.robot_id, self.obstacle_id, physicsClientId=self.client_id)
        collision = float(len(contacts) > 0)
        soft_keepout = float(distance_to_obstacle <= self.cfg.safe_margin)
        cost = float(max(soft_keepout, collision))
        return {
            "cost": cost,
            "violation": cost,
            "collision": collision,
            "soft_keepout": soft_keepout,
            "success": float(distance_to_goal < self.cfg.success_radius),
            "distance_to_goal": distance_to_goal,
            "distance_to_obstacle": distance_to_obstacle,
            "min_clearance": distance_to_obstacle - self.cfg.obstacle_radius,
            "path_length_increment": float(self._last_path_increment),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.step_count = 0
        self.start_pos, self.goal_pos, self.obstacle_pos = self._sample_positions()
        if options:
            if "start_pos" in options:
                self.start_pos = np.asarray(options["start_pos"], dtype=np.float32)
            if "goal_pos" in options:
                self.goal_pos = np.asarray(options["goal_pos"], dtype=np.float32)
            if "obstacle_pos" in options:
                self.obstacle_pos = np.asarray(options["obstacle_pos"], dtype=np.float32)
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            self.obstacle_pos.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        self._reset_robot_to(self.start_pos)
        self.prev_ee_pos = self._ee_position()
        self.trajectory = [self.prev_ee_pos.copy()]
        self._last_path_increment = 0.0
        return self._observation(), self._safety_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        old_ee = self._ee_position()
        target = old_ee + action * self.cfg.action_scale
        target = np.clip(target, np.asarray(self.cfg.workspace_low), np.asarray(self.cfg.workspace_high))
        self._set_ee_target(target)
        ee = self._ee_position()
        self._last_path_increment = float(np.linalg.norm(ee - old_ee))
        self.trajectory.append(ee.copy())
        self.step_count += 1

        info = self._safety_info()
        reward = -info["distance_to_goal"] + self.cfg.success_bonus * info["success"]
        reward -= self.cfg.action_penalty * float(np.square(action).sum())
        terminated = bool(info["success"] >= 1.0)
        truncated = self.step_count >= self.cfg.max_episode_steps
        return self._observation(), float(reward), terminated, truncated, info

    def debug_set_positions(
        self,
        *,
        ee_pos: np.ndarray | None = None,
        goal_pos: np.ndarray | None = None,
        obstacle_pos: np.ndarray | None = None,
    ) -> None:
        if goal_pos is not None:
            self.goal_pos = np.asarray(goal_pos, dtype=np.float32)
        if obstacle_pos is not None:
            self.obstacle_pos = np.asarray(obstacle_pos, dtype=np.float32)
            p.resetBasePositionAndOrientation(
                self.obstacle_id,
                self.obstacle_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )
        if ee_pos is not None:
            self._reset_robot_to(np.asarray(ee_pos, dtype=np.float32))
            self.prev_ee_pos = self._ee_position()

    def render(self):
        width, height = 640, 480
        view = p.computeViewMatrix(
            cameraEyePosition=[0.75, -0.75, 0.85],
            cameraTargetPosition=[0.50, 0.0, 0.25],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        proj = p.computeProjectionMatrixFOV(fov=50, aspect=width / height, nearVal=0.01, farVal=4.0)
        image = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.client_id)
        rgba = np.reshape(image[2], (height, width, 4))
        return rgba[:, :, :3].astype(np.uint8)

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
            self.client_id = None


class FlattenPandaObsWrapper(gym.ObservationWrapper):
    """Flatten Panda-style dict observations when a backend returns them."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            size = 0
            for space in env.observation_space.spaces.values():
                size += int(np.prod(space.shape))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def observation(self, observation):
        if isinstance(observation, dict):
            parts = []
            for key in sorted(observation):
                parts.append(np.asarray(observation[key], dtype=np.float32).reshape(-1))
            return np.concatenate(parts).astype(np.float32)
        return np.asarray(observation, dtype=np.float32).reshape(-1)


def plot_topdown_trajectory(
    trajectories: dict[str, np.ndarray],
    *,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle: np.ndarray,
    obstacle_radius: float,
    safe_margin: float,
    output_path: str | Path,
    shadow_points: np.ndarray | None = None,
    shadow_risk: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    colors = {
        "SAC-Lag": "#4C78A8",
        "Current-only-N": "#F58518",
        "STAR": "#54A24B",
        "STAR+Exec": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.add_patch(Circle(obstacle[:2], safe_margin, color="#F2B8B5", alpha=0.35, label="unsafe keep-out"))
    ax.add_patch(Circle(obstacle[:2], obstacle_radius, color="#C9352B", alpha=0.9, label="obstacle"))
    ax.scatter(start[0], start[1], marker="o", s=70, color="black", label="start", zorder=5)
    ax.scatter(goal[0], goal[1], marker="*", s=160, color="#2F6FED", label="target", zorder=6)
    for label, points in trajectories.items():
        points = np.asarray(points)
        if points.size == 0:
            continue
        ax.plot(points[:, 0], points[:, 1], lw=2.2, color=colors.get(label, None), label=label)
        ax.scatter(points[-1, 0], points[-1, 1], s=36, color=colors.get(label, None), zorder=6)
    if shadow_points is not None and len(shadow_points):
        risk = np.zeros(len(shadow_points)) if shadow_risk is None else np.asarray(shadow_risk)
        sc = ax.scatter(shadow_points[:, 0], shadow_points[:, 1], c=risk, cmap="magma_r", s=24, alpha=0.8, label="shadow endpoints")
        fig.colorbar(sc, ax=ax, label="predicted risk")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Panda obstacle-reaching safety showcase")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def register_env() -> None:
    try:
        gym.spec(ENV_ID)
    except gym.error.Error:
        gym.register(
            id=ENV_ID,
            entry_point="envs.safety_panda_reach_obstacle:SafetyPandaReachObstacleEnv",
            max_episode_steps=100,
        )


register_env()
