from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("pybullet")

from envs.safety_panda_reach_obstacle import ENV_ID, FlattenPandaObsWrapper, SafetyPandaReachObstacleEnv


def make_env():
    return gym.make(ENV_ID)


def test_env_reset_works():
    env = make_env()
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
    finally:
        env.close()


def test_env_step_works_and_info_contains_cost():
    env = make_env()
    try:
        obs, _ = env.reset(seed=123)
        next_obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert next_obs.shape == obs.shape
        assert np.isfinite(next_obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "cost" in info
    finally:
        env.close()


def test_obs_is_finite_flat_vector():
    env = make_env()
    try:
        obs, _ = env.reset(seed=3)
        assert obs.ndim == 1
        assert np.isfinite(obs).all()
    finally:
        env.close()


def test_action_space_is_three_dimensional_box():
    env = make_env()
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (3,)
    finally:
        env.close()


def test_cost_becomes_one_inside_obstacle_safe_margin():
    env = make_env()
    try:
        env.reset(seed=4)
        base = env.unwrapped
        ee = base._ee_position()
        base.debug_set_positions(obstacle_pos=ee)
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["cost"] == 1.0
        assert info["soft_keepout"] == 1.0
    finally:
        env.close()


def test_success_triggers_near_goal():
    env = make_env()
    try:
        env.reset(seed=5)
        base = env.unwrapped
        ee = base._ee_position()
        base.debug_set_positions(goal_pos=ee)
        _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["success"] == 1.0
        assert terminated
    finally:
        env.close()


def test_random_seeds_are_reproducible():
    env1 = make_env()
    env2 = make_env()
    try:
        obs1, info1 = env1.reset(seed=7)
        obs2, info2 = env2.reset(seed=7)
        np.testing.assert_allclose(obs1, obs2, atol=1e-5)
        assert info1["distance_to_goal"] == pytest.approx(info2["distance_to_goal"], abs=1e-5)
    finally:
        env1.close()
        env2.close()


def test_flatten_wrapper_keeps_existing_flat_obs_compatible():
    env = FlattenPandaObsWrapper(SafetyPandaReachObstacleEnv())
    try:
        obs, _ = env.reset(seed=8)
        assert obs.ndim == 1
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
    finally:
        env.close()


def test_existing_agent_code_can_read_cost_from_gym_step():
    env = make_env()
    try:
        env.reset(seed=9)
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        cost = float(info.get("cost", 0.0))
        assert np.isfinite(reward)
        assert cost in (0.0, 1.0)
        assert isinstance(terminated or truncated, bool)
    finally:
        env.close()


def test_random_steps_run_without_crash():
    env = make_env()
    try:
        obs, _ = env.reset(seed=10)
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert info["cost"] in (0.0, 1.0)
            if terminated or truncated:
                obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()
