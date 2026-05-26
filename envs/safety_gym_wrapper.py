import gymnasium as gym
import safety_gymnasium
from safety_gymnasium.wrappers import SafeRescaleAction


class BinaryCostWrapper(gym.Wrapper):
    """Convert Safety-Gymnasium cost to binary cost and keep it in info."""

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        cost = float(cost > 0)
        info = dict(info)
        info["cost"] = cost
        return obs, reward, cost, terminated, truncated, info


class RewardScalingWrapper(gym.Wrapper):
    """Scale rewards for some Safety-Gymnasium tasks."""

    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.reward_scale = float(reward_scale)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.reward_scale, cost, terminated, truncated, info


def get_reward_scale(env_id: str) -> float:
    """Use a small reward scale for high-return velocity tasks."""
    if "Velocity" in env_id and "Swimmer" not in env_id:
        return 0.01
    return 1.0


def make_safe_env(env_id: str, train: bool = True):
    """Create a Safety-Gymnasium env with action rescaling and cost wrapper."""
    env = safety_gymnasium.make(env_id)
    env = SafeRescaleAction(env, -1.0, 1.0)
    env = BinaryCostWrapper(env)
    if train:
        env = RewardScalingWrapper(env, get_reward_scale(env_id))
    return env
