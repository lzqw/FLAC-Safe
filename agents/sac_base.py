from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model.model import QNetwork
from model.utils import hard_update, soft_update
from utilis.utils import RunningMeanStd

from .gaussian_policy import SquashedGaussianPolicy


class SACBase:
    def __init__(self, num_inputs: int, action_space, config) -> None:
        self.num_inputs = int(num_inputs)
        self.action_space = action_space
        self.action_dim = int(action_space.shape[0])
        self.gamma = float(config.gamma)
        self.cost_gamma = float(config.cost_gamma)
        self.target_tau = float(config.target_tau)
        self.target_update_interval = int(getattr(config, "target_update_interval", 1))
        self.policy_update_interval = int(config.policy_update_interval)
        self.batch_size = int(config.batch_size)
        self.device = torch.device(
            f"cuda:{getattr(config, 'device', 0)}"
            if bool(config.cuda) and torch.cuda.is_available()
            else "cpu"
        )

        self.obs_norm_clip = float(getattr(config, "obs_norm_clip", 10.0))
        self.obs_norm_eps = float(getattr(config, "obs_norm_eps", 1e-8))
        self.normalize_obs = bool(getattr(config, "normalize_obs", True))
        self.obs_rms = RunningMeanStd(num_inputs, device=self.device) if self.normalize_obs else None

        hidden = int(config.hidden_size)
        self.policy = SquashedGaussianPolicy(num_inputs, self.action_dim, hidden, action_space).to(self.device)
        self.reward_critic = QNetwork(num_inputs, self.action_dim, hidden).to(self.device)
        self.reward_critic_target = QNetwork(num_inputs, self.action_dim, hidden).to(self.device)
        self.cost_critic = QNetwork(num_inputs, self.action_dim, hidden).to(self.device)
        self.cost_critic_target = QNetwork(num_inputs, self.action_dim, hidden).to(self.device)
        hard_update(self.reward_critic_target, self.reward_critic)
        hard_update(self.cost_critic_target, self.cost_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=float(config.actor_lr))
        self.reward_critic_optim = Adam(self.reward_critic.parameters(), lr=float(config.critic_lr))
        self.cost_critic_optim = Adam(self.cost_critic.parameters(), lr=float(config.cost_critic_lr))

        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=float(config.alpha_lr))

        self.cost_critic_mode = str(config.cost_critic_mode)
        if self.cost_critic_mode not in ("reachability", "discounted_cost"):
            raise ValueError(f"unknown cost_critic_mode: {self.cost_critic_mode}")
        self.cost_critic_reduce = str(getattr(config, "cost_critic_reduce", "max"))
        if self.cost_critic_reduce not in ("max", "mean"):
            raise ValueError(f"unknown cost_critic_reduce: {self.cost_critic_reduce}")
        self.binary_cost = bool(config.binary_cost)
        self.risk_threshold = float(config.star_risk_threshold)

        self.lagrange_multiplier = torch.zeros(1, requires_grad=True, device=self.device)
        self.lagrange_optim = Adam([self.lagrange_multiplier], lr=float(getattr(config, "lagrange_lr", config.actor_lr)))

        self.total_updates = 0
        self.actor_updates = 0
        self.cost_critic_forward_calls = 0
        self.last_action_info: Dict[str, float | bool | str] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def observe(self, state) -> None:
        if self.obs_rms is None:
            return
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.obs_rms.update(tensor)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return state
        return self.obs_rms.normalize(state, clip=self.obs_norm_clip, eps=self.obs_norm_eps)

    def set_requires_grad(self, module: torch.nn.Module, requires_grad: bool) -> list[bool]:
        flags = []
        for parameter in module.parameters():
            flags.append(parameter.requires_grad)
            parameter.requires_grad_(requires_grad)
        return flags

    def restore_requires_grad(self, module: torch.nn.Module, flags: Iterable[bool]) -> None:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)

    def _tensor_batch(self, memory, batch_size: int):
        state, action, reward, cost, next_state, mask = memory.sample(batch_size)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        cost = torch.as_tensor(cost, dtype=torch.float32, device=self.device).view(-1, 1)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device).view(-1, 1)
        return (
            self.normalize_state(state),
            action,
            reward,
            cost,
            self.normalize_state(next_state),
            mask,
        )

    def reduce_cost_values(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if self.cost_critic_reduce == "mean":
            return 0.5 * (q1 + q2)
        return torch.max(q1, q2)

    def _cost_target(self, cost: torch.Tensor, mask: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action, _, _ = self.policy.sample(next_state)
            q1_next, q2_next = self.cost_critic_target(next_state, next_action)
            self.cost_critic_forward_calls += int(next_state.shape[0])
            conservative_next = self.reduce_cost_values(q1_next, q2_next)
            if self.cost_critic_mode == "reachability":
                c_bin = (cost > 0).float()
                target = c_bin + (1.0 - c_bin) * mask * self.cost_gamma * conservative_next
                return torch.clamp(target, 0.0, 1.0)
            return cost + mask * self.cost_gamma * conservative_next

    def update_reward_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            q1_next, q2_next = self.reward_critic_target(next_state, next_action)
            min_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_pi
            target = reward + mask * self.gamma * min_next
        q1, q2 = self.reward_critic(state, action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.reward_critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.reward_critic_optim.step()
        return loss.detach()

    def update_cost_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        next_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self._cost_target(cost, mask, next_state)
        q1, q2 = self.cost_critic(state, action)
        self.cost_critic_forward_calls += int(state.shape[0])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.cost_critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.cost_critic_optim.step()
        return loss.detach()

    def sac_actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        action, log_pi, _ = self.policy.sample(state)
        q1, q2 = self.reward_critic(state, action)
        min_q = torch.min(q1, q2)
        loss = (self.alpha.detach() * log_pi - min_q).mean()
        return loss, {"action": action, "log_pi": log_pi, "reward_q": min_q}

    def update_alpha(self, log_pi: torch.Tensor) -> torch.Tensor:
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()
        return alpha_loss.detach()

    def update_targets(self) -> None:
        soft_update(self.reward_critic_target, self.reward_critic, self.target_tau)
        soft_update(self.cost_critic_target, self.cost_critic, self.target_tau)

    def checkpoint_state(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "reward_critic": self.reward_critic.state_dict(),
            "reward_critic_target": self.reward_critic_target.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "cost_critic_target": self.cost_critic_target.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "reward_critic_optim": self.reward_critic_optim.state_dict(),
            "cost_critic_optim": self.cost_critic_optim.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "lagrange_multiplier": self.lagrange_multiplier.detach().cpu(),
            "lagrange_optim": self.lagrange_optim.state_dict(),
            "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
            "total_updates": self.total_updates,
            "actor_updates": self.actor_updates,
            "cost_critic_forward_calls": self.cost_critic_forward_calls,
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.reward_critic.load_state_dict(state["reward_critic"])
        self.reward_critic_target.load_state_dict(state["reward_critic_target"])
        self.cost_critic.load_state_dict(state["cost_critic"])
        self.cost_critic_target.load_state_dict(state["cost_critic_target"])
        self.policy_optim.load_state_dict(state["policy_optim"])
        self.reward_critic_optim.load_state_dict(state["reward_critic_optim"])
        self.cost_critic_optim.load_state_dict(state["cost_critic_optim"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.alpha_optim.load_state_dict(state["alpha_optim"])
        self.lagrange_multiplier.data.copy_(state["lagrange_multiplier"].to(self.device))
        self.lagrange_optim.load_state_dict(state["lagrange_optim"])
        if self.obs_rms is not None and state.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])
        self.total_updates = int(state.get("total_updates", 0))
        self.actor_updates = int(state.get("actor_updates", 0))
        self.cost_critic_forward_calls = int(state.get("cost_critic_forward_calls", 0))
