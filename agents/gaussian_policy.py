from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from model.model import weights_init_


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


class SquashedGaussianPolicy(nn.Module):
    """Standard diagonal Gaussian actor with tanh squashing and Box rescaling."""

    def __init__(self, num_inputs: int, num_actions: int, hidden_size: int, action_space=None) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.apply(weights_init_)

        if action_space is None:
            action_scale = torch.ones(num_actions, dtype=torch.float32)
            action_bias = torch.zeros(num_actions, dtype=torch.float32)
            action_low = -torch.ones(num_actions, dtype=torch.float32)
            action_high = torch.ones(num_actions, dtype=torch.float32)
        else:
            action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            action_bias = torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
            action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
            action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def distribution_parameters(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(state)

    def squash(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def _log_prob_from_pre_tanh(
        self, mean: torch.Tensor, log_std: torch.Tensor, pre_tanh: torch.Tensor
    ) -> torch.Tensor:
        normal = Normal(mean, log_std.exp())
        log_prob = normal.log_prob(pre_tanh)
        squashed = torch.tanh(pre_tanh)
        correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + EPS)
        while correction.ndim < log_prob.ndim:
            correction = correction.unsqueeze(1)
        return (log_prob - correction).sum(dim=-1, keepdim=True)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        eps = torch.randn_like(mean)
        pre_tanh = mean + log_std.exp() * eps
        action = self.squash(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(mean, log_std, pre_tanh)
        mean_action = self.squash(mean)
        return action, log_prob, mean_action

    def sample_n(
        self, state: torch.Tensor, k: int, eps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k <= 0:
            raise ValueError("k must be positive")
        mean, log_std = self.forward(state)
        mean_k = mean.unsqueeze(1).expand(-1, k, -1)
        log_std_k = log_std.unsqueeze(1).expand(-1, k, -1)
        if eps is None:
            eps = torch.randn_like(mean_k)
        pre_tanh = mean_k + log_std_k.exp() * eps
        action = self.squash(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(mean_k, log_std_k, pre_tanh)
        mean_action = self.squash(mean).unsqueeze(1).expand(-1, k, -1)
        return action, log_prob, mean_action

    def pre_tanh_from_noise(self, state: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(state)
        if eps.ndim == mean.ndim + 1:
            mean = mean.unsqueeze(1)
            log_std = log_std.unsqueeze(1)
        return mean + log_std.exp() * eps

    def analytic_kl_to(self, reference_policy: "SquashedGaussianPolicy", state: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(state)
        with torch.no_grad():
            ref_mean, ref_log_std = reference_policy.distribution_parameters(state)
        var = torch.exp(2.0 * log_std)
        ref_var = torch.exp(2.0 * ref_log_std)
        kl = ref_log_std - log_std + (var + (mean - ref_mean).pow(2)) / (2.0 * ref_var + EPS) - 0.5
        return kl.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
