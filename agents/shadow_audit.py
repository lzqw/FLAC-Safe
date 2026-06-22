from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ShadowBatch:
    actions: torch.Tensor
    beta: torch.Tensor
    spread: torch.Tensor


def log_mean_exp_risk(q_values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-smoothed maximum over candidate risks."""
    if q_values.ndim < 2:
        raise ValueError("q_values must include a candidate dimension")
    if q_values.shape[1] == 1:
        return q_values[:, 0]
    if temperature <= 1e-8:
        return q_values.max(dim=1).values
    q_float = q_values.float()
    t = float(temperature)
    return t * (torch.logsumexp(q_float / t, dim=1) - torch.log(torch.tensor(q_float.shape[1], device=q_float.device)))


def aggregate_shadow_risk(q_values: torch.Tensor, mode: str, temperature: float) -> torch.Tensor:
    if mode == "log_mean_exp":
        return log_mean_exp_risk(q_values, temperature)
    if mode == "mean":
        return q_values.mean(dim=1)
    if mode == "max":
        return q_values.max(dim=1).values
    raise ValueError(f"unknown shadow aggregation: {mode}")


class ShadowAuditModule(nn.Module):
    """Sampled policy-update corridor audit.

    The interpolation below is a counterfactual local actor-update corridor. It is
    not a transport model and does not interact with the environment.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        shadow_k: int = 16,
        shadow_aggregation: str = "log_mean_exp",
        shadow_temperature: float = 0.05,
        shadow_local_std: float = 0.0,
        shadow_beta_mode: str = "linspace",
        shadow_reference_mode: str = "corridor",
        shadow_chunk_size: int = 0,
        cost_critic_reduce: str = "max",
    ) -> None:
        super().__init__()
        if shadow_k <= 0:
            raise ValueError("shadow_k must be positive")
        if shadow_aggregation not in ("mean", "log_mean_exp", "max"):
            raise ValueError(f"unknown shadow_aggregation: {shadow_aggregation}")
        if shadow_beta_mode not in ("linspace", "random_stratified"):
            raise ValueError(f"unknown shadow_beta_mode: {shadow_beta_mode}")
        if shadow_reference_mode not in ("corridor", "current_only"):
            raise ValueError(f"unknown shadow_reference_mode: {shadow_reference_mode}")
        if cost_critic_reduce not in ("max", "mean"):
            raise ValueError(f"unknown cost_critic_reduce: {cost_critic_reduce}")
        self.action_dim = int(action_dim)
        self.shadow_k = int(shadow_k)
        self.shadow_aggregation = str(shadow_aggregation)
        self.shadow_temperature = float(shadow_temperature)
        self.shadow_local_std = float(shadow_local_std)
        self.shadow_beta_mode = str(shadow_beta_mode)
        self.shadow_reference_mode = str(shadow_reference_mode)
        self.shadow_chunk_size = int(shadow_chunk_size)
        self.cost_critic_reduce = str(cost_critic_reduce)

    def beta_grid(self, batch_size: int, k: Optional[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(k or self.shadow_k)
        if count <= 0:
            raise ValueError("candidate count must be positive")
        if self.shadow_beta_mode == "linspace":
            if count == 1:
                beta = torch.ones(1, device=device, dtype=dtype)
            else:
                beta = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
            return beta.view(1, count, 1).expand(batch_size, count, self.action_dim)

        base = torch.arange(count, device=device, dtype=dtype).view(1, count, 1)
        jitter = torch.rand(batch_size, count, 1, device=device, dtype=dtype)
        beta = (base + jitter) / float(count)
        if count > 1:
            beta[:, 0, :] = 0.0
            beta[:, -1, :] = 1.0
        return beta.expand(batch_size, count, self.action_dim)

    def generate_shadow_actions(
        self,
        policy,
        reference_policy,
        state: torch.Tensor,
        *,
        k: Optional[int] = None,
        eps: Optional[torch.Tensor] = None,
        xi: Optional[torch.Tensor] = None,
    ) -> ShadowBatch:
        count = int(k or self.shadow_k)
        mu, log_std = policy.distribution_parameters(state)
        if self.shadow_reference_mode == "current_only":
            beta = torch.ones(state.shape[0], count, self.action_dim, device=state.device, dtype=state.dtype)
            mu_beta = mu.unsqueeze(1).expand(-1, count, -1)
            log_std_beta = log_std.unsqueeze(1).expand(-1, count, -1)
        else:
            with torch.no_grad():
                mu_ref, log_std_ref = reference_policy.distribution_parameters(state)
            mu_ref = mu_ref.detach()
            log_std_ref = log_std_ref.detach()

            beta = self.beta_grid(state.shape[0], count, state.device, state.dtype)
            mu_beta = (1.0 - beta) * mu_ref.unsqueeze(1) + beta * mu.unsqueeze(1)
            log_std_beta = (1.0 - beta) * log_std_ref.unsqueeze(1) + beta * log_std.unsqueeze(1)
        if eps is None:
            eps = torch.randn_like(mu_beta)
        if xi is None:
            xi = torch.randn_like(mu_beta)
        pre_tanh = mu_beta + log_std_beta.exp() * eps + self.shadow_local_std * xi
        actions = policy.squash(pre_tanh)
        spread = actions.std(dim=1).mean(dim=-1, keepdim=True)
        return ShadowBatch(actions=actions, beta=beta, spread=spread)

    def conservative_cost(self, cost_critic, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, count, action_dim = actions.shape
        state_flat = state.unsqueeze(1).expand(-1, count, -1).reshape(batch * count, -1)
        action_flat = actions.reshape(batch * count, action_dim)
        chunk = int(self.shadow_chunk_size)
        if chunk <= 0:
            q1, q2 = cost_critic(state_flat, action_flat)
            return self._reduce_cost(q1, q2).view(batch, count)

        outputs = []
        for start in range(0, state_flat.shape[0], chunk):
            end = start + chunk
            q1, q2 = cost_critic(state_flat[start:end], action_flat[start:end])
            outputs.append(self._reduce_cost(q1, q2))
        return torch.cat(outputs, dim=0).view(batch, count)

    def shadow_risk(self, q_shadow: torch.Tensor) -> torch.Tensor:
        return aggregate_shadow_risk(q_shadow, self.shadow_aggregation, self.shadow_temperature)

    def _reduce_cost(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if self.cost_critic_reduce == "mean":
            return 0.5 * (q1 + q2)
        return torch.max(q1, q2)
