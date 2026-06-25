from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShadowBatch:
    # Flattened candidate actions used by critics: [B, N, A], N=K_beta*L.
    actions: torch.Tensor
    # Canonical beta grid before sample flattening: [B, K_beta, 1].
    beta: torch.Tensor
    # Flattened stratum/sample indices: [B, N].
    stratum_index: torch.Tensor
    sample_index: torch.Tensor
    spread: torch.Tensor
    # Optional replayable noises: [B, K_beta, L, A].
    eps: Optional[torch.Tensor] = None
    xi: Optional[torch.Tensor] = None
    # Diagnostic-only reference endpoint. It must not enter rho/actor loss.
    reference_endpoint_action: Optional[torch.Tensor] = None
    reference_endpoint_risk: Optional[torch.Tensor] = None


@dataclass
class MatchedAuditBatch:
    corridor: ShadowBatch
    current_only: ShadowBatch
    eps: torch.Tensor
    xi: torch.Tensor


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


def exceedance_penalty(risk: torch.Tensor, threshold: float, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared STAR exceedance penalty.

    STAR-v1 uses ``linear``. Canonical STAR-v2 uses ``squared``:
    0.5 * mean(relu(risk-threshold)^2).
    """
    excess = F.relu(risk - float(threshold))
    if mode == "linear":
        return excess.mean(), excess
    if mode == "squared":
        return 0.5 * excess.square().mean(), excess
    raise ValueError(f"unknown shadow penalty mode: {mode}")


class ShadowAuditModule(nn.Module):
    """Sampled policy-update corridor audit.

    This interpolation is a finite, sampled policy-update corridor. It is not a
    transport flow, flow policy, Bellman particle method, or environment model.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        shadow_k: int = 16,
        shadow_num_strata: Optional[int] = None,
        shadow_samples_per_stratum: int = 1,
        shadow_aggregation: str = "log_mean_exp",
        shadow_temperature: float = 0.05,
        shadow_local_std: float = 0.0,
        shadow_beta_mode: str = "legacy_endpoints",
        shadow_reference_mode: str = "corridor",
        shadow_chunk_size: int = 0,
        cost_critic_reduce: str = "max",
    ) -> None:
        super().__init__()
        num_strata = int(shadow_num_strata if shadow_num_strata is not None else shadow_k)
        if num_strata <= 0:
            raise ValueError("shadow_num_strata must be positive")
        if shadow_samples_per_stratum <= 0:
            raise ValueError("shadow_samples_per_stratum must be positive")
        if shadow_aggregation not in ("mean", "log_mean_exp", "max"):
            raise ValueError(f"unknown shadow_aggregation: {shadow_aggregation}")
        if shadow_beta_mode == "linspace":
            shadow_beta_mode = "legacy_endpoints"
        if shadow_beta_mode == "random_stratified":
            shadow_beta_mode = "random_stratified_positive"
        if shadow_beta_mode not in ("positive_linspace", "legacy_endpoints", "random_stratified_positive"):
            raise ValueError(f"unknown shadow_beta_mode: {shadow_beta_mode}")
        if shadow_reference_mode not in ("corridor", "current_only"):
            raise ValueError(f"unknown shadow_reference_mode: {shadow_reference_mode}")
        if cost_critic_reduce not in ("max", "mean"):
            raise ValueError(f"unknown cost_critic_reduce: {cost_critic_reduce}")
        self.action_dim = int(action_dim)
        self.shadow_num_strata = num_strata
        self.shadow_samples_per_stratum = int(shadow_samples_per_stratum)
        self.shadow_k = self.shadow_num_strata  # backward-compatible log field
        self.shadow_aggregation = str(shadow_aggregation)
        self.shadow_temperature = float(shadow_temperature)
        self.shadow_local_std = float(shadow_local_std)
        self.shadow_beta_mode = str(shadow_beta_mode)
        self.shadow_reference_mode = str(shadow_reference_mode)
        self.shadow_chunk_size = int(shadow_chunk_size)
        self.cost_critic_reduce = str(cost_critic_reduce)

    @property
    def candidate_count(self) -> int:
        return int(self.shadow_num_strata * self.shadow_samples_per_stratum)

    def beta_grid(
        self,
        batch_size: int,
        k: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
        *,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        count = int(k or self.shadow_num_strata)
        if count <= 0:
            raise ValueError("candidate count must be positive")
        beta_mode = mode or self.shadow_beta_mode
        if beta_mode == "linspace":
            beta_mode = "legacy_endpoints"
        if beta_mode == "positive_linspace":
            beta = torch.arange(1, count + 1, device=device, dtype=dtype) / float(count)
            return beta.view(1, count, 1).expand(batch_size, count, 1)
        if beta_mode == "legacy_endpoints":
            if count == 1:
                beta = torch.ones(1, device=device, dtype=dtype)
            else:
                beta = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
            return beta.view(1, count, 1).expand(batch_size, count, 1)
        if beta_mode == "random_stratified_positive":
            base = torch.arange(count, device=device, dtype=dtype).view(1, count, 1)
            jitter = torch.rand(batch_size, count, 1, device=device, dtype=dtype)
            beta = (base + jitter + 1.0) / float(count)
            beta[:, -1, :] = 1.0
            return beta.clamp_min(torch.finfo(dtype).eps)
        raise ValueError(f"unknown shadow_beta_mode: {beta_mode}")

    def _index_tensors(self, batch_size: int, k: int, l: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        stratum = torch.arange(k, device=device).view(1, k, 1).expand(batch_size, k, l).reshape(batch_size, k * l)
        sample = torch.arange(l, device=device).view(1, 1, l).expand(batch_size, k, l).reshape(batch_size, k * l)
        return stratum, sample

    def generate_shadow_actions(
        self,
        policy,
        reference_policy,
        state: torch.Tensor,
        *,
        k: Optional[int] = None,
        samples_per_stratum: Optional[int] = None,
        eps: Optional[torch.Tensor] = None,
        xi: Optional[torch.Tensor] = None,
        reference_mode: Optional[str] = None,
        beta_mode: Optional[str] = None,
        include_reference_endpoint: bool = True,
    ) -> ShadowBatch:
        batch_size = int(state.shape[0])
        count = int(k or self.shadow_num_strata)
        l_count = int(samples_per_stratum or self.shadow_samples_per_stratum)
        mode = reference_mode or self.shadow_reference_mode
        if mode not in ("corridor", "current_only"):
            raise ValueError(f"unknown shadow_reference_mode: {mode}")

        mu, log_std = policy.distribution_parameters(state)
        with torch.no_grad():
            mu_ref, log_std_ref = reference_policy.distribution_parameters(state)
        mu_ref = mu_ref.detach()
        log_std_ref = log_std_ref.detach()

        if mode == "current_only":
            beta = torch.ones(batch_size, count, 1, device=state.device, dtype=state.dtype)
        else:
            beta = self.beta_grid(batch_size, count, state.device, state.dtype, mode=beta_mode)

        beta4 = beta.view(batch_size, count, 1, 1)
        mu_beta = (1.0 - beta4) * mu_ref.view(batch_size, 1, 1, -1) + beta4 * mu.view(batch_size, 1, 1, -1)
        log_std_beta = (1.0 - beta4) * log_std_ref.view(batch_size, 1, 1, -1) + beta4 * log_std.view(batch_size, 1, 1, -1)
        mu_beta = mu_beta.expand(-1, -1, l_count, -1)
        log_std_beta = log_std_beta.expand(-1, -1, l_count, -1)

        noise_shape = (batch_size, count, l_count, self.action_dim)
        if eps is None:
            eps = torch.randn(noise_shape, device=state.device, dtype=state.dtype)
        if xi is None:
            xi = torch.randn(noise_shape, device=state.device, dtype=state.dtype)
        pre_tanh = mu_beta + log_std_beta.exp() * eps + self.shadow_local_std * xi
        actions4 = policy.squash(pre_tanh)
        actions = actions4.reshape(batch_size, count * l_count, self.action_dim)
        spread = actions.std(dim=1, unbiased=False).mean(dim=-1, keepdim=True)
        stratum_index, sample_index = self._index_tensors(batch_size, count, l_count, state.device)

        reference_endpoint_action = None
        if include_reference_endpoint:
            with torch.no_grad():
                reference_endpoint_action = policy.squash(mu_ref).detach()

        return ShadowBatch(
            actions=actions,
            beta=beta,
            stratum_index=stratum_index,
            sample_index=sample_index,
            spread=spread,
            eps=eps,
            xi=xi,
            reference_endpoint_action=reference_endpoint_action,
        )

    def generate_matched_audits(
        self,
        policy,
        reference_policy,
        state: torch.Tensor,
        *,
        eps: Optional[torch.Tensor] = None,
        xi: Optional[torch.Tensor] = None,
    ) -> MatchedAuditBatch:
        if eps is None:
            eps = torch.randn(
                state.shape[0],
                self.shadow_num_strata,
                self.shadow_samples_per_stratum,
                self.action_dim,
                device=state.device,
                dtype=state.dtype,
            )
        if xi is None:
            xi = torch.randn_like(eps)
        corridor = self.generate_shadow_actions(
            policy,
            reference_policy,
            state,
            eps=eps,
            xi=xi,
            reference_mode="corridor",
            include_reference_endpoint=True,
        )
        current_only = self.generate_shadow_actions(
            policy,
            reference_policy,
            state,
            eps=eps,
            xi=xi,
            reference_mode="current_only",
            include_reference_endpoint=True,
        )
        return MatchedAuditBatch(corridor=corridor, current_only=current_only, eps=eps, xi=xi)

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
