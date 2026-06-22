from __future__ import annotations

import copy
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model.utils import hard_update

from .sac_base import SACBase
from .shadow_audit import ShadowAuditModule


class STARAgent(SACBase):
    """Safety-Shadow Trust-Region Actor-Critic."""

    def __init__(self, num_inputs: int, action_space, config) -> None:
        super().__init__(num_inputs, action_space, config)
        self.method = str(config.method)
        valid_methods = {"sac", "pointwise", "sac_lag", "star_actor", "star_exec", "star"}
        if self.method not in valid_methods:
            raise ValueError(f"unknown method: {self.method}")

        self.reference_policy = copy.deepcopy(self.policy).to(self.device)
        self.reference_policy.freeze()
        self.reference_age = 0
        self.reference_update_count = 0
        self.star_ref_update_interval = int(config.star_ref_update_interval)

        self.audit = ShadowAuditModule(
            self.action_dim,
            shadow_k=int(config.shadow_k),
            shadow_aggregation=str(getattr(config, "shadow_aggregation", "log_mean_exp")),
            shadow_temperature=float(config.shadow_temperature),
            shadow_local_std=float(config.shadow_local_std),
            shadow_beta_mode=str(config.shadow_beta_mode),
            shadow_reference_mode=str(getattr(config, "shadow_reference_mode", "corridor")),
            shadow_chunk_size=int(config.shadow_chunk_size),
            cost_critic_reduce=str(getattr(config, "cost_critic_reduce", "max")),
        )
        self.star_lambda = float(config.star_lambda)
        self.star_kl_coef = float(config.star_kl_coef)
        self.star_kl_target = float(config.star_kl_target)
        self.star_kl_mode = str(config.star_kl_mode)
        if self.star_kl_mode not in ("hinge", "plain"):
            raise ValueError(f"unknown star_kl_mode: {self.star_kl_mode}")

        self.star_exec = bool(config.star_exec)
        self.star_exec_candidates = int(config.star_exec_candidates)
        self.star_exec_margin = float(config.star_exec_margin)
        self.star_exec_start_steps = int(config.star_exec_start_steps)
        self.boundary_epsilon = float(config.boundary_epsilon)

    def _method_uses_shadow_loss(self) -> bool:
        return self.method in {"star", "star_actor"}

    def _method_uses_kl(self) -> bool:
        return self.method == "star"

    def _method_uses_execution(self) -> bool:
        return self.method in {"star", "star_exec"} and self.star_exec

    def _reference_update_if_needed(self) -> None:
        if self.star_ref_update_interval <= 0:
            return
        if self.actor_updates > 0 and self.actor_updates % self.star_ref_update_interval == 0:
            hard_update(self.reference_policy, self.policy)
            self.reference_policy.freeze()
            self.reference_age = 0
            self.reference_update_count += 1

    def _cost_plus(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.cost_critic(state, action)
        self.cost_critic_forward_calls += int(state.shape[0])
        return self.reduce_cost_values(q1, q2)

    def _reward_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.reward_critic(state, action)
        return torch.min(q1, q2)

    def _shadow_actor_terms(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        shadow = self.audit.generate_shadow_actions(self.policy, self.reference_policy, state)
        q_shadow = self.audit.conservative_cost(self.cost_critic, state, shadow.actions)
        self.cost_critic_forward_calls += int(q_shadow.numel())
        rho = self.audit.shadow_risk(q_shadow)
        threshold = self.risk_threshold
        penalty = F.relu(rho - threshold).mean()
        unsafe = (q_shadow > threshold).float()

        _, _, mean_action = self.policy.sample(state)
        mean_q = self._cost_plus(state, mean_action).view(-1)
        any_unsafe = unsafe.max(dim=1).values
        hidden = ((mean_q <= threshold) & (any_unsafe > 0)).float()
        stats = {
            "penalty": penalty,
            "rho": rho,
            "q_shadow": q_shadow,
            "p_svr": unsafe.mean(),
            "any_unsafe": any_unsafe.mean(),
            "hidden": hidden.mean(),
            "action_spread": shadow.spread.mean(),
            "actor_mean_action_risk": mean_q.mean(),
        }
        return penalty, stats

    def _kl_terms(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        kl = self.policy.analytic_kl_to(self.reference_policy, state)
        if self.star_kl_mode == "plain":
            loss = kl.mean()
        else:
            loss = F.relu(kl - self.star_kl_target).mean()
        return loss, {
            "kl": kl,
            "kl_exceed": (kl > self.star_kl_target).float().mean(),
        }

    def _pointwise_penalty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_cost = self._cost_plus(state, action)
        return F.relu(q_cost - self.risk_threshold).mean()

    def _lagrange_value(self) -> torch.Tensor:
        return F.softplus(self.lagrange_multiplier)

    def _update_lagrange(self, residual: torch.Tensor) -> torch.Tensor:
        loss = -(self._lagrange_value() * residual.detach()).mean()
        self.lagrange_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.lagrange_optim.step()
        return loss.detach()

    def update_actor(self, state: torch.Tensor) -> Dict[str, float]:
        reward_flags = self.set_requires_grad(self.reward_critic, False)
        cost_flags = self.set_requires_grad(self.cost_critic, False)
        self.cost_critic_optim.zero_grad(set_to_none=True)
        try:
            sac_loss, sac_info = self.sac_actor_loss(state)
            log_pi = sac_info["log_pi"]
            action = sac_info["action"]
            actor_loss = sac_loss
            pointwise = torch.zeros((), device=self.device)
            lagrange_mean_qc = torch.zeros((), device=self.device)
            lagrange_residual = torch.zeros((), device=self.device)
            shadow_penalty = torch.zeros((), device=self.device)
            kl_loss = torch.zeros((), device=self.device)
            shadow_stats: Dict[str, torch.Tensor] = {}
            kl_stats: Dict[str, torch.Tensor] = {}

            if self.method == "pointwise":
                pointwise = self._pointwise_penalty(state, action)
                actor_loss = actor_loss + self.star_lambda * pointwise
            elif self.method == "sac_lag":
                lagrange_mean_qc = self._cost_plus(state, action).mean()
                lagrange_residual = lagrange_mean_qc - self.risk_threshold
                actor_loss = actor_loss + self._lagrange_value().detach() * lagrange_mean_qc
            elif self._method_uses_shadow_loss():
                shadow_penalty, shadow_stats = self._shadow_actor_terms(state)
                actor_loss = actor_loss + self.star_lambda * shadow_penalty

            if self._method_uses_kl():
                kl_loss, kl_stats = self._kl_terms(state)
                actor_loss = actor_loss + self.star_kl_coef * kl_loss

            self.policy_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            cost_grad_leak = any(p.grad is not None for p in self.cost_critic.parameters())
            self.policy_optim.step()
            alpha_loss = self.update_alpha(log_pi)

            if self.method == "sac_lag":
                self._update_lagrange(lagrange_residual)

            self.actor_updates += 1
            self.reference_age += 1
            self._reference_update_if_needed()

            q_shadow = shadow_stats.get("q_shadow")
            rho = shadow_stats.get("rho")
            kl = kl_stats.get("kl")
            log = {
                "loss/actor": float(actor_loss.detach().item()),
                "loss/sac_actor": float(sac_loss.detach().item()),
                "loss/alpha": float(alpha_loss.item()),
                "train/alpha": float(self.alpha.detach().item()),
                "star/shadow_penalty": float(shadow_penalty.detach().item()),
                "star/kl_mean": float(kl.mean().detach().item()) if kl is not None else 0.0,
                "star/kl_max": float(kl.max().detach().item()) if kl is not None else 0.0,
                "star/kl_exceed_rate": float(kl_stats.get("kl_exceed", torch.zeros((), device=self.device)).detach().item()),
                "star/reference_age": float(self.reference_age),
                "star/reference_update_count": float(self.reference_update_count),
                "star/action_spread": float(shadow_stats.get("action_spread", torch.zeros((), device=self.device)).detach().item()),
                "star/shadow_k": float(self.audit.shadow_k),
                "star/shadow_aggregation": self.audit.shadow_aggregation,
                "star/shadow_temperature": float(self.audit.shadow_temperature),
                "star/shadow_reference_mode": self.audit.shadow_reference_mode,
                "star/pSVR": float(shadow_stats.get("p_svr", torch.zeros((), device=self.device)).detach().item()),
                "star/any_unsafe_shadow_rate": float(shadow_stats.get("any_unsafe", torch.zeros((), device=self.device)).detach().item()),
                "star/hidden_unsafe_rate": float(shadow_stats.get("hidden", torch.zeros((), device=self.device)).detach().item()),
                "star/shadow_risk_mean": float(rho.mean().detach().item()) if rho is not None else 0.0,
                "star/shadow_risk_batch_max": float(rho.max().detach().item()) if rho is not None else 0.0,
                "star/shadow_risk_max_mean": float(q_shadow.max(dim=1).values.mean().detach().item()) if q_shadow is not None else 0.0,
                "star/shadow_q_mean": float(q_shadow.mean().detach().item()) if q_shadow is not None else 0.0,
                "star/shadow_q_std": float(q_shadow.std(unbiased=False).detach().item()) if q_shadow is not None else 0.0,
                "star/actor_mean_action_risk": float(shadow_stats.get("actor_mean_action_risk", torch.zeros((), device=self.device)).detach().item()),
                "star/cost_grad_leak": float(cost_grad_leak),
                "star/pointwise_penalty": float(pointwise.detach().item()),
                "star/lagrange": float(self._lagrange_value().detach().item()),
                "star/lagrange_value": float(self._lagrange_value().detach().item()),
                "star/lagrange_residual": float(lagrange_residual.detach().item()),
                "star/lagrange_mean_qc": float(lagrange_mean_qc.detach().item()),
                "efficiency/cost_critic_forward_calls": float(self.cost_critic_forward_calls),
            }
            return log
        finally:
            self.restore_requires_grad(self.reward_critic, reward_flags)
            self.restore_requires_grad(self.cost_critic, cost_flags)

    def update_parameters(self, memory, batch_size: int, updates: int, total_numsteps: int | None = None) -> Dict[str, float]:
        state, action, reward, cost, next_state, mask = self._tensor_batch(memory, batch_size)
        reward_loss = self.update_reward_critic(state, action, reward, next_state, mask)
        cost_loss = self.update_cost_critic(state, action, cost, next_state, mask)
        log = {
            "loss/reward_critic": float(reward_loss.item()),
            "loss/cost_critic": float(cost_loss.item()),
        }

        if updates % self.policy_update_interval == 0:
            log.update(self.update_actor(state))

        if updates % self.target_update_interval == 0:
            self.update_targets()
        self.total_updates += 1
        return log

    @staticmethod
    def _candidate_choice(
        candidates: torch.Tensor,
        reward_values: torch.Tensor,
        risk_values: torch.Tensor,
        threshold: float,
        margin: float,
        mean_index: int,
    ) -> Tuple[int, Dict[str, float | bool]]:
        safe = risk_values <= (threshold - margin)
        if torch.any(safe):
            safe_scores = reward_values.clone()
            safe_scores[~safe] = -torch.inf
            index = int(torch.argmax(safe_scores).item())
            fallback = False
        else:
            min_risk = torch.min(risk_values)
            tied = torch.isclose(risk_values, min_risk)
            scores = reward_values.clone()
            scores[~tied] = -torch.inf
            index = int(torch.argmax(scores).item())
            fallback = True
        spread = candidates.std(dim=0).mean()
        return index, {
            "safe_candidate_fraction": float(safe.float().mean().item()),
            "execution_fallback": bool(fallback),
            "selected_predicted_risk": float(risk_values[index].item()),
            "selected_predicted_reward": float(reward_values[index].item()),
            "mean_action_predicted_risk": float(risk_values[mean_index].item()),
            "any_shadow_predicted_unsafe": bool(torch.any(risk_values > threshold).item()),
            "shadow_predicted_unsafe_fraction": float((risk_values > threshold).float().mean().item()),
            "selected_from_mean": bool(index == mean_index),
            "action_candidate_spread": float(spread.item()),
        }

    @torch.no_grad()
    def _raw_action_info(self, state_tensor: torch.Tensor, action: torch.Tensor, mean_action: torch.Tensor) -> Dict[str, float | bool | str]:
        selected_risk = self._cost_plus(state_tensor, action).view(-1)[0]
        selected_reward = self._reward_min(state_tensor, action).view(-1)[0]
        mean_risk = self._cost_plus(state_tensor, mean_action).view(-1)[0]
        return {
            "execution_mode": "raw",
            "candidate_count": 1,
            "safe_candidate_fraction": 0.0,
            "execution_fallback": False,
            "selected_predicted_risk": float(selected_risk.item()),
            "selected_predicted_reward": float(selected_reward.item()),
            "mean_action_predicted_risk": float(mean_risk.item()),
            "any_shadow_predicted_unsafe": bool(selected_risk.item() > self.risk_threshold),
            "shadow_predicted_unsafe_fraction": float(selected_risk.item() > self.risk_threshold),
            "selected_from_mean": bool(torch.allclose(action, mean_action, atol=1e-6)),
            "action_candidate_spread": 0.0,
        }

    @torch.no_grad()
    def select_action(
        self,
        state,
        evaluate: bool = False,
        execution_mode: str = "raw",
        total_numsteps: int | None = None,
    ):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_tensor = self.normalize_state(state_tensor)

        use_execution = execution_mode == "star_exec" and self._method_uses_execution()
        if total_numsteps is not None and total_numsteps < self.star_exec_start_steps:
            use_execution = False

        if execution_mode == "raw" or not use_execution:
            _, _, mean_action = self.policy.sample(state_tensor)
            if evaluate:
                action = mean_action
            else:
                action, _, _ = self.policy.sample(state_tensor)
            self.last_action_info = self._raw_action_info(state_tensor, action, mean_action)
            return action.cpu().numpy()[0].clip(self.action_space.low, self.action_space.high)

        _, _, mean_action = self.policy.sample(state_tensor)
        count = max(1, self.star_exec_candidates - 1)
        shadow = self.audit.generate_shadow_actions(self.policy, self.reference_policy, state_tensor, k=count)
        candidates = torch.cat([mean_action.unsqueeze(1), shadow.actions], dim=1).squeeze(0)
        state_rep = state_tensor.expand(candidates.shape[0], -1)
        reward_values = self._reward_min(state_rep, candidates).view(-1)
        risk_values = self._cost_plus(state_rep, candidates).view(-1)
        index, info = self._candidate_choice(
            candidates,
            reward_values,
            risk_values,
            self.risk_threshold,
            self.star_exec_margin,
            mean_index=0,
        )
        info["execution_mode"] = "star_exec"
        info["candidate_count"] = int(candidates.shape[0])
        self.last_action_info = info
        return candidates[index].cpu().numpy().clip(self.action_space.low, self.action_space.high)

    def save_checkpoint(self, checkpoint_path: str, suffix: str = "checkpoint") -> str:
        os.makedirs(checkpoint_path, exist_ok=True)
        path = os.path.join(checkpoint_path, f"{suffix}.torch")
        state = self.checkpoint_state()
        state.update(
            {
                "reference_policy": self.reference_policy.state_dict(),
                "reference_age": self.reference_age,
                "reference_update_count": self.reference_update_count,
                "method": self.method,
            }
        )
        torch.save(state, path)
        print(f"Saving STAR checkpoint to {path}", flush=True)
        return path

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_checkpoint_state(state)
        self.reference_policy.load_state_dict(state["reference_policy"])
        self.reference_policy.freeze()
        self.reference_age = int(state.get("reference_age", 0))
        self.reference_update_count = int(state.get("reference_update_count", 0))
