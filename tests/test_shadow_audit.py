import math
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.gaussian_policy import SquashedGaussianPolicy
from agents.shadow_audit import ShadowAuditModule, log_mean_exp_risk


def make_policy(obs_dim=3, act_dim=2):
    torch.manual_seed(0)
    action_space = Box(low=-np.ones(act_dim), high=np.ones(act_dim), dtype=np.float32)
    return SquashedGaussianPolicy(obs_dim, act_dim, 32, action_space=action_space)


def test_shadow_action_shape_and_bounds():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    state = torch.randn(5, 3)
    audit = ShadowAuditModule(action_dim=2, shadow_k=7, shadow_temperature=0.05)

    batch = audit.generate_shadow_actions(policy, reference, state)

    assert batch.actions.shape == (5, 7, 2)
    assert torch.all(batch.actions <= 1.000001)
    assert torch.all(batch.actions >= -1.000001)


def test_beta_grid_contains_reference_and_current_endpoints():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    state = torch.randn(4, 3)
    audit = ShadowAuditModule(action_dim=2, shadow_k=5)

    eps = torch.zeros(4, 5, 2)
    xi = torch.zeros(4, 5, 2)
    batch = audit.generate_shadow_actions(policy, reference, state, eps=eps, xi=xi)
    ref_mean, _ = reference.distribution_parameters(state)
    cur_mean, _ = policy.distribution_parameters(state)

    assert torch.allclose(batch.beta[0, :, 0], torch.linspace(0, 1, 5))
    assert torch.allclose(batch.actions[:, 0, :], reference.squash(ref_mean), atol=1e-6)
    assert torch.allclose(batch.actions[:, -1, :], policy.squash(cur_mean), atol=1e-6)


def test_log_mean_exp_k1_equals_single_value():
    q = torch.tensor([[0.2], [0.7]])
    rho = log_mean_exp_risk(q, temperature=0.05)
    assert torch.allclose(rho, q.squeeze(1))


def test_shadow_spread_k1_is_zero_not_nan():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    state = torch.randn(3, 3)
    audit = ShadowAuditModule(action_dim=2, shadow_k=1)

    batch = audit.generate_shadow_actions(policy, reference, state)

    assert torch.all(torch.isfinite(batch.spread))
    assert torch.allclose(batch.spread, torch.zeros_like(batch.spread))


def test_log_mean_exp_matches_manual_result():
    q = torch.tensor([[0.1, 0.2, 0.4], [0.0, 0.5, 0.8]])
    temperature = 0.2
    expected = temperature * (torch.logsumexp(q / temperature, dim=1) - math.log(q.shape[1]))
    rho = log_mean_exp_risk(q, temperature=temperature)
    assert torch.allclose(rho, expected)


def test_log_mean_exp_bounds():
    q = torch.tensor([[0.1, 0.2, 0.4], [0.0, 0.5, 0.8]])
    temperature = 0.05
    rho = log_mean_exp_risk(q, temperature=temperature)
    qmax = q.max(dim=1).values
    lower = qmax - temperature * math.log(q.shape[1])
    assert torch.all(rho <= qmax + 1e-6)
    assert torch.all(rho >= lower - 1e-6)
