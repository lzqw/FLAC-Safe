import math
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.gaussian_policy import SquashedGaussianPolicy
from agents.shadow_audit import ShadowAuditModule, exceedance_penalty, log_mean_exp_risk


def make_policy(obs_dim=3, act_dim=2):
    torch.manual_seed(0)
    action_space = Box(low=-np.ones(act_dim), high=np.ones(act_dim), dtype=np.float32)
    return SquashedGaussianPolicy(obs_dim, act_dim, 32, action_space=action_space)


def test_shadow_action_shape_and_bounds_with_samples_per_stratum():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    state = torch.randn(5, 3)
    audit = ShadowAuditModule(
        action_dim=2,
        shadow_num_strata=7,
        shadow_samples_per_stratum=3,
        shadow_beta_mode="positive_linspace",
    )

    batch = audit.generate_shadow_actions(policy, reference, state)

    assert batch.actions.shape == (5, 21, 2)
    assert batch.beta.shape == (5, 7, 1)
    assert batch.stratum_index.shape == (5, 21)
    assert batch.sample_index.shape == (5, 21)
    assert torch.all(batch.actions <= 1.000001)
    assert torch.all(batch.actions >= -1.000001)


def test_positive_linspace_excludes_reference_and_includes_current_endpoint():
    audit = ShadowAuditModule(action_dim=2, shadow_num_strata=5, shadow_beta_mode="positive_linspace")
    beta = audit.beta_grid(4, None, torch.device("cpu"), torch.float32)
    assert torch.all(beta > 0)
    assert torch.allclose(beta[0, :, 0], torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]))
    assert torch.isclose(beta[0, -1, 0], torch.tensor(1.0))


def test_positive_linspace_k1_is_current_endpoint():
    audit = ShadowAuditModule(action_dim=2, shadow_num_strata=1, shadow_beta_mode="positive_linspace")
    beta = audit.beta_grid(3, None, torch.device("cpu"), torch.float32)
    assert torch.allclose(beta[:, :, 0], torch.ones(3, 1))


def test_legacy_endpoints_reproduce_old_grid():
    audit = ShadowAuditModule(action_dim=2, shadow_num_strata=5, shadow_beta_mode="legacy_endpoints")
    beta = audit.beta_grid(4, None, torch.device("cpu"), torch.float32)
    assert torch.allclose(beta[0, :, 0], torch.linspace(0, 1, 5))


def test_reference_endpoint_is_diagnostic_only_not_in_shadow_actions():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.01)
    state = torch.randn(4, 3)
    audit = ShadowAuditModule(action_dim=2, shadow_num_strata=5, shadow_beta_mode="positive_linspace")

    eps = torch.zeros(4, 5, 1, 2)
    xi = torch.zeros(4, 5, 1, 2)
    batch = audit.generate_shadow_actions(policy, reference, state, eps=eps, xi=xi)
    ref_mean, _ = reference.distribution_parameters(state)
    cur_mean, _ = policy.distribution_parameters(state)

    assert batch.actions.shape[1] == 5
    assert batch.reference_endpoint_action is not None
    assert torch.allclose(batch.reference_endpoint_action, reference.squash(ref_mean), atol=1e-6)
    assert not torch.allclose(batch.actions[:, 0, :], reference.squash(ref_mean), atol=1e-6)
    assert torch.allclose(batch.actions[:, -1, :], policy.squash(cur_mean), atol=1e-6)


def test_matched_audits_share_eps_and_xi():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    state = torch.randn(4, 3)
    audit = ShadowAuditModule(
        action_dim=2,
        shadow_num_strata=4,
        shadow_samples_per_stratum=2,
        shadow_beta_mode="positive_linspace",
    )

    matched = audit.generate_matched_audits(policy, reference, state)

    assert torch.allclose(matched.corridor.eps, matched.current_only.eps)
    assert torch.allclose(matched.corridor.xi, matched.current_only.xi)
    assert matched.corridor.actions.shape == matched.current_only.actions.shape == (4, 8, 2)
    assert torch.all(matched.corridor.beta > 0)
    assert torch.allclose(matched.current_only.beta, torch.ones_like(matched.current_only.beta))


def test_log_mean_exp_k1_equals_single_value():
    q = torch.tensor([[0.2], [0.7]])
    rho = log_mean_exp_risk(q, temperature=0.05)
    assert torch.allclose(rho, q.squeeze(1))


def test_shadow_spread_k1_is_zero_not_nan():
    policy = make_policy()
    reference = make_policy()
    reference.freeze()
    state = torch.randn(3, 3)
    audit = ShadowAuditModule(action_dim=2, shadow_num_strata=1, shadow_beta_mode="positive_linspace")

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


def test_squared_exceedance_penalty_value_and_gradients():
    risk = torch.tensor([0.1, 0.3, 0.5], requires_grad=True)
    penalty, excess = exceedance_penalty(risk, threshold=0.2, mode="squared")
    expected = 0.5 * torch.tensor([0.0, 0.1, 0.3]).square().mean()
    assert torch.allclose(penalty, expected)
    penalty.backward()
    assert risk.grad[0].item() == 0.0
    assert risk.grad[2].item() > risk.grad[1].item() > 0.0
    assert torch.allclose(excess.detach(), torch.tensor([0.0, 0.1, 0.3]), atol=1e-6)


def test_threshold_changes_squared_hinge_gradient_magnitude():
    risk1 = torch.tensor([0.6, 0.7], requires_grad=True)
    loss1, _ = exceedance_penalty(risk1, threshold=0.1, mode="squared")
    loss1.backward()
    grad1 = risk1.grad.clone()

    risk2 = torch.tensor([0.6, 0.7], requires_grad=True)
    loss2, _ = exceedance_penalty(risk2, threshold=0.5, mode="squared")
    loss2.backward()
    grad2 = risk2.grad.clone()

    assert grad1.norm() > grad2.norm()


def test_linear_exceedance_reproduces_v1_behavior():
    risk = torch.tensor([0.1, 0.3, 0.5])
    penalty, excess = exceedance_penalty(risk, threshold=0.2, mode="linear")
    assert torch.allclose(penalty, torch.tensor([0.0, 0.1, 0.3]).mean())
    assert torch.allclose(excess, torch.tensor([0.0, 0.1, 0.3]), atol=1e-6)
