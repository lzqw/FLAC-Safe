import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.star_agent import STARAgent
from utilis.SafeReplaybuffer import SafeReplayMemory
from utilis.star_default_config import star_default_config


def make_config(**kwargs):
    cfg = star_default_config.copy()
    cfg.cuda = False
    cfg.hidden_size = 32
    cfg.batch_size = 16
    cfg.shadow_k = 4
    cfg.shadow_num_strata = 4
    cfg.shadow_samples_per_stratum = 1
    cfg.star_exec_candidates = 4
    cfg.replay_size = 256
    cfg.normalize_obs = False
    cfg.policy_update_interval = 1
    cfg.updates_per_step = 1
    cfg.update(kwargs)
    return cfg


def make_spaces(obs_dim=3, act_dim=2):
    obs_space = Box(low=-np.ones(obs_dim) * 10, high=np.ones(obs_dim) * 10, dtype=np.float32)
    action_space = Box(low=-np.ones(act_dim), high=np.ones(act_dim), dtype=np.float32)
    return obs_space, action_space


class ParamCostCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(action_dim))

    def forward(self, state, action):
        q = (action * self.weight).sum(dim=-1, keepdim=True)
        return q, q


class PositiveActionCostCritic(nn.Module):
    def forward(self, state, action):
        q = action.pow(2).sum(dim=-1, keepdim=True) + 0.5
        return q, q


def test_shadow_loss_gives_actor_gradient_but_not_cost_critic_gradient():
    torch.manual_seed(1)
    obs_space, action_space = make_spaces()
    agent = STARAgent(obs_space.shape[0], action_space, make_config(method="star"))
    agent.cost_critic = ParamCostCritic(action_space.shape[0])
    state = torch.randn(8, obs_space.shape[0])

    agent.cost_critic.zero_grad(set_to_none=True)
    for p in agent.policy.parameters():
        p.grad = None

    flags = agent.set_requires_grad(agent.cost_critic, False)
    try:
        loss, _ = agent._shadow_actor_terms(state)
        loss.backward()
    finally:
        agent.restore_requires_grad(agent.cost_critic, flags)

    actor_grad = sum(
        float(p.grad.detach().abs().sum())
        for p in agent.policy.parameters()
        if p.grad is not None
    )
    assert actor_grad > 0.0
    assert all(p.grad is None for p in agent.cost_critic.parameters())
    assert all(p.grad is None for p in agent.reference_policy.parameters())


def test_star_v2_all_positive_strata_have_actor_gradient_path():
    torch.manual_seed(6)
    obs_space, action_space = make_spaces()
    cfg = make_config(
        method="star_v2",
        star_algorithm_version="star_v2",
        shadow_beta_mode="positive_linspace",
        star_shadow_penalty_mode="squared",
        star_risk_threshold=-10.0,
    )
    agent = STARAgent(obs_space.shape[0], action_space, cfg)
    agent.cost_critic = ParamCostCritic(action_space.shape[0])
    state = torch.randn(8, obs_space.shape[0])

    shadow = agent.audit.generate_shadow_actions(agent.policy, agent.reference_policy, state)
    assert torch.all(shadow.beta > 0)
    assert torch.isclose(shadow.beta[0, -1, 0], torch.tensor(1.0))

    flags = agent.set_requires_grad(agent.cost_critic, False)
    try:
        loss, _ = agent._shadow_actor_terms(state)
        agent.policy_optim.zero_grad(set_to_none=True)
        loss.backward()
    finally:
        agent.restore_requires_grad(agent.cost_critic, flags)

    actor_grad = sum(
        float(p.grad.detach().abs().sum())
        for p in agent.policy.parameters()
        if p.grad is not None
    )
    assert actor_grad > 0.0
    assert all(p.grad is None for p in agent.cost_critic.parameters())


def test_cost_critic_action_input_gradient_nonzero():
    torch.manual_seed(7)
    obs_space, action_space = make_spaces()
    agent = STARAgent(obs_space.shape[0], action_space, make_config(method="star_v2", star_algorithm_version="star_v2"))
    agent.cost_critic = ParamCostCritic(action_space.shape[0])
    state = torch.randn(4, obs_space.shape[0])
    action = torch.randn(4, action_space.shape[0], requires_grad=True)
    q = agent._cost_plus(state, action).sum()
    grad = torch.autograd.grad(q, action, retain_graph=False)[0]
    assert float(grad.abs().sum()) > 0.0


def test_after_reference_refresh_corridor_and_current_only_match_with_same_noise():
    torch.manual_seed(8)
    obs_space, action_space = make_spaces()
    cfg = make_config(
        method="star_v2",
        star_algorithm_version="star_v2",
        shadow_beta_mode="positive_linspace",
        star_ref_update_interval=1,
    )
    agent = STARAgent(obs_space.shape[0], action_space, cfg)
    state = torch.randn(5, obs_space.shape[0])
    hard_state = agent.policy.state_dict()
    agent.reference_policy.load_state_dict(hard_state)
    agent.reference_policy.freeze()
    matched = agent.audit.generate_matched_audits(agent.policy, agent.reference_policy, state)
    assert torch.allclose(matched.corridor.actions, matched.current_only.actions, atol=1e-6)


def test_current_only_v2_and_star_v2_have_same_candidate_count():
    torch.manual_seed(9)
    obs_space, action_space = make_spaces()
    base = dict(star_algorithm_version="star_v2", shadow_num_strata=4, shadow_samples_per_stratum=3)
    star = STARAgent(obs_space.shape[0], action_space, make_config(method="star_v2", **base))
    current = STARAgent(obs_space.shape[0], action_space, make_config(method="current_only_v2", **base))
    state = torch.randn(6, obs_space.shape[0])
    star_batch = star.audit.generate_shadow_actions(star.policy, star.reference_policy, state, reference_mode="corridor")
    current_batch = current.audit.generate_shadow_actions(current.policy, current.reference_policy, state, reference_mode="current_only")
    assert star_batch.actions.shape[1] == current_batch.actions.shape[1] == 12


def test_executor_selects_best_safe_candidate():
    candidates = torch.zeros(3, 2)
    rewards = torch.tensor([[1.0, 3.0, 2.0]])
    risks = torch.tensor([[0.4, 0.08, 0.09]])
    selected, info = STARAgent._candidate_choice(
        candidates, rewards.view(-1), risks.view(-1), threshold=0.10, margin=0.0, mean_index=0
    )
    assert selected == 1
    assert info["execution_fallback"] is False


def test_executor_falls_back_to_minimum_risk_candidate():
    candidates = torch.zeros(3, 2)
    rewards = torch.tensor([[10.0, 1.0, 8.0]])
    risks = torch.tensor([[0.4, 0.2, 0.2]])
    selected, info = STARAgent._candidate_choice(
        candidates, rewards.view(-1), risks.view(-1), threshold=0.10, margin=0.0, mean_index=0
    )
    assert selected == 2
    assert info["execution_fallback"] is True


def test_raw_select_action_records_predicted_risk_without_changing_execution():
    torch.manual_seed(4)
    obs_space, action_space = make_spaces()
    agent = STARAgent(obs_space.shape[0], action_space, make_config(method="sac"))
    agent.cost_critic = PositiveActionCostCritic()
    state = np.random.randn(obs_space.shape[0]).astype(np.float32)

    action = agent.select_action(state, evaluate=True, execution_mode="raw")
    info = agent.last_action_info

    assert action.shape == action_space.shape
    assert info["execution_mode"] == "raw"
    assert info["candidate_count"] == 1
    assert info["selected_predicted_risk"] > 0.0
    assert info["mean_action_predicted_risk"] > 0.0
    assert "selected_predicted_reward" in info


def test_sac_lag_local_logs_residual_and_mean_qc():
    torch.manual_seed(5)
    obs_space, action_space = make_spaces()
    cfg = make_config(method="sac_lag")
    agent = STARAgent(obs_space.shape[0], action_space, cfg)
    state = torch.randn(cfg.batch_size, obs_space.shape[0])

    stats = agent.update_actor(state)

    assert "star/lagrange_residual" in stats
    assert "star/lagrange_value" in stats
    assert "star/lagrange_mean_qc" in stats


def test_checkpoint_save_load_restores_training_state(tmp_path):
    torch.manual_seed(2)
    obs_space, action_space = make_spaces()
    cfg = make_config(method="star", normalize_obs=True)
    agent = STARAgent(obs_space.shape[0], action_space, cfg)
    agent.reference_age = 7
    agent.reference_update_count = 3
    agent.total_updates = 11
    agent.actor_updates = 5
    agent.obs_rms.update(torch.randn(12, obs_space.shape[0]))
    for p in agent.policy.parameters():
        p.data.add_(0.01)

    path = tmp_path / "star_ckpt.pt"
    saved_path = agent.save_checkpoint(str(tmp_path), suffix="star_ckpt")

    restored = STARAgent(obs_space.shape[0], action_space, cfg)
    restored.load_checkpoint(saved_path)

    assert restored.reference_age == 7
    assert restored.reference_update_count == 3
    assert restored.total_updates == 11
    assert restored.actor_updates == 5
    for p1, p2 in zip(agent.policy.parameters(), restored.policy.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(agent.reference_policy.parameters(), restored.reference_policy.parameters()):
        assert torch.allclose(p1, p2)
    assert torch.allclose(agent.obs_rms.mean, restored.obs_rms.mean)
    assert torch.allclose(agent.log_alpha, restored.log_alpha)


def test_star_v2_checkpoint_preserves_algorithm_metadata(tmp_path):
    torch.manual_seed(10)
    obs_space, action_space = make_spaces()
    cfg = make_config(
        method="star_v2",
        star_algorithm_version="star_v2",
        shadow_beta_mode="positive_linspace",
        star_shadow_penalty_mode="squared",
        shadow_num_strata=3,
        shadow_samples_per_stratum=2,
        normalize_obs=False,
    )
    agent = STARAgent(obs_space.shape[0], action_space, cfg)
    agent.reference_age = 4
    path = agent.save_checkpoint(str(tmp_path), suffix="star_v2_ckpt")
    state = torch.load(path, map_location="cpu")
    assert state["star_algorithm_version"] == "star_v2"
    assert state["shadow_beta_mode"] == "positive_linspace"
    assert state["shadow_penalty_mode"] == "squared"
    assert state["shadow_num_strata"] == 3
    assert state["shadow_samples_per_stratum"] == 2

    restored = STARAgent(obs_space.shape[0], action_space, cfg)
    restored.load_checkpoint(path)
    assert restored.reference_age == 4


def test_loading_star_v1_checkpoint_as_v2_requires_override(tmp_path):
    torch.manual_seed(11)
    obs_space, action_space = make_spaces()
    v1 = STARAgent(obs_space.shape[0], action_space, make_config(method="star", normalize_obs=False))
    path = v1.save_checkpoint(str(tmp_path), suffix="star_v1_ckpt")
    v2_cfg = make_config(method="star_v2", star_algorithm_version="star_v2", normalize_obs=False)
    v2 = STARAgent(obs_space.shape[0], action_space, v2_cfg)
    try:
        v2.load_checkpoint(path)
    except ValueError as exc:
        assert "allow_star_v1_checkpoint" in str(exc)
    else:
        raise AssertionError("STAR-v1 checkpoint loaded as STAR-v2 without explicit override")


def test_lambda_zero_kl_zero_exec_false_degenerates_to_sac_actor_update():
    torch.manual_seed(3)
    obs_space, action_space = make_spaces()
    cfg = make_config(method="sac", star_lambda=0.0, star_kl_coef=0.0, star_exec=False)
    agent = STARAgent(obs_space.shape[0], action_space, cfg)
    memory = SafeReplayMemory(128, seed=3, recent_fraction=0.0)
    for _ in range(64):
        state = np.random.randn(obs_space.shape[0]).astype(np.float32)
        action = np.random.uniform(-1, 1, size=action_space.shape[0]).astype(np.float32)
        reward = np.float32(np.random.randn())
        cost = np.float32(np.random.rand() > 0.8)
        next_state = np.random.randn(obs_space.shape[0]).astype(np.float32)
        done = np.float32(np.random.rand() > 0.9)
        memory.push(state, action, reward, cost, next_state, done)

    stats = agent.update_parameters(memory, cfg.batch_size, updates=0)
    assert torch.isfinite(torch.tensor(stats["loss/actor"]))
    assert stats["star/shadow_penalty"] == 0.0
    assert stats["star/kl_mean"] == 0.0
