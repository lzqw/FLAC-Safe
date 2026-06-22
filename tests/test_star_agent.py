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
