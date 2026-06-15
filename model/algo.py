import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import copy
from .utils import soft_update, hard_update
from .model import QNetwork, ValueNetwork, Policy_flow, C51QNetwork
import time
from torch.optim import Adam
import torch.optim as optim
import numpy as np

from utilis.utils import RunningMeanStd

try:
    from torch.func import jvp as torch_func_jvp
except Exception:  # pragma: no cover - fallback for older torch versions
    torch_func_jvp = None


mode = "max-autotune"

class flowAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.num_inputs = num_inputs
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_level = args.epsilon
        self.action_space = action_space
        self.sample_count = 0

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.device = torch.device(f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu")
        self.amp_enabled = args.cuda and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        self.obs_norm_clip = getattr(args, "obs_norm_clip", 10.0)
        self.obs_norm_eps = getattr(args, "obs_norm_eps", 1e-8)
        self.normalize_obs = bool(getattr(args, "normalize_obs", False))
        self.obs_rms = RunningMeanStd(num_inputs, device=self.device) if self.normalize_obs else None

        self.safe_env = bool(getattr(args, "safe_env", False))
        self.cost_gamma = float(getattr(args, "cost_gamma", 0.97))
        self.safe_threshold = float(getattr(args, "safe_threshold", 0.1))
        self.safety_critic_mode = str(getattr(args, "safety_critic_mode", "cumulative"))
        if self.safety_critic_mode not in ("cumulative", "cdf"):
            raise ValueError(f"Unknown safety_critic_mode: {self.safety_critic_mode}")
        self.qc_geom_mode = str(getattr(args, "qc_geom_mode", "max"))
        if self.qc_geom_mode not in ("max", "mean"):
            raise ValueError(f"Unknown qc_geom_mode: {self.qc_geom_mode}")
        self.cdf_binarize_cost = bool(getattr(args, "cdf_binarize_cost", True))
        self.cdf_target_clip = bool(getattr(args, "cdf_target_clip", True))
        self.safe_bandwidth = float(getattr(args, "safe_bandwidth", 0.05))
        self.lambda_safe = float(getattr(args, "lambda_safe", 1.0))
        self.lambda_jvp = float(getattr(args, "lambda_jvp", 0.05))
        self.lambda_jvp_schedule = bool(getattr(args, "lambda_jvp_schedule", False))
        self.lambda_jvp_start = float(getattr(args, "lambda_jvp_start", 0.0))
        self.lambda_jvp_end = float(getattr(args, "lambda_jvp_end", 0.003))
        self.lambda_jvp_warmup_steps = int(getattr(args, "lambda_jvp_warmup_steps", 30000))
        self.lambda_jvp_ramp_steps = int(getattr(args, "lambda_jvp_ramp_steps", 70000))
        self.jvp_warmup_steps = int(getattr(args, "jvp_warmup_steps", 20000))
        self.safe_policy_loss = bool(getattr(args, "safe_policy_loss", True))

        # Safety-critical directional derivative options.
        # jvp_mode="forward" uses torch.func.jvp when available; it falls back to
        # grad-dot-vector if the local PyTorch build does not support forward-mode JVP.
        self.jvp_mode = str(getattr(args, "jvp_mode", "forward"))
        self.normalize_jvp = bool(getattr(args, "normalize_jvp", False))
        self.jvp_norm_mode = str(getattr(args, "jvp_norm_mode", "hutchinson"))
        self.jvp_hutchinson_samples = int(getattr(args, "jvp_hutchinson_samples", 1))
        self.jvp_eps = float(getattr(args, "jvp_eps", 1e-6))
        self.soft_feasibility_gate = bool(getattr(args, "soft_feasibility_gate", False))
        self.feas_gate_tau = float(getattr(args, "feas_gate_tau", 0.05))
        self.feas_gate_detach = bool(getattr(args, "feas_gate_detach", True))
        self.feas_gate_reward_floor = float(getattr(args, "feas_gate_reward_floor", 0.2))
        if self.feas_gate_tau <= 0:
            raise ValueError("feas_gate_tau must be positive")
        if not 0.0 <= self.feas_gate_reward_floor <= 1.0:
            raise ValueError("feas_gate_reward_floor must be in [0, 1]")

        self.high_fidelity_safety_q = bool(getattr(args, "high_fidelity_safety_q", False))
        self.safety_q_priority = bool(getattr(args, "safety_q_priority", False))
        self.safety_q_cost_weight = float(getattr(args, "safety_q_cost_weight", 2.0))
        self.safety_q_boundary_weight = float(getattr(args, "safety_q_boundary_weight", 3.0))
        self.safety_q_td_weight = float(getattr(args, "safety_q_td_weight", 0.0))
        self.safety_q_max_weight = float(getattr(args, "safety_q_max_weight", 5.0))
        self.safety_q_extra_updates = int(getattr(args, "safety_q_extra_updates", 0))
        self.safety_q_boundary_width = float(getattr(args, "safety_q_boundary_width", 0.05))
        self.diagnose_safety_q_geometry = bool(getattr(args, "diagnose_safety_q_geometry", False))
        self.safety_q_fd_eps = float(getattr(args, "safety_q_fd_eps", 0.01))
        if self.safety_q_max_weight < 1.0:
            raise ValueError("safety_q_max_weight must be >= 1")
        if self.safety_q_extra_updates < 0:
            raise ValueError("safety_q_extra_updates must be >= 0")
        if self.safety_q_boundary_width <= 0:
            raise ValueError("safety_q_boundary_width must be positive")
        if self.safety_q_fd_eps <= 0:
            raise ValueError("safety_q_fd_eps must be positive")

        # Optional real-interaction soft normal masking. This is deliberately
        # separated from the training-time JVP-SCD objective: training can use
        # JVP only, while environment sampling can use an explicit VJP normal.
        self.soft_normal_masking = bool(getattr(args, "soft_normal_masking", False))
        self.masking_warmup_steps = int(getattr(args, "masking_warmup_steps", self.jvp_warmup_steps))
        self.mask_beta_max = float(getattr(args, "mask_beta_max", 0.5))
        self.mask_beta_tau = float(getattr(args, "mask_beta_tau", 10000.0))
        self.mask_noise_scale = float(getattr(args, "mask_noise_scale", 0.01))
        self.mask_noise_clip = float(getattr(args, "mask_noise_clip", 0.25))

        self.directional_ref_noise = bool(getattr(args, "directional_ref_noise", False))
        self.directional_noise_mode = str(getattr(args, "directional_noise_mode", "none"))
        self.tangent_noise_scale = float(getattr(args, "tangent_noise_scale", 0.05))
        self.ref_noise_scale = float(getattr(args, "ref_noise_scale", 0.02))
        self.normal_noise_scale = float(getattr(args, "normal_noise_scale", 0.01))
        self.directional_noise_warmup_steps = int(getattr(args, "directional_noise_warmup_steps", 10000))
        self.directional_noise_beta_max = float(getattr(args, "directional_noise_beta_max", 0.5))
        self.directional_noise_eps = float(getattr(args, "directional_noise_eps", 1e-6))
        self.directional_noise_clip = float(getattr(args, "directional_noise_clip", 0.3))
        self._last_explore_stats = self._empty_explore_stats()

        # LAC: Target kinetic energy (coef * action_dim)
        target_kinetic_coef = float(getattr(args, "target_kinetic_coef", 2.5))
        self.target_kinetic = target_kinetic_coef * action_space.shape[0]

        # LAC: Adaptive temperature parameter (alpha = exp(log_alpha))
        init_log_alpha = float(getattr(args, "init_log_alpha", 0.0))
        self.auto_alpha = bool(getattr(args, "auto_alpha", True))
        self.log_alpha = torch.tensor(
            [init_log_alpha],
            requires_grad=self.auto_alpha,
            device=self.device,
        )
        # Use a smaller LR for alpha to avoid overreacting.
        self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr * 0.1) if self.auto_alpha else None

        self.distributional_critic = bool(getattr(args, "distributional_critic", False))
        if self.distributional_critic:
            self.critic_num_atoms = int(getattr(args, "critic_num_atoms", 101))
            self.critic_v_min = float(getattr(args, "critic_v_min", -150.0))
            self.critic_v_max = float(getattr(args, "critic_v_max", 150.0))
            self.c51_atoms = torch.linspace(
                self.critic_v_min, self.critic_v_max, self.critic_num_atoms, device=self.device
            )
            self.c51_delta = (self.critic_v_max - self.critic_v_min) / (self.critic_num_atoms - 1)

        # ---------------------- Policy Network ----------------------
        if self.policy_type == "Flow":
            self.policy = Policy_flow(num_inputs, action_space.shape[0], args.hidden_size, args.steps, action_space).to(self.device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        else:
            pass

        # ---------------------- Critic Networks ----------------------
        if self.distributional_critic:
            self.critic = C51QNetwork(
                num_inputs,
                action_space.shape[0],
                args.hidden_size,
                num_atoms=self.critic_num_atoms,
            ).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        if self.distributional_critic:
            self.critic_target = C51QNetwork(
                num_inputs,
                action_space.shape[0],
                args.hidden_size,
                num_atoms=self.critic_num_atoms,
            ).to(self.device)
        else:
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.safe_env:
            self.safety_critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.safety_critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.safety_critic_optim = optim.Adam(self.safety_critic.parameters(), lr=args.lr)
            hard_update(self.safety_critic_target, self.safety_critic)
        else:
            self.safety_critic = None
            self.safety_critic_target = None
            self.safety_critic_optim = None

        # ---------------------- Compile Models ----------------------
        self.compile_model = bool(getattr(args, "compile_model", False))
        if self.compile_model:
            self.critic = torch.compile(self.critic,mode=mode)
            self.critic_target = torch.compile(self.critic_target, mode=mode)
            # self.policy = torch.compile(self.policy, mode=mode)

    def _safe_bandwidth(self):
        return max(self.safe_bandwidth, 1e-6)

    def _compute_g_mid_from_qc(self, qc):
        bandwidth = self._safe_bandwidth()
        return torch.exp(-((qc - self.safe_threshold) ** 2) / (2.0 * bandwidth ** 2))

    def _lambda_jvp_eff(self, total_numsteps):
        if not self.lambda_jvp_schedule:
            return self.lambda_jvp
        if total_numsteps < self.lambda_jvp_warmup_steps:
            return self.lambda_jvp_start
        ramp_steps = max(1.0, float(self.lambda_jvp_ramp_steps))
        progress = (float(total_numsteps) - float(self.lambda_jvp_warmup_steps)) / ramp_steps
        progress = min(1.0, max(0.0, progress))
        return self.lambda_jvp_start + progress * (self.lambda_jvp_end - self.lambda_jvp_start)

    def _mask_beta(self):
        if self.sample_count < self.masking_warmup_steps:
            return 0.0
        tau = max(self.mask_beta_tau, 1.0)
        x = (float(self.sample_count) - float(self.masking_warmup_steps)) / tau
        beta = self.mask_beta_max / (1.0 + np.exp(-x))
        return float(beta)

    def _directional_noise_mode_id(self):
        return {
            "none": 0.0,
            "tangent": 1.0,
            "reward_ref": 2.0,
            "ref_normal": 3.0,
        }.get(self.directional_noise_mode, 0.0)

    def _empty_explore_stats(self):
        return {
            "explore/directional_noise_enabled": 0.0,
            "explore/directional_noise_mode_id": 0.0,
            "explore/noise_tangent_norm": 0.0,
            "explore/noise_ref_norm": 0.0,
            "explore/noise_normal_norm": 0.0,
            "explore/ref_grad_norm": 0.0,
            "explore/qc_grad_norm": 0.0,
            "explore/tangent_ratio": 0.0,
            "explore/g_mid_mean": 0.0,
            "explore/noise_total_norm": 0.0,
            "explore/action_delta_norm": 0.0,
        }

    def _directional_noise_beta(self, total_numsteps):
        if total_numsteps < self.directional_noise_warmup_steps:
            return 0.0
        tau = max(float(self.mask_beta_tau), 1.0)
        progress = min(1.0, (float(total_numsteps) - float(self.directional_noise_warmup_steps)) / tau)
        return float(self.directional_noise_beta_max * progress)

    @staticmethod
    def _is_finite_tensor(value):
        return value is not None and torch.isfinite(value).all()

    def _critic_scalar(self, state, action):
        q1, q2 = self.critic(state, action)
        if self.distributional_critic:
            q1 = (F.softmax(q1.float(), dim=-1) * self.c51_atoms).sum(dim=-1, keepdim=True)
            q2 = (F.softmax(q2.float(), dim=-1) * self.c51_atoms).sum(dim=-1, keepdim=True)
        return torch.min(q1, q2)

    def _directional_reference_noise_action(self, state, action, total_numsteps):
        stats = self._empty_explore_stats()
        mode = self.directional_noise_mode
        stats["explore/directional_noise_mode_id"] = self._directional_noise_mode_id()

        enabled = (
            self.directional_ref_noise
            and mode in ("tangent", "reward_ref", "ref_normal")
            and (not self.safe_env or self.safe_policy_loss)
            and self.safe_env
            and self.safety_critic is not None
            and total_numsteps >= self.directional_noise_warmup_steps
        )
        if not enabled:
            self._last_explore_stats = stats
            return action

        stats["explore/directional_noise_enabled"] = 1.0
        safety_flags = self.set_requires_grad(self.safety_critic, False)
        critic_flags = self.set_requires_grad(self.critic, False)
        try:
            with torch.enable_grad():
                action_for_grad = action.detach().requires_grad_(True)
                qc = self._qc_scalar(state.detach(), action_for_grad)
                grad_qc = torch.autograd.grad(
                    outputs=qc.sum(),
                    inputs=action_for_grad,
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True,
                )[0].detach()

                qc_norm = grad_qc.norm(dim=-1, keepdim=True)
                stats["explore/qc_grad_norm"] = float(qc_norm.mean().item())
                if (not self._is_finite_tensor(grad_qc)) or torch.any(qc_norm <= self.directional_noise_eps):
                    self._last_explore_stats = stats
                    return action

                normal = grad_qc / (qc_norm + self.directional_noise_eps)
                xi = torch.randn_like(action)
                xi_norm = xi.norm(dim=-1, keepdim=True)
                normal_xi = (normal * xi).sum(dim=-1, keepdim=True)
                tangent_raw = xi - normal * normal_xi
                tangent = torch.clamp(
                    self.tangent_noise_scale * tangent_raw,
                    -self.directional_noise_clip,
                    self.directional_noise_clip,
                )

                ref = torch.zeros_like(action)
                ref_grad_norm = torch.tensor(0.0, device=self.device)
                if mode in ("reward_ref", "ref_normal"):
                    q_reward = self._critic_scalar(state.detach(), action_for_grad)
                    grad_r = torch.autograd.grad(
                        outputs=q_reward.sum(),
                        inputs=action_for_grad,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True,
                    )[0].detach()
                    ref_grad_norm = grad_r.norm(dim=-1, keepdim=True)
                    if self._is_finite_tensor(grad_r) and torch.all(ref_grad_norm > self.directional_noise_eps):
                        ref_raw = grad_r - normal * (normal * grad_r).sum(dim=-1, keepdim=True)
                        ref_norm = ref_raw.norm(dim=-1, keepdim=True)
                        if self._is_finite_tensor(ref_raw) and torch.all(ref_norm > self.directional_noise_eps):
                            ref = torch.clamp(
                                self.ref_noise_scale * ref_raw / (ref_norm + self.directional_noise_eps),
                                -self.directional_noise_clip,
                                self.directional_noise_clip,
                            )

                normal_noise = torch.zeros_like(action)
                g_mid = self._compute_g_mid_from_qc(qc.detach())
                if mode == "ref_normal":
                    beta = self._directional_noise_beta(total_numsteps)
                    normal_noise = self.normal_noise_scale * (1.0 - beta * g_mid) * normal * normal_xi
                    normal_noise = torch.clamp(normal_noise, -self.directional_noise_clip, self.directional_noise_clip)

                total_noise = tangent + ref + normal_noise
                if not self._is_finite_tensor(total_noise):
                    total_noise = torch.zeros_like(action)
                noisy_action = action + total_noise

                stats.update({
                    "explore/noise_tangent_norm": float(tangent.norm(dim=-1).mean().item()),
                    "explore/noise_ref_norm": float(ref.norm(dim=-1).mean().item()),
                    "explore/noise_normal_norm": float(normal_noise.norm(dim=-1).mean().item()),
                    "explore/ref_grad_norm": float(ref_grad_norm.mean().item()),
                    "explore/tangent_ratio": float((tangent_raw.norm(dim=-1, keepdim=True) / (xi_norm + self.directional_noise_eps)).mean().item()),
                    "explore/g_mid_mean": float(g_mid.mean().item()),
                    "explore/noise_total_norm": float(total_noise.norm(dim=-1).mean().item()),
                    "explore/action_delta_norm": float((noisy_action - action).norm(dim=-1).mean().item()),
                })
                self._last_explore_stats = stats
                return noisy_action.detach()
        finally:
            self.restore_requires_grad(self.safety_critic, safety_flags)
            self.restore_requires_grad(self.critic, critic_flags)

    def _soft_normal_mask_action(self, state, action, noise):
        """Softly attenuate exploration noise along the learned risk normal.

        epsilon_mask = epsilon - beta * g_mid * n_C * (n_C^T epsilon)

        This module is only for real environment sampling. The training-time
        policy loss can still use JVP-SCD without explicitly constructing n_C.
        """
        if (not self.safe_env) or (not self.safe_policy_loss) or (not self.soft_normal_masking):
            return action + noise, {
                "beta": 0.0,
                "g_mid": torch.tensor(0.0, device=self.device),
                "normal_norm": torch.tensor(0.0, device=self.device),
            }
        if self.safety_critic is None:
            return action + noise, {
                "beta": 0.0,
                "g_mid": torch.tensor(0.0, device=self.device),
                "normal_norm": torch.tensor(0.0, device=self.device),
            }

        beta = self._mask_beta()
        if beta <= 0.0:
            return action + noise, {
                "beta": 0.0,
                "g_mid": torch.tensor(0.0, device=self.device),
                "normal_norm": torch.tensor(0.0, device=self.device),
            }

        # Freeze safety-critic parameters, but keep gradient w.r.t. action.
        flags = self.set_requires_grad(self.safety_critic, False)
        try:
            action_for_grad = action.detach().requires_grad_(True)
            qc1, qc2 = self.safety_critic(state.detach(), action_for_grad)
            qc = torch.max(qc1, qc2)
            g_mid = self._compute_g_mid_from_qc(qc.detach())
            grad_q = torch.autograd.grad(
                outputs=qc.sum(),
                inputs=action_for_grad,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0].detach()
            normal = grad_q / (grad_q.norm(dim=-1, keepdim=True) + self.jvp_eps)
            normal_component = (normal * noise).sum(dim=-1, keepdim=True)
            masked_noise = noise - beta * g_mid * normal * normal_component
            masked_action = action + masked_noise
            return masked_action, {
                "beta": beta,
                "g_mid": g_mid.detach().mean(),
                "normal_norm": grad_q.norm(dim=-1).detach().mean(),
            }
        finally:
            self.restore_requires_grad(self.safety_critic, flags)

    # only use for env step 
    def select_action(self, state, evaluate=False, total_numsteps=None):

        # Noise schedule for exploration: In all tasks, we set the noise to 0.
        if not evaluate:
            self.sample_count += 1
            if self.sample_count % 1e5 == 0:
                self.noise_level = self.noise_level*0.8

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = self._normalize_obs(state)

        if not evaluate:
            action, _, _ = self.policy.sample_env(state)
            step = self.sample_count if total_numsteps is None else int(total_numsteps)
            if self.directional_ref_noise and self.directional_noise_mode != "none":
                action = self._directional_reference_noise_action(state, action, step)
            else:
                self._last_explore_stats = self._empty_explore_stats()
                noise = torch.randn_like(action) * self.mask_noise_scale * self.noise_level
                noise = torch.clamp(noise, -self.mask_noise_clip, self.mask_noise_clip)
                action, _ = self._soft_normal_mask_action(state, action, noise)
        else:
            self._last_explore_stats = self._empty_explore_stats()
            with torch.no_grad():
                action, _, _ = self.policy.sample_env(state)
        
        return action.detach().cpu().numpy()[0].clip(self.action_space.low, self.action_space.high)

    @torch.no_grad()
    def observe(self, state, next_state=None):
        if self.obs_rms is None:
            return

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.obs_rms.update(state_tensor)
        if next_state is not None:
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            self.obs_rms.update(next_state_tensor)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs, clip=self.obs_norm_clip, eps=self.obs_norm_eps)

    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """
        Critic update.
        - If distributional_critic: C51 cross-entropy on projected distribution.
        - Else: MSE TD error on scalar Q.
        Both include LAC kinetic penalty in the target:  r + gamma * (Q - alpha * kinetic).
        """
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            with torch.no_grad():
                next_state_action, next_kinetic, _ = self.policy.sample(next_state_batch)
                alpha = self.log_alpha.exp()

                if self.distributional_critic:
                    qf1_next_target_logits, qf2_next_target_logits = self.critic_target(
                        next_state_batch, next_state_action
                    )
                    next_prob_1 = F.softmax(qf1_next_target_logits.float(), dim=-1)
                    next_prob_2 = F.softmax(qf2_next_target_logits.float(), dim=-1)

                    qf1_next_target = (next_prob_1 * self.c51_atoms).sum(dim=-1, keepdim=True)
                    qf2_next_target = (next_prob_2 * self.c51_atoms).sum(dim=-1, keepdim=True)
                    use_q1 = (qf1_next_target <= qf2_next_target)
                    next_prob = torch.where(use_q1, next_prob_1, next_prob_2)

                    # Project (r + gamma * (z - alpha * kinetic)) onto fixed support.
                    target_z = reward_batch + mask_batch * self.gamma * self.c51_atoms.view(1, -1)
                    target_z = target_z - (mask_batch * self.gamma * alpha * next_kinetic)
                    target_z = target_z.clamp(self.critic_v_min, self.critic_v_max)

                    b = (target_z - self.critic_v_min) / self.c51_delta
                    l = b.floor().to(torch.int64)
                    u = b.ceil().to(torch.int64)
                    l = l.clamp(0, self.critic_num_atoms - 1)
                    u = u.clamp(0, self.critic_num_atoms - 1)

                    m = torch.zeros_like(next_prob)
                    m_l = (u.to(b.dtype) - b)
                    m_u = (b - l.to(b.dtype))
                    eq = (u == l)
                    m_l = torch.where(eq, torch.ones_like(m_l), m_l)
                    m_u = torch.where(eq, torch.zeros_like(m_u), m_u)
                    m.scatter_add_(1, l, next_prob * m_l)
                    m.scatter_add_(1, u, next_prob * m_u)
                    target_dist = m
                else:
                    qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target - alpha * next_kinetic)

            # Update critic
            if self.distributional_critic:
                qf1_logits, qf2_logits = self.critic(state_batch, action_batch)
                log_p1 = F.log_softmax(qf1_logits.float(), dim=-1)
                log_p2 = F.log_softmax(qf2_logits.float(), dim=-1)
                qf1_loss = -(target_dist * log_p1).sum(dim=-1).mean()
                qf2_loss = -(target_dist * log_p2).sum(dim=-1).mean()
                qf_loss = qf1_loss + qf2_loss
            else:
                qf1, qf2 = self.critic(state_batch, action_batch)
                # Keep two independent targets to avoid accidental graph aliasing.
                qf1_loss = F.mse_loss(qf1, next_q_value)
                qf2_loss = F.mse_loss(qf2, next_q_value.clone())
                qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        self.scaler.scale(qf_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        return {
            "loss/critic": float(qf_loss.detach().item()),
        }

    def update_safety_critic(self, state_batch, action_batch, cost_batch, next_state_batch, mask_batch):
        with torch.no_grad():
            next_action, _, _ = self.policy.sample(next_state_batch)
            qc1_next, qc2_next = self.safety_critic_target(next_state_batch, next_action)
            qc_next = torch.max(qc1_next, qc2_next)

            if self.safety_critic_mode == "cdf":
                cost = (cost_batch > 0).float() if self.cdf_binarize_cost else cost_batch.clamp(0.0, 1.0)
                qc_target = cost + (1.0 - cost) * mask_batch * self.cost_gamma * qc_next
                if self.cdf_target_clip:
                    qc_target = qc_target.clamp(0.0, 1.0)
            else:
                # Preserve the historical FLAC-Safe safety target exactly for
                # old runs and checkpoints.
                qc_target = cost_batch + mask_batch * (1.0 - cost_batch) * self.cost_gamma * qc_next
                qc_target = torch.clamp(qc_target, 0.0, 1.0)
            if self.safety_critic_mode == "cdf" and self.cdf_target_clip:
                qc_target_clip_frac = ((qc_target <= 0.0) | (qc_target >= 1.0)).float().mean()
            else:
                qc_target_clip_frac = torch.tensor(0.0, device=self.device)

        qc1, qc2 = self.safety_critic(state_batch, action_batch)
        qc_risk = torch.max(qc1, qc2)
        td1 = qc1 - qc_target
        td2 = qc2 - qc_target
        loss1_unweighted = td1.pow(2)
        loss2_unweighted = td2.pow(2)
        qc_loss_unweighted = loss1_unweighted.mean() + loss2_unweighted.mean()

        priority_enabled = self.high_fidelity_safety_q and self.safety_q_priority
        if priority_enabled:
            with torch.no_grad():
                q_score = qc_risk.detach()
                boundary_mask = (torch.abs(q_score - self.safe_threshold) < self.safety_q_boundary_width).float()
                cost_mask = (cost_batch > 0).float()
                td_abs = torch.max(td1.detach().abs(), td2.detach().abs())
                weight = (
                    1.0
                    + self.safety_q_cost_weight * cost_mask
                    + self.safety_q_boundary_weight * boundary_mask
                    + self.safety_q_td_weight * td_abs
                )
                weight = weight.clamp(1.0, self.safety_q_max_weight)
            qc_loss = (weight * loss1_unweighted).mean() + (weight * loss2_unweighted).mean()
        else:
            boundary_mask = torch.zeros_like(qc_risk)
            cost_mask = (cost_batch > 0).float()
            td_abs = torch.max(td1.detach().abs(), td2.detach().abs())
            weight = torch.ones_like(qc_risk)
            qc_loss = qc_loss_unweighted

        self.safety_critic_optim.zero_grad()
        qc_loss.backward()
        self.safety_critic_optim.step()

        return {
            "loss/safety_critic": qc_loss.item(),
            "safety/qc_loss": qc_loss.detach().item(),
            "safety/safety_critic_mode_id": 1.0 if self.safety_critic_mode == "cdf" else 0.0,
            "safety/qc_geom_mode_id": 1.0 if self.qc_geom_mode == "mean" else 0.0,
            "safety/qc_mean": qc_risk.detach().mean().item(),
            "safety/qc_target_mean": qc_target.detach().mean().item(),
            "safety/qc_target_min": qc_target.detach().min().item(),
            "safety/qc_target_max": qc_target.detach().max().item(),
            "safety/qc_target_clip_frac": qc_target_clip_frac.detach().item(),
            "safety/cost_batch": cost_batch.detach().mean().item(),
            "safety_q/priority_enabled": 1.0 if priority_enabled else 0.0,
            "safety_q/weight_mean": weight.detach().mean().item(),
            "safety_q/weight_max": weight.detach().max().item(),
            "safety_q/cost_mask_frac": cost_mask.detach().mean().item(),
            "safety_q/boundary_mask_frac": boundary_mask.detach().mean().item(),
            "safety_q/td_abs_mean": td_abs.detach().mean().item(),
            "safety_q/loss_weighted": qc_loss.detach().item(),
            "safety_q/loss_unweighted": qc_loss_unweighted.detach().item(),
        }

    def safety_q_geometry_diagnostics(self, state_batch, action_batch, velocity_state_batch=None):
        if (not self.safe_env) or self.safety_critic is None:
            return {}
        batch_size = int(state_batch.shape[0])
        if batch_size == 0:
            return {}

        low = torch.as_tensor(self.action_space.low, dtype=action_batch.dtype, device=self.device).view(1, -1)
        high = torch.as_tensor(self.action_space.high, dtype=action_batch.dtype, device=self.device).view(1, -1)
        fd_eps = max(self.safety_q_fd_eps, 1e-8)
        eps = max(self.jvp_eps, 1e-8)

        flags = self.set_requires_grad(self.safety_critic, False)
        try:
            def scalar_from_outputs(q1, q2, mode):
                if mode == "mean":
                    return 0.5 * (q1 + q2)
                return torch.max(q1, q2)

            def fd_stats(q_ref, grad, mode):
                grad_norm_local = grad.detach().norm(dim=-1, keepdim=True)
                normal = grad.detach() / (grad_norm_local + eps)
                action_plus = torch.max(torch.min(action_batch.detach() + fd_eps * normal, high), low)
                action_minus = torch.max(torch.min(action_batch.detach() - fd_eps * normal, high), low)
                with torch.no_grad():
                    q1_plus, q2_plus = self.safety_critic(state_batch.detach(), action_plus)
                    q1_minus, q2_minus = self.safety_critic(state_batch.detach(), action_minus)
                    q_plus = scalar_from_outputs(q1_plus, q2_plus, mode)
                    q_minus = scalar_from_outputs(q1_minus, q2_minus, mode)
                q_detached_local = q_ref.detach()
                q_plus_delta_local = q_plus.detach() - q_detached_local
                q_minus_delta_local = q_minus.detach() - q_detached_local
                fd_slope_local = (q_plus.detach() - q_minus.detach()) / (2.0 * fd_eps)
                finite_grad_local = torch.isfinite(grad.detach()).all(dim=-1, keepdim=True).float()
                zero_grad_local = (grad_norm_local <= eps).float()
                boundary_local = (
                    torch.abs(q_detached_local - self.safe_threshold) < self.safety_q_boundary_width
                ).float()
                return {
                    "grad_norm_mean": grad_norm_local.mean(),
                    "grad_norm_std": grad_norm_local.std(unbiased=False),
                    "zero_grad_frac": zero_grad_local.mean(),
                    "grad_nan_inf_frac": (1.0 - finite_grad_local).mean(),
                    "q_plus_minus_q_mean": q_plus_delta_local.mean(),
                    "q_minus_minus_q_mean": q_minus_delta_local.mean(),
                    "fd_slope_mean": fd_slope_local.mean(),
                    "mono_plus_frac": (q_plus.detach() > q_detached_local).float().mean(),
                    "mono_minus_frac": (q_minus.detach() < q_detached_local).float().mean(),
                    "boundary_frac": boundary_local.mean(),
                }

            with torch.enable_grad():
                action_for_grad = action_batch.detach().requires_grad_(True)
                qc1, qc2 = self.safety_critic(state_batch.detach(), action_for_grad)
                q_max = torch.max(qc1, qc2)
                q_mean = 0.5 * (qc1 + qc2)
                q_geom = self._qc_geom_scalar(state_batch.detach(), action_for_grad)
                grad_geom = torch.autograd.grad(
                    outputs=q_geom.sum(),
                    inputs=action_for_grad,
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                if self.qc_geom_mode == "max":
                    grad_max = grad_geom
                else:
                    grad_max = torch.autograd.grad(
                        outputs=q_max.sum(),
                        inputs=action_for_grad,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True,
                    )[0]

            geom_stats = fd_stats(q_geom, grad_geom, self.qc_geom_mode)
            max_stats = fd_stats(q_max, grad_max, "max")

            jvp_mean = torch.tensor(0.0, device=self.device)
            normalized_jvp_mean = torch.tensor(0.0, device=self.device)
            if velocity_state_batch is not None:
                with torch.no_grad():
                    action_pi, _, _, velocity_action = self.policy.sample(velocity_state_batch.detach(), return_velocity=True)
                with torch.enable_grad():
                    action_pi_grad = action_pi.detach().requires_grad_(True)
                    qc_policy = self._qc_geom_scalar(velocity_state_batch.detach(), action_pi_grad)
                    grad_policy = torch.autograd.grad(
                        outputs=qc_policy.sum(),
                        inputs=action_pi_grad,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True,
                    )[0].detach()
                jvp = (grad_policy * velocity_action.detach()).sum(dim=-1, keepdim=True)
                grad_policy_norm = grad_policy.norm(dim=-1, keepdim=True)
                jvp_mean = jvp.abs().mean()
                normalized_jvp_mean = (jvp.pow(2) / (grad_policy_norm.pow(2) + eps)).mean()

            return {
                "safety_q/q_mean": q_mean.detach().mean().item(),
                "safety_q/q_std": q_mean.detach().std(unbiased=False).item(),
                "safety_q/q_max_mean": q_max.detach().mean().item(),
                "safety_q/q_max_std": q_max.detach().std(unbiased=False).item(),
                "safety_q/diag_qc_geom_mode_id": 1.0 if self.qc_geom_mode == "mean" else 0.0,
                "safety_q/grad_norm_mean": geom_stats["grad_norm_mean"].item(),
                "safety_q/grad_norm_std": geom_stats["grad_norm_std"].item(),
                "safety_q/zero_grad_frac": geom_stats["zero_grad_frac"].item(),
                "safety_q/grad_nan_inf_frac": geom_stats["grad_nan_inf_frac"].item(),
                "safety_q/q_plus_minus_q_mean": geom_stats["q_plus_minus_q_mean"].item(),
                "safety_q/q_minus_minus_q_mean": geom_stats["q_minus_minus_q_mean"].item(),
                "safety_q/fd_slope_mean": geom_stats["fd_slope_mean"].item(),
                "safety_q/mono_plus_frac": geom_stats["mono_plus_frac"].item(),
                "safety_q/mono_minus_frac": geom_stats["mono_minus_frac"].item(),
                "safety_q/boundary_frac": geom_stats["boundary_frac"].item(),
                "safety_q/geom_grad_norm_mean": geom_stats["grad_norm_mean"].item(),
                "safety_q/geom_grad_norm_std": geom_stats["grad_norm_std"].item(),
                "safety_q/geom_zero_grad_frac": geom_stats["zero_grad_frac"].item(),
                "safety_q/geom_q_plus_minus_q_mean": geom_stats["q_plus_minus_q_mean"].item(),
                "safety_q/geom_q_minus_minus_q_mean": geom_stats["q_minus_minus_q_mean"].item(),
                "safety_q/geom_fd_slope_mean": geom_stats["fd_slope_mean"].item(),
                "safety_q/geom_mono_plus_frac": geom_stats["mono_plus_frac"].item(),
                "safety_q/geom_mono_minus_frac": geom_stats["mono_minus_frac"].item(),
                "safety_q/geom_boundary_frac": geom_stats["boundary_frac"].item(),
                "safety_q/max_grad_norm_mean": max_stats["grad_norm_mean"].item(),
                "safety_q/max_zero_grad_frac": max_stats["zero_grad_frac"].item(),
                "safety_q/max_mono_plus_frac": max_stats["mono_plus_frac"].item(),
                "safety_q/max_mono_minus_frac": max_stats["mono_minus_frac"].item(),
                "safety_q/max_fd_slope_mean": max_stats["fd_slope_mean"].item(),
                "safety_q/jvp_mean": jvp_mean.detach().item(),
                "safety_q/normalized_jvp_mean": normalized_jvp_mean.detach().item(),
            }
        finally:
            self.restore_requires_grad(self.safety_critic, flags)

    def _qc_risk_scalar(self, state_batch, action_batch):
        qc1, qc2 = self.safety_critic(state_batch, action_batch)
        return torch.max(qc1, qc2)

    def _qc_geom_scalar(self, state_batch, action_batch):
        qc1, qc2 = self.safety_critic(state_batch, action_batch)
        if self.qc_geom_mode == "mean":
            return 0.5 * (qc1 + qc2)
        return torch.max(qc1, qc2)

    def _qc_scalar(self, state_batch, action_batch):
        return self._qc_risk_scalar(state_batch, action_batch)

    def _forward_jvp_directional(self, state_batch, action_base, velocity_action):
        """Compute d Q_C(x,u)[velocity_action] by true forward-mode JVP.

        The primal action is detached so the actor update uses Q_C only as a
        fixed risk geometry; gradients still flow to velocity_action through the
        tangent output of JVP.
        """
        if torch_func_jvp is None:
            raise RuntimeError("torch.func.jvp is unavailable in this PyTorch version")
        state_detached = state_batch.detach()
        action_detached = action_base.detach()

        def qc_fn(action_in):
            return self._qc_geom_scalar(state_detached, action_in)

        _, directional = torch_func_jvp(qc_fn, (action_detached,), (velocity_action,))
        return directional

    def _grad_dot_directional(self, state_batch, action_pi, velocity_action):
        """Fallback: VJP-style grad-dot-vector directional derivative."""
        action_for_grad = action_pi.detach().requires_grad_(True)
        qc = self._qc_geom_scalar(state_batch.detach(), action_for_grad)
        grad_q = torch.autograd.grad(
            outputs=qc.sum(),
            inputs=action_for_grad,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0].detach()
        directional = (grad_q * velocity_action).sum(dim=-1, keepdim=True)
        grad_norm_sq = grad_q.pow(2).sum(dim=-1, keepdim=True).detach()
        return directional, grad_norm_sq

    def _hutchinson_grad_norm_sq(self, state_batch, action_pi):
        """Estimate ||grad_u Q_C||^2 using forward-mode JVP probes."""
        samples = max(1, self.jvp_hutchinson_samples)
        terms = []
        for _ in range(samples):
            xi = torch.randn_like(action_pi)
            try:
                d_xi = self._forward_jvp_directional(state_batch, action_pi, xi)
                terms.append(d_xi.pow(2))
            except Exception:
                _, grad_norm_sq = self._grad_dot_directional(state_batch, action_pi, xi)
                return grad_norm_sq
        return torch.stack(terms, dim=0).mean(dim=0).detach()

    def compute_jvp_scd(self, state_batch, action_pi, velocity_action, g_mid):
        """Safety-critical directional derivative penalty.

        Preferred mode: true forward-mode JVP of Q_C along flow velocity.
        Fallback mode: grad-dot-vector, mathematically the same directional
        derivative but computed by reverse-mode autograd.

        If normalize_jvp=True, the loss uses an estimated ||grad_u Q_C||^2
        denominator. With jvp_norm_mode='hutchinson', that denominator is
        estimated by random JVP probes; with 'exact' it uses the fallback exact
        reverse-mode gradient norm.
        """
        directional_source = "forward_jvp"
        try:
            if self.jvp_mode == "forward":
                directional = self._forward_jvp_directional(state_batch, action_pi, velocity_action)
                grad_norm_sq = None
            else:
                directional, grad_norm_sq = self._grad_dot_directional(state_batch, action_pi, velocity_action)
                directional_source = "grad_dot"
        except Exception:
            directional, grad_norm_sq = self._grad_dot_directional(state_batch, action_pi, velocity_action)
            directional_source = "grad_dot_fallback"

        if self.normalize_jvp:
            if self.jvp_norm_mode == "hutchinson":
                denom = self._hutchinson_grad_norm_sq(state_batch, action_pi)
            else:
                if grad_norm_sq is None:
                    _, grad_norm_sq = self._grad_dot_directional(state_batch, action_pi, torch.zeros_like(velocity_action))
                denom = grad_norm_sq
            loss_terms = directional.pow(2) / (denom.detach() + self.jvp_eps)
            denom_mean = denom.detach().mean()
            jvp_loss = (g_mid.detach() * loss_terms).mean()
            grad_norm = torch.sqrt(denom.detach() + self.jvp_eps).mean()
        else:
            jvp_loss = (g_mid.detach() * directional.pow(2)).mean()
            denom_mean = torch.tensor(0.0, device=self.device)
            if grad_norm_sq is None:
                # Avoid an extra reverse-mode pass only for logging in forward JVP mode.
                grad_norm = torch.tensor(0.0, device=self.device)
            else:
                grad_norm = torch.sqrt(grad_norm_sq + self.jvp_eps).mean().detach()

        source_code = {
            "forward_jvp": 1.0,
            "grad_dot": 0.0,
            "grad_dot_fallback": -1.0,
        }[directional_source]
        source_tensor = torch.tensor(source_code, device=self.device)
        directional_abs = directional.detach().abs().mean()
        return jvp_loss, grad_norm.detach(), denom_mean.detach(), directional_abs, source_tensor

    @staticmethod
    def set_requires_grad(module, requires_grad):
        old_flags = []
        for p in module.parameters():
            old_flags.append(p.requires_grad)
            p.requires_grad_(requires_grad)
        return old_flags

    @staticmethod
    def restore_requires_grad(module, old_flags):
        for p, flag in zip(module.parameters(), old_flags):
            p.requires_grad_(flag)

    def update_policy(self, state_batch, current_step_or_updates=0):
        """
        LAC policy + temperature update.
        Actor loss:  E[ -Q(s,a) + alpha * kinetic ]
        Alpha update (SAC-style on log_alpha): match mean kinetic to target_kinetic.
        """
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            if self.safe_env and self.safe_policy_loss:
                action, kinetic, _, velocity_action = self.policy.sample(state_batch, return_velocity=True)
            else:
                action, kinetic, _ = self.policy.sample(state_batch)
            alpha = self.log_alpha.exp()

            if self.distributional_critic:
                qf1_pi_logits, qf2_pi_logits = self.critic(state_batch, action)
                qf1_pi = (F.softmax(qf1_pi_logits.float(), dim=-1) * self.c51_atoms).sum(dim=-1, keepdim=True)
                qf2_pi = (F.softmax(qf2_pi_logits.float(), dim=-1) * self.c51_atoms).sum(dim=-1, keepdim=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            else:
                qf1_pi, qf2_pi = self.critic(state_batch, action)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

            safety_penalty = torch.zeros_like(min_qf_pi)
            jvp_loss = torch.tensor(0.0, device=self.device)
            grad_q_norm = torch.tensor(0.0, device=self.device)
            jvp_denom_mean = torch.tensor(0.0, device=self.device)
            jvp_directional_abs = torch.tensor(0.0, device=self.device)
            jvp_source = torch.tensor(0.0, device=self.device)
            g_mid_mean = torch.tensor(0.0, device=self.device)
            feas_gate_mean = torch.tensor(0.0, device=self.device)
            feas_gate_min = torch.tensor(0.0, device=self.device)
            feas_gate_max = torch.tensor(0.0, device=self.device)
            reward_weight_mean = torch.tensor(1.0, device=self.device)
            safe_weight_mean = torch.tensor(0.0, device=self.device)
            feas_gate_risky_frac = torch.tensor(0.0, device=self.device)
            reward_weight = torch.ones_like(min_qf_pi)
            safe_weight = torch.ones_like(min_qf_pi)
            jvp_enabled = self.safe_env and self.safe_policy_loss and current_step_or_updates >= self.jvp_warmup_steps
            lambda_jvp_eff_value = self._lambda_jvp_eff(current_step_or_updates)
            lambda_jvp_eff = torch.tensor(lambda_jvp_eff_value, device=self.device)

            if self.safe_env and self.safe_policy_loss:
                safety_flags = self.set_requires_grad(self.safety_critic, False)
                try:
                    qc_pi_risk = self._qc_risk_scalar(state_batch, action)
                    qc_pi_geom = self._qc_geom_scalar(state_batch, action)
                    safety_penalty = F.relu(qc_pi_risk - self.safe_threshold)
                    g_mid = self._compute_g_mid_from_qc(qc_pi_risk.detach())
                    if jvp_enabled:
                        jvp_loss, grad_q_norm, jvp_denom_mean, jvp_directional_abs, jvp_source = self.compute_jvp_scd(
                            state_batch, action, velocity_action, g_mid
                        )
                    g_mid_mean = g_mid.detach().mean()
                    qc_pi_risk_mean = qc_pi_risk.detach().mean()
                    qc_pi_geom_mean = qc_pi_geom.detach().mean()
                    qc_pi_risk_over_threshold = (qc_pi_risk.detach() > self.safe_threshold).float().mean()
                    if self.soft_feasibility_gate:
                        zeta = torch.sigmoid((self.safe_threshold - qc_pi_risk) / self.feas_gate_tau)
                        if self.feas_gate_detach:
                            zeta = zeta.detach()
                        reward_weight = self.feas_gate_reward_floor + (1.0 - self.feas_gate_reward_floor) * zeta
                        safe_weight = 1.0 - zeta
                        zeta_detached = zeta.detach()
                        feas_gate_mean = zeta_detached.mean()
                        feas_gate_min = zeta_detached.min()
                        feas_gate_max = zeta_detached.max()
                        reward_weight_mean = reward_weight.detach().mean()
                        safe_weight_mean = safe_weight.detach().mean()
                        feas_gate_risky_frac = (qc_pi_risk.detach() > self.safe_threshold).float().mean()
                finally:
                    self.restore_requires_grad(self.safety_critic, safety_flags)
            else:
                qc_pi_risk_mean = torch.tensor(0.0, device=self.device)
                qc_pi_geom_mean = torch.tensor(0.0, device=self.device)
                qc_pi_risk_over_threshold = torch.tensor(0.0, device=self.device)

            if self.safe_env and self.safe_policy_loss and self.soft_feasibility_gate:
                policy_loss_terms = -reward_weight * min_qf_pi + alpha.detach() * kinetic
            else:
                policy_loss_terms = -min_qf_pi + alpha.detach() * kinetic
            if self.safe_env and self.safe_policy_loss:
                if self.soft_feasibility_gate:
                    policy_loss_terms = policy_loss_terms + safe_weight * self.lambda_safe * safety_penalty
                else:
                    policy_loss_terms = policy_loss_terms + self.lambda_safe * safety_penalty

            policy_loss = policy_loss_terms.mean()
            if jvp_enabled:
                policy_loss = policy_loss + lambda_jvp_eff * jvp_loss

        # Update policy
        self.policy_optim.zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.policy_optim)
        self.scaler.update()

        if self.auto_alpha:
            # Update alpha (SAC-style on log_alpha; stable when alpha is small).
            # We intentionally detach kinetic to avoid gradients flowing into the policy.
            kinetic_mean = kinetic.detach().mean()
            alpha_loss = self.log_alpha * (self.target_kinetic - kinetic_mean)

            self.alpha_optim.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.alpha_optim)
            self.scaler.update()
        jvp_weighted = lambda_jvp_eff * jvp_loss if jvp_enabled else torch.tensor(0.0, device=self.device)
        jvp_scd_value = float(jvp_loss.detach().item())
        jvp_weighted_value = float(jvp_weighted.detach().item())
        return {
            "loss/policy": float(policy_loss.detach().item()),
            "loss/alpha": float(alpha_loss.detach().item()) if self.auto_alpha else 0.0,
            "train/kinetic": float(kinetic.detach().mean().item()),
            "safety/safety_penalty": safety_penalty.detach().mean().item(),
            "safety/safety_critic_mode_id": 1.0 if self.safety_critic_mode == "cdf" else 0.0,
            "safety/qc_geom_mode_id": 1.0 if self.qc_geom_mode == "mean" else 0.0,
            "safety/qc_pi_risk_mean": float(qc_pi_risk_mean.detach().item()),
            "safety/qc_pi_geom_mean": float(qc_pi_geom_mean.detach().item()),
            "safety/qc_pi_risk_over_threshold": float(qc_pi_risk_over_threshold.detach().item()),
            "safety/soft_feasibility_gate_enabled": 1.0 if self.soft_feasibility_gate else 0.0,
            "safety/feas_gate_mean": float(feas_gate_mean.detach().item()),
            "safety/feas_gate_min": float(feas_gate_min.detach().item()),
            "safety/feas_gate_max": float(feas_gate_max.detach().item()),
            "safety/reward_weight_mean": float(reward_weight_mean.detach().item()),
            "safety/safe_weight_mean": float(safe_weight_mean.detach().item()),
            "safety/feas_gate_risky_frac": float(feas_gate_risky_frac.detach().item()),
            "safety/lambda_jvp_schedule_enabled": 1.0 if self.lambda_jvp_schedule else 0.0,
            "safety/lambda_jvp_eff": float(lambda_jvp_eff.detach().item()),
            "safety/lambda_jvp_start": float(self.lambda_jvp_start),
            "safety/lambda_jvp_end": float(self.lambda_jvp_end),
            "loss/jvp_scd": jvp_scd_value,
            "loss/jvp_scd_x1e6": jvp_scd_value * 1e6,
            "loss/jvp_weighted": jvp_weighted_value,
            "loss/jvp_weighted_x1e6": jvp_weighted_value * 1e6,
            "safety/g_mid_mean": float(g_mid_mean.detach().item()),
            "safety/grad_q_norm": float(grad_q_norm.detach().item()),
            "safety/jvp_denom_mean": float(jvp_denom_mean.detach().item()),
            "safety/jvp_directional_abs": float(jvp_directional_abs.detach().item()),
            "safety/jvp_source": float(jvp_source.detach().item()),
        }


    def update_parameters(self, memory, batch_size, updates, total_numsteps=None):
        """
        Update: Critic and Policy updates
        """
        if self.safe_env:
            state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = self.ensure_column(torch.FloatTensor(reward_batch).to(self.device))
        mask_batch = self.ensure_column(torch.FloatTensor(mask_batch).to(self.device))
        if self.safe_env:
            cost_batch = self.ensure_column(torch.FloatTensor(cost_batch).to(self.device))

        state_batch = self._normalize_obs(state_batch)
        next_state_batch = self._normalize_obs(next_state_batch)
        
        log_info = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        if self.safe_env:
            log_info.update(
                self.update_safety_critic(state_batch, action_batch, cost_batch, next_state_batch, mask_batch)
            )
            extra_losses = []
            if self.high_fidelity_safety_q and self.safety_q_extra_updates > 0:
                for _ in range(self.safety_q_extra_updates):
                    extra_state, extra_action, _, extra_cost, extra_next_state, extra_mask = memory.sample(batch_size=batch_size)
                    extra_state = self._normalize_obs(torch.FloatTensor(extra_state).to(self.device))
                    extra_next_state = self._normalize_obs(torch.FloatTensor(extra_next_state).to(self.device))
                    extra_action = torch.FloatTensor(extra_action).to(self.device)
                    extra_cost = self.ensure_column(torch.FloatTensor(extra_cost).to(self.device))
                    extra_mask = self.ensure_column(torch.FloatTensor(extra_mask).to(self.device))
                    extra_info = self.update_safety_critic(
                        extra_state, extra_action, extra_cost, extra_next_state, extra_mask
                    )
                    extra_losses.append(float(extra_info["loss/safety_critic"]))
            log_info["safety_q/extra_updates"] = float(self.safety_q_extra_updates if self.high_fidelity_safety_q else 0)
            log_info["safety_q/extra_loss_mean"] = float(np.mean(extra_losses)) if extra_losses else 0.0
            if self.high_fidelity_safety_q and self.diagnose_safety_q_geometry:
                log_info.update(self.safety_q_geometry_diagnostics(state_batch, action_batch, state_batch))

        # Update policy and alpha (with delayed update)
        if updates % self.target_update_interval == 0:
            step_for_jvp = total_numsteps if total_numsteps is not None else updates
            log_info.update(self.update_policy(state_batch, step_for_jvp))
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                if self.safe_env:
                    soft_update(self.safety_critic_target, self.safety_critic, self.tau)

        if self._last_explore_stats:
            log_info.update(self._last_explore_stats)

        return log_info

    @staticmethod
    def ensure_column(x):
        if x.dim() == 1:
            return x.unsqueeze(1)
        return x

    # Save model parameters
    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        checkpoint = {'policy_state_dict': self.policy.state_dict(),
                      'critic_state_dict': self.critic.state_dict(),
                      'critic_target_state_dict': self.critic_target.state_dict(),
                      'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                      'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                      'alpha_optimizer_state_dict': self.alpha_optim.state_dict() if self.alpha_optim else None,
                      'log_alpha': self.log_alpha,
                      'obs_rms_state_dict': self.obs_rms.state_dict() if self.obs_rms is not None else None,
                      }
        if self.safe_env:
            checkpoint.update({
                'safety_critic_state_dict': self.safety_critic.state_dict(),
                'safety_critic_target_state_dict': self.safety_critic_target.state_dict(),
                'safety_critic_optimizer_state_dict': self.safety_critic_optim.state_dict(),
            })
        torch.save(checkpoint, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, path, i_episode, evaluate=False):
        # ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        ckpt_path = path + '/' + 'checkpoint/'+'best.torch'
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            # Load alpha state if available
            if 'log_alpha' in checkpoint:
                self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
            if self.alpha_optim is not None and checkpoint.get('alpha_optimizer_state_dict') is not None:
                self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

            if self.safe_env and checkpoint.get('safety_critic_state_dict') is not None:
                self.safety_critic.load_state_dict(checkpoint['safety_critic_state_dict'])
                self.safety_critic_target.load_state_dict(checkpoint['safety_critic_target_state_dict'])
                if checkpoint.get('safety_critic_optimizer_state_dict') is not None:
                    self.safety_critic_optim.load_state_dict(checkpoint['safety_critic_optimizer_state_dict'])

            obs_rms_state_dict = checkpoint.get('obs_rms_state_dict')
            if obs_rms_state_dict is not None:
                if self.obs_rms is None:
                    self.normalize_obs = True
                    self.obs_rms = RunningMeanStd(self.num_inputs, device=self.device)
                self.obs_rms.load_state_dict(obs_rms_state_dict)

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                if self.safe_env:
                    self.safety_critic.eval()
                    self.safety_critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                if self.safe_env:
                    self.safety_critic.train()
                    self.safety_critic_target.train()
