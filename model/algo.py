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
        self.safe_bandwidth = float(getattr(args, "safe_bandwidth", 0.05))
        self.lambda_safe = float(getattr(args, "lambda_safe", 1.0))
        self.lambda_jvp = float(getattr(args, "lambda_jvp", 0.05))
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

        # Optional real-interaction soft normal masking. This is deliberately
        # separated from the training-time JVP-SCD objective: training can use
        # JVP only, while environment sampling can use an explicit VJP normal.
        self.soft_normal_masking = bool(getattr(args, "soft_normal_masking", False))
        self.masking_warmup_steps = int(getattr(args, "masking_warmup_steps", self.jvp_warmup_steps))
        self.mask_beta_max = float(getattr(args, "mask_beta_max", 0.5))
        self.mask_beta_tau = float(getattr(args, "mask_beta_tau", 10000.0))
        self.mask_noise_scale = float(getattr(args, "mask_noise_scale", 0.01))
        self.mask_noise_clip = float(getattr(args, "mask_noise_clip", 0.25))

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

    def _mask_beta(self):
        if self.sample_count < self.masking_warmup_steps:
            return 0.0
        tau = max(self.mask_beta_tau, 1.0)
        x = (float(self.sample_count) - float(self.masking_warmup_steps)) / tau
        beta = self.mask_beta_max / (1.0 + np.exp(-x))
        return float(beta)

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
    def select_action(self, state, evaluate=False):

        # Noise schedule for exploration: In all tasks, we set the noise to 0.
        if not evaluate:
            self.sample_count += 1
            if self.sample_count % 1e5 == 0:
                self.noise_level = self.noise_level*0.8

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = self._normalize_obs(state)

        if not evaluate:
            action, _, _ = self.policy.sample_env(state)
            noise = torch.randn_like(action) * self.mask_noise_scale * self.noise_level
            noise = torch.clamp(noise, -self.mask_noise_clip, self.mask_noise_clip)
            action, _ = self._soft_normal_mask_action(state, action, noise)
        else:
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

            qc_target = cost_batch + mask_batch * (1.0 - cost_batch) * self.cost_gamma * qc_next
            qc_target = torch.clamp(qc_target, 0.0, 1.0)

        qc1, qc2 = self.safety_critic(state_batch, action_batch)
        qc_loss = F.mse_loss(qc1, qc_target) + F.mse_loss(qc2, qc_target)

        self.safety_critic_optim.zero_grad()
        qc_loss.backward()
        self.safety_critic_optim.step()

        return {
            "loss/safety_critic": qc_loss.item(),
            "safety/qc_mean": torch.max(qc1, qc2).detach().mean().item(),
            "safety/qc_target_mean": qc_target.detach().mean().item(),
            "safety/cost_batch": cost_batch.detach().mean().item(),
        }

    def _qc_scalar(self, state_batch, action_batch):
        qc1, qc2 = self.safety_critic(state_batch, action_batch)
        return torch.max(qc1, qc2)

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
            return self._qc_scalar(state_detached, action_in)

        _, directional = torch_func_jvp(qc_fn, (action_detached,), (velocity_action,))
        return directional

    def _grad_dot_directional(self, state_batch, action_pi, velocity_action):
        """Fallback: VJP-style grad-dot-vector directional derivative."""
        action_for_grad = action_pi.detach().requires_grad_(True)
        qc = self._qc_scalar(state_batch.detach(), action_for_grad)
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
            jvp_enabled = self.safe_env and self.safe_policy_loss and current_step_or_updates >= self.jvp_warmup_steps

            if self.safe_env and self.safe_policy_loss:
                safety_flags = self.set_requires_grad(self.safety_critic, False)
                try:
                    qc1_pi, qc2_pi = self.safety_critic(state_batch, action)
                    qc_pi = torch.max(qc1_pi, qc2_pi)
                    safety_penalty = F.relu(qc_pi - self.safe_threshold)
                    g_mid = self._compute_g_mid_from_qc(qc_pi.detach())
                    if jvp_enabled:
                        jvp_loss, grad_q_norm, jvp_denom_mean, jvp_directional_abs, jvp_source = self.compute_jvp_scd(
                            state_batch, action, velocity_action, g_mid
                        )
                    g_mid_mean = g_mid.detach().mean()
                finally:
                    self.restore_requires_grad(self.safety_critic, safety_flags)

            policy_loss_terms = -min_qf_pi + alpha.detach() * kinetic
            if self.safe_env and self.safe_policy_loss:
                policy_loss_terms = policy_loss_terms + self.lambda_safe * safety_penalty

            policy_loss = policy_loss_terms.mean()
            if jvp_enabled:
                policy_loss = policy_loss + self.lambda_jvp * jvp_loss

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
        jvp_weighted = self.lambda_jvp * jvp_loss if jvp_enabled else torch.tensor(0.0, device=self.device)
        jvp_scd_value = float(jvp_loss.detach().item())
        jvp_weighted_value = float(jvp_weighted.detach().item())
        return {
            "loss/policy": float(policy_loss.detach().item()),
            "loss/alpha": float(alpha_loss.detach().item()) if self.auto_alpha else 0.0,
            "train/kinetic": float(kinetic.detach().mean().item()),
            "safety/safety_penalty": safety_penalty.detach().mean().item(),
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

        # Update policy and alpha (with delayed update)
        if updates % self.target_update_interval == 0:
            step_for_jvp = total_numsteps if total_numsteps is not None else updates
            log_info.update(self.update_policy(state_batch, step_for_jvp))
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                if self.safe_env:
                    soft_update(self.safety_critic_target, self.safety_critic, self.tau)

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
