#!/usr/bin/env python3
"""Run safety-geometry diagnostics for CDF/JVP safety critics.

This script is diagnostic-only: it does not change actor/critic losses.  The
current Safety-Gym wrapper does not expose a hard safety filter correction, so
the filter-direction fields use the only available executed-action projection:
action-space clipping.  Reports explicitly mark this as an action-bound proxy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.algo import flowAC
from utilis.SafeReplaybuffer import SafeReplayMemory
from utilis.default_config import default_config


LOG_DIR = ROOT / "logs" / "safety_geometry_diag"
REPORT_DIR = ROOT / "reports" / "safety_geometry_diag"
AUTO_REPORT_DIR = ROOT / "reports" / "auto_goal_tuning"
ERR_RE = ("Traceback", "RuntimeError", "NaN", "nan", "OOM", "out of memory")


RUNS: dict[str, dict[str, Any]] = {
    "DIAG_PG1_G4": {
        "task": "SafetyPointGoal1-v0",
        "config": "G4_fixed_main",
        "seed": 0,
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 60000,
    },
    "DIAG_CG1_G4": {
        "task": "SafetyCarGoal1-v0",
        "config": "G4_fixed_main",
        "seed": 0,
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 60000,
    },
    "DIAG_CG1_C2": {
        "task": "SafetyCarGoal1-v0",
        "config": "CG1_C2_safe05",
        "seed": 0,
        "safe_threshold": 0.05,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 60000,
    },
    "DIAG_CG1_C4_long": {
        "task": "SafetyCarGoal1-v0",
        "config": "CG1_C4_long_G4",
        "seed": 0,
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 60000,
    },
}

DEFAULT_RUNS = ("DIAG_PG1_G4", "DIAG_CG1_G4", "DIAG_CG1_C2")


def reset_env(env, seed: int | None = None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    next_state, reward, cost, terminated, truncated, info = env.step(action)
    return next_state, reward, float(cost), terminated, truncated, info


def ensure_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid:
        return None, None
    if len(valid) == 1:
        return valid[0], 0.0
    return mean(valid), stdev(valid)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_config(run_name: str, overrides: argparse.Namespace):
    spec = RUNS[run_name]
    cfg = default_config.copy()
    cfg.update(
        {
            "task": spec["task"],
            "safe_env": True,
            "safety_critic_mode": "cdf",
            "qc_geom_mode": "mean",
            "safe_threshold": spec["safe_threshold"],
            "lambda_safe": spec["lambda_safe"],
            "lambda_jvp": spec["lambda_jvp"],
            "safe_bandwidth": spec["safe_bandwidth"],
            "normalize_jvp": True,
            "jvp_norm_mode": "exact",
            "jvp_mode": "grad",
            "cdf_binarize_cost": True,
            "cdf_target_clip": True,
            "soft_feasibility_gate": False,
            "soft_normal_masking": False,
            "directional_ref_noise": False,
            "epsilon": 0.0,
            "batch_size": overrides.batch_size,
            "updates_per_step": overrides.updates_per_step,
            "hidden_size": overrides.hidden_size,
            "num_steps": overrides.num_steps or spec["num_steps"],
            "start_steps": overrides.start_steps,
            "eval": True,
            "eval_numsteps": overrides.eval_numsteps,
            "eval_times": overrides.eval_times,
            "distributional_critic": False,
            "compile_model": False,
            "save": False,
            "steps": 1,
            "seed": spec["seed"],
            "cuda": not overrides.cpu,
            "device": overrides.device,
            "algo": "SafetyGeometryDiag",
            "tag": run_name,
            "diagnose_safety_geometry": True,
        }
    )
    return cfg


def evaluate(agent: flowAC, env, cfg) -> tuple[float, float]:
    total_reward = 0.0
    total_cost = 0.0
    for _ in range(cfg.eval_times):
        state = reset_env(env)
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, cost, terminated, truncated, _ = step_env(env, action)
            done = terminated or truncated
            total_reward += float(reward)
            total_cost += float(cost)
            state = next_state
    avg_reward = total_reward / float(cfg.eval_times)
    avg_cost = total_cost / float(cfg.eval_times)
    print(
        f"Env: {cfg.task}, Test Episodes: {cfg.eval_times}, "
        f"Avg. Reward: {avg_reward:.2f}, Avg. Cost: {avg_cost:.2f}"
    )
    return avg_reward, avg_cost


def diagnostic_snapshot(
    agent: flowAC,
    memory: SafeReplayMemory,
    cfg,
    *,
    total_numsteps: int,
    run_name: str,
    eval_reward: float,
    eval_cost: float,
    batch_size: int,
    eps_filter: float,
    eps_grad: float,
) -> dict[str, Any]:
    batch = min(batch_size, len(memory))
    record: dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run": run_name,
        "task": cfg.task,
        "config": RUNS[run_name]["config"],
        "seed": int(cfg.seed),
        "step": int(total_numsteps),
        "eval_reward": float(eval_reward),
        "eval_cost": float(eval_cost),
        "filter_source": "action_bounds_clip_proxy",
        "hard_filter_available": False,
    }
    if batch < 8:
        record["status"] = "insufficient_replay"
        return record

    state, action_replay, _, cost, _, _ = memory.sample(batch_size=batch)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
    action_replay_t = torch.as_tensor(action_replay, dtype=torch.float32, device=agent.device)
    cost_t = agent.ensure_column(torch.as_tensor(cost, dtype=torch.float32, device=agent.device))
    state_t = agent._normalize_obs(state_t)

    low = torch.as_tensor(agent.action_space.low, dtype=torch.float32, device=agent.device).view(1, -1)
    high = torch.as_tensor(agent.action_space.high, dtype=torch.float32, device=agent.device).view(1, -1)

    safety_flags = agent.set_requires_grad(agent.safety_critic, False)
    try:
        with torch.enable_grad():
            action_pi, _, _, velocity_action = agent.policy.sample(state_t, return_velocity=True)
            raw_action = action_pi.detach()
            filtered_action = torch.max(torch.min(raw_action, high), low)
            delta_f = raw_action - filtered_action
            d_f = delta_f.norm(dim=-1, keepdim=True)
            intervention = d_f > eps_filter

            action_for_grad = raw_action.detach().requires_grad_(True)
            qc = agent._qc_risk_scalar(state_t.detach(), action_for_grad)
            grad_qc = torch.autograd.grad(
                outputs=qc.sum(),
                inputs=action_for_grad,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            grad_norm = grad_qc.norm(dim=-1, keepdim=True)
            n_c = grad_qc / (grad_norm + eps_grad)
            n_f = delta_f / (d_f + eps_filter)
            cos_qc_filter_all = (n_c * n_f).sum(dim=-1, keepdim=True)
            v_norm = velocity_action.detach().norm(dim=-1, keepdim=True)
            n_v = velocity_action.detach() / (v_norm + eps_grad)
            cos_v_filter_all = (n_v * n_f).sum(dim=-1, keepdim=True)
            jvp_mag = ((grad_qc.detach() * velocity_action.detach()).sum(dim=-1, keepdim=True).pow(2) /
                       (grad_norm.detach().pow(2) + eps_grad))

            qc_replay = agent._qc_risk_scalar(state_t.detach(), action_replay_t.detach())

        d_f_np = d_f.detach().cpu().numpy().reshape(-1)
        qc_np = qc.detach().cpu().numpy().reshape(-1)
        qc_replay_np = qc_replay.detach().cpu().numpy().reshape(-1)
        cost_np = (cost_t.detach().cpu().numpy().reshape(-1) > 0).astype(np.float64)
        grad_norm_np = grad_norm.detach().cpu().numpy().reshape(-1)
        finite_grad = np.isfinite(grad_qc.detach().cpu().numpy()).all(axis=1)
        intervention_np = intervention.detach().cpu().numpy().reshape(-1).astype(bool)
        cos_qc_np = cos_qc_filter_all.detach().cpu().numpy().reshape(-1)
        cos_v_np = cos_v_filter_all.detach().cpu().numpy().reshape(-1)

        record.update(
            {
                "status": "ok",
                "batch_size": int(batch),
                "intervention_frac": float(intervention_np.mean()),
                "projection_distance_mean": float(np.mean(d_f_np)),
                "projection_distance_max": float(np.max(d_f_np)),
                "qc_raw_mean": float(np.mean(qc_np)),
                "qc_raw_std": float(np.std(qc_np)),
                "corr_QC_filter": pearson(qc_np, d_f_np),
                "corr_QC_cost": pearson(qc_replay_np, cost_np),
                "grad_norm_mean": float(np.mean(grad_norm_np)),
                "grad_norm_median": float(np.median(grad_norm_np)),
                "zero_grad_frac": float(np.mean(grad_norm_np <= eps_grad)),
                "grad_nan_inf_frac": float(np.mean(~finite_grad)),
                "cost_event_frac": float(np.mean(cost_np)),
                "jvp_magnitude_mean": float(jvp_mag.detach().mean().item()),
            }
        )
        if intervention_np.any():
            record["cos_QC_filter"] = float(np.mean(cos_qc_np[intervention_np]))
            record["cos_v_filter"] = float(np.mean(cos_v_np[intervention_np]))
        else:
            record["cos_QC_filter"] = None
            record["cos_v_filter"] = None
    finally:
        agent.restore_requires_grad(agent.safety_critic, safety_flags)
    return record


def run_diagnostic(run_name: str, args: argparse.Namespace) -> None:
    from envs.safety_gym_wrapper import make_safe_env

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    torch.set_float32_matmul_precision("high")

    cfg = build_config(run_name, args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    env = make_safe_env(cfg.task, train=True)
    eval_env = make_safe_env(cfg.task, train=False)
    env.action_space.seed(cfg.seed)
    eval_env.action_space.seed(cfg.seed)

    agent = flowAC(env.observation_space.shape[0], env.action_space, cfg)
    memory = SafeReplayMemory(cfg.replay_size, cfg.seed)
    jsonl_path = LOG_DIR / f"{run_name}_seed{cfg.seed}.jsonl"
    train_log_path = LOG_DIR / f"{run_name}_seed{cfg.seed}.log"

    total_numsteps = 0
    updates = 0
    with train_log_path.open("w", encoding="utf-8") as train_log:
        train_log.write(
            f"===== {dt.datetime.now():%F %T} START {run_name} "
            f"task={cfg.task} seed={cfg.seed} =====\n"
        )
        for episode in range(1, 10**9):
            state = reset_env(env, seed=cfg.seed if episode == 1 else None)
            agent.observe(state)
            done = False
            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
            while not done:
                if cfg.start_steps > total_numsteps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, total_numsteps=total_numsteps)

                if cfg.start_steps <= total_numsteps and len(memory) >= cfg.batch_size:
                    for _ in range(cfg.updates_per_step):
                        agent.update_parameters(memory, cfg.batch_size, updates, total_numsteps)
                        updates += 1

                next_state, reward, cost, terminated, truncated, _ = step_env(env, action)
                done = terminated or truncated
                agent.observe(next_state)
                mask = 0.0 if terminated else 1.0
                memory.push(state, action, reward, cost, next_state, mask)
                state = next_state
                total_numsteps += 1
                episode_steps += 1
                episode_reward += float(reward)
                episode_cost += float(cost)

                if total_numsteps % cfg.eval_numsteps == 0:
                    eval_reward, eval_cost = evaluate(agent, eval_env, cfg)
                    rec = diagnostic_snapshot(
                        agent,
                        memory,
                        cfg,
                        total_numsteps=total_numsteps,
                        run_name=run_name,
                        eval_reward=eval_reward,
                        eval_cost=eval_cost,
                        batch_size=args.diag_batch_size,
                        eps_filter=args.eps_filter,
                        eps_grad=args.eps_grad,
                    )
                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, sort_keys=True) + "\n")
                    train_log.write(
                        "DIAG step={} eval_reward={:.3f} eval_cost={:.3f} "
                        "intervention_frac={} corr_QC_filter={} corr_QC_cost={} "
                        "cos_QC_filter={} zero_grad_frac={}\n".format(
                            total_numsteps,
                            eval_reward,
                            eval_cost,
                            fmt(ensure_float(rec.get("intervention_frac"))),
                            fmt(ensure_float(rec.get("corr_QC_filter"))),
                            fmt(ensure_float(rec.get("corr_QC_cost"))),
                            fmt(ensure_float(rec.get("cos_QC_filter"))),
                            fmt(ensure_float(rec.get("zero_grad_frac"))),
                        )
                    )
                    train_log.flush()

                if total_numsteps >= cfg.num_steps:
                    break

            if episode % 10 == 0:
                train_log.write(
                    f"Episode: {episode}, total numsteps: {total_numsteps}, "
                    f"episode steps: {episode_steps}, reward: {episode_reward:.2f}, "
                    f"cost: {episode_cost:.2f}\n"
                )
                train_log.flush()
            if total_numsteps >= cfg.num_steps:
                break

        train_log.write(f"===== {dt.datetime.now():%F %T} END {run_name} seed={cfg.seed} =====\n")
    env.close()
    eval_env.close()
    write_reports()


def load_latest_records() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(LOG_DIR.glob("DIAG_*_seed*.jsonl")):
        latest = None
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                latest = json.loads(line)
        if latest is not None:
            latest["path"] = str(path)
            rows.append(latest)
    return rows


def decide(row: dict[str, Any]) -> str:
    if not row.get("hard_filter_available", False):
        corr_cost = ensure_float(row.get("corr_QC_cost"))
        if corr_cost is not None and corr_cost <= 0.10:
            return "hard_filter_unavailable_and_critic_not_calibrated"
        return "hard_filter_unavailable"
    corr_filter = ensure_float(row.get("corr_QC_filter"))
    corr_cost = ensure_float(row.get("corr_QC_cost"))
    cos_filter = ensure_float(row.get("cos_QC_filter"))
    zero_frac = ensure_float(row.get("zero_grad_frac"))
    if corr_filter is not None and corr_cost is not None and corr_cost <= 0.10 and corr_filter <= 0.10:
        return "critic_not_calibrated"
    if (cos_filter is not None and cos_filter < 0.0) or (cos_filter is not None and cos_filter <= 0.05):
        return "wrong_safety_normal"
    if corr_filter is not None and corr_filter <= 0.10:
        return "wrong_safety_normal"
    if zero_frac is not None and zero_frac > 0.50:
        return "wrong_safety_normal"
    if (
        corr_filter is not None
        and corr_cost is not None
        and cos_filter is not None
        and zero_frac is not None
        and corr_filter >= 0.25
        and corr_cost >= 0.20
        and cos_filter >= 0.20
        and zero_frac <= 0.30
    ):
        return "healthy_safety_geometry"
    return "inconclusive"


def write_reports() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_latest_records()
    for row in rows:
        row["decision"] = decide(row)

    lines = [
        "# Safety Geometry Diagnostic Summary",
        "",
        "Note: current Safety-Gym wrapper exposes no hard safety-filter correction. "
        "The filter fields use an action-bound clipping proxy and are marked as "
        "`hard_filter_available=False`.",
        "",
        "| Task | Config | Reward | Cost | intervention_frac | corr_QC_filter | corr_QC_cost | grad_norm | zero_grad_frac | cos_QC_filter | cos_v_filter | Decision |",
        "| ---- | ------ | -----: | ---: | ----------------: | -------------: | -----------: | --------: | -------------: | ------------: | -----------: | -------- |",
    ]
    for row in rows:
        lines.append(
            "| {task} | {config} | {reward} | {cost} | {intervention} | {corr_filter} | "
            "{corr_cost} | {grad_norm} | {zero_grad} | {cos_qc} | {cos_v} | {decision} |".format(
                task=row.get("task", "n/a"),
                config=row.get("config", row.get("run", "n/a")),
                reward=fmt(ensure_float(row.get("eval_reward"))),
                cost=fmt(ensure_float(row.get("eval_cost"))),
                intervention=fmt(ensure_float(row.get("intervention_frac"))),
                corr_filter=fmt(ensure_float(row.get("corr_QC_filter"))),
                corr_cost=fmt(ensure_float(row.get("corr_QC_cost"))),
                grad_norm=fmt(ensure_float(row.get("grad_norm_mean"))),
                zero_grad=fmt(ensure_float(row.get("zero_grad_frac"))),
                cos_qc=fmt(ensure_float(row.get("cos_QC_filter"))),
                cos_v=fmt(ensure_float(row.get("cos_v_filter"))),
                decision=row.get("decision", "n/a"),
            )
        )

    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no_runs |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `hard_filter_available=False`, filter-grounded JVP cannot be validated from the current environment wrapper.",
            "- If `corr_QC_cost` is also low, the next branch should be risk-CDF critic rather than more soft-gate/lambda tuning.",
            "- If filter correction is required, the environment wrapper must expose or implement an actual safety filter first.",
        ]
    )
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Safety Geometry Diagnostic Decision Log",
        "",
        f"timestamp: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for row in rows:
        decision_lines.extend(
            [
                f"## {row.get('run')}",
                f"- task: {row.get('task')}",
                f"- config: {row.get('config')}",
                f"- step: {row.get('step')}",
                f"- hard_filter_available: {row.get('hard_filter_available')}",
                f"- filter_source: {row.get('filter_source')}",
                f"- intervention_frac: {fmt(ensure_float(row.get('intervention_frac')))}",
                f"- corr_QC_filter: {fmt(ensure_float(row.get('corr_QC_filter')))}",
                f"- corr_QC_cost: {fmt(ensure_float(row.get('corr_QC_cost')))}",
                f"- cos_QC_filter: {fmt(ensure_float(row.get('cos_QC_filter')))}",
                f"- zero_grad_frac: {fmt(ensure_float(row.get('zero_grad_frac')))}",
                f"- decision: {row.get('decision')}",
                "",
            ]
        )
    if not rows:
        decision_lines.append("- decision: no diagnostic runs collected yet")
    (REPORT_DIR / "decision_log.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="default", help="Run name, default, all, or report-only")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--eval-numsteps", type=int, default=5000)
    parser.add_argument("--eval-times", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--diag-batch-size", type=int, default=1024)
    parser.add_argument("--eps-filter", type=float, default=1e-6)
    parser.add_argument("--eps-grad", type=float, default=1e-6)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run == "report-only":
        write_reports()
        return
    if args.run == "default":
        run_names = list(DEFAULT_RUNS)
    elif args.run == "all":
        run_names = list(DEFAULT_RUNS)
        if args.include_optional:
            run_names.append("DIAG_CG1_C4_long")
    else:
        if args.run not in RUNS:
            raise SystemExit(f"Unknown run {args.run}. Choices: {', '.join(RUNS)}")
        run_names = [args.run]
    if len(run_names) > 4:
        raise SystemExit("Refusing to run more than 4 diagnostic runs")
    for run_name in run_names:
        run_diagnostic(run_name, args)


if __name__ == "__main__":
    main()
