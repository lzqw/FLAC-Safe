#!/usr/bin/env python3
"""Diagnose safety-Q action-gradient geometry with finite differences."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
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


LOG_DIR = ROOT / "logs" / "safety_q_geometry_diag"
REPORT_DIR = ROOT / "reports" / "safety_q_geometry_diag"

RUNS: dict[str, dict[str, Any]] = {
    "SQ_DIAG_PG1_G4": {
        "task": "SafetyPointGoal1-v0",
        "config": "G4_fixed_main",
        "seed": 0,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
    },
    "SQ_DIAG_CG1_G4": {
        "task": "SafetyCarGoal1-v0",
        "config": "G4_fixed_main",
        "seed": 0,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
    },
    "SQ_DIAG_CG1_C2": {
        "task": "SafetyCarGoal1-v0",
        "config": "CG1_C2_safe05",
        "seed": 0,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
    },
}
DEFAULT_RUNS = ("SQ_DIAG_PG1_G4", "SQ_DIAG_CG1_G4", "SQ_DIAG_CG1_C2")


def reset_env(env, seed: int | None = None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    next_state, reward, cost, terminated, truncated, info = env.step(action)
    return next_state, float(reward), float(cost), terminated, truncated, info


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value_f):
        return "n/a"
    return f"{value_f:.3f}"


def build_config(run_name: str, args: argparse.Namespace):
    spec = RUNS[run_name]
    cfg = default_config.copy()
    cfg.update(
        {
            "task": spec["task"],
            "safe_env": True,
            "safe_policy_loss": True,
            "safety_critic_mode": "cdf",
            "qc_geom_mode": "mean",
            "safe_threshold": 0.05,
            "lambda_safe": spec["lambda_safe"],
            "lambda_jvp": spec["lambda_jvp"],
            "safe_bandwidth": 0.05,
            "normalize_jvp": True,
            "jvp_norm_mode": "exact",
            "jvp_mode": "grad",
            "cdf_binarize_cost": True,
            "cdf_target_clip": True,
            "soft_feasibility_gate": False,
            "soft_normal_masking": False,
            "directional_ref_noise": False,
            "epsilon": 0.0,
            "batch_size": args.batch_size,
            "updates_per_step": args.updates_per_step,
            "hidden_size": args.hidden_size,
            "num_steps": args.num_steps,
            "start_steps": args.start_steps,
            "eval": True,
            "eval_numsteps": args.eval_numsteps,
            "eval_times": args.eval_times,
            "distributional_critic": False,
            "compile_model": False,
            "save": False,
            "steps": 1,
            "seed": spec["seed"],
            "cuda": not args.cpu,
            "device": args.device,
            "algo": "SafetyQGeometryDiag",
            "tag": run_name,
            "diagnose_safety_q_geometry": True,
            "safety_q_fd_eps": args.safety_q_fd_eps,
            "safety_q_boundary_width": args.safety_q_boundary_width,
            "high_fidelity_safety_q": args.high_fidelity_safety_q,
            "safety_q_priority": args.safety_q_priority,
            "safety_q_cost_weight": args.safety_q_cost_weight,
            "safety_q_boundary_weight": args.safety_q_boundary_weight,
            "safety_q_td_weight": args.safety_q_td_weight,
            "safety_q_max_weight": args.safety_q_max_weight,
            "safety_q_extra_updates": args.safety_q_extra_updates,
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
            total_reward += reward
            total_cost += cost
            state = next_state
    return total_reward / cfg.eval_times, total_cost / cfg.eval_times


def diagnostic_snapshot(agent: flowAC, memory: SafeReplayMemory, cfg, run_name: str, step: int, reward: float, cost: float, batch_size: int):
    record: dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run": run_name,
        "task": cfg.task,
        "config": RUNS[run_name]["config"],
        "seed": int(cfg.seed),
        "step": int(step),
        "eval_reward": float(reward),
        "eval_cost": float(cost),
    }
    batch = min(batch_size, len(memory))
    if batch < 8:
        record["status"] = "insufficient_replay"
        return record

    state, action, _, _, _, _ = memory.sample(batch_size=batch)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=agent.device)
    state_t = agent._normalize_obs(state_t)
    record.update(agent.safety_q_geometry_diagnostics(state_t, action_t, state_t))
    record["status"] = "ok"
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
        train_log.write(f"===== {dt.datetime.now():%F %T} START {run_name} task={cfg.task} seed={cfg.seed} =====\n")
        for episode in range(1, 10**9):
            state = reset_env(env, seed=cfg.seed if episode == 1 else None)
            agent.observe(state)
            done = False
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
                memory.push(state, action, reward, cost, next_state, 0.0 if terminated else 1.0)
                state = next_state
                total_numsteps += 1

                if total_numsteps % cfg.eval_numsteps == 0:
                    eval_reward, eval_cost = evaluate(agent, eval_env, cfg)
                    rec = diagnostic_snapshot(agent, memory, cfg, run_name, total_numsteps, eval_reward, eval_cost, args.diag_batch_size)
                    with jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(rec, sort_keys=True) + "\n")
                    train_log.write(
                        "DIAG step={} eval_reward={:.3f} eval_cost={:.3f} grad_norm={} "
                        "zero_grad_frac={} mono_plus={} mono_minus={} fd_slope={}\n".format(
                            total_numsteps,
                            eval_reward,
                            eval_cost,
                            fmt(rec.get("safety_q/grad_norm_mean")),
                            fmt(rec.get("safety_q/zero_grad_frac")),
                            fmt(rec.get("safety_q/mono_plus_frac")),
                            fmt(rec.get("safety_q/mono_minus_frac")),
                            fmt(rec.get("safety_q/fd_slope_mean")),
                        )
                    )
                    train_log.flush()

                if total_numsteps >= cfg.num_steps:
                    break
            if total_numsteps >= cfg.num_steps:
                break
        train_log.write(f"===== {dt.datetime.now():%F %T} END {run_name} seed={cfg.seed} =====\n")
    env.close()
    eval_env.close()
    write_reports()


def load_latest_records() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(LOG_DIR.glob("SQ_DIAG_*_seed*.jsonl")):
        latest = None
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    latest = json.loads(line)
        if latest is not None:
            latest["path"] = str(path)
            rows.append(latest)
    return rows


def decide(row: dict[str, Any]) -> str:
    grad_norm = row.get("safety_q/grad_norm_mean")
    zero_grad = row.get("safety_q/zero_grad_frac")
    mono_plus = row.get("safety_q/mono_plus_frac")
    mono_minus = row.get("safety_q/mono_minus_frac")
    fd_slope = row.get("safety_q/fd_slope_mean")
    if any(value is None for value in (grad_norm, zero_grad, mono_plus, mono_minus, fd_slope)):
        return "no_data"
    grad_norm = float(grad_norm)
    zero_grad = float(zero_grad)
    mono_plus = float(mono_plus)
    mono_minus = float(mono_minus)
    fd_slope = float(fd_slope)
    if zero_grad > 0.50 or mono_plus <= 0.50 or fd_slope <= 0.0 or grad_norm <= 1e-8:
        return "bad_q_geometry"
    if grad_norm > 1e-4 and zero_grad < 0.30 and mono_plus > 0.60 and mono_minus > 0.60 and fd_slope > 0.0:
        return "healthy_q_geometry"
    return "inconclusive"


def write_reports() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_latest_records()
    for row in rows:
        row["decision"] = decide(row)

    lines = [
        "# Safety-Q Geometry Diagnostic Summary",
        "",
        "| Task | Config | Reward | Cost | q_mean | q_std | grad_norm | zero_grad_frac | mono_plus_frac | mono_minus_frac | fd_slope_mean | boundary_frac | jvp_mean | Decision |",
        "| ---- | ------ | -----: | ---: | -----: | ----: | --------: | -------------: | -------------: | --------------: | ------------: | ------------: | -------: | -------- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('task')} | {row.get('config')} | {fmt(row.get('eval_reward'))} | {fmt(row.get('eval_cost'))} | "
            f"{fmt(row.get('safety_q/q_mean'))} | {fmt(row.get('safety_q/q_std'))} | "
            f"{fmt(row.get('safety_q/grad_norm_mean'))} | {fmt(row.get('safety_q/zero_grad_frac'))} | "
            f"{fmt(row.get('safety_q/mono_plus_frac'))} | {fmt(row.get('safety_q/mono_minus_frac'))} | "
            f"{fmt(row.get('safety_q/fd_slope_mean'))} | {fmt(row.get('safety_q/boundary_frac'))} | "
            f"{fmt(row.get('safety_q/jvp_mean'))} | {row.get('decision')} |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no_runs |")
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Safety-Q Geometry Diagnostic Decision Log",
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
                f"- grad_norm_mean: {fmt(row.get('safety_q/grad_norm_mean'))}",
                f"- zero_grad_frac: {fmt(row.get('safety_q/zero_grad_frac'))}",
                f"- mono_plus_frac: {fmt(row.get('safety_q/mono_plus_frac'))}",
                f"- mono_minus_frac: {fmt(row.get('safety_q/mono_minus_frac'))}",
                f"- fd_slope_mean: {fmt(row.get('safety_q/fd_slope_mean'))}",
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
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--eval-numsteps", type=int, default=5000)
    parser.add_argument("--eval-times", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--diag-batch-size", type=int, default=1024)
    parser.add_argument("--safety-q-fd-eps", type=float, default=0.01)
    parser.add_argument("--safety-q-boundary-width", type=float, default=0.05)
    parser.add_argument("--high-fidelity-safety-q", action="store_true")
    parser.add_argument("--safety-q-priority", action="store_true")
    parser.add_argument("--safety-q-cost-weight", type=float, default=2.0)
    parser.add_argument("--safety-q-boundary-weight", type=float, default=3.0)
    parser.add_argument("--safety-q-td-weight", type=float, default=0.0)
    parser.add_argument("--safety-q-max-weight", type=float, default=5.0)
    parser.add_argument("--safety-q-extra-updates", type=int, default=0)
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
        run_names = list(RUNS)
    else:
        if args.run not in RUNS:
            raise SystemExit(f"Unknown run {args.run}. Choices: {', '.join(RUNS)}")
        run_names = [args.run]
    for run_name in run_names:
        run_diagnostic(run_name, args)


if __name__ == "__main__":
    main()
