#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.star_agent import STARAgent
from main_star import make_env, reset_env, step_env
from utilis.star_default_config import star_default_config


STATE_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "checkpoint_path",
    "checkpoint_name",
    "eval_seed",
    "state_index",
    "mean_action_risk",
    "shadow_q_min",
    "shadow_q_mean",
    "shadow_q_max",
    "rho_shadow",
    "audit_gap",
    "audit_gap_positive",
    "shadow_action_spread",
    "kl_current_reference",
]

SUMMARY_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "checkpoint_path",
    "checkpoint_name",
    "states",
    "mean_action_risk_mean",
    "shadow_q_max_mean",
    "rho_shadow_mean",
    "audit_gap_mean",
    "audit_gap_positive_rate",
    "shadow_action_spread_mean",
    "kl_current_reference_mean",
    "mean_action_risk_p01",
    "mean_action_risk_p05",
    "mean_action_risk_p10",
    "mean_action_risk_p25",
    "mean_action_risk_p50",
    "mean_action_risk_p75",
    "mean_action_risk_p90",
    "mean_action_risk_p95",
    "mean_action_risk_p99",
    "shadow_q_max_p01",
    "shadow_q_max_p05",
    "shadow_q_max_p10",
    "shadow_q_max_p25",
    "shadow_q_max_p50",
    "shadow_q_max_p75",
    "shadow_q_max_p90",
    "shadow_q_max_p95",
    "shadow_q_max_p99",
    "rho_shadow_p01",
    "rho_shadow_p05",
    "rho_shadow_p10",
    "rho_shadow_p25",
    "rho_shadow_p50",
    "rho_shadow_p75",
    "rho_shadow_p90",
    "rho_shadow_p95",
    "rho_shadow_p99",
    "audit_gap_p01",
    "audit_gap_p05",
    "audit_gap_p10",
    "audit_gap_p25",
    "audit_gap_p50",
    "audit_gap_p75",
    "audit_gap_p90",
    "audit_gap_p95",
    "audit_gap_p99",
]

GRID_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "checkpoint_path",
    "checkpoint_name",
    "threshold",
    "mean_action_unsafe_rate",
    "pSVR",
    "rho_active_rate",
    "hidden_unsafe_rate",
    "audit_gap_positive_rate",
]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_config(metadata: dict):
    config = star_default_config.copy()
    config.update(metadata)
    config.cuda = bool(config.cuda and torch.cuda.is_available())
    return config


def append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def final_checkpoint(run_dir: Path) -> Path | None:
    path = run_dir / "checkpoint" / "final.torch"
    return path if path.exists() else None


def quantiles(values: list[float], prefix: str) -> dict:
    if not values:
        return {f"{prefix}_p{q:02d}": math.nan for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_p01": float(np.quantile(arr, 0.01)),
        f"{prefix}_p05": float(np.quantile(arr, 0.05)),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p25": float(np.quantile(arr, 0.25)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p75": float(np.quantile(arr, 0.75)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_p99": float(np.quantile(arr, 0.99)),
    }


def collect_states(agent: STARAgent, env, config, seeds: list[int], max_states: int) -> list[tuple[np.ndarray, int]]:
    states: list[tuple[np.ndarray, int]] = []
    per_seed = max(1, math.ceil(max_states / max(1, len(seeds))))
    for seed in seeds:
        state = reset_env(env, seed=seed)
        done = False
        steps = 0
        while not done and steps < int(config.eval_numsteps) and len(states) < max_states:
            states.append((np.asarray(state, dtype=np.float32).copy(), seed))
            action = agent.select_action(state, evaluate=True, execution_mode="raw", total_numsteps=int(config.num_steps), diagnostics=False)
            next_state, _reward, _cost, terminated, truncated, _info = step_env(env, action, config.safe_env)
            done = terminated or truncated
            state = next_state
            steps += 1
            if steps >= per_seed and len(states) >= max_states:
                break
        if len(states) >= max_states:
            break
    return states[:max_states]


@torch.no_grad()
def diagnose_state_batch(agent: STARAgent, states: np.ndarray) -> dict[str, np.ndarray]:
    device = agent.device
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    norm_state = agent.normalize_state(state_tensor)
    _action, _logp, mean_action = agent.policy.sample(norm_state)
    mean_q = agent._cost_plus(norm_state, mean_action).view(-1)
    shadow = agent.audit.generate_shadow_actions(agent.policy, agent.reference_policy, norm_state)
    q_shadow = agent.audit.conservative_cost(agent.cost_critic, norm_state, shadow.actions)
    rho = agent.audit.shadow_risk(q_shadow)
    kl = agent.policy.analytic_kl_to(agent.reference_policy, norm_state)
    q_max = q_shadow.max(dim=1).values
    return {
        "mean_action_risk": mean_q.detach().cpu().numpy(),
        "shadow_q_min": q_shadow.min(dim=1).values.detach().cpu().numpy(),
        "shadow_q_mean": q_shadow.mean(dim=1).detach().cpu().numpy(),
        "shadow_q_max": q_max.detach().cpu().numpy(),
        "rho_shadow": rho.detach().cpu().numpy(),
        "audit_gap": (q_max - mean_q).detach().cpu().numpy(),
        "shadow_action_spread": shadow.spread.view(-1).detach().cpu().numpy(),
        "kl_current_reference": kl.detach().cpu().numpy(),
    }


def diagnose_run(run_dir: Path, seeds: list[int], max_states: int) -> tuple[list[dict], dict, list[dict]]:
    metadata = read_metadata(run_dir)
    if not metadata:
        return [], {}, []
    checkpoint = final_checkpoint(run_dir)
    if checkpoint is None:
        return [], {}, []
    config = build_config(metadata)
    env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    try:
        agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
        agent.load_checkpoint(str(checkpoint))
        states_with_seed = collect_states(agent, env, config, seeds, max_states)
        if not states_with_seed:
            return [], {}, []
        states = np.stack([item[0] for item in states_with_seed], axis=0)
        eval_seeds = [item[1] for item in states_with_seed]
        arrays = diagnose_state_batch(agent, states)
    finally:
        env.close()

    base = {
        "run_name": metadata.get("run_name", run_dir.name),
        "task": metadata.get("task", ""),
        "method": metadata.get("method", ""),
        "seed": metadata.get("seed", ""),
        "checkpoint_path": str(checkpoint),
        "checkpoint_name": checkpoint.stem,
    }
    state_rows = []
    for i in range(states.shape[0]):
        row = dict(base)
        row.update({
            "eval_seed": eval_seeds[i],
            "state_index": i,
            "mean_action_risk": float(arrays["mean_action_risk"][i]),
            "shadow_q_min": float(arrays["shadow_q_min"][i]),
            "shadow_q_mean": float(arrays["shadow_q_mean"][i]),
            "shadow_q_max": float(arrays["shadow_q_max"][i]),
            "rho_shadow": float(arrays["rho_shadow"][i]),
            "audit_gap": float(arrays["audit_gap"][i]),
            "audit_gap_positive": float(arrays["audit_gap"][i] > 0),
            "shadow_action_spread": float(arrays["shadow_action_spread"][i]),
            "kl_current_reference": float(arrays["kl_current_reference"][i]),
        })
        state_rows.append(row)

    summary = dict(base)
    summary["states"] = len(state_rows)
    for key in ("mean_action_risk", "shadow_q_max", "rho_shadow", "audit_gap", "shadow_action_spread", "kl_current_reference"):
        vals = [float(v) for v in arrays[key]]
        summary[f"{key}_mean"] = mean(vals) if vals else math.nan
    summary["audit_gap_positive_rate"] = mean([float(v > 0) for v in arrays["audit_gap"]])
    summary.update(quantiles([float(v) for v in arrays["mean_action_risk"]], "mean_action_risk"))
    summary.update(quantiles([float(v) for v in arrays["shadow_q_max"]], "shadow_q_max"))
    summary.update(quantiles([float(v) for v in arrays["rho_shadow"]], "rho_shadow"))
    summary.update(quantiles([float(v) for v in arrays["audit_gap"]], "audit_gap"))

    grid_rows = []
    for threshold in [round(x * 0.05, 2) for x in range(1, 20)]:
        row = dict(base)
        row["threshold"] = threshold
        row["mean_action_unsafe_rate"] = mean([float(v > threshold) for v in arrays["mean_action_risk"]])
        row["pSVR"] = mean([float(v > threshold) for v in arrays["shadow_q_max"]])
        row["rho_active_rate"] = mean([float(v > threshold) for v in arrays["rho_shadow"]])
        row["hidden_unsafe_rate"] = mean([
            float(mean_v <= threshold and max_v > threshold)
            for mean_v, max_v in zip(arrays["mean_action_risk"], arrays["shadow_q_max"])
        ])
        row["audit_gap_positive_rate"] = summary["audit_gap_positive_rate"]
        grid_rows.append(row)
    return state_rows, summary, grid_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/star_tune_actor_stage1_3393af8"))
    parser.add_argument("--eval-seeds", default="200000,200001,200002,200003,200004")
    parser.add_argument("--max-states-per-run", type=int, default=1000)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/star_goal"))
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.eval_seeds.split(",") if item.strip()]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        args.report_dir / "shadow_risk_states.csv",
        args.report_dir / "shadow_risk_summary.csv",
        args.report_dir / "shadow_threshold_grid.csv",
    ):
        if path.exists():
            path.unlink()

    runs = 0
    state_total = 0
    for meta_path in sorted(args.root.rglob("run_metadata.json")):
        state_rows, summary, grid_rows = diagnose_run(meta_path.parent, seeds, args.max_states_per_run)
        if not state_rows:
            continue
        append_csv(args.report_dir / "shadow_risk_states.csv", state_rows, STATE_FIELDS)
        append_csv(args.report_dir / "shadow_risk_summary.csv", [summary], SUMMARY_FIELDS)
        append_csv(args.report_dir / "shadow_threshold_grid.csv", grid_rows, GRID_FIELDS)
        runs += 1
        state_total += len(state_rows)
    print(f"Diagnosed {runs} checkpoints, states={state_total}, report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
