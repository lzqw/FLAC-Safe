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
from diagnostics.shadow_oracle import ShadowOracleAccumulator, shadow_oracle_step, snapshot_supported
from main_star import make_env, reset_env, step_env
from utilis.star_default_config import star_default_config


ROW_FIELDS = [
    "run_name",
    "task",
    "train_seed",
    "checkpoint_path",
    "eval_seed",
    "sample_index",
    "horizon",
    "evaluation_only",
    "supported",
    "unsupported_reason",
    "mean_predicted_risk",
    "executed_predicted_risk",
    "shadow_predicted_risk",
    "mean_actual_cost",
    "executed_actual_cost",
    "shadow_actual_cost",
    "mean_violation",
    "executed_violation",
    "shadow_violation",
    "shadow_predicted_unsafe",
    "executed_predicted_unsafe",
    "unsafe_found_but_not_deployed",
]

SUMMARY_FIELDS = [
    "run_name",
    "task",
    "train_seed",
    "checkpoint_path",
    "horizon",
    "supported",
    "unsupported_reason",
    "oracle_samples",
    "oracle_shadow_violation_rate",
    "oracle_mean_violation_rate",
    "oracle_executed_violation_rate",
    "shadow_minus_mean_violation",
    "shadow_risk_precision",
    "shadow_risk_recall",
    "shadow_risk_AUROC",
    "predicted_actual_cost_correlation",
    "unsafe_found_but_not_deployed_rate",
]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_meta(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_config(meta: dict):
    config = star_default_config.copy()
    config.update(meta)
    config.cuda = bool(config.cuda and torch.cuda.is_available())
    config.star_exec = True
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


def advance_raw(agent: STARAgent, env, state, config):
    action = agent.select_action(state, evaluate=True, execution_mode="raw", total_numsteps=int(config.num_steps), diagnostics=False)
    next_state, _reward, _cost, terminated, truncated, _info = step_env(env, action, config.safe_env)
    return next_state, bool(terminated or truncated)


def run_checkpoint(run_dir: Path, report_dir: Path, eval_seeds: list[int], horizons: list[int], max_states: int) -> None:
    meta = read_meta(run_dir)
    if meta.get("method") != "star_actor":
        return
    checkpoint = final_checkpoint(run_dir)
    if checkpoint is None:
        return
    config = build_config(meta)
    env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    supported, reason = snapshot_supported(env)
    run_name = meta.get("run_name", run_dir.name)
    base = {
        "run_name": run_name,
        "task": config.task,
        "train_seed": meta.get("seed", ""),
        "checkpoint_path": str(checkpoint),
    }
    if not supported:
        rows = []
        for horizon in horizons:
            row = dict(base)
            row.update({
                "horizon": horizon,
                "supported": False,
                "unsupported_reason": reason,
                "oracle_samples": 0,
            })
            rows.append(row)
        append_csv(report_dir / "oracle" / "oracle_summary.csv", rows, SUMMARY_FIELDS)
        env.close()
        return
    try:
        agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
        agent.load_checkpoint(str(checkpoint))
        # Force candidate executor for STAR-Actor checkpoints only during
        # evaluation-only oracle probing.
        agent.method = "star"
        agent.star_exec = True
        per_horizon_rows = {h: [] for h in horizons}
        sample_index = 0
        for seed in eval_seeds:
            state = reset_env(env, seed=seed)
            done = False
            steps = 0
            while not done and steps < int(config.eval_numsteps) and sample_index < max_states:
                for horizon in horizons:
                    result = shadow_oracle_step(
                        agent=agent,
                        env=env,
                        state=state,
                        total_numsteps=int(config.num_steps),
                        horizon=horizon,
                        threshold=float(config.star_risk_threshold),
                    )
                    row = dict(base)
                    row.update({
                        "eval_seed": seed,
                        "sample_index": sample_index,
                    })
                    row.update(result)
                    per_horizon_rows[horizon].append(row)
                state, done = advance_raw(agent, env, state, config)
                steps += 1
                sample_index += 1
                if sample_index >= max_states:
                    break
            if sample_index >= max_states:
                break
        all_rows = [row for rows in per_horizon_rows.values() for row in rows]
        append_csv(report_dir / "oracle" / "oracle_rows.csv", all_rows, ROW_FIELDS)
        summaries = []
        for horizon, rows in per_horizon_rows.items():
            acc = ShadowOracleAccumulator(float(config.star_risk_threshold))
            for row in rows:
                acc.add(row)
            summary = acc.summary()
            mean_v = [float(row["mean_violation"]) for row in rows if row.get("supported")]
            shadow_v = [float(row["shadow_violation"]) for row in rows if row.get("supported")]
            summary_row = dict(base)
            summary_row.update({
                "horizon": horizon,
                "supported": bool(rows),
                "unsupported_reason": "",
                "oracle_mean_violation_rate": mean(mean_v) if mean_v else math.nan,
                "shadow_minus_mean_violation": (mean(shadow_v) - mean(mean_v)) if shadow_v and mean_v else math.nan,
                "predicted_actual_cost_correlation": summary.get("predicted_vs_actual_shadow_cost", math.nan),
            })
            summary_row.update(summary)
            summaries.append(summary_row)
        append_csv(report_dir / "oracle" / "oracle_summary.csv", summaries, SUMMARY_FIELDS)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-seeds", default="200000,200001,200002,200003,200004")
    parser.add_argument("--horizons", default="1,5")
    parser.add_argument("--max-states-per-run", type=int, default=200)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/star_goal"))
    args = parser.parse_args()
    out_dir = args.report_dir / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (out_dir / "oracle_rows.csv", out_dir / "oracle_summary.csv"):
        if path.exists():
            path.unlink()
    for meta_path in sorted(args.root.rglob("run_metadata.json")):
        run_checkpoint(
            meta_path.parent,
            args.report_dir,
            parse_seed_list(args.eval_seeds),
            parse_int_list(args.horizons),
            args.max_states_per_run,
        )
    print(f"wrote {out_dir / 'oracle_rows.csv'}")
    print(f"wrote {out_dir / 'oracle_summary.csv'}")


if __name__ == "__main__":
    main()
