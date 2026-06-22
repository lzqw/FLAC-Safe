#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


PERFORMANCE_COLUMNS = [
    "raw_return",
    "filtered_return",
    "raw_episode_cost",
    "filtered_episode_cost",
    "raw_EVR",
    "filtered_EVR",
    "success_rate",
    "constraint_satisfaction_rate",
]

AUDIT_COLUMNS = [
    "pSVR",
    "any_unsafe_shadow_rate",
    "hidden_unsafe_rate",
    "shadow_risk_mean",
    "shadow_risk_max_mean",
    "actor_mean_action_risk",
    "kl_mean",
    "candidate_spread",
]

EXECUTION_COLUMNS = [
    "safe_candidate_fraction",
    "fallback_rate",
    "executed_predicted_risk",
    "found_but_not_executed_rate",
    "boundary_safe_coverage",
    "boundary_safety_rate",
]

EFFICIENCY_COLUMNS = [
    "wall_clock_time",
    "update_time",
    "env_steps_per_second",
    "gpu_memory_peak_mb",
    "cost_critic_forward_calls",
    "approx_star_overhead_vs_pointwise",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=math.nan) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def finite(values):
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def mean_ci(values, *, bootstrap: int = 2000, seed: int = 0) -> dict:
    vals = finite(values)
    if not vals:
        return {"mean": "", "std": "", "ci95_low": "", "ci95_high": "", "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "ci95_low": vals[0], "ci95_high": vals[0], "n": 1}
    rng = random.Random(seed)
    samples = []
    for _ in range(bootstrap):
        draw = [vals[rng.randrange(len(vals))] for _ in vals]
        samples.append(mean(draw))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {"mean": mean(vals), "std": pstdev(vals), "ci95_low": low, "ci95_high": high, "n": len(vals)}


def latest_by_step(rows: list[dict], step_key: str) -> list[dict]:
    if not rows:
        return []
    max_step = max(to_int(row.get(step_key)) for row in rows)
    return [row for row in rows if to_int(row.get(step_key)) == max_step]


def avg(rows: list[dict], key: str) -> float:
    vals = finite([to_float(row.get(key)) for row in rows])
    return mean(vals) if vals else math.nan


def load_runs(root: Path) -> list[dict]:
    runs = []
    for meta_path in sorted(root.rglob("run_metadata.json")):
        run_dir = meta_path.parent
        with meta_path.open() as f:
            meta = json.load(f)
        eval_rows = read_csv(run_dir / "eval_episodes.csv")
        train_rows = read_csv(run_dir / "train_episodes.csv")
        mechanism_rows = read_csv(run_dir / "mechanism.csv")
        efficiency_rows = read_csv(run_dir / "efficiency.csv")
        runs.append(
            {
                "run_dir": run_dir,
                "meta": meta,
                "eval": eval_rows,
                "train": train_rows,
                "mechanism": mechanism_rows,
                "efficiency": efficiency_rows,
            }
        )
    return runs


def final_eval(rows: list[dict], mode: str) -> list[dict]:
    mode_rows = [row for row in rows if row.get("mode") == mode]
    return latest_by_step(mode_rows, "global_step")


def final_mechanism(rows: list[dict]) -> list[dict]:
    return latest_by_step(rows, "step")


def final_efficiency(rows: list[dict]) -> list[dict]:
    return latest_by_step(rows, "step")


def run_summary(run: dict) -> dict:
    meta = run["meta"]
    raw = final_eval(run["eval"], "raw")
    filtered = final_eval(run["eval"], "star_exec")
    mech = final_mechanism(run["mechanism"])
    eff = final_efficiency(run["efficiency"])
    train = latest_by_step(run["train"], "end_step")
    return {
        "run_name": meta.get("run_name", run["run_dir"].name),
        "run_dir": str(run["run_dir"]),
        "task": meta.get("task", ""),
        "method": meta.get("method", ""),
        "seed": meta.get("seed", ""),
        "ablation_group": meta.get("ablation_group", "main"),
        "ablation_name": meta.get("ablation_name", meta.get("method", "")),
        "final_step": max([to_int(row.get("global_step")) for row in raw + filtered] + [to_int(row.get("end_step")) for row in train] + [0]),
        "raw_return": avg(raw, "episode_reward"),
        "filtered_return": avg(filtered, "episode_reward"),
        "raw_episode_cost": avg(raw, "episode_cost"),
        "filtered_episode_cost": avg(filtered, "episode_cost"),
        "raw_EVR": avg(raw, "violation_rate"),
        "filtered_EVR": avg(filtered, "violation_rate"),
        "success_rate": avg(filtered, "success"),
        "constraint_satisfaction_rate": avg(filtered, "constraint_satisfied"),
        "train_cost_rate": avg(train, "train_total_cost_rate"),
        "pSVR": avg(mech, "pSVR"),
        "any_unsafe_shadow_rate": avg(mech, "any_unsafe_shadow_rate"),
        "hidden_unsafe_rate": avg(mech, "hidden_unsafe_rate"),
        "shadow_risk_mean": avg(mech, "shadow_risk_mean"),
        "shadow_risk_max_mean": avg(mech, "shadow_risk_max_mean"),
        "actor_mean_action_risk": avg(mech, "actor_mean_action_risk"),
        "kl_mean": avg(mech, "kl_mean"),
        "candidate_spread": avg(mech, "candidate_spread"),
        "safe_candidate_fraction": avg(filtered, "safe_candidate_fraction"),
        "fallback_rate": avg(filtered, "execution_fallback_rate"),
        "executed_predicted_risk": avg(filtered, "executed_predicted_risk"),
        "found_but_not_executed_rate": avg(filtered, "found_but_not_executed_rate"),
        "boundary_safe_coverage": avg(filtered, "boundary_safe_coverage"),
        "boundary_safety_rate": avg(filtered, "boundary_safety_rate"),
        "wall_clock_time": avg(eff, "wall_clock_time"),
        "update_time": avg(eff, "update_time"),
        "env_steps_per_second": avg(eff, "env_steps_per_second"),
        "updates_per_second": avg(eff, "updates_per_second"),
        "gpu_memory_peak_mb": avg(eff, "gpu_memory_peak_mb"),
        "cost_critic_forward_calls": avg(eff, "cost_critic_forward_calls"),
    }


def grouped_ci(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for group_key, group_rows in sorted(groups.items()):
        base = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            stats = mean_ci([to_float(row.get(metric)) for row in group_rows])
            base[f"{metric}_mean"] = stats["mean"]
            base[f"{metric}_std"] = stats["std"]
            base[f"{metric}_ci95_low"] = stats["ci95_low"]
            base[f"{metric}_ci95_high"] = stats["ci95_high"]
            base[f"{metric}_n"] = stats["n"]
        out.append(base)
    return out


def build_learning_curves(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        meta = run["meta"]
        for row in run["eval"]:
            rows.append(
                {
                    "task": meta.get("task", ""),
                    "method": meta.get("method", ""),
                    "ablation_group": meta.get("ablation_group", ""),
                    "ablation_name": meta.get("ablation_name", ""),
                    "seed": meta.get("seed", ""),
                    "step": row.get("global_step", ""),
                    "mode": "filtered" if row.get("mode") == "star_exec" else "raw",
                    "return": to_float(row.get("episode_reward")),
                    "episode_cost": to_float(row.get("episode_cost")),
                    "EVR": to_float(row.get("violation_rate")),
                    "success": to_float(row.get("success")),
                    "constraint_satisfied": to_float(row.get("constraint_satisfied")),
                }
            )
    return grouped_ci(rows, ["task", "method", "ablation_group", "ablation_name", "step", "mode"], ["return", "episode_cost", "EVR", "success", "constraint_satisfied"])


def build_mechanism_curves(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        meta = run["meta"]
        for row in run["mechanism"]:
            item = {
                "task": meta.get("task", ""),
                "method": meta.get("method", ""),
                "ablation_group": meta.get("ablation_group", ""),
                "ablation_name": meta.get("ablation_name", ""),
                "seed": meta.get("seed", ""),
                "step": row.get("step", ""),
            }
            for key in AUDIT_COLUMNS + EXECUTION_COLUMNS:
                item[key] = to_float(row.get(key))
            rows.append(item)
    return grouped_ci(rows, ["task", "method", "ablation_group", "ablation_name", "step"], AUDIT_COLUMNS + EXECUTION_COLUMNS)


def add_efficiency_overhead(summary_rows: list[dict]) -> None:
    baseline = {}
    for row in summary_rows:
        if row.get("method") != "pointwise":
            continue
        step = to_float(row.get("final_step"))
        wall = to_float(row.get("wall_clock_time"))
        if step > 0 and wall > 0:
            baseline.setdefault(row.get("task", ""), []).append(wall / step)
    baseline_mean = {task: mean(vals) for task, vals in baseline.items() if vals}
    for row in summary_rows:
        task = row.get("task", "")
        step = to_float(row.get("final_step"))
        wall = to_float(row.get("wall_clock_time"))
        if task in baseline_mean and step > 0 and wall > 0:
            row["approx_star_overhead_vs_pointwise"] = wall / step / baseline_mean[task] - 1.0
        else:
            row["approx_star_overhead_vs_pointwise"] = math.nan


def build_raw_vs_filtered(summary_rows: list[dict]) -> list[dict]:
    rows = []
    for row in summary_rows:
        rows.append(
            {
                "run_name": row["run_name"],
                "task": row["task"],
                "method": row["method"],
                "seed": row["seed"],
                "ablation_group": row["ablation_group"],
                "ablation_name": row["ablation_name"],
                "raw_return": row["raw_return"],
                "filtered_return": row["filtered_return"],
                "return_delta_filtered_minus_raw": to_float(row["filtered_return"]) - to_float(row["raw_return"]),
                "raw_episode_cost": row["raw_episode_cost"],
                "filtered_episode_cost": row["filtered_episode_cost"],
                "cost_delta_filtered_minus_raw": to_float(row["filtered_episode_cost"]) - to_float(row["raw_episode_cost"]),
                "raw_EVR": row["raw_EVR"],
                "filtered_EVR": row["filtered_EVR"],
                "EVR_delta_filtered_minus_raw": to_float(row["filtered_EVR"]) - to_float(row["raw_EVR"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/star_aaai")
    parser.add_argument("--report-dir", default="reports/star_aaai")
    args = parser.parse_args()

    root = Path(args.root)
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    summary_rows = [run_summary(run) for run in runs]
    add_efficiency_overhead(summary_rows)

    summary_fields = [
        "run_name",
        "run_dir",
        "task",
        "method",
        "seed",
        "ablation_group",
        "ablation_name",
        "final_step",
        "train_cost_rate",
    ] + PERFORMANCE_COLUMNS + AUDIT_COLUMNS + EXECUTION_COLUMNS + EFFICIENCY_COLUMNS
    write_csv(report / "summary_by_seed.csv", summary_rows, summary_fields)

    learning = build_learning_curves(runs)
    write_csv(report / "learning_curves.csv", learning, list(learning[0].keys()) if learning else ["task", "method", "step", "mode"])

    mechanism = build_mechanism_curves(runs)
    write_csv(report / "mechanism_curves.csv", mechanism, list(mechanism[0].keys()) if mechanism else ["task", "method", "step"])

    ablation = grouped_ci(summary_rows, ["task", "ablation_group", "ablation_name"], PERFORMANCE_COLUMNS + AUDIT_COLUMNS + EXECUTION_COLUMNS)
    write_csv(report / "ablation_summary.csv", ablation, list(ablation[0].keys()) if ablation else ["task", "ablation_group", "ablation_name"])

    efficiency = grouped_ci(summary_rows, ["task", "method", "ablation_group", "ablation_name"], EFFICIENCY_COLUMNS)
    write_csv(report / "efficiency_summary.csv", efficiency, list(efficiency[0].keys()) if efficiency else ["task", "method"])

    raw_vs_filtered = build_raw_vs_filtered(summary_rows)
    write_csv(report / "raw_vs_filtered.csv", raw_vs_filtered, list(raw_vs_filtered[0].keys()) if raw_vs_filtered else ["run_name", "task", "method"])

    with (report / "summary.md").open("w") as f:
        f.write("# AAAI STAR Experiment Summary\n\n")
        f.write(f"Runs parsed: {len(summary_rows)}\n\n")
        f.write("Generated CSV files:\n\n")
        for name in [
            "summary_by_seed.csv",
            "learning_curves.csv",
            "mechanism_curves.csv",
            "ablation_summary.csv",
            "efficiency_summary.csv",
            "raw_vs_filtered.csv",
        ]:
            f.write(f"- {name}\n")

    print(f"Parsed {len(summary_rows)} runs from {root}")
    print(f"Wrote reports to {report}")


if __name__ == "__main__":
    main()
