#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_run(run_dir: Path, expected_steps: int) -> dict:
    train_rows = read_csv(run_dir / "train_episodes.csv")
    mech_rows = read_csv(run_dir / "mechanism.csv")
    metadata = {}
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            metadata = {}
    final_ckpt = run_dir / "checkpoint" / "final.torch"
    last = train_rows[-1] if train_rows else {}
    last10 = train_rows[-10:] if train_rows else []
    mech_last10 = mech_rows[-10:] if mech_rows else []
    final_step = int(fnum(last.get("end_step", 0), 0))
    completed = final_ckpt.exists() and final_step >= expected_steps
    return {
        "run_name": run_dir.name,
        "task": metadata.get("task", last.get("task", "")),
        "method": metadata.get("method", last.get("method", "")),
        "seed": int(fnum(metadata.get("seed", last.get("seed", 0)), 0)),
        "ablation_group": metadata.get("ablation_group", ""),
        "ablation_name": metadata.get("ablation_name", ""),
        "star_algorithm_version": metadata.get("star_algorithm_version", ""),
        "shadow_beta_mode": metadata.get("shadow_beta_mode", ""),
        "shadow_penalty_mode": metadata.get("star_shadow_penalty_mode", metadata.get("shadow_penalty_mode", "")),
        "shadow_num_strata": metadata.get("shadow_num_strata", ""),
        "shadow_samples_per_stratum": metadata.get("shadow_samples_per_stratum", ""),
        "shadow_temperature": metadata.get("shadow_temperature", ""),
        "star_risk_threshold": metadata.get("star_risk_threshold", ""),
        "star_lambda": metadata.get("star_lambda", ""),
        "cost_gamma": metadata.get("cost_gamma", ""),
        "final_step": final_step,
        "completed": completed,
        "final_checkpoint": str(final_ckpt) if final_ckpt.exists() else "",
        "latest_reward": fnum(last.get("episode_reward", 0)),
        "latest_cost": fnum(last.get("episode_cost", 0)),
        "avg_last10_reward": mean([fnum(r.get("episode_reward", 0)) for r in last10]),
        "avg_last10_cost": mean([fnum(r.get("episode_cost", 0)) for r in last10]),
        "train_total_cost": fnum(last.get("train_total_cost", 0)),
        "train_total_cost_rate": fnum(last.get("train_total_cost_rate", 0)),
        "wall_clock_time": fnum(last.get("wall_clock_time", 0)),
        "shadow_excess_mean": mean([fnum(r.get("shadow_excess_mean", 0)) for r in mech_last10]),
        "actor_gradient_norm": mean([fnum(r.get("actor_gradient_norm", 0)) for r in mech_last10]),
        "effective_beta": mean([fnum(r.get("effective_beta", 0)) for r in mech_last10]),
        "reference_update_count": max([fnum(r.get("reference_update_count", 0)) for r in mech_rows], default=0.0),
        "mechanism_rows": len(mech_rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_star(rows: list[dict]) -> tuple[str | None, str]:
    complete = [r for r in rows if r["completed"] and r["star_algorithm_version"] == "star_v2"]
    expected = 16
    if len(complete) < expected:
        return None, f"STAR calibration incomplete: {len(complete)}/{expected} completed."
    by_config_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in complete:
        by_config_task[(row["ablation_name"], row["task"])].append(row)
    tasks = sorted({r["task"] for r in complete})
    configs = sorted({r["ablation_name"] for r in complete})
    best_return_by_task = {}
    for task in tasks:
        best_return_by_task[task] = max(
            mean([r["avg_last10_reward"] for r in by_config_task[(cfg, task)]])
            for cfg in configs
        )
    candidates = []
    for cfg in configs:
        task_reward_ok = True
        task_returns = []
        task_costs = []
        total_costs = []
        for task in tasks:
            cfg_rows = by_config_task[(cfg, task)]
            r_mean = mean([r["avg_last10_reward"] for r in cfg_rows])
            c_mean = mean([r["avg_last10_cost"] for r in cfg_rows])
            task_returns.append(r_mean)
            task_costs.append(c_mean)
            total_costs.extend([r["train_total_cost"] for r in cfg_rows])
            if r_mean < 0.85 * best_return_by_task[task]:
                task_reward_ok = False
        if task_reward_ok:
            candidates.append((mean(task_costs), mean(total_costs), cfg, mean(task_returns)))
    if not candidates:
        return None, "No STAR calibration config retained at least 85% of best task return on every development task."
    candidates.sort()
    selected = candidates[0][2]
    return selected, (
        f"Selected {selected}: lowest average development cost among configs satisfying "
        "per-task 85% return retention."
    )


def select_baseline_method(rows: list[dict], method: str) -> tuple[str | None, str]:
    complete = [r for r in rows if r["completed"] and r["method"] == method and r["star_algorithm_version"] == "star_v2"]
    tasks = sorted({r["task"] for r in complete})
    configs = sorted({r["ablation_name"] for r in complete})
    if len(tasks) < 2 or not configs or len(complete) < len(tasks) * len(configs):
        return None, f"{method} calibration incomplete: {len(complete)} completed rows."
    by_config_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in complete:
        by_config_task[(row["ablation_name"], row["task"])].append(row)
    best_return_by_task = {
        task: max(mean([r["avg_last10_reward"] for r in by_config_task[(cfg, task)]]) for cfg in configs)
        for task in tasks
    }
    candidates = []
    fallback = []
    for cfg in configs:
        task_reward_ok = True
        task_costs = []
        total_costs = []
        task_returns = []
        for task in tasks:
            cfg_rows = by_config_task[(cfg, task)]
            r_mean = mean([r["avg_last10_reward"] for r in cfg_rows])
            c_mean = mean([r["avg_last10_cost"] for r in cfg_rows])
            task_returns.append(r_mean)
            task_costs.append(c_mean)
            total_costs.extend([r["train_total_cost"] for r in cfg_rows])
            if r_mean < 0.85 * best_return_by_task[task]:
                task_reward_ok = False
        item = (mean(task_costs), mean(total_costs), cfg, mean(task_returns))
        fallback.append(item)
        if task_reward_ok:
            candidates.append(item)
    pool = candidates or fallback
    pool.sort()
    selected = pool[0][2]
    if candidates:
        return selected, f"Selected {selected} for {method}: lowest development cost with 85% per-task return retention."
    return selected, f"Selected {selected} for {method}: no config met 85% return retention, using lowest development cost fallback."


def update_baseline_config(path: Path, selected: dict[str, str], source_sha: str) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data["algorithm_git_sha"] = source_sha
    data["selection_source"] = "reports/star_v2_final/calibration/baseline_selection.md"
    methods = data.setdefault("methods", {})
    pointwise = selected.get("pointwise_v2")
    if pointwise:
        methods.setdefault("pointwise_v2", {})["method"] = "pointwise_v2"
        methods["pointwise_v2"]["selected_calibration_name"] = pointwise
        methods["pointwise_v2"]["star_lambda"] = 1.0 if "lam10" in pointwise else 0.5
    sac_lag = selected.get("sac_lag")
    if sac_lag:
        methods.setdefault("sac_lag", {})["method"] = "sac_lag"
        methods["sac_lag"]["selected_calibration_name"] = sac_lag
        methods["sac_lag"]["lagrange_lr"] = 1e-3 if "lr1e3" in sac_lag else 3e-4
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/star_v2_final")
    parser.add_argument("--report-dir", default="reports/star_v2_final/calibration")
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    report = Path(args.report_dir)
    star_dirs = sorted((root / "calibration_star100k").glob("*/*/starv2_*"))
    baseline_dirs = sorted((root / "calibration_baseline100k").glob("*/*/starv2_*"))
    star_rows = [summarize_run(path, 100000) for path in star_dirs if path.is_dir()]
    baseline_rows = [summarize_run(path, 100000) for path in baseline_dirs if path.is_dir()]
    write_csv(report / "star_calibration.csv", star_rows)
    write_csv(report / "baseline_calibration.csv", baseline_rows)
    selected, rationale = select_star(star_rows)
    lines = [
        "# STAR-v2 calibration selection",
        "",
        f"STAR runs completed: {sum(1 for r in star_rows if r['completed'])}/16",
        f"Baseline screen runs completed: {sum(1 for r in baseline_rows if r['completed'])}/8",
        "",
        f"Selection: {selected or 'PENDING'}",
        "",
        rationale,
        "",
        "Selection uses avg_last10_reward/cost from training episodes. Offline evaluation remains required before the 100k core gate.",
    ]
    (report / "star_selection.md").write_text("\n".join(lines) + "\n")
    baseline_completed = sum(1 for r in baseline_rows if r["completed"])
    selected_baselines: dict[str, str] = {}
    baseline_lines = [
        "# Baseline calibration",
        "",
        f"Baseline screen runs completed: {baseline_completed}/8",
        "",
    ]
    if baseline_completed >= 8:
        for method in ("pointwise_v2", "sac_lag"):
            choice, reason = select_baseline_method(baseline_rows, method)
            baseline_lines.extend([f"## {method}", "", f"Selection: {choice or 'PENDING'}", "", reason, ""])
            if choice:
                selected_baselines[method] = choice
        if args.update_config and selected_baselines:
            import subprocess

            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            update_baseline_config(Path("configs/star_v2_selected_baselines.json"), selected_baselines, sha)
    else:
        baseline_lines.append("Selection remains pending until all baseline screen runs complete.")
    (report / "baseline_selection.md").write_text("\n".join(baseline_lines) + "\n")
    print(f"wrote {report / 'star_calibration.csv'} rows={len(star_rows)}")
    print(f"wrote {report / 'baseline_calibration.csv'} rows={len(baseline_rows)}")
    print(f"selected={selected or 'PENDING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
