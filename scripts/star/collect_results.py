#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PERFORMANCE_COLUMNS = [
    "raw_return",
    "filtered_return",
    "raw_episode_cost",
    "filtered_episode_cost",
    "raw_EVR",
    "filtered_EVR",
    "raw_success_rate",
    "filtered_success_rate",
    "raw_constraint_satisfaction_rate",
    "filtered_constraint_satisfaction_rate",
]

AUDIT_COLUMNS = [
    "pSVR",
    "any_unsafe_shadow_rate",
    "hidden_unsafe_rate",
    "shadow_risk_mean",
    "shadow_risk_batch_max",
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

ERROR_RE = re.compile(
    r"No space left|Traceback|RuntimeError|OOM|out of memory|CUDA error|Segmentation fault|KeyboardInterrupt|invalid loss",
    re.IGNORECASE,
)


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


def method_display_name(method: str) -> str:
    if method == "sac_lag":
        return "SAC-Lag-local"
    return method


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
    return {"mean": mean(vals), "std": stdev(vals), "ci95_low": low, "ci95_high": high, "n": len(vals)}


def latest_by_step(rows: list[dict], step_key: str) -> list[dict]:
    if not rows:
        return []
    max_step = max(to_int(row.get(step_key)) for row in rows)
    return [row for row in rows if to_int(row.get(step_key)) == max_step]


def avg(rows: list[dict], key: str) -> float:
    vals = finite([to_float(row.get(key)) for row in rows])
    return mean(vals) if vals else math.nan


def scan_error(run_dir: Path, log_root: Path | None, run_name: str) -> tuple[bool, str]:
    candidates = []
    for pattern in ("*.log", "*.out", "*.err", "stderr*", "stdout*"):
        candidates.extend(run_dir.rglob(pattern))
    if log_root and log_root.exists():
        candidates.extend(path for path in log_root.rglob("*") if path.is_file() and run_name in path.name)
    for path in sorted(set(candidates)):
        try:
            for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
                if ERROR_RE.search(line):
                    return True, f"{path}:{lineno}:{line[:200]}"
        except OSError:
            continue
    return False, ""


def load_runs(root: Path, log_root: Path | None) -> list[dict]:
    runs = []
    for meta_path in sorted(root.rglob("run_metadata.json")):
        run_dir = meta_path.parent
        with meta_path.open() as f:
            meta = json.load(f)
        eval_path = run_dir / "corrected_eval_episodes.csv"
        if not eval_path.exists():
            eval_path = run_dir / "eval_episodes.csv"
        run_name = meta.get("run_name", run_dir.name)
        has_error, error_detail = scan_error(run_dir, log_root, run_name)
        runs.append(
            {
                "run_dir": run_dir,
                "meta": meta,
                "eval": read_csv(eval_path),
                "eval_source": str(eval_path),
                "train": read_csv(run_dir / "train_episodes.csv"),
                "mechanism": read_csv(run_dir / "mechanism.csv"),
                "efficiency": read_csv(run_dir / "efficiency.csv"),
                "has_error": has_error,
                "error_detail": error_detail,
            }
        )
    return runs


def final_eval(rows: list[dict], mode: str) -> list[dict]:
    return latest_by_step([row for row in rows if row.get("mode") == mode], "global_step")


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
    final_step = max(
        [to_int(row.get("global_step")) for row in raw + filtered]
        + [to_int(row.get("end_step")) for row in train]
        + [0]
    )
    target_steps = to_int(meta.get("num_steps"))
    completed = bool(target_steps > 0 and final_step >= target_steps)
    env_cost_limit = to_float(meta.get("env_cost_limit"), 25.0)
    run_name = meta.get("run_name", run["run_dir"].name)
    method = meta.get("method", "")
    return {
        "run_name": run_name,
        "run_dir": str(run["run_dir"]),
        "eval_source": run.get("eval_source", ""),
        "task": meta.get("task", ""),
        "method": method,
        "method_display_name": method_display_name(method),
        "seed": meta.get("seed", ""),
        "env_cost_limit": env_cost_limit,
        "ablation_group": meta.get("ablation_group", "main"),
        "ablation_name": meta.get("ablation_name", meta.get("method", "")),
        "target_steps": target_steps,
        "final_step": final_step,
        "completed": completed,
        "has_error": bool(run["has_error"]),
        "error_detail": run["error_detail"],
        "mtime": run["run_dir"].stat().st_mtime,
        "start_time": meta.get("start_time", ""),
        "raw_return": avg(raw, "episode_reward"),
        "filtered_return": avg(filtered, "episode_reward"),
        "raw_episode_cost": avg(raw, "episode_cost"),
        "filtered_episode_cost": avg(filtered, "episode_cost"),
        "raw_EVR": avg(raw, "violation_rate"),
        "filtered_EVR": avg(filtered, "violation_rate"),
        "raw_success_rate": avg(raw, "success"),
        "filtered_success_rate": avg(filtered, "success"),
        "raw_constraint_satisfaction_rate": avg(raw, "constraint_satisfied"),
        "filtered_constraint_satisfaction_rate": avg(filtered, "constraint_satisfied"),
        "train_cost_rate": avg(train, "train_total_cost_rate"),
        "pSVR": avg(mech, "pSVR"),
        "any_unsafe_shadow_rate": avg(mech, "any_unsafe_shadow_rate"),
        "hidden_unsafe_rate": avg(mech, "hidden_unsafe_rate"),
        "shadow_risk_mean": avg(mech, "shadow_risk_mean"),
        "shadow_risk_batch_max": avg(mech, "shadow_risk_batch_max"),
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


def deduplicate(summary_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in summary_rows:
        key = (
            row.get("task", ""),
            row.get("method", ""),
            str(row.get("seed", "")),
            row.get("ablation_group", ""),
            row.get("ablation_name", ""),
        )
        groups[key].append(row)
    selected = []
    duplicates = []
    for key, rows in sorted(groups.items()):
        qualified = [row for row in rows if row["completed"] and not row["has_error"]]
        for row in rows:
            row["qualified_for_default"] = row in qualified
        if len(qualified) > 1:
            for row in qualified:
                dup = dict(row)
                dup["duplicate_key"] = "|".join(key)
                duplicates.append(dup)
        if qualified:
            chosen = sorted(qualified, key=lambda row: (str(row.get("start_time", "")), float(row.get("mtime", 0))))[-1]
            chosen["selected_for_default"] = True
            selected.append(chosen)
        else:
            for row in rows:
                row["selected_for_default"] = False
    return selected, duplicates


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


def build_learning_curves(runs: list[dict], selected_names: set[str]) -> list[dict]:
    rows = []
    for run in runs:
        meta = run["meta"]
        run_name = meta.get("run_name", run["run_dir"].name)
        if run_name not in selected_names:
            continue
        for row in run["eval"]:
            rows.append(
                {
                    "task": meta.get("task", ""),
                    "method": meta.get("method", ""),
                    "method_display_name": method_display_name(meta.get("method", "")),
                    "env_cost_limit": to_float(meta.get("env_cost_limit"), 25.0),
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
    return grouped_ci(
        rows,
        ["task", "method", "method_display_name", "env_cost_limit", "ablation_group", "ablation_name", "step", "mode"],
        ["return", "episode_cost", "EVR", "success", "constraint_satisfied"],
    )


def build_mechanism_curves(runs: list[dict], selected_names: set[str]) -> list[dict]:
    rows = []
    for run in runs:
        meta = run["meta"]
        run_name = meta.get("run_name", run["run_dir"].name)
        if run_name not in selected_names:
            continue
        base = {
            "task": meta.get("task", ""),
            "method": meta.get("method", ""),
            "method_display_name": method_display_name(meta.get("method", "")),
            "env_cost_limit": to_float(meta.get("env_cost_limit"), 25.0),
            "ablation_group": meta.get("ablation_group", ""),
            "ablation_name": meta.get("ablation_name", ""),
            "seed": meta.get("seed", ""),
        }
        for row in run["mechanism"]:
            item = dict(base)
            item["step"] = row.get("step", "")
            for key in AUDIT_COLUMNS:
                item[key] = to_float(row.get(key))
            rows.append(item)
        for row in run["eval"]:
            if row.get("mode") != "star_exec":
                continue
            item = dict(base)
            item["step"] = row.get("global_step", "")
            item.update(
                {
                    "safe_candidate_fraction": to_float(row.get("safe_candidate_fraction")),
                    "fallback_rate": to_float(row.get("execution_fallback_rate")),
                    "executed_predicted_risk": to_float(row.get("executed_predicted_risk")),
                    "found_but_not_executed_rate": to_float(row.get("found_but_not_executed_rate")),
                    "boundary_safe_coverage": to_float(row.get("boundary_safe_coverage")),
                    "boundary_safety_rate": to_float(row.get("boundary_safety_rate")),
                }
            )
            rows.append(item)
    return grouped_ci(
        rows,
        ["task", "method", "method_display_name", "env_cost_limit", "ablation_group", "ablation_name", "step"],
        AUDIT_COLUMNS + EXECUTION_COLUMNS,
    )


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
                "method_display_name": row["method_display_name"],
                "seed": row["seed"],
                "env_cost_limit": row["env_cost_limit"],
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


def completeness_matrix(manifest_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in manifest_rows:
        key = (row["task"], row["method"], row["method_display_name"], row["ablation_group"], row["ablation_name"])
        groups[key].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        selected = [row for row in rows if row.get("selected_for_default")]
        qualified = [row for row in rows if row.get("qualified_for_default")]
        out.append(
            {
                "task": key[0],
                "method": key[1],
                "method_display_name": key[2],
                "ablation_group": key[3],
                "ablation_name": key[4],
                "runs_total": len(rows),
                "runs_qualified": len(qualified),
                "runs_selected": len(selected),
                "seeds_selected": ",".join(str(row["seed"]) for row in sorted(selected, key=lambda r: str(r["seed"]))),
                "complete_5_seeds": len({str(row["seed"]) for row in selected}) >= 5,
            }
        )
    return out


def write_claim_audit(path: Path, manifest_rows: list[dict], selected_rows: list[dict], duplicate_rows: list[dict]) -> None:
    missing = [row for row in manifest_rows if not row.get("qualified_for_default")]
    with path.open("w") as f:
        f.write("# STAR Final Claim Audit\n\n")
        f.write(f"Runs discovered: {len(manifest_rows)}\n\n")
        f.write(f"Runs selected for default aggregation: {len(selected_rows)}\n\n")
        f.write(f"Qualified duplicate runs: {len(duplicate_rows)}\n\n")
        f.write("Default aggregation includes only runs with `completed=True`, `has_error=False`, and final step >= configured `num_steps`.\n\n")
        f.write("SAC-Lag baseline display name is `SAC-Lag-local`; it is an internal local baseline, not an external official implementation.\n\n")
        if missing:
            f.write("## Excluded or Incomplete Runs\n\n")
            f.write("| Run | Task | Method | Seed | Completed | Error | Final Step | Target Steps |\n")
            f.write("| --- | --- | --- | ---: | --- | --- | ---: | ---: |\n")
            for row in missing[:200]:
                f.write(
                    f"| {row['run_name']} | {row['task']} | {row['method_display_name']} | {row['seed']} | "
                    f"{row['completed']} | {row['has_error']} | {row['final_step']} | {row['target_steps']} |\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/star_aaai")
    parser.add_argument("--report-dir", default="reports/star_aaai")
    parser.add_argument("--final-report-dir", default="reports/star_final")
    parser.add_argument("--log-root", default="logs/star_aaai")
    args = parser.parse_args()

    root = Path(args.root)
    report = Path(args.report_dir)
    final_report = Path(args.final_report_dir)
    log_root = Path(args.log_root) if args.log_root else None
    report.mkdir(parents=True, exist_ok=True)
    final_report.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root, log_root)
    manifest_rows = [run_summary(run) for run in runs]
    selected_rows, duplicate_rows = deduplicate(manifest_rows)
    add_efficiency_overhead(selected_rows)
    selected_names = {row["run_name"] for row in selected_rows}

    summary_fields = [
        "run_name",
        "run_dir",
        "eval_source",
        "task",
        "method",
        "method_display_name",
        "seed",
        "env_cost_limit",
        "ablation_group",
        "ablation_name",
        "target_steps",
        "final_step",
        "completed",
        "has_error",
        "train_cost_rate",
    ] + PERFORMANCE_COLUMNS + AUDIT_COLUMNS + EXECUTION_COLUMNS + EFFICIENCY_COLUMNS
    write_csv(report / "summary_by_seed.csv", selected_rows, summary_fields)

    learning = build_learning_curves(runs, selected_names)
    write_csv(report / "learning_curves.csv", learning, list(learning[0].keys()) if learning else ["task", "method", "step", "mode"])

    mechanism = build_mechanism_curves(runs, selected_names)
    write_csv(report / "mechanism_curves.csv", mechanism, list(mechanism[0].keys()) if mechanism else ["task", "method", "step"])

    ablation = grouped_ci(selected_rows, ["task", "ablation_group", "ablation_name"], PERFORMANCE_COLUMNS + AUDIT_COLUMNS + EXECUTION_COLUMNS)
    write_csv(report / "ablation_summary.csv", ablation, list(ablation[0].keys()) if ablation else ["task", "ablation_group", "ablation_name"])

    efficiency = grouped_ci(selected_rows, ["task", "method", "method_display_name", "ablation_group", "ablation_name"], EFFICIENCY_COLUMNS)
    write_csv(report / "efficiency_summary.csv", efficiency, list(efficiency[0].keys()) if efficiency else ["task", "method"])

    raw_vs_filtered = build_raw_vs_filtered(selected_rows)
    write_csv(report / "raw_vs_filtered.csv", raw_vs_filtered, list(raw_vs_filtered[0].keys()) if raw_vs_filtered else ["run_name", "task", "method"])

    manifest_fields = [
        "run_name",
        "run_dir",
        "eval_source",
        "task",
        "method",
        "method_display_name",
        "seed",
        "env_cost_limit",
        "ablation_group",
        "ablation_name",
        "target_steps",
        "final_step",
        "completed",
        "has_error",
        "error_detail",
        "qualified_for_default",
        "selected_for_default",
    ]
    write_csv(final_report / "run_manifest.csv", manifest_rows, manifest_fields)
    write_csv(final_report / "duplicate_runs.csv", duplicate_rows, manifest_fields + ["duplicate_key"])
    comp = completeness_matrix(manifest_rows)
    write_csv(final_report / "completeness_matrix.csv", comp, list(comp[0].keys()) if comp else ["task", "method"])
    write_claim_audit(final_report / "claim_audit.md", manifest_rows, selected_rows, duplicate_rows)

    with (report / "summary.md").open("w") as f:
        f.write("# AAAI STAR Experiment Summary\n\n")
        f.write(f"Runs discovered: {len(manifest_rows)}\n\n")
        f.write(f"Runs selected: {len(selected_rows)}\n\n")
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

    print(f"Parsed {len(manifest_rows)} runs from {root}")
    print(f"Selected {len(selected_rows)} completed error-free runs")
    print(f"Wrote reports to {report} and audit files to {final_report}")


if __name__ == "__main__":
    main()
