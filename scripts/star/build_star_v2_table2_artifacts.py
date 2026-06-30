#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TASKS = ["SafetyPointGoal1-v0", "SafetyCarButton1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"]
PRIMARY_TASKS = ["SafetyPointGoal1-v0", "SafetyCarButton1-v0"]
METHODS = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]
SEEDS = [10, 11, 12, 13, 14]
AGE_BINS = ["age=0", "age=1-5", "age=6-10", "age=11-20", "age>20"]


def fnum(value, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"], extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(values) -> list[float]:
    out = []
    for value in values:
        x = fnum(value)
        if not math.isnan(x):
            out.append(x)
    return out


def avg(values) -> float:
    vals = finite(values)
    return mean(vals) if vals else math.nan


def sd(values) -> float:
    vals = finite(values)
    return stdev(vals) if len(vals) > 1 else (0.0 if vals else math.nan)


def age_bin(age: float) -> str:
    if math.isnan(age):
        return "missing"
    if age == 0:
        return "age=0"
    if 1 <= age <= 5:
        return "age=1-5"
    if 6 <= age <= 10:
        return "age=6-10"
    if 11 <= age <= 20:
        return "age=11-20"
    return "age>20"


def run_dirs(root: Path, phase: str = "resume_300k") -> list[Path]:
    return sorted((root / phase).glob("*/*/*"))


def build_mechanism(root: Path, report_dir: Path) -> None:
    rows = []
    age_values = []
    for run_dir in run_dirs(root):
        meta_rows = read_csv(run_dir / "mechanism.csv")
        if not meta_rows:
            continue
        for row in meta_rows:
            if row.get("method") != "star_v2":
                continue
            task = row.get("task", "")
            seed = int(fnum(row.get("seed", 0), 0))
            if task not in TASKS or seed not in SEEDS:
                continue
            age = fnum(row.get("reference_age_post_update"), fnum(row.get("reference_age_pre_update")))
            age_values.append(age)
            rho_cur = fnum(row.get("paired_current_risk"))
            rho_cor = fnum(row.get("paired_corridor_risk"))
            out = {
                "task": task,
                "seed": seed,
                "run_name": row.get("run_name", run_dir.name),
                "step": int(fnum(row.get("step"), 0)),
                "reference_age": age,
                "reference_age_bin": age_bin(age),
                "rho_cur": rho_cur,
                "rho_cor": rho_cor,
                "corridor_risk_lift": fnum(row.get("paired_corridor_risk_lift"), rho_cor - rho_cur),
                "lift_positive": fnum(row.get("paired_lift_positive_rate")),
                "shadow_excess": fnum(row.get("shadow_excess_mean")),
                "effective_beta": fnum(row.get("effective_beta")),
                "redteam_entropy": fnum(row.get("redteam_weight_entropy")),
                "reference_kl": fnum(row.get("reference_kl_mean"), fnum(row.get("kl_mean"))),
                "highest_risk_current": rho_cur,
                "highest_risk_corridor": rho_cor,
                "actor_mean_risk": fnum(row.get("actor_mean_action_risk")),
                "source_kind": "training_mechanism_log_batch",
                "source_file": str(run_dir / "mechanism.csv"),
            }
            rows.append(out)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["seed"], row["reference_age_bin"])].append(row)
    summary = []
    for (task, seed, bin_name), vals in sorted(groups.items()):
        summary.append(
            {
                "task": task,
                "seed": seed,
                "reference_age_bin": bin_name,
                "rows": len(vals),
                "rho_cur_mean": avg(v["rho_cur"] for v in vals),
                "rho_cor_mean": avg(v["rho_cor"] for v in vals),
                "corridor_risk_lift_mean": avg(v["corridor_risk_lift"] for v in vals),
                "corridor_risk_lift_std": sd(v["corridor_risk_lift"] for v in vals),
                "lift_positive_rate_mean": avg(v["lift_positive"] for v in vals),
                "shadow_excess_mean": avg(v["shadow_excess"] for v in vals),
                "effective_beta_mean": avg(v["effective_beta"] for v in vals),
                "redteam_entropy_mean": avg(v["redteam_entropy"] for v in vals),
                "reference_kl_mean": avg(v["reference_kl"] for v in vals),
                "actor_mean_risk_mean": avg(v["actor_mean_risk"] for v in vals),
            }
        )
    out_dir = report_dir / "mechanism"
    write_csv(out_dir / "corridor_mechanism_by_age.csv", rows)
    write_csv(out_dir / "corridor_mechanism_summary.csv", summary)

    finite_ages = sorted({int(v) for v in finite(age_values)})
    if len(finite_ages) <= 1:
        (out_dir / "MECHANISM_REFERENCE_AGE_UNAVAILABLE.md").write_text(
            "# Mechanism Reference-Age Dynamics Unavailable\n\n"
            "Final resume_300k mechanism logs contain paired corridor/current audit diagnostics, "
            "but `reference_age_post_update` is collapsed to a single value. The Figure 2 mechanism "
            "plot therefore uses logged paired-audit batches as mechanism validation evidence, while "
            "age-dynamic claims are not made.\n\n"
            f"- observed_reference_age_values: `{finite_ages}`\n"
            f"- mechanism_rows_used: `{len(rows)}`\n",
            encoding="utf-8",
        )


def build_efficiency(root: Path, report_dir: Path) -> None:
    rows = []
    for run_dir in run_dirs(root):
        eff = read_csv(run_dir / "efficiency.csv")
        mech = read_csv(run_dir / "mechanism.csv")
        if not eff:
            continue
        final = eff[-1]
        task = final.get("task", "")
        method = final.get("method", "")
        seed = int(fnum(final.get("seed", 0), 0))
        if task not in TASKS or method not in METHODS or seed not in SEEDS:
            continue
        step = fnum(final.get("step"))
        update_time = fnum(final.get("update_time"))
        calls = fnum(final.get("cost_critic_forward_calls"))
        actor_update_ms = math.nan
        critic_update_ms = math.nan
        # The training logger records aggregate update_time, not separate actor
        # and critic timers. Keep separate timers unavailable rather than
        # inventing a split.
        rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "steps": step,
                "training_steps_per_sec": fnum(final.get("env_steps_per_second")),
                "updates_per_sec": fnum(final.get("updates_per_second")),
                "update_ms_per_env_step": 1000.0 * update_time / step if step and not math.isnan(update_time) else math.nan,
                "actor_update_ms": actor_update_ms,
                "critic_update_ms": critic_update_ms,
                "cost_critic_calls_per_env_step": calls / step if step and not math.isnan(calls) else math.nan,
                "cost_critic_forward_calls": calls,
                "mechanism_rows": len(mech),
                "source_file": str(run_dir / "efficiency.csv"),
            }
        )
    base = {}
    executor_rows = read_csv(report_dir / "executor" / "executor_summary.csv")
    star_exec_latency_by_task = {
        row.get("task", ""): row.get("latency_ms", "")
        for row in executor_rows
        if row.get("task")
    }
    for task in TASKS:
        point = [r for r in rows if r["task"] == task and r["method"] == "pointwise_v2"]
        current = [r for r in rows if r["task"] == task and r["method"] == "current_only_v2"]
        base[(task, "pointwise_v2")] = avg(r["training_steps_per_sec"] for r in point)
        base[(task, "current_only_v2")] = avg(r["training_steps_per_sec"] for r in current)
    summary = []
    for (task, method), vals in sorted(defaultdict(list, {}).items()):
        pass
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"])].append(row)
    for (task, method), vals in sorted(grouped.items()):
        speed = avg(v["training_steps_per_sec"] for v in vals)
        summary.append(
            {
                "task": task,
                "method": method,
                "seeds": len(vals),
                "training_steps_per_sec_mean": speed,
                "training_steps_per_sec_std": sd(v["training_steps_per_sec"] for v in vals),
                "update_ms_per_env_step_mean": avg(v["update_ms_per_env_step"] for v in vals),
                "actor_update_ms": "",
                "critic_update_ms": "",
                "timing_note": "training logs record aggregate update_time only; actor/critic split unavailable",
                "cost_critic_calls_per_env_step_mean": avg(v["cost_critic_calls_per_env_step"] for v in vals),
                "relative_overhead_vs_pointwise": (base[(task, "pointwise_v2")] / speed if speed else math.nan),
                "relative_overhead_vs_current_only": (base[(task, "current_only_v2")] / speed if speed else math.nan),
                "star_exec_latency_ms": star_exec_latency_by_task.get(task, "") if method == "star_v2" else "",
            }
        )
    out_dir = report_dir / "efficiency"
    write_csv(out_dir / "efficiency_by_seed.csv", rows)
    write_csv(out_dir / "efficiency_summary.csv", summary)


def enrich_ablation(report_dir: Path, result_root: Path) -> None:
    path = report_dir / "ablation" / "ablation_by_seed.csv"
    rows = read_csv(path)
    for row in rows:
        run_dir = Path(row.get("run_dir", ""))
        mech = read_csv(run_dir / "mechanism.csv")
        eff = read_csv(run_dir / "efficiency.csv")
        if mech:
            final = mech[-1]
            row["shadow_excess"] = final.get("shadow_excess_mean", "")
            row["effective_beta"] = final.get("effective_beta", "")
        if eff:
            final_eff = eff[-1]
            row["steps_per_sec"] = final_eff.get("env_steps_per_second", "")
            row["actor_update_ms"] = ""
            row["actor_update_note"] = "aggregate update_time only"
    write_csv(path, rows)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("task", ""), row.get("ablation_name", ""))].append(row)
    summary = []
    for (task, ablation), vals in sorted(grouped.items()):
        summary.append(
            {
                "task": task,
                "ablation_name": ablation,
                "seeds": len(vals),
                "return_mean": avg(v.get("raw_return_mean") for v in vals),
                "cost_mean": avg(v.get("raw_cost_mean") for v in vals),
                "EVR_mean": avg(v.get("raw_evr_mean") for v in vals),
                "shadow_excess_mean": avg(v.get("shadow_excess") for v in vals),
                "effective_beta_mean": avg(v.get("effective_beta") for v in vals),
                "steps_per_sec_mean": avg(v.get("steps_per_sec") for v in vals),
                "actor_update_ms": "",
            }
        )
    write_csv(report_dir / "ablation" / "ablation_summary.csv", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/star_v2_final"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/star_v2_final"))
    args = parser.parse_args()
    build_mechanism(args.root, args.report_dir)
    build_efficiency(args.root, args.report_dir)
    enrich_ablation(args.report_dir, args.root)
    print(f"wrote STAR-v2 Table 2 mechanism/efficiency artifacts under {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
