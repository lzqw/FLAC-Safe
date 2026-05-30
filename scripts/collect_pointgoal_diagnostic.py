#!/usr/bin/env python3
"""Collect PointGoal diagnostic metrics."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_diagnostic"
REPORT = ROOT / "reports" / "pointgoal_diagnostic" / "summary.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

GROUPS = {
    "PGD0_S0_extend": {"lambda_safe": 0.5, "lambda_jvp": 0.0, "safe_bandwidth": 0.05, "updates_per_step": 2, "seeds": [5, 6, 7]},
    "PGD1_S1_R2D_extend": {"lambda_safe": 0.5, "lambda_jvp": 0.005, "safe_bandwidth": 0.05, "updates_per_step": 2, "seeds": [5, 6, 7]},
    "PGD2_R2D_update1": {"lambda_safe": 0.5, "lambda_jvp": 0.005, "safe_bandwidth": 0.05, "updates_per_step": 1, "seeds": [0, 1, 2]},
    "PGD3_mid_jvp": {"lambda_safe": 0.5, "lambda_jvp": 0.003, "safe_bandwidth": 0.05, "updates_per_step": 2, "seeds": [0, 1, 2]},
}

OLD = {
    "S0": {"avg_last3_reward": 22.06, "avg_last3_cost": 53.69, "n": 5},
    "S1": {"avg_last3_reward": 22.24, "avg_last3_cost": 48.29, "n": 5},
}

EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")
TRACE_KEYS = [
    "safety/qc_mean",
    "safety/qc_target_mean",
    "safety/g_mid_mean",
    "safety/grad_q_norm",
    "loss/jvp_scd",
    "safety/safety_penalty",
]


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), stdev(clean)


def parse_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "evals": [], "traces": {}}
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in EVAL_RE.finditer(text)]
    lowered = text.lower()
    if "oom" in lowered or "out of memory" in lowered:
        status = "failed_oom"
    elif "nan" in text or "NaN" in text:
        status = "failed_nan"
    elif ERR_RE.search(text):
        status = "failed_error"
    elif " END " in text and evals:
        status = "completed"
    elif evals:
        status = "partial"
    else:
        status = "no evals"
    traces = {}
    for key in TRACE_KEYS:
        matches = re.findall(re.escape(key) + r"\s+([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", text, re.IGNORECASE)
        traces[key] = float(matches[-1]) if matches else None
    return {"status": status, "evals": evals, "traces": traces}


def summarize_evals(evals: list[tuple[float, float, float]]) -> dict[str, float | None]:
    if not evals:
        return {"final_reward": None, "final_cost": None, "avg_last3_reward": None, "avg_last3_cost": None}
    final_reward, final_cost, _ = evals[-1]
    last3 = evals[-3:]
    return {
        "final_reward": final_reward,
        "final_cost": final_cost,
        "avg_last3_reward": sum(row[0] for row in last3) / len(last3),
        "avg_last3_cost": sum(row[1] for row in last3) / len(last3),
    }


def parse_monitor() -> dict[str, str]:
    if not MONITOR.exists():
        return {}
    used = []
    util = []
    total = None
    with MONITOR.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                used.append(float(row["memory.used [MiB]"]))
                util.append(float(row["utilization.gpu [%]"]))
                total = float(row["memory.total [MiB]"])
            except (KeyError, TypeError, ValueError):
                continue
    if not used:
        return {}
    return {
        "gpu_total_memory": fmt(total / 1024 if total else None),
        "peak_memory_used": fmt(max(used) / 1024),
        "avg_memory_used": fmt(sum(used) / len(used) / 1024),
        "avg_gpu_utilization": fmt(sum(util) / len(util)) if util else "n/a",
    }


def combine_old_new(old_name: str, new_group: str, rows: list[dict[str, object]], metric: str) -> float | None:
    vals = [row[metric] for row in rows if row["group"] == new_group and row["status"] == "completed" and row[metric] is not None]
    old = OLD[old_name]
    if not vals:
        return old[metric]  # type: ignore[index]
    return (old[metric] * old["n"] + sum(vals)) / (old["n"] + len(vals))  # type: ignore[operator]


def group_stats(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    complete = [row for row in rows if row["group"] == group and row["status"] == "completed"]
    group_rows = [row for row in rows if row["group"] == group]
    fr_m, fr_s = mean_std([row["final_reward"] for row in complete])  # type: ignore[list-item]
    fc_m, fc_s = mean_std([row["final_cost"] for row in complete])  # type: ignore[list-item]
    ar_m, ar_s = mean_std([row["avg_last3_reward"] for row in complete])  # type: ignore[list-item]
    ac_m, ac_s = mean_std([row["avg_last3_cost"] for row in complete])  # type: ignore[list-item]
    return {
        "completed": len(complete),
        "failed": len(group_rows) - len(complete),
        "final_reward_mean": fr_m,
        "final_reward_std": fr_s,
        "final_cost_mean": fc_m,
        "final_cost_std": fc_s,
        "avg_last3_reward_mean": ar_m,
        "avg_last3_reward_std": ar_s,
        "avg_last3_cost_mean": ac_m,
        "avg_last3_cost_std": ac_s,
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for group, cfg in GROUPS.items():
        for seed in cfg["seeds"]:
            path = LOG_DIR / f"{group}_seed{seed}.log"
            parsed = parse_log(path)
            metrics = summarize_evals(parsed["evals"])  # type: ignore[arg-type]
            row = {
                "group": group,
                "seed": seed,
                "lambda_safe": cfg["lambda_safe"],
                "lambda_jvp": cfg["lambda_jvp"],
                "safe_bandwidth": cfg["safe_bandwidth"],
                "updates_per_step": cfg["updates_per_step"],
                "status": parsed["status"],
                "path": path,
                **metrics,
            }
            for key in TRACE_KEYS:
                row[key] = parsed["traces"].get(key)  # type: ignore[index,union-attr]
            rows.append(row)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PointGoal Diagnostic Summary",
        "",
        "| Group | Seed | lambda_safe | lambda_jvp | safe_bandwidth | updates_per_step | Final Eval Reward | Final Eval Cost | Avg Last 3 Reward | Avg Last 3 Cost | Status | Log Path |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {row['lambda_safe']:g} | {row['lambda_jvp']:g} | "
            f"{row['safe_bandwidth']:g} | {row['updates_per_step']} | {fmt(row['final_reward'])} | "
            f"{fmt(row['final_cost'])} | {fmt(row['avg_last3_reward'])} | {fmt(row['avg_last3_cost'])} | "
            f"{row['status']} | {row['path']} |"
        )

    lines += [
        "",
        "## Group Statistics",
        "",
        "| Group | Completed | Failed | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    stats = {group: group_stats(rows, group) for group in GROUPS}
    for group, stat in stats.items():
        lines.append(
            f"| {group} | {stat['completed']} | {stat['failed']} | "
            f"{fmt(stat['final_reward_mean'])} / {fmt(stat['final_reward_std'])} | "
            f"{fmt(stat['final_cost_mean'])} / {fmt(stat['final_cost_std'])} | "
            f"{fmt(stat['avg_last3_reward_mean'])} / {fmt(stat['avg_last3_reward_std'])} | "
            f"{fmt(stat['avg_last3_cost_mean'])} / {fmt(stat['avg_last3_cost_std'])} |"
        )

    monitor = parse_monitor()
    any_oom = any(row["status"] == "failed_oom" for row in rows)
    any_nan = any(row["status"] == "failed_nan" for row in rows)

    s0_combined_cost = combine_old_new("S0", "PGD0_S0_extend", rows, "avg_last3_cost")
    s0_combined_reward = combine_old_new("S0", "PGD0_S0_extend", rows, "avg_last3_reward")
    s1_combined_cost = combine_old_new("S1", "PGD1_S1_R2D_extend", rows, "avg_last3_cost")
    s1_combined_reward = combine_old_new("S1", "PGD1_S1_R2D_extend", rows, "avg_last3_reward")
    s1_validated = (
        s0_combined_cost is not None
        and s1_combined_cost is not None
        and s0_combined_reward is not None
        and s1_combined_reward is not None
        and s1_combined_cost <= 0.85 * s0_combined_cost
        and s1_combined_reward >= 0.90 * s0_combined_reward
    )

    pgd2_cost = stats["PGD2_R2D_update1"]["avg_last3_cost_mean"]
    pgd2_reward = stats["PGD2_R2D_update1"]["avg_last3_reward_mean"]
    pgd1_cost = stats["PGD1_S1_R2D_extend"]["avg_last3_cost_mean"]
    pgd1_reward = stats["PGD1_S1_R2D_extend"]["avg_last3_reward_mean"]
    pgd3_cost = stats["PGD3_mid_jvp"]["avg_last3_cost_mean"]
    pgd3_reward = stats["PGD3_mid_jvp"]["avg_last3_reward_mean"]

    lines += [
        "",
        "## Diagnostic Traces",
        "",
        "| Group | Seed | safety/qc_mean | safety/qc_target_mean | safety/g_mid_mean | safety/grad_q_norm | loss/jvp_scd | safety/safety_penalty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | "
            f"{fmt(row['safety/qc_mean'])} | {fmt(row['safety/qc_target_mean'])} | "
            f"{fmt(row['safety/g_mid_mean'])} | {fmt(row['safety/grad_q_norm'])} | "
            f"{fmt(row['loss/jvp_scd'])} | {fmt(row['safety/safety_penalty'])} |"
        )

    lines += [
        "",
        "## Comparisons",
        "",
        f"- PGD1 + old S1 combined Avg Last 3 Cost: {fmt(s1_combined_cost)}",
        f"- PGD0 + old S0 combined Avg Last 3 Cost: {fmt(s0_combined_cost)}",
        f"- PGD1 + old S1 combined Avg Last 3 Reward: {fmt(s1_combined_reward)}",
        f"- PGD0 + old S0 combined Avg Last 3 Reward: {fmt(s0_combined_reward)}",
        f"- S1 validated by combined rule: {s1_validated}",
        f"- PGD2 update1 Avg Last 3 Cost/Reward: {fmt(pgd2_cost)} / {fmt(pgd2_reward)}",
        f"- PGD1 update2 Avg Last 3 Cost/Reward: {fmt(pgd1_cost)} / {fmt(pgd1_reward)}",
        f"- PGD3 lambda_jvp=0.003 Avg Last 3 Cost/Reward: {fmt(pgd3_cost)} / {fmt(pgd3_reward)}",
        "",
        "## GPU",
        "",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        f"- OOM observed: {any_oom}",
        f"- NaN observed: {any_nan}",
        "",
        "## Recommendation",
        "",
        "- Move to CarGoal pilot only if the combined diagnostic rule validates S1 or another JVP group clearly beats penalty-only.",
        "- Keep normalized JVP, forward JVP, and soft normal masking disabled for now.",
    ]

    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
