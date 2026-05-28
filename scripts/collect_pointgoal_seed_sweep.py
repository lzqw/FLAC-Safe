#!/usr/bin/env python3
"""Collect PointGoal seed-sweep metrics."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_seed_sweep"
REPORT = ROOT / "reports" / "pointgoal_seed_sweep" / "summary.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

GROUPS = {
    "S0_penalty_only": {"lambda_safe": 0.5, "lambda_jvp": 0.0, "safe_bandwidth": 0.05},
    "S1_main_R2D": {"lambda_safe": 0.5, "lambda_jvp": 0.005, "safe_bandwidth": 0.05},
    "S2_bw010_R2E": {"lambda_safe": 0.5, "lambda_jvp": 0.001, "safe_bandwidth": 0.10},
}
SEEDS = [0, 1, 2, 3, 4]

EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def parse_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "evals": []}
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in EVAL_RE.finditer(text)]
    err = ERR_RE.search(text)
    lowered = text.lower()
    if "oom" in lowered or "out of memory" in lowered:
        status = "failed_oom"
    elif "nan" in text or "NaN" in text:
        status = "failed_nan"
    elif err:
        status = "failed_error"
    elif " END " in text and evals:
        status = "completed"
    elif evals:
        status = "partial"
    else:
        status = "no evals"
    return {"status": status, "evals": evals}


def summarize_evals(evals: list[tuple[float, float, float]]) -> dict[str, float | None]:
    if not evals:
        return {
            "final_reward": None,
            "final_cost": None,
            "avg_last3_reward": None,
            "avg_last3_cost": None,
        }
    final_reward, final_cost, _ = evals[-1]
    last3 = evals[-3:]
    return {
        "final_reward": final_reward,
        "final_cost": final_cost,
        "avg_last3_reward": sum(row[0] for row in last3) / len(last3),
        "avg_last3_cost": sum(row[1] for row in last3) / len(last3),
    }


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), stdev(clean)


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


def main() -> None:
    rows = []
    for group, cfg in GROUPS.items():
        for seed in SEEDS:
            tag = f"{group}_seed{seed}"
            path = LOG_DIR / f"{tag}.log"
            parsed = parse_log(path)
            metrics = summarize_evals(parsed["evals"])  # type: ignore[arg-type]
            rows.append({"group": group, "seed": seed, "path": path, "status": parsed["status"], **cfg, **metrics})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PointGoal Seed Sweep Summary",
        "",
        "| Group | Seed | lambda_safe | lambda_jvp | safe_bandwidth | Final Eval Reward | Final Eval Cost | Avg Last 3 Reward | Avg Last 3 Cost | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {row['lambda_safe']:g} | {row['lambda_jvp']:g} | "
            f"{row['safe_bandwidth']:g} | {fmt(row['final_reward'])} | {fmt(row['final_cost'])} | "
            f"{fmt(row['avg_last3_reward'])} | {fmt(row['avg_last3_cost'])} | {row['status']} |"
        )

    lines += [
        "",
        "## Group Statistics",
        "",
        "| Group | Success | Failure | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        complete = [row for row in group_rows if row["status"] == "completed"]
        failures = len(group_rows) - len(complete)
        fr_m, fr_s = mean_std([row["final_reward"] for row in complete])
        fc_m, fc_s = mean_std([row["final_cost"] for row in complete])
        ar_m, ar_s = mean_std([row["avg_last3_reward"] for row in complete])
        ac_m, ac_s = mean_std([row["avg_last3_cost"] for row in complete])
        lines.append(
            f"| {group} | {len(complete)} | {failures} | {fmt(fr_m)} / {fmt(fr_s)} | "
            f"{fmt(fc_m)} / {fmt(fc_s)} | {fmt(ar_m)} / {fmt(ar_s)} | {fmt(ac_m)} / {fmt(ac_s)} |"
        )

    monitor = parse_monitor()
    any_oom = any(row["status"] == "failed_oom" for row in rows)
    any_nan = any(row["status"] == "failed_nan" for row in rows)
    completed_rows = [row for row in rows if row["status"] == "completed"]
    best_tradeoff = max(
        completed_rows,
        key=lambda row: (row["avg_last3_reward"] or -999) - 0.05 * (row["avg_last3_cost"] or 999),
        default=None,
    )
    lowest_cost = min(completed_rows, key=lambda row: row["avg_last3_cost"] or float("inf"), default=None)
    highest_reward = max(completed_rows, key=lambda row: row["final_reward"] or float("-inf"), default=None)

    lines += [
        "",
        "## GPU Scheduling",
        "",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        f"- OOM observed: {any_oom}",
        f"- NaN observed: {any_nan}",
        "",
        "## Conclusion",
        "",
        f"- Lowest Avg Last 3 Cost: {lowest_cost['group'] + ' seed ' + str(lowest_cost['seed']) if lowest_cost else 'n/a'}",
        f"- Highest Final Reward: {highest_reward['group'] + ' seed ' + str(highest_reward['seed']) if highest_reward else 'n/a'}",
        f"- Best reward/cost tradeoff: {best_tradeoff['group'] + ' seed ' + str(best_tradeoff['seed']) if best_tradeoff else 'n/a'}",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
