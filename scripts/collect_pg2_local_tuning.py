#!/usr/bin/env python3
"""Collect PointGoal2 local tuning metrics and decisions."""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pg2_local_tuning"
REPORT_DIR = ROOT / "reports" / "pg2_local_tuning"
SUMMARY = REPORT_DIR / "summary.md"
DECISION_LOG = REPORT_DIR / "decision_log.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

PASS_COST = 45.64
PASS_REWARD = 19.85
IMPROVED_COST = 60.0
IMPROVED_REWARD = 18.0
BAD_COST = 100.0
BAD_REWARD = 15.0

GROUPS = {
    "PG2_A1_less_conservative": {
        "safe_threshold": 0.10,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 120000,
        "seeds": [0, 1],
    },
    "PG2_A2_stronger_safe": {
        "safe_threshold": 0.05,
        "lambda_safe": 1.0,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 120000,
        "seeds": [0, 1],
    },
    "PG2_A3_stronger_jvp": {
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.0035,
        "safe_bandwidth": 0.05,
        "num_steps": 120000,
        "seeds": [0, 1],
    },
    "PG2_A4_wider_band": {
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.075,
        "num_steps": 120000,
        "seeds": [0, 1],
    },
    "PG2_A5_longer_G4": {
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "num_steps": 200000,
        "seeds": [0, 1],
    },
}

EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")
NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
METRICS = [
    "loss/jvp_weighted",
    "safety/qc_target_mean",
    "safety/qc_pi_risk_over_threshold",
]


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def sci(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3e}"


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), stdev(clean)


def fmt_mean_std(pair: tuple[float | None, float | None]) -> str:
    return f"{fmt(pair[0])} / {fmt(pair[1])}"


def parse_metric(text: str, key: str) -> float | None:
    values = re.findall(r"wandb:\s+" + re.escape(key) + r"\s+(" + NUM_RE + r")\b", text, flags=re.I)
    return float(values[-1]) if values else None


def parse_log(group: str, seed: int) -> dict[str, object]:
    path = LOG_DIR / f"{group}_seed{seed}.log"
    row: dict[str, object] = {"group": group, "seed": seed, "path": str(path)}
    if not path.exists():
        row["status"] = "missing"
        return row
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2))) for m in EVAL_RE.finditer(text)]
    lowered = text.lower()
    if "oom" in lowered or "out of memory" in lowered:
        status = "failed_oom"
    elif "nan" in text or "NaN" in text:
        status = "failed_nan"
    elif ERR_RE.search(text):
        status = "failed_error"
    elif f" END {group} seed={seed} " in text and evals:
        status = "completed"
    elif evals:
        status = "partial"
    else:
        status = "no evals"
    row.update(
        {
            "status": status,
            "final_reward": evals[-1][0] if evals else None,
            "final_cost": evals[-1][1] if evals else None,
            "avg_last3_reward": mean(v[0] for v in evals[-3:]) if evals else None,
            "avg_last3_cost": mean(v[1] for v in evals[-3:]) if evals else None,
        }
    )
    for key in METRICS:
        row[key] = parse_metric(text, key)
    return row


def metric_mean(rows: list[dict[str, object]], key: str) -> float | None:
    return mean_std([row.get(key) for row in rows])[0]  # type: ignore[list-item]


def decide(avg_reward: float | None, avg_cost: float | None, complete: int, expected: int, failed: bool) -> str:
    if failed:
        return "failed"
    if complete < expected:
        return "pending"
    if avg_cost is None or avg_reward is None:
        return "no_data"
    if avg_cost <= PASS_COST and avg_reward >= PASS_REWARD:
        return "passed"
    if avg_cost <= IMPROVED_COST and avg_reward >= IMPROVED_REWARD:
        return "improved"
    if avg_cost > BAD_COST or avg_reward < BAD_REWARD:
        return "bad"
    return "not_passed"


def group_stats(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    group_rows = [row for row in rows if row["group"] == group]
    complete = [row for row in group_rows if row["status"] == "completed"]
    failed = [row for row in group_rows if str(row["status"]).startswith("failed")]
    expected = len(GROUPS[group]["seeds"])
    fr = mean_std([row.get("final_reward") for row in complete])  # type: ignore[list-item]
    fc = mean_std([row.get("final_cost") for row in complete])  # type: ignore[list-item]
    ar = mean_std([row.get("avg_last3_reward") for row in complete])  # type: ignore[list-item]
    ac = mean_std([row.get("avg_last3_cost") for row in complete])  # type: ignore[list-item]
    return {
        "rows": group_rows,
        "complete": complete,
        "failed": failed,
        "completed_count": len(complete),
        "expected": expected,
        "final_reward": fr,
        "final_cost": fc,
        "avg_last3_reward": ar,
        "avg_last3_cost": ac,
        "weighted_jvp": metric_mean(complete, "loss/jvp_weighted"),
        "qc_target_mean": metric_mean(complete, "safety/qc_target_mean"),
        "qc_risk_over_threshold": metric_mean(complete, "safety/qc_pi_risk_over_threshold"),
        "status": "failed" if failed else ("completed" if len(complete) >= expected else "pending"),
        "decision": decide(ar[0], ac[0], len(complete), expected, bool(failed)),
    }


def numeric(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(NUM_RE, value)
    return float(match.group(0)) if match else None


def parse_monitor() -> dict[str, str]:
    if not MONITOR.exists():
        return {}
    used: list[float] = []
    util: list[float] = []
    total = None
    with MONITOR.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_row = {key.strip(): value for key, value in row.items() if key is not None}
            memory_used = numeric(clean_row.get("memory.used [MiB]"))
            memory_total = numeric(clean_row.get("memory.total [MiB]"))
            gpu_util = numeric(clean_row.get("utilization.gpu [%]"))
            if memory_used is not None:
                used.append(memory_used)
            if gpu_util is not None:
                util.append(gpu_util)
            if memory_total is not None:
                total = memory_total
    if not used:
        return {}
    return {
        "gpu_total_memory": fmt(total / 1024 if total else None),
        "peak_memory_used": fmt(max(used) / 1024),
        "avg_memory_used": fmt(sum(used) / len(used) / 1024),
        "avg_gpu_utilization": fmt(sum(util) / len(util)) if util else "n/a",
    }


def best_group(stats: dict[str, dict[str, object]]) -> str:
    ranked = [
        group for group, stat in stats.items() if stat["completed_count"] and stat["avg_last3_reward"][0] is not None
    ]
    if not ranked:
        return "n/a"
    preferred = [group for group in ranked if stats[group]["decision"] in ("passed", "improved")]
    pool = preferred if preferred else ranked
    return max(
        pool,
        key=lambda group: (
            stats[group]["decision"] == "passed",
            stats[group]["decision"] == "improved",
            -(stats[group]["avg_last3_cost"][0] or float("inf")),
            stats[group]["avg_last3_reward"][0] or float("-inf"),
        ),
    )


def main() -> None:
    rows = [parse_log(group, seed) for group, cfg in GROUPS.items() for seed in cfg["seeds"]]
    stats = {group: group_stats(rows, group) for group in GROUPS}
    monitor = parse_monitor()
    best = best_group(stats)
    any_passed = any(stat["decision"] == "passed" for stat in stats.values())
    any_improved = any(stat["decision"] == "improved" for stat in stats.values())

    lines = [
        "# PointGoal2 Local Tuning Summary",
        "",
        f"Passed: Avg Last 3 Cost <= {PASS_COST} and Avg Last 3 Reward >= {PASS_REWARD}.",
        f"Improved: Avg Last 3 Cost <= {IMPROVED_COST} and Avg Last 3 Reward >= {IMPROVED_REWARD}.",
        f"Bad: Avg Last 3 Cost > {BAD_COST} or Avg Last 3 Reward < {BAD_REWARD}.",
        "",
        "| Group | safe_threshold | lambda_safe | lambda_jvp | safe_bandwidth | num_steps | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP | qc target mean | qc risk over threshold | Status | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group, cfg in GROUPS.items():
        stat = stats[group]
        lines.append(
            f"| {group} | {cfg['safe_threshold']} | {cfg['lambda_safe']} | {cfg['lambda_jvp']} | "
            f"{cfg['safe_bandwidth']} | {cfg['num_steps']} | {stat['completed_count']}/{stat['expected']} | "
            f"{fmt_mean_std(stat['final_reward'])} | {fmt_mean_std(stat['final_cost'])} | "
            f"{fmt_mean_std(stat['avg_last3_reward'])} | {fmt_mean_std(stat['avg_last3_cost'])} | "
            f"{sci(stat['weighted_jvp'])} | {sci(stat['qc_target_mean'])} | {sci(stat['qc_risk_over_threshold'])} | "
            f"{stat['status']} | {stat['decision']} |"
        )

    lines += [
        "",
        "## Per-Run Results",
        "",
        "| Group | Seed | Final Reward | Final Cost | Avg Last 3 Reward | Avg Last 3 Cost | Status | Log Path |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {fmt(row.get('final_reward'))} | {fmt(row.get('final_cost'))} | "
            f"{fmt(row.get('avg_last3_reward'))} | {fmt(row.get('avg_last3_cost'))} | {row['status']} | {row['path']} |"
        )

    lines += [
        "",
        "## Decisions",
        f"- Best PointGoal2 group: {best}",
        f"- Whether PointGoal2 passed: {'yes' if any_passed else 'no'}",
        f"- Whether PointGoal2 improved: {'yes' if any_improved else 'no'}",
        "- Whether to expand seeds: "
        + ("passed group seeds 2,3,4" if any_passed else ("best improved group seed2" if any_improved else "no")),
        "- Whether Stage B is needed: no automatic Stage B; wait for decision.",
        "",
        "## GPU",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        "",
        "## Error Scan",
    ]
    failed_rows = [row for row in rows if str(row["status"]).startswith("failed")]
    if failed_rows:
        for row in failed_rows:
            lines.append(f"- {row['path']}: {row['status']}")
    else:
        lines.append("- No error patterns found in parsed logs.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n")

    decision_lines = [
        "# PointGoal2 Local Tuning Decision Log",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Best PointGoal2 group: {best}",
        f"- PointGoal2 passed: {'yes' if any_passed else 'no'}",
        f"- PointGoal2 improved: {'yes' if any_improved else 'no'}",
    ]
    for group, stat in stats.items():
        decision_lines.append(
            f"- {group}: seeds={stat['completed_count']}/{stat['expected']}, "
            f"avg_last3_reward={fmt_mean_std(stat['avg_last3_reward'])}, "
            f"avg_last3_cost={fmt_mean_std(stat['avg_last3_cost'])}, decision={stat['decision']}"
        )
    DECISION_LOG.write_text("\n".join(decision_lines) + "\n")

    print(SUMMARY)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
