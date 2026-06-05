#!/usr/bin/env python3
"""Collect PointGoal goal-mode tuning metrics and decisions."""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_goal_tuning"
REPORT_DIR = ROOT / "reports" / "pointgoal_goal_tuning"
REPORT = REPORT_DIR / "summary.md"
DECISION_LOG = REPORT_DIR / "decision_log.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

THRESHOLD_COST = 45.64
THRESHOLD_REWARD = 19.85
NEAR_COST = 48.0
NEAR_REWARD = 23.0

GROUPS = {
    "G0_legacy_best": {
        "stage": "A",
        "safety_critic_mode": "cumulative",
        "qc_geom_mode": "max",
        "safe_threshold": 0.1,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "soft_gate": False,
        "risk_gate": False,
        "seeds": [0, 1, 2],
    },
    "G1_cdf_mean": {
        "stage": "A",
        "safety_critic_mode": "cdf",
        "qc_geom_mode": "mean",
        "safe_threshold": 0.1,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "soft_gate": False,
        "risk_gate": False,
        "seeds": [0, 1, 2],
    },
    "G2_cdf_mean_thr005": {
        "stage": "A",
        "safety_critic_mode": "cdf",
        "qc_geom_mode": "mean",
        "safe_threshold": 0.05,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "soft_gate": False,
        "risk_gate": False,
        "seeds": [0, 1, 2],
    },
    "G3_cdf_mean_safe07": {
        "stage": "A",
        "safety_critic_mode": "cdf",
        "qc_geom_mode": "mean",
        "safe_threshold": 0.1,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "soft_gate": False,
        "risk_gate": False,
        "seeds": [0, 1, 2],
    },
    "G4_cdf_mean_thr005_safe07": {
        "stage": "A",
        "safety_critic_mode": "cdf",
        "qc_geom_mode": "mean",
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "soft_gate": False,
        "risk_gate": False,
        "seeds": [0, 1, 2],
    },
    "G5_cdf_mean_jvp0035": {
        "stage": "A",
        "safety_critic_mode": "cdf",
        "qc_geom_mode": "mean",
        "safe_threshold": 0.1,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.0035,
        "soft_gate": False,
        "risk_gate": False,
        "seeds": [0, 1, 2],
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
    "safety/feas_gate_mean",
    "safety/risk_gate_mean",
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


def parse_metric(text: str, key: str) -> float | None:
    values = re.findall(r"wandb:\s+" + re.escape(key) + r"\s+(" + NUM_RE + r")\b", text, flags=re.I)
    return float(values[-1]) if values else None


def parse_log(group: str, seed: int) -> dict[str, object]:
    path = LOG_DIR / f"{group}_seed{seed}.log"
    if not path.exists():
        return {"group": group, "seed": seed, "status": "missing", "path": str(path)}
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
    row = {
        "group": group,
        "seed": seed,
        "status": status,
        "path": str(path),
        "final_reward": evals[-1][0] if evals else None,
        "final_cost": evals[-1][1] if evals else None,
        "avg_last3_reward": mean(v[0] for v in evals[-3:]) if evals else None,
        "avg_last3_cost": mean(v[1] for v in evals[-3:]) if evals else None,
    }
    for key in METRICS:
        row[key] = parse_metric(text, key)
    return row


def metric_mean(rows: list[dict[str, object]], key: str) -> float | None:
    return mean_std([r.get(key) for r in rows])[0]  # type: ignore[list-item]


def group_stats(rows: list[dict[str, object]], group: str, expected: int) -> dict[str, object]:
    group_rows = [r for r in rows if r["group"] == group]
    complete = [r for r in group_rows if r["status"] == "completed"]
    failed = [r for r in group_rows if str(r["status"]).startswith("failed")]
    fr = mean_std([r.get("final_reward") for r in complete])  # type: ignore[list-item]
    fc = mean_std([r.get("final_cost") for r in complete])  # type: ignore[list-item]
    ar = mean_std([r.get("avg_last3_reward") for r in complete])  # type: ignore[list-item]
    ac = mean_std([r.get("avg_last3_cost") for r in complete])  # type: ignore[list-item]
    if failed:
        decision = "failed"
    elif len(complete) < expected:
        decision = "pending"
    elif ac[0] is not None and ar[0] is not None and ac[0] <= THRESHOLD_COST and ar[0] >= THRESHOLD_REWARD:
        decision = "passed"
    elif ac[0] is not None and ar[0] is not None and ac[0] <= NEAR_COST and ar[0] >= NEAR_REWARD:
        decision = "near_pass"
    else:
        decision = "not_passed"
    return {
        "complete": complete,
        "failed": failed,
        "completed_count": len(complete),
        "expected": expected,
        "final_reward": fr,
        "final_cost": fc,
        "avg_last3_reward": ar,
        "avg_last3_cost": ac,
        "decision": decision,
        "status": "failed" if failed else ("completed" if len(complete) >= expected else "pending"),
    }


def fmt_mean_std(pair: tuple[float | None, float | None]) -> str:
    return f"{fmt(pair[0])} / {fmt(pair[1])}"


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


def best_groups(stats: dict[str, dict[str, object]]) -> tuple[str | None, str | None, str | None]:
    completed = [g for g, s in stats.items() if s["completed_count"]]
    if not completed:
        return None, None, None
    best_cost = min(
        completed,
        key=lambda g: stats[g]["avg_last3_cost"][0] if stats[g]["avg_last3_cost"][0] is not None else float("inf"),
    )
    best_tradeoff = min(
        completed,
        key=lambda g: (
            stats[g]["avg_last3_cost"][0] if stats[g]["avg_last3_cost"][0] is not None else float("inf"),
            -(stats[g]["avg_last3_reward"][0] if stats[g]["avg_last3_reward"][0] is not None else float("-inf")),
        ),
    )
    stable = next((g for g, s in stats.items() if s["decision"] == "passed"), None)
    if stable is None:
        stable = next((g for g, s in stats.items() if s["decision"] == "near_pass"), None)
    if stable is None:
        stable = best_tradeoff
    return best_cost, best_tradeoff, stable


def append_decision_log(stats: dict[str, dict[str, object]], best_group: str | None) -> None:
    completed = [g for g, s in stats.items() if s["status"] == "completed"]
    passed = [g for g, s in stats.items() if s["decision"] == "passed"]
    near = [g for g, s in stats.items() if s["decision"] == "near_pass"]
    if passed:
        decision = "passed"
        next_action = f"expand seeds 3,4 for {passed[0]}"
    elif near:
        decision = "near_pass"
        next_action = f"consider expanding {near[0]} or Stage B"
    elif all(s["status"] in ("completed", "failed") for s in stats.values()):
        decision = "stage_a_no_pass"
        next_action = "consider Stage B soft feasibility gate"
    else:
        decision = "pending"
        next_action = "wait for Stage A completion"
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG.open("a") as handle:
        handle.write(f"\n## {dt.datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"- completed groups: {', '.join(completed) if completed else 'none'}\n")
        handle.write(f"- best group: {best_group or 'n/a'}\n")
        handle.write(f"- decision: {decision}\n")
        handle.write(f"- next action: {next_action}\n")


def main() -> None:
    rows = [parse_log(group, seed) for group, cfg in GROUPS.items() for seed in cfg["seeds"]]
    stats = {group: group_stats(rows, group, len(cfg["seeds"])) for group, cfg in GROUPS.items()}
    failed_rows = [r for r in rows if str(r["status"]).startswith("failed")]
    monitor = parse_monitor()
    best_cost, best_tradeoff, stable = best_groups(stats)

    lines = [
        "# PointGoal Goal-Mode Tuning Summary",
        "",
        f"Threshold: Avg Last 3 Cost <= {THRESHOLD_COST} and Avg Last 3 Reward >= {THRESHOLD_REWARD}.",
        f"Near-pass: Avg Last 3 Cost <= {NEAR_COST} and Avg Last 3 Reward >= {NEAR_REWARD}.",
        "",
        "Note: Current cumulative path is legacy CDF-like target. C0_cumulative and CDF max are expected to be nearly identical under binary cost; Stage A focuses on geometry, threshold, and safety weights.",
        "",
        "| Group | Stage | Safety Critic Mode | QC Geom Mode | safe_threshold | lambda_safe | lambda_jvp | soft_gate | risk_gate | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP | qc target mean | qc risk over threshold | feas gate mean | risk gate mean | Decision | Status |",
        "|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group, cfg in GROUPS.items():
        group_stat = stats[group]
        complete = group_stat["complete"]  # type: ignore[assignment]
        lines.append(
            f"| {group} | {cfg['stage']} | {cfg['safety_critic_mode']} | {cfg['qc_geom_mode']} | "
            f"{cfg['safe_threshold']} | {cfg['lambda_safe']} | {cfg['lambda_jvp']} | "
            f"{cfg['soft_gate']} | {cfg['risk_gate']} | {group_stat['completed_count']}/{group_stat['expected']} | "
            f"{fmt_mean_std(group_stat['final_reward'])} | {fmt_mean_std(group_stat['final_cost'])} | "
            f"{fmt_mean_std(group_stat['avg_last3_reward'])} | {fmt_mean_std(group_stat['avg_last3_cost'])} | "
            f"{sci(metric_mean(complete, 'loss/jvp_weighted'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_target_mean'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_pi_risk_over_threshold'))} | "
            f"{sci(metric_mean(complete, 'safety/feas_gate_mean'))} | "
            f"{sci(metric_mean(complete, 'safety/risk_gate_mean'))} | "
            f"{group_stat['decision']} | {group_stat['status']} |"
        )

    lines += [
        "",
        "## Per-Run Results",
        "",
        "| Group | Seed | Final Reward | Final Cost | Avg Last 3 Reward | Avg Last 3 Cost | weighted JVP | qc target mean | qc risk over threshold | Status | Log Path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {fmt(row.get('final_reward'))} | {fmt(row.get('final_cost'))} | "
            f"{fmt(row.get('avg_last3_reward'))} | {fmt(row.get('avg_last3_cost'))} | "
            f"{sci(row.get('loss/jvp_weighted'))} | {sci(row.get('safety/qc_target_mean'))} | "
            f"{sci(row.get('safety/qc_pi_risk_over_threshold'))} | {row['status']} | {row['path']} |"
        )

    lines += [
        "",
        "## Decisions",
        f"- Best by Avg Last 3 Cost: {best_cost or 'n/a'}",
        f"- Best by reward-cost tradeoff: {best_tradeoff or 'n/a'}",
        f"- Best stable candidate: {stable or 'n/a'}",
        "- Whether to expand seeds: "
        + ("yes" if any(s["decision"] == "passed" for s in stats.values()) else "not yet"),
        "- Whether to stop: "
        + ("yes" if failed_rows else "no"),
        "",
        "## GPU",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        f"- OOM observed: {any(r['status'] == 'failed_oom' for r in rows)}",
        f"- NaN observed: {any(r['status'] == 'failed_nan' for r in rows)}",
        "",
        "## Error Scan",
    ]
    if failed_rows:
        for row in failed_rows:
            lines.append(f"- {row['path']}: {row['status']}")
    else:
        lines.append("- No error patterns found in parsed logs.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    append_decision_log(stats, stable)
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
