#!/usr/bin/env python3
"""Collect CarGoal1 soft feasibility gate metrics and decisions."""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "cargoal1_soft_gate"
REPORT_DIR = ROOT / "reports" / "cargoal1_soft_gate"
AUTO_DIR = ROOT / "reports" / "auto_goal_tuning"
SUMMARY = REPORT_DIR / "summary.md"
DECISION_LOG = REPORT_DIR / "decision_log.md"
AUTO_SUMMARY = AUTO_DIR / "summary.md"
AUTO_DECISION_LOG = AUTO_DIR / "decision_log.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

G4_REWARD = 17.98
G4_COST = 50.24
C2_REWARD = 18.62
C2_COST = 63.62
PPO_REWARD = 25.49
PPO_COST = 54.218

GROUPS = {
    "SG_CG1_1_C2_tau010_floor02": {
        "base": "C2",
        "safe_threshold": 0.05,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "feas_gate_tau": 0.10,
        "feas_gate_reward_floor": 0.2,
        "planned_seeds": [0, 1],
    },
    "SG_CG1_2_C2_tau010_floor03": {
        "base": "C2",
        "safe_threshold": 0.05,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "feas_gate_tau": 0.10,
        "feas_gate_reward_floor": 0.3,
        "planned_seeds": [0, 1],
    },
    "SG_CG1_3_G4_tau010_floor02": {
        "base": "G4",
        "safe_threshold": 0.05,
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "feas_gate_tau": 0.10,
        "feas_gate_reward_floor": 0.2,
        "planned_seeds": [0, 1],
    },
    "SG_CG1_4_C2_tau005_floor02": {
        "base": "C2",
        "safe_threshold": 0.05,
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "safe_bandwidth": 0.05,
        "feas_gate_tau": 0.05,
        "feas_gate_reward_floor": 0.2,
        "planned_seeds": [0, 1],
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
    "safety/feas_gate_mean",
    "safety/reward_weight_mean",
    "safety/safe_weight_mean",
    "safety/feas_gate_risky_frac",
    "safety/qc_pi_risk_over_threshold",
    "loss/jvp_weighted",
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


def discover_seeds(group: str, planned: list[int]) -> list[int]:
    seeds = set(planned)
    if LOG_DIR.exists():
        prefix = f"{group}_seed"
        for path in LOG_DIR.glob(f"{prefix}*.log"):
            suffix = path.name.removeprefix(prefix).removesuffix(".log")
            if suffix.isdigit():
                seeds.add(int(suffix))
    return sorted(seeds)


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
        status = "no_evals"
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
    if avg_reward is None or avg_cost is None:
        return "no_data"
    if avg_reward >= 20.0 and avg_cost <= PPO_COST:
        return "ppo_competitive"
    if avg_reward >= 19.5 and avg_cost <= PPO_COST:
        return "near_ppo_competitive"
    if avg_reward >= 18.0 and avg_cost <= 50.0:
        return "pilot_good"
    if avg_reward > C2_REWARD and avg_cost < C2_COST:
        return "improved_candidate"
    if avg_reward >= 16.0 and avg_cost <= 60.0:
        return "pilot_ok"
    if avg_cost > 80.0 or avg_reward < 12.0:
        return "bad"
    return "not_good"


def group_stats(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    group_rows = [row for row in rows if row["group"] == group]
    complete = [row for row in group_rows if row["status"] == "completed"]
    failed = [row for row in group_rows if str(row["status"]).startswith("failed")]
    expected = len(group_rows)
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
        "feas_gate_mean": metric_mean(complete, "safety/feas_gate_mean"),
        "reward_weight_mean": metric_mean(complete, "safety/reward_weight_mean"),
        "safe_weight_mean": metric_mean(complete, "safety/safe_weight_mean"),
        "feas_gate_risky_frac": metric_mean(complete, "safety/feas_gate_risky_frac"),
        "qc_risk_over_threshold": metric_mean(complete, "safety/qc_pi_risk_over_threshold"),
        "weighted_jvp": metric_mean(complete, "loss/jvp_weighted"),
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
            clean = {key.strip(): value for key, value in row.items() if key is not None}
            memory_used = numeric(clean.get("memory.used [MiB]"))
            memory_total = numeric(clean.get("memory.total [MiB]"))
            gpu_util = numeric(clean.get("utilization.gpu [%]"))
            if memory_used is not None:
                used.append(memory_used)
            if memory_total is not None:
                total = memory_total
            if gpu_util is not None:
                util.append(gpu_util)
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
    rank = {
        "ppo_competitive": 5,
        "near_ppo_competitive": 4,
        "pilot_good": 3,
        "improved_candidate": 2,
        "pilot_ok": 1,
    }
    return max(
        ranked,
        key=lambda group: (
            rank.get(str(stats[group]["decision"]), 0),
            -(stats[group]["avg_last3_cost"][0] or float("inf")),
            stats[group]["avg_last3_reward"][0] or float("-inf"),
        ),
    )


def write_auto_reports(stats: dict[str, dict[str, object]], best: str) -> None:
    AUTO_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Autonomous Goal Tuning Summary",
        "",
        "| Stage | Group | Seeds | Reward | Cost | Decision |",
        "| ----- | ----- | ----: | -----: | ---: | -------- |",
        f"| Reference | G4_fixed_main | 3 | {G4_REWARD:.2f} | {G4_COST:.2f} | current_best_reference |",
        f"| Reference | CG1_C2_safe05 | 4 | {C2_REWARD:.2f} | {C2_COST:.2f} | parameter_search_bottleneck |",
    ]
    for group, stat in stats.items():
        rows.append(
            f"| Stage B | {group} | {stat['completed_count']}/{stat['expected']} | "
            f"{fmt_mean_std(stat['avg_last3_reward'])} | {fmt_mean_std(stat['avg_last3_cost'])} | {stat['decision']} |"
        )
    AUTO_SUMMARY.write_text("\n".join(rows) + "\n")

    completed = [group for group, stat in stats.items() if stat["status"] == "completed"]
    expand = [
        group
        for group, stat in stats.items()
        if stat["decision"] in ("ppo_competitive", "near_ppo_competitive", "pilot_good")
    ]
    all_done = all(stat["status"] in ("completed", "failed") for stat in stats.values())
    if expand:
        decision = "expand seed2"
        reason = "At least one Stage B group reached near/PPO/pilot_good criteria."
        next_action = ", ".join(f"{group} seed2" for group in expand)
    elif all_done:
        decision = "next stage"
        reason = "No Stage B group reached usable criteria."
        next_action = "Implement Stage C risk-side gate."
    else:
        decision = "continue"
        reason = "Stage B runs are still pending."
        next_action = "Wait for Stage B completion."

    decision_lines = [
        "# Autonomous Goal Tuning Decision Log",
        "",
        f"timestamp: {dt.datetime.now().isoformat(timespec='seconds')}",
        "stage: Stage B - soft feasibility gate",
        f"completed groups: {', '.join(completed) if completed else 'none'}",
        f"best group: {best}",
        f"comparison against G4_fixed: reward={G4_REWARD:.2f}, cost={G4_COST:.2f}",
        f"comparison against C2: reward={C2_REWARD:.2f}, cost={C2_COST:.2f}",
        f"comparison against PPO: reward={PPO_REWARD:.2f}, cost={PPO_COST:.3f}",
        f"decision: {decision}",
        f"reason: {reason}",
        f"next action: {next_action}",
    ]
    AUTO_DECISION_LOG.write_text("\n".join(decision_lines) + "\n")


def main() -> None:
    group_seeds = {
        group: discover_seeds(group, list(cfg["planned_seeds"]))  # type: ignore[arg-type]
        for group, cfg in GROUPS.items()
    }
    rows = [parse_log(group, seed) for group, seeds in group_seeds.items() for seed in seeds]
    stats = {group: group_stats(rows, group) for group in GROUPS}
    monitor = parse_monitor()
    best = best_group(stats)
    expand_groups = [
        group
        for group, stat in stats.items()
        if stat["decision"] in ("ppo_competitive", "near_ppo_competitive", "pilot_good")
    ]

    lines = [
        "# CarGoal1 Soft Feasibility Gate Summary",
        "",
        "PPO-competitive: Avg Last 3 Reward >= 20.0 and Avg Last 3 Cost <= 54.218.",
        "Near PPO-competitive: Avg Last 3 Reward >= 19.5 and Avg Last 3 Cost <= 54.218.",
        "Pilot good: Avg Last 3 Reward >= 18.0 and Avg Last 3 Cost <= 50.0.",
        "Improved candidate: Avg Last 3 Reward > 18.62 and Avg Last 3 Cost < 63.62.",
        f"G4 fixed reference: Avg Last 3 Reward = {G4_REWARD}, Avg Last 3 Cost = {G4_COST}.",
        f"C2 reference: Avg Last 3 Reward = {C2_REWARD}, Avg Last 3 Cost = {C2_COST}.",
        f"PPO baseline reference: Ret = {PPO_REWARD}, Cost = {PPO_COST}.",
        "",
        "| Group | Base | safe_threshold | lambda_safe | tau | floor | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | feas gate | reward weight | safe weight | risky frac | weighted JVP | Status | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group, cfg in GROUPS.items():
        stat = stats[group]
        lines.append(
            f"| {group} | {cfg['base']} | {cfg['safe_threshold']} | {cfg['lambda_safe']} | "
            f"{cfg['feas_gate_tau']} | {cfg['feas_gate_reward_floor']} | {stat['completed_count']}/{stat['expected']} | "
            f"{fmt_mean_std(stat['final_reward'])} | {fmt_mean_std(stat['final_cost'])} | "
            f"{fmt_mean_std(stat['avg_last3_reward'])} | {fmt_mean_std(stat['avg_last3_cost'])} | "
            f"{sci(stat['feas_gate_mean'])} | {sci(stat['reward_weight_mean'])} | {sci(stat['safe_weight_mean'])} | "
            f"{sci(stat['feas_gate_risky_frac'])} | {sci(stat['weighted_jvp'])} | {stat['status']} | {stat['decision']} |"
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
        f"- Best Stage B group: {best}",
        "- Whether to expand seeds: " + (", ".join(f"{group} seed2" for group in expand_groups) if expand_groups else "no"),
        "- Whether Stage C is needed: " + ("no, expand usable Stage B candidate first" if expand_groups else "yes if all Stage B groups are completed/failed"),
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
        "# CarGoal1 Soft Feasibility Gate Decision Log",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Best Stage B group: {best}",
        "- Expand seeds: " + (", ".join(f"{group} seed2" for group in expand_groups) if expand_groups else "no"),
    ]
    for group, stat in stats.items():
        decision_lines.append(
            f"- {group}: seeds={stat['completed_count']}/{stat['expected']}, "
            f"avg_last3_reward={fmt_mean_std(stat['avg_last3_reward'])}, "
            f"avg_last3_cost={fmt_mean_std(stat['avg_last3_cost'])}, decision={stat['decision']}"
        )
    DECISION_LOG.write_text("\n".join(decision_lines) + "\n")
    write_auto_reports(stats, best)

    print(SUMMARY)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
