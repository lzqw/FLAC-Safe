#!/usr/bin/env python3
"""Collect high-fidelity safety-Q CarGoal1 metrics."""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "high_fidelity_q_cargoal1"
REPORT_DIR = ROOT / "reports" / "high_fidelity_q_cargoal1"
SUMMARY = REPORT_DIR / "summary.md"
DECISION_LOG = REPORT_DIR / "decision_log.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

G4_REWARD = 17.98
G4_COST = 50.24
C2_REWARD = 18.62
C2_COST = 63.62
C4_REWARD = 20.76
C4_COST = 70.57
PPO_REWARD = 25.49
PPO_COST = 54.218

GROUPS = {
    "HQC_CG1_1_G4_priority": {
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safety_q_extra_updates": 0,
        "planned_seeds": [0, 1],
    },
    "HQC_CG1_2_C2_priority": {
        "lambda_safe": 0.5,
        "lambda_jvp": 0.003,
        "safety_q_extra_updates": 0,
        "planned_seeds": [0, 1],
    },
    "HQC_CG1_3_G4_priority_extra1": {
        "lambda_safe": 0.7,
        "lambda_jvp": 0.003,
        "safety_q_extra_updates": 1,
        "planned_seeds": [0, 1],
    },
    "HQC_CG1_4_C2_priority_weak_jvp": {
        "lambda_safe": 0.5,
        "lambda_jvp": 0.001,
        "safety_q_extra_updates": 1,
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
    "safety_q/weight_mean",
    "safety_q/boundary_mask_frac",
    "safety_q/cost_mask_frac",
    "safety_q/grad_norm_mean",
    "safety_q/zero_grad_frac",
    "safety_q/mono_plus_frac",
    "safety_q/mono_minus_frac",
    "safety_q/fd_slope_mean",
    "safety_q/jvp_mean",
    "safety_q/normalized_jvp_mean",
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
    values.extend(re.findall(re.escape(key) + r"=(" + NUM_RE + r")\b", text))
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


def geometry_improved(stat: dict[str, object]) -> bool:
    mono_plus = stat.get("mono_plus_frac")
    mono_minus = stat.get("mono_minus_frac")
    fd_slope = stat.get("fd_slope_mean")
    zero_grad = stat.get("zero_grad_frac")
    if None in (mono_plus, mono_minus, fd_slope, zero_grad):
        return False
    return (
        float(mono_plus) > 0.60
        and float(mono_minus) > 0.60
        and float(fd_slope) > 0.0
        and float(zero_grad) < 0.30
    )


def decide(avg_reward: float | None, avg_cost: float | None, complete: int, expected: int, failed: bool, stat: dict[str, object]) -> str:
    if failed:
        return "failed"
    if complete < expected:
        return "pending"
    if avg_reward is None or avg_cost is None:
        return "no_data"
    zero_grad = stat.get("zero_grad_frac")
    if avg_reward < 12.0 or avg_cost > 90.0 or (zero_grad is not None and float(zero_grad) > 0.50):
        return "bad"
    if avg_reward >= 20.0 and avg_cost <= PPO_COST:
        return "ppo_competitive"
    if avg_reward >= 19.5 and avg_cost <= PPO_COST:
        return "near_ppo_competitive"
    if avg_reward >= 18.0 and avg_cost <= 50.0:
        return "pilot_good"
    if avg_reward > C2_REWARD and avg_cost < C2_COST and geometry_improved(stat):
        return "improved_candidate"
    if geometry_improved(stat):
        return "geometry_improved_only"
    return "not_good"


def group_stats(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    group_rows = [row for row in rows if row["group"] == group]
    complete = [row for row in group_rows if row["status"] == "completed"]
    failed = [row for row in group_rows if str(row["status"]).startswith("failed")]
    expected = len(group_rows)
    stat = {
        "rows": group_rows,
        "complete": complete,
        "failed": failed,
        "completed_count": len(complete),
        "expected": expected,
        "final_reward": mean_std([row.get("final_reward") for row in complete]),  # type: ignore[list-item]
        "final_cost": mean_std([row.get("final_cost") for row in complete]),  # type: ignore[list-item]
        "avg_last3_reward": mean_std([row.get("avg_last3_reward") for row in complete]),  # type: ignore[list-item]
        "avg_last3_cost": mean_std([row.get("avg_last3_cost") for row in complete]),  # type: ignore[list-item]
        "safety_q_weight_mean": metric_mean(complete, "safety_q/weight_mean"),
        "safety_q_boundary_frac": metric_mean(complete, "safety_q/boundary_mask_frac"),
        "safety_q_cost_frac": metric_mean(complete, "safety_q/cost_mask_frac"),
        "grad_norm_mean": metric_mean(complete, "safety_q/grad_norm_mean"),
        "zero_grad_frac": metric_mean(complete, "safety_q/zero_grad_frac"),
        "mono_plus_frac": metric_mean(complete, "safety_q/mono_plus_frac"),
        "mono_minus_frac": metric_mean(complete, "safety_q/mono_minus_frac"),
        "fd_slope_mean": metric_mean(complete, "safety_q/fd_slope_mean"),
        "jvp_mean": metric_mean(complete, "safety_q/jvp_mean"),
        "normalized_jvp_mean": metric_mean(complete, "safety_q/normalized_jvp_mean"),
        "weighted_jvp": metric_mean(complete, "loss/jvp_weighted"),
        "status": "failed" if failed else ("completed" if len(complete) >= expected else "pending"),
    }
    stat["geometry_decision"] = "geometry_improved" if geometry_improved(stat) else "geometry_not_improved"
    stat["decision"] = decide(
        stat["avg_last3_reward"][0],  # type: ignore[index]
        stat["avg_last3_cost"][0],  # type: ignore[index]
        len(complete),
        expected,
        bool(failed),
        stat,
    )
    return stat


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
    decision_rank = {
        "ppo_competitive": 6,
        "near_ppo_competitive": 5,
        "pilot_good": 4,
        "improved_candidate": 3,
        "geometry_improved_only": 2,
        "not_good": 1,
    }
    return max(
        ranked,
        key=lambda group: (
            decision_rank.get(str(stats[group]["decision"]), 0),
            -(stats[group]["avg_last3_cost"][0] or float("inf")),
            stats[group]["avg_last3_reward"][0] or float("-inf"),
        ),
    )


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
        group for group, stat in stats.items()
        if stat["decision"] in ("ppo_competitive", "near_ppo_competitive", "pilot_good", "improved_candidate")
    ]

    lines = [
        "# High-Fidelity Safety-Q CarGoal1 Summary",
        "",
        f"G4 fixed: Reward = {G4_REWARD}, Cost = {G4_COST}.",
        f"C2: Reward = {C2_REWARD}, Cost = {C2_COST}.",
        f"C4 long: Reward = {C4_REWARD}, Cost = {C4_COST}.",
        f"PPO baseline: Reward = {PPO_REWARD}, Cost = {PPO_COST}.",
        "",
        "| Group | Seeds | Reward | Cost | lambda_safe | lambda_jvp | safety_q_extra_updates | safety_q_weight_mean | safety_q_boundary_frac | safety_q_cost_frac | grad_norm_mean | zero_grad_frac | mono_plus_frac | mono_minus_frac | fd_slope_mean | jvp_mean | Decision |",
        "| ----- | ----: | -----: | ---: | ----------: | ---------: | ---------------------: | -------------------: | ---------------------: | -----------------: | -------------: | -------------: | -------------: | --------------: | ------------: | -------: | -------- |",
    ]
    for group, cfg in GROUPS.items():
        stat = stats[group]
        lines.append(
            f"| {group} | {stat['completed_count']}/{stat['expected']} | "
            f"{fmt_mean_std(stat['avg_last3_reward'])} | {fmt_mean_std(stat['avg_last3_cost'])} | "
            f"{cfg['lambda_safe']} | {cfg['lambda_jvp']} | {cfg['safety_q_extra_updates']} | "
            f"{fmt(stat['safety_q_weight_mean'])} | {fmt(stat['safety_q_boundary_frac'])} | {fmt(stat['safety_q_cost_frac'])} | "
            f"{sci(stat['grad_norm_mean'])} | {fmt(stat['zero_grad_frac'])} | {fmt(stat['mono_plus_frac'])} | "
            f"{fmt(stat['mono_minus_frac'])} | {sci(stat['fd_slope_mean'])} | {sci(stat['jvp_mean'])} | {stat['decision']} |"
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
        f"- Best group: {best}",
        "- Expand seeds: " + (", ".join(f"{group} seed2" for group in expand_groups) if expand_groups else "no"),
        "- If geometry improves but performance does not, next branch should tune actor/JVP schedule.",
        "- If geometry does not improve, next HQC round should increase boundary weight to 5.0 or extra updates to 2.",
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
        "# High-Fidelity Safety-Q CarGoal1 Decision Log",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Best group: {best}",
        "- Expand seeds: " + (", ".join(f"{group} seed2" for group in expand_groups) if expand_groups else "no"),
    ]
    for group, stat in stats.items():
        decision_lines.append(
            f"- {group}: seeds={stat['completed_count']}/{stat['expected']}, "
            f"reward={fmt_mean_std(stat['avg_last3_reward'])}, cost={fmt_mean_std(stat['avg_last3_cost'])}, "
            f"geometry={stat['geometry_decision']}, decision={stat['decision']}"
        )
    DECISION_LOG.write_text("\n".join(decision_lines) + "\n")

    print(SUMMARY)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
