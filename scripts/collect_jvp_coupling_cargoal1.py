#!/usr/bin/env python3
"""Collect JVP coupling CarGoal1 experiment metrics."""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "jvp_coupling_cargoal1"
REPORT_DIR = ROOT / "reports" / "jvp_coupling_cargoal1"
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
    "JSC_CG1_1_C2_safe045_bw025": {
        "lambda_safe": 0.45,
        "safe_bandwidth": 0.025,
        "lambda_jvp_start": 0.001,
        "lambda_jvp_end": 0.003,
        "planned_seeds": [0, 1],
    },
    "JSC_CG1_2_C2_safe045_bw030": {
        "lambda_safe": 0.45,
        "safe_bandwidth": 0.030,
        "lambda_jvp_start": 0.001,
        "lambda_jvp_end": 0.003,
        "planned_seeds": [0, 1],
    },
    "JSC_CG1_3_C2_safe050_bw025": {
        "lambda_safe": 0.50,
        "safe_bandwidth": 0.025,
        "lambda_jvp_start": 0.001,
        "lambda_jvp_end": 0.003,
        "planned_seeds": [0, 1],
    },
    "JSC_CG1_4_G4_safe060_bw025": {
        "lambda_safe": 0.60,
        "safe_bandwidth": 0.025,
        "lambda_jvp_start": 0.001,
        "lambda_jvp_end": 0.003,
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
    "safety/lambda_jvp_schedule_enabled",
    "safety/lambda_jvp_eff",
    "safety/lambda_jvp_start",
    "safety/lambda_jvp_end",
    "loss/jvp_weighted",
    "safety_q/weight_mean",
    "safety_q/boundary_mask_frac",
    "safety_q/cost_mask_frac",
    "safety_q/diag_qc_geom_mode_id",
    "safety_q/grad_norm_mean",
    "safety_q/zero_grad_frac",
    "safety_q/mono_plus_frac",
    "safety_q/mono_minus_frac",
    "safety_q/fd_slope_mean",
    "safety_q/geom_grad_norm_mean",
    "safety_q/geom_zero_grad_frac",
    "safety_q/geom_mono_plus_frac",
    "safety_q/geom_mono_minus_frac",
    "safety_q/geom_fd_slope_mean",
    "safety_q/max_grad_norm_mean",
    "safety_q/max_zero_grad_frac",
    "safety_q/max_mono_plus_frac",
    "safety_q/max_mono_minus_frac",
    "safety_q/max_fd_slope_mean",
    "safety_q/jvp_mean",
    "safety_q/normalized_jvp_mean",
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
    values = [row.get(key) for row in rows]
    return mean_std([v for v in values if isinstance(v, float)])[0]


def metric_mean_fallback(rows: list[dict[str, object]], primary: str, fallback: str) -> float | None:
    primary_value = metric_mean(rows, primary)
    return primary_value if primary_value is not None else metric_mean(rows, fallback)


def geometry_improved(stat: dict[str, object]) -> bool:
    mono_plus = stat.get("geom_mono_plus_frac")
    mono_minus = stat.get("geom_mono_minus_frac")
    fd_slope = stat.get("geom_fd_slope_mean")
    zero_grad = stat.get("geom_zero_grad_frac")
    if None in (mono_plus, mono_minus, fd_slope, zero_grad):
        return False
    return (
        float(mono_plus) > 0.60
        and float(mono_minus) > 0.60
        and float(fd_slope) > 0.0
        and float(zero_grad) < 0.30
    )


def decide(
    avg_reward: float | None,
    avg_cost: float | None,
    complete: int,
    expected: int,
    failed: bool,
    stat: dict[str, object],
) -> str:
    if failed:
        return "failed"
    if complete < expected:
        return "pending"
    if avg_reward is None or avg_cost is None:
        return "no_data"
    zero_grad = stat.get("geom_zero_grad_frac")
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
    stat: dict[str, object] = {
        "rows": group_rows,
        "complete": complete,
        "failed": failed,
        "completed_count": len(complete),
        "expected": expected,
        "final_reward": mean_std([row.get("final_reward") for row in complete if isinstance(row.get("final_reward"), float)]),
        "final_cost": mean_std([row.get("final_cost") for row in complete if isinstance(row.get("final_cost"), float)]),
        "avg_last3_reward": mean_std(
            [row.get("avg_last3_reward") for row in complete if isinstance(row.get("avg_last3_reward"), float)]
        ),
        "avg_last3_cost": mean_std(
            [row.get("avg_last3_cost") for row in complete if isinstance(row.get("avg_last3_cost"), float)]
        ),
        "lambda_jvp_eff": metric_mean(complete, "safety/lambda_jvp_eff"),
        "lambda_jvp_schedule_enabled": metric_mean(complete, "safety/lambda_jvp_schedule_enabled"),
        "safety_q_weight_mean": metric_mean(complete, "safety_q/weight_mean"),
        "safety_q_boundary_frac": metric_mean(complete, "safety_q/boundary_mask_frac"),
        "safety_q_cost_frac": metric_mean(complete, "safety_q/cost_mask_frac"),
        "diag_qc_geom_mode_id": metric_mean(complete, "safety_q/diag_qc_geom_mode_id"),
        "geom_grad_norm_mean": metric_mean_fallback(complete, "safety_q/geom_grad_norm_mean", "safety_q/grad_norm_mean"),
        "geom_zero_grad_frac": metric_mean_fallback(complete, "safety_q/geom_zero_grad_frac", "safety_q/zero_grad_frac"),
        "geom_mono_plus_frac": metric_mean_fallback(complete, "safety_q/geom_mono_plus_frac", "safety_q/mono_plus_frac"),
        "geom_mono_minus_frac": metric_mean_fallback(complete, "safety_q/geom_mono_minus_frac", "safety_q/mono_minus_frac"),
        "geom_fd_slope_mean": metric_mean_fallback(complete, "safety_q/geom_fd_slope_mean", "safety_q/fd_slope_mean"),
        "max_grad_norm_mean": metric_mean_fallback(complete, "safety_q/max_grad_norm_mean", "safety_q/grad_norm_mean"),
        "max_zero_grad_frac": metric_mean_fallback(complete, "safety_q/max_zero_grad_frac", "safety_q/zero_grad_frac"),
        "max_mono_plus_frac": metric_mean_fallback(complete, "safety_q/max_mono_plus_frac", "safety_q/mono_plus_frac"),
        "max_mono_minus_frac": metric_mean_fallback(complete, "safety_q/max_mono_minus_frac", "safety_q/mono_minus_frac"),
        "max_fd_slope_mean": metric_mean_fallback(complete, "safety_q/max_fd_slope_mean", "safety_q/fd_slope_mean"),
        "jvp_mean": metric_mean(complete, "safety_q/jvp_mean"),
        "normalized_jvp_mean": metric_mean(complete, "safety_q/normalized_jvp_mean"),
        "weighted_jvp": metric_mean(complete, "loss/jvp_weighted"),
        "status": "failed" if failed else ("completed" if len(complete) >= expected else "pending"),
    }
    stat["geometry_decision"] = "geometry_improved" if geometry_improved(stat) else "geometry_not_improved"
    avg_reward = stat["avg_last3_reward"][0]  # type: ignore[index]
    avg_cost = stat["avg_last3_cost"][0]  # type: ignore[index]
    stat["decision"] = decide(avg_reward, avg_cost, len(complete), expected, bool(failed), stat)
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
        "# JVP Coupling CarGoal1 Summary",
        "",
        f"G4 fixed: Reward = {G4_REWARD}, Cost = {G4_COST}.",
        f"C2: Reward = {C2_REWARD}, Cost = {C2_COST}.",
        f"C4 long: Reward = {C4_REWARD}, Cost = {C4_COST}.",
        f"PPO baseline: Reward = {PPO_REWARD}, Cost = {PPO_COST}.",
        "",
        "| Group | Seeds | Reward | Cost | lambda_safe | safe_bandwidth | lambda_jvp_eff | safety_q_weight_mean | safety_q_boundary_frac | safety_q_cost_frac | geom_grad_norm_mean | geom_zero_grad_frac | geom_mono_plus_frac | geom_mono_minus_frac | geom_fd_slope_mean | max_mono_plus_frac | max_mono_minus_frac | weighted_jvp | Decision |",
        "| ----- | ----: | -----: | ---: | ----------: | -------------: | -------------: | -------------------: | ---------------------: | -----------------: | ------------------: | ------------------: | ------------------: | -------------------: | -----------------: | -----------------: | ------------------: | -----------: | -------- |",
    ]
    for group, cfg in GROUPS.items():
        stat = stats[group]
        lines.append(
            f"| {group} | {stat['completed_count']}/{stat['expected']} | "
            f"{fmt_mean_std(stat['avg_last3_reward'])} | {fmt_mean_std(stat['avg_last3_cost'])} | "
            f"{cfg['lambda_safe']} | {cfg['safe_bandwidth']} | {fmt(stat['lambda_jvp_eff'])} | "
            f"{fmt(stat['safety_q_weight_mean'])} | {fmt(stat['safety_q_boundary_frac'])} | {fmt(stat['safety_q_cost_frac'])} | "
            f"{sci(stat['geom_grad_norm_mean'])} | {fmt(stat['geom_zero_grad_frac'])} | {fmt(stat['geom_mono_plus_frac'])} | "
            f"{fmt(stat['geom_mono_minus_frac'])} | {sci(stat['geom_fd_slope_mean'])} | "
            f"{fmt(stat['max_mono_plus_frac'])} | {fmt(stat['max_mono_minus_frac'])} | {sci(stat['weighted_jvp'])} | {stat['decision']} |"
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
        "- If weak-JVP/scheduled coupling improves reward while cost remains controlled, expand the best group first.",
        "- If geometry improves but performance does not, inspect actor/JVP coupling rather than changing the safety target.",
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
        "# JVP Coupling CarGoal1 Decision Log",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Hypothesis: scheduled JVP coupling plus narrower safety bandwidth can reduce early actor suppression while keeping the geom-Q safety normal active.",
        f"- Best group: {best}",
        "- Expand seeds: " + (", ".join(f"{group} seed2" for group in expand_groups) if expand_groups else "no"),
    ]
    for group, stat in stats.items():
        decision_lines.append(
            f"- {group}: seeds={stat['completed_count']}/{stat['expected']}, "
            f"reward={fmt_mean_std(stat['avg_last3_reward'])}, cost={fmt_mean_std(stat['avg_last3_cost'])}, "
            f"lambda_jvp_eff={fmt(stat['lambda_jvp_eff'])}, geometry={stat['geometry_decision']}, decision={stat['decision']}"
        )
    DECISION_LOG.write_text("\n".join(decision_lines) + "\n")

    print(SUMMARY)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
