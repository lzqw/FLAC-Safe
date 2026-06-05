#!/usr/bin/env python3
"""Collect PointGoal CDF safety critic experiment metrics."""

from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_cdf_safety"
REPORT = ROOT / "reports" / "pointgoal_cdf_safety" / "summary.md"
THRESHOLD_COST = 45.64
THRESHOLD_REWARD = 19.85

GROUPS = {
    "C0_cumulative": {"safety_critic_mode": "cumulative", "qc_geom_mode": "max", "seeds": [0, 1, 2]},
    "C1_cdf_max": {"safety_critic_mode": "cdf", "qc_geom_mode": "max", "seeds": [0, 1, 2]},
    "C2_cdf_mean": {"safety_critic_mode": "cdf", "qc_geom_mode": "mean", "seeds": [0, 1, 2]},
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
    "safety/qc_target_min",
    "safety/qc_target_max",
    "safety/qc_target_clip_frac",
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


def main() -> None:
    rows = [parse_log(group, seed) for group, cfg in GROUPS.items() for seed in cfg["seeds"]]
    stats = {group: group_stats(rows, group, len(cfg["seeds"])) for group, cfg in GROUPS.items()}
    any_logs = any(r["status"] != "missing" for r in rows)
    failed_rows = [r for r in rows if str(r["status"]).startswith("failed")]

    lines = [
        "# PointGoal CDF Safety Critic Summary",
        "",
        f"Threshold: Avg Last 3 Cost <= {THRESHOLD_COST} and Avg Last 3 Reward >= {THRESHOLD_REWARD}.",
        "",
        "| Group | Safety Critic Mode | QC Geom Mode | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP | qc target mean | qc target min | qc target max | qc target clip frac | qc risk over threshold | Decision | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group, cfg in GROUPS.items():
        group_stat = stats[group]
        complete = group_stat["complete"]  # type: ignore[assignment]
        lines.append(
            f"| {group} | {cfg['safety_critic_mode']} | {cfg['qc_geom_mode']} | "
            f"{group_stat['completed_count']}/{group_stat['expected']} | "
            f"{fmt_mean_std(group_stat['final_reward'])} | {fmt_mean_std(group_stat['final_cost'])} | "
            f"{fmt_mean_std(group_stat['avg_last3_reward'])} | {fmt_mean_std(group_stat['avg_last3_cost'])} | "
            f"{sci(metric_mean(complete, 'loss/jvp_weighted'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_target_mean'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_target_min'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_target_max'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_target_clip_frac'))} | "
            f"{sci(metric_mean(complete, 'safety/qc_pi_risk_over_threshold'))} | "
            f"{group_stat['decision']} | {group_stat['status']} |"
        )

    lines += [
        "",
        "## Per-Run Results",
        "",
        "| Group | Seed | Final Reward | Final Cost | Avg Last 3 Reward | Avg Last 3 Cost | weighted JVP | qc target mean | qc target min | qc target max | qc target clip frac | qc risk over threshold | Status | Log Path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {fmt(row.get('final_reward'))} | {fmt(row.get('final_cost'))} | "
            f"{fmt(row.get('avg_last3_reward'))} | {fmt(row.get('avg_last3_cost'))} | "
            f"{sci(row.get('loss/jvp_weighted'))} | {sci(row.get('safety/qc_target_mean'))} | "
            f"{sci(row.get('safety/qc_target_min'))} | {sci(row.get('safety/qc_target_max'))} | "
            f"{sci(row.get('safety/qc_target_clip_frac'))} | {sci(row.get('safety/qc_pi_risk_over_threshold'))} | "
            f"{row['status']} | {row['path']} |"
        )

    lines += [
        "",
        "## Error Scan",
        "",
    ]
    if failed_rows:
        lines.append("Failed logs:")
        for row in failed_rows:
            lines.append(f"- {row['path']}: {row['status']}")
    else:
        lines.append("- No error patterns found in parsed logs." if any_logs else "- No logs found.")

    lines += [
        "",
        "## Interpretation hints",
        "- C1 vs C0: whether CDF target improves safety critic learning.",
        "- C2 vs C1: whether mean-normal geometry improves JVP stability.",
        "- If C2 improves cost without hurting reward, expand C2 to seeds 3,4.",
        "- If CDF modes are worse, revert to cumulative and test soft feasibility gate separately.",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
