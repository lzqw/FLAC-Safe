#!/usr/bin/env python3
"""Collect PointGoal directional-noise experiment metrics."""

from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_directional_noise"
REPORT = ROOT / "reports" / "pointgoal_directional_noise" / "summary.md"
THRESHOLD_COST = 45.64
THRESHOLD_REWARD = 19.85
ACTIVE_EPS = 1e-8

GROUPS = {
    "DN0_none": {"mode": "none", "seeds": [0, 1, 2]},
    "DN1_tangent": {"mode": "tangent", "seeds": [0, 1, 2]},
    "DN2_reward_ref": {"mode": "reward_ref", "seeds": [0, 1, 2]},
    "DN3_ref_normal": {"mode": "ref_normal", "seeds": [0, 1, 2]},
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
    "explore/noise_tangent_norm",
    "explore/noise_ref_norm",
    "explore/noise_normal_norm",
    "explore/ref_grad_norm",
    "explore/qc_grad_norm",
    "explore/tangent_ratio",
    "explore/g_mid_mean",
    "explore/noise_total_norm",
    "explore/action_delta_norm",
]

METRIC_LABELS = {
    "loss/jvp_weighted": "weighted JVP",
    "explore/noise_tangent_norm": "tangent norm",
    "explore/noise_ref_norm": "ref norm",
    "explore/noise_normal_norm": "normal norm",
    "explore/ref_grad_norm": "ref grad norm",
    "explore/qc_grad_norm": "qc grad norm",
    "explore/tangent_ratio": "tangent ratio",
    "explore/g_mid_mean": "g_mid mean",
    "explore/noise_total_norm": "noise total norm",
    "explore/action_delta_norm": "action delta norm",
}


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


def metric_mean(rows: list[dict[str, object]], key: str) -> float | None:
    return mean_std([r.get(key) for r in rows])[0]  # type: ignore[list-item]


def parse_metric(text: str, key: str) -> float | None:
    values = re.findall(r"wandb:\s+" + re.escape(key) + r"\s+(" + NUM_RE + r")\b", text, flags=re.I)
    return float(values[-1]) if values else None


def has_error(text: str) -> bool:
    return bool(ERR_RE.search(text))


def parse_log(group: str, seed: int) -> dict[str, object]:
    path = LOG_DIR / f"{group}_seed{seed}.log"
    if not path.exists():
        return {"group": group, "seed": seed, "status": "missing", "path": str(path)}
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2))) for m in EVAL_RE.finditer(text)]
    if "oom" in text.lower() or "out of memory" in text.lower():
        status = "failed_oom"
    elif "NaN" in text or "nan" in text:
        status = "failed_nan"
    elif has_error(text):
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


def active(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if abs(value) > ACTIVE_EPS else "no"


def main() -> None:
    rows = [parse_log(group, seed) for group, cfg in GROUPS.items() for seed in cfg["seeds"]]
    stats = {group: group_stats(rows, group, len(cfg["seeds"])) for group, cfg in GROUPS.items()}
    any_logs = any(r["status"] != "missing" for r in rows)

    lines = [
        "# PointGoal Directional Noise Summary",
        "",
        "## Main results",
        "",
        f"Threshold: Avg Last 3 Cost <= {THRESHOLD_COST} and Avg Last 3 Reward >= {THRESHOLD_REWARD}.",
        "",
        "| Group | Mode | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP | tangent norm | ref norm | normal norm | ref grad norm | qc grad norm | tangent ratio | action delta norm | Decision | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group, cfg in GROUPS.items():
        group_stat = stats[group]
        complete = group_stat["complete"]  # type: ignore[assignment]
        lines.append(
            f"| {group} | {cfg['mode']} | {group_stat['completed_count']}/{group_stat['expected']} | "
            f"{fmt_mean_std(group_stat['final_reward'])} | {fmt_mean_std(group_stat['final_cost'])} | "
            f"{fmt_mean_std(group_stat['avg_last3_reward'])} | {fmt_mean_std(group_stat['avg_last3_cost'])} | "
            f"{sci(metric_mean(complete, 'loss/jvp_weighted'))} | "
            f"{sci(metric_mean(complete, 'explore/noise_tangent_norm'))} | "
            f"{sci(metric_mean(complete, 'explore/noise_ref_norm'))} | "
            f"{sci(metric_mean(complete, 'explore/noise_normal_norm'))} | "
            f"{sci(metric_mean(complete, 'explore/ref_grad_norm'))} | "
            f"{sci(metric_mean(complete, 'explore/qc_grad_norm'))} | "
            f"{sci(metric_mean(complete, 'explore/tangent_ratio'))} | "
            f"{sci(metric_mean(complete, 'explore/action_delta_norm'))} | "
            f"{group_stat['decision']} | {group_stat['status']} |"
        )

    lines += [
        "",
        "## Exploration statistics",
        "",
        "| Group | " + " | ".join(METRIC_LABELS.values()) + " |",
        "|---|" + "|".join(["---:"] * len(METRICS)) + "|",
    ]
    for group in GROUPS:
        complete = stats[group]["complete"]  # type: ignore[assignment]
        values = " | ".join(sci(metric_mean(complete, key)) for key in METRICS)
        lines.append(f"| {group} | {values} |")

    lines += [
        "",
        "## Threshold decision",
        "",
        f"- threshold cost: {THRESHOLD_COST}",
        f"- threshold reward: {THRESHOLD_REWARD}",
        "- passed groups: "
        + (", ".join(group for group, group_stat in stats.items() if group_stat["decision"] == "passed") or "none"),
    ]
    completed_groups = [group for group, group_stat in stats.items() if group_stat["completed_count"]]
    if completed_groups:
        best_tradeoff = min(
            completed_groups,
            key=lambda group: (
                stats[group]["avg_last3_cost"][0] if stats[group]["avg_last3_cost"][0] is not None else float("inf"),
                -(stats[group]["avg_last3_reward"][0] if stats[group]["avg_last3_reward"][0] is not None else float("-inf")),
            ),
        )
        lowest_cost = min(
            completed_groups,
            key=lambda group: stats[group]["avg_last3_cost"][0]
            if stats[group]["avg_last3_cost"][0] is not None
            else float("inf"),
        )
        highest_reward = max(
            completed_groups,
            key=lambda group: stats[group]["avg_last3_reward"][0]
            if stats[group]["avg_last3_reward"][0] is not None
            else float("-inf"),
        )
        lines += [
            f"- best reward-cost tradeoff: {best_tradeoff}",
            f"- lowest Avg Last 3 Cost: {lowest_cost}",
            f"- highest Avg Last 3 Reward: {highest_reward}",
        ]
    else:
        lines += [
            "- best reward-cost tradeoff: n/a",
            "- lowest Avg Last 3 Cost: n/a",
            "- highest Avg Last 3 Reward: n/a",
            "- note: no completed directional-noise runs found.",
        ]

    dn1 = stats["DN1_tangent"]["complete"]  # type: ignore[assignment]
    dn2 = stats["DN2_reward_ref"]["complete"]  # type: ignore[assignment]
    dn3 = stats["DN3_ref_normal"]["complete"]  # type: ignore[assignment]
    lines += [
        "",
        "## Exploration activation check",
        "",
        "- DN0 should have near-zero exploration stats.",
        "- DN1 should have nonzero tangent norm and near-zero ref/normal norm.",
        "- DN2 should have nonzero tangent norm and ref norm, near-zero normal norm.",
        "- DN3 should have nonzero tangent/ref/normal norm.",
        f"- DN1 tangent active: {active(metric_mean(dn1, 'explore/noise_tangent_norm'))}",
        f"- DN2 reward reference active: {active(metric_mean(dn2, 'explore/noise_ref_norm'))}",
        f"- DN3 suppressed normal active: {active(metric_mean(dn3, 'explore/noise_normal_norm'))}",
    ]

    failed_rows = [r for r in rows if str(r["status"]).startswith("failed")]
    lines += [
        "",
        "## Error scan",
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
        "## Per-Run Results",
        "",
        "| Group | Seed | Final Reward | Final Cost | Avg Last 3 Reward | Avg Last 3 Cost | weighted JVP | tangent norm | ref norm | normal norm | ref grad norm | qc grad norm | tangent ratio | action delta norm | Status | Log Path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {fmt(row.get('final_reward'))} | "
            f"{fmt(row.get('final_cost'))} | {fmt(row.get('avg_last3_reward'))} | "
            f"{fmt(row.get('avg_last3_cost'))} | {sci(row.get('loss/jvp_weighted'))} | "
            f"{sci(row.get('explore/noise_tangent_norm'))} | {sci(row.get('explore/noise_ref_norm'))} | "
            f"{sci(row.get('explore/noise_normal_norm'))} | {sci(row.get('explore/ref_grad_norm'))} | "
            f"{sci(row.get('explore/qc_grad_norm'))} | {sci(row.get('explore/tangent_ratio'))} | "
            f"{sci(row.get('explore/action_delta_norm'))} | {row['status']} | {row['path']} |"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
