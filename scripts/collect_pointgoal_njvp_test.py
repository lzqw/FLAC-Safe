#!/usr/bin/env python3
"""Collect normalized JVP PointGoal test metrics."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_njvp_test"
REPORT = ROOT / "reports" / "pointgoal_njvp_test" / "summary.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

GROUPS = {
    "N0_raw_R2D": {"normalize_jvp": False, "jvp_norm_mode": "exact", "lambda_jvp": 0.005, "seeds": [0, 1, 2]},
    "N1_njvp_0005": {"normalize_jvp": True, "jvp_norm_mode": "exact", "lambda_jvp": 0.0005, "seeds": [0, 1, 2]},
    "N2_njvp_001": {"normalize_jvp": True, "jvp_norm_mode": "exact", "lambda_jvp": 0.001, "seeds": [0, 1, 2]},
    "N3_njvp_003": {"normalize_jvp": True, "jvp_norm_mode": "exact", "lambda_jvp": 0.003, "seeds": [0, 1, 2]},
    "N4_njvp_005": {"normalize_jvp": True, "jvp_norm_mode": "exact", "lambda_jvp": 0.005, "seeds": [0, 1, 2]},
}
S0_BASELINE = {"avg_last3_reward": 22.06, "avg_last3_cost": 53.69}

EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")
METRIC_KEYS = [
    "loss/jvp_scd",
    "loss/jvp_weighted",
    "safety/jvp_denom_mean",
    "safety/jvp_directional_abs",
    "safety/g_mid_mean",
]
NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


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


def parse_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "evals": [], "metrics": {}}
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
    metrics: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        vals = re.findall(r"wandb:\s+" + re.escape(key) + r"\s+(" + NUM_RE + r")\b", text, flags=re.I)
        metrics[key] = float(vals[-1]) if vals else None
    return {"status": status, "evals": evals, "metrics": metrics}


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


def group_stats(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    complete = [row for row in rows if row["group"] == group and row["status"] == "completed"]
    group_rows = [row for row in rows if row["group"] == group]
    fr_m, fr_s = mean_std([row["final_reward"] for row in complete])  # type: ignore[list-item]
    fc_m, fc_s = mean_std([row["final_cost"] for row in complete])  # type: ignore[list-item]
    ar_m, ar_s = mean_std([row["avg_last3_reward"] for row in complete])  # type: ignore[list-item]
    ac_m, ac_s = mean_std([row["avg_last3_cost"] for row in complete])  # type: ignore[list-item]
    jw_m, jw_s = mean_std([row["loss/jvp_weighted"] for row in complete])  # type: ignore[list-item]
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
        "jvp_weighted_mean": jw_m,
        "jvp_weighted_std": jw_s,
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for group, cfg in GROUPS.items():
        for seed in cfg["seeds"]:
            path = LOG_DIR / f"{group}_seed{seed}.log"
            parsed = parse_log(path)
            metrics = summarize_evals(parsed["evals"])  # type: ignore[arg-type]
            metric_values = parsed["metrics"]  # type: ignore[assignment]
            row = {
                "group": group,
                "seed": seed,
                "normalize_jvp": cfg["normalize_jvp"],
                "jvp_norm_mode": cfg["jvp_norm_mode"],
                "lambda_jvp": cfg["lambda_jvp"],
                "status": parsed["status"],
                "path": path,
                **metrics,
            }
            for key in METRIC_KEYS:
                row[key] = metric_values.get(key)  # type: ignore[union-attr]
            rows.append(row)

    stats = {group: group_stats(rows, group) for group in GROUPS}
    monitor = parse_monitor()
    any_oom = any(row["status"] == "failed_oom" for row in rows)
    any_nan = any(row["status"] == "failed_nan" for row in rows)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PointGoal Normalized JVP Test Summary",
        "",
        "| Group | Seed | normalize_jvp | jvp_norm_mode | lambda_jvp | Final Eval Reward | Final Eval Cost | Avg Last 3 Reward | Avg Last 3 Cost | loss/jvp_scd | loss/jvp_weighted | safety/jvp_denom_mean | safety/jvp_directional_abs | safety/g_mid_mean | Status |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {row['normalize_jvp']} | {row['jvp_norm_mode']} | "
            f"{row['lambda_jvp']:g} | {fmt(row['final_reward'])} | {fmt(row['final_cost'])} | "
            f"{fmt(row['avg_last3_reward'])} | {fmt(row['avg_last3_cost'])} | "
            f"{sci(row['loss/jvp_scd'])} | {sci(row['loss/jvp_weighted'])} | "
            f"{sci(row['safety/jvp_denom_mean'])} | {sci(row['safety/jvp_directional_abs'])} | "
            f"{sci(row['safety/g_mid_mean'])} | {row['status']} |"
        )

    lines += [
        "",
        "## Group Statistics",
        "",
        "| Group | Completed | Failed | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | loss/jvp_weighted mean/std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, stat in stats.items():
        lines.append(
            f"| {group} | {stat['completed']} | {stat['failed']} | "
            f"{fmt(stat['final_reward_mean'])} / {fmt(stat['final_reward_std'])} | "
            f"{fmt(stat['final_cost_mean'])} / {fmt(stat['final_cost_std'])} | "
            f"{fmt(stat['avg_last3_reward_mean'])} / {fmt(stat['avg_last3_reward_std'])} | "
            f"{fmt(stat['avg_last3_cost_mean'])} / {fmt(stat['avg_last3_cost_std'])} | "
            f"{sci(stat['jvp_weighted_mean'])} / {sci(stat['jvp_weighted_std'])} |"
        )

    lines += [
        "",
        "## Decision Checks",
        "",
    ]
    for group, stat in stats.items():
        if stat["completed"]:
            cost_ok = stat["avg_last3_cost_mean"] is not None and stat["avg_last3_cost_mean"] <= 0.85 * S0_BASELINE["avg_last3_cost"]
            reward_ok = stat["avg_last3_reward_mean"] is not None and stat["avg_last3_reward_mean"] >= 0.90 * S0_BASELINE["avg_last3_reward"]
            weighted_ok = stat["jvp_weighted_mean"] is not None and stat["jvp_weighted_mean"] >= 1e-5
            lines.append(f"- {group}: cost_ok={cost_ok}, reward_ok={reward_ok}, weighted_jvp_ge_1e-5={weighted_ok}")
    lines += [
        "",
        "## GPU",
        "",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        f"- OOM observed: {any_oom}",
        f"- NaN observed: {any_nan}",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
