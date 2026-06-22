#!/usr/bin/env python3
"""Collect PointGoal normalized-JVP overnight search metrics."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_njvp_overnight"
REPORT = ROOT / "reports" / "pointgoal_njvp_overnight" / "summary.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

GROUPS = {
    "ON0_N3_extend": {"lambda_safe": 0.5, "lambda_jvp": 0.003, "bandwidth": 0.05, "seeds": [3, 4, 5, 6, 7, 8, 9]},
    "ON1_jvp0035": {"lambda_safe": 0.5, "lambda_jvp": 0.0035, "bandwidth": 0.05, "seeds": [0, 1, 2, 3, 4]},
    "ON2_jvp0040": {"lambda_safe": 0.5, "lambda_jvp": 0.004, "bandwidth": 0.05, "seeds": [0, 1, 2, 3, 4]},
    "ON3_bw0075": {"lambda_safe": 0.5, "lambda_jvp": 0.003, "bandwidth": 0.075, "seeds": [0, 1, 2, 3, 4]},
    "ON4_safe07": {"lambda_safe": 0.7, "lambda_jvp": 0.003, "bandwidth": 0.05, "seeds": [0, 1, 2, 3, 4]},
}

BASELINE_COST_THRESHOLD = 45.64
BASELINE_REWARD_THRESHOLD = 19.85
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")
EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
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


def parse_metric(text: str, key: str) -> float | None:
    values = re.findall(r"wandb:\s+" + re.escape(key) + r"\s+(" + NUM_RE + r")\b", text, flags=re.I)
    return float(values[-1]) if values else None


def parse_log(path: Path, group: str, seed: int) -> dict[str, object]:
    if not path.exists():
        return {"group": group, "seed": seed, "status": "missing", "log_path": str(path)}
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in EVAL_RE.finditer(text)]
    lower = text.lower()
    if "oom" in lower or "out of memory" in lower:
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
    if evals:
        final_reward, final_cost, _ = evals[-1]
        last3 = evals[-3:]
        avg_last3_reward = mean(row[0] for row in last3)
        avg_last3_cost = mean(row[1] for row in last3)
    else:
        final_reward = final_cost = avg_last3_reward = avg_last3_cost = None
    return {
        "group": group,
        "seed": seed,
        "status": status,
        "final_reward": final_reward,
        "final_cost": final_cost,
        "avg_last3_reward": avg_last3_reward,
        "avg_last3_cost": avg_last3_cost,
        "loss/jvp_weighted": parse_metric(text, "loss/jvp_weighted"),
        "log_path": str(path),
    }


def group_stats(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    complete = [row for row in rows if row["group"] == group and row["status"] == "completed"]
    all_rows = [row for row in rows if row["group"] == group]
    fr_m, fr_s = mean_std([row.get("final_reward") for row in complete])  # type: ignore[list-item]
    fc_m, fc_s = mean_std([row.get("final_cost") for row in complete])  # type: ignore[list-item]
    ar_m, ar_s = mean_std([row.get("avg_last3_reward") for row in complete])  # type: ignore[list-item]
    ac_m, ac_s = mean_std([row.get("avg_last3_cost") for row in complete])  # type: ignore[list-item]
    jw_m, jw_s = mean_std([row.get("loss/jvp_weighted") for row in complete])  # type: ignore[list-item]
    passed = (
        ar_m is not None
        and ac_m is not None
        and ac_m <= BASELINE_COST_THRESHOLD
        and ar_m >= BASELINE_REWARD_THRESHOLD
    )
    return {
        "completed": len(complete),
        "failed": len(all_rows) - len(complete),
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
        "passed": passed,
    }


def parse_monitor() -> dict[str, object]:
    if not MONITOR.exists():
        return {}
    used: list[float] = []
    util: list[float] = []
    with MONITOR.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            def number(raw: str) -> float | None:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
                return float(match.group()) if match else None
            mem = number(row[1])
            gpu = number(row[3])
            if mem is not None:
                used.append(mem)
            if gpu is not None:
                util.append(gpu)
    if not used:
        return {}
    return {
        "peak_memory_mib": max(used),
        "avg_memory_mib": mean(used),
        "avg_gpu_util": mean(util) if util else None,
        "samples": len(used),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for group, cfg in GROUPS.items():
        for seed in cfg["seeds"]:
            rows.append(parse_log(LOG_DIR / f"{group}_seed{seed}.log", group, seed))

    stats = {group: group_stats(rows, group) for group in GROUPS}
    monitor = parse_monitor()
    any_oom = any(row["status"] == "failed_oom" for row in rows)
    any_nan = any(row["status"] == "failed_nan" for row in rows)
    any_error = any(str(row["status"]).startswith("failed") for row in rows)

    lines = [
        "# PointGoal NJVP Overnight Summary",
        "",
        "| Group | lambda_safe | lambda_jvp | bandwidth | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP mean/std | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group, cfg in GROUPS.items():
        stat = stats[group]
        decision = "pass_threshold" if stat["passed"] else "not_passed"
        if stat["failed"]:
            decision += "_with_incomplete_or_failed"
        lines.append(
            f"| {group} | {cfg['lambda_safe']:g} | {cfg['lambda_jvp']:g} | {cfg['bandwidth']:g} | "
            f"{stat['completed']}/{len(cfg['seeds'])} | "
            f"{fmt(stat['final_reward_mean'])} / {fmt(stat['final_reward_std'])} | "
            f"{fmt(stat['final_cost_mean'])} / {fmt(stat['final_cost_std'])} | "
            f"{fmt(stat['avg_last3_reward_mean'])} / {fmt(stat['avg_last3_reward_std'])} | "
            f"{fmt(stat['avg_last3_cost_mean'])} / {fmt(stat['avg_last3_cost_std'])} | "
            f"{sci(stat['jvp_weighted_mean'])} / {sci(stat['jvp_weighted_std'])} | {decision} |"
        )

    lines += [
        "",
        "## Per-Run Results",
        "",
        "| Group | Seed | Final Reward | Final Cost | Avg Last 3 Reward | Avg Last 3 Cost | weighted JVP | Status | Log Path |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {fmt(row.get('final_reward'))} | "
            f"{fmt(row.get('final_cost'))} | {fmt(row.get('avg_last3_reward'))} | "
            f"{fmt(row.get('avg_last3_cost'))} | {sci(row.get('loss/jvp_weighted'))} | "
            f"{row['status']} | {row['log_path']} |"
        )

    lines += [
        "",
        "## Comparisons",
        "",
        f"- Transfer threshold: Avg Last 3 Cost <= {BASELINE_COST_THRESHOLD:.2f} and Avg Last 3 Reward >= {BASELINE_REWARD_THRESHOLD:.2f}.",
        f"- ON0_N3_extend passed: {stats['ON0_N3_extend']['passed']}",
        f"- ON1_jvp0035 passed: {stats['ON1_jvp0035']['passed']}",
        f"- ON2_jvp0040 passed: {stats['ON2_jvp0040']['passed']}",
        f"- ON3_bw0075 passed: {stats['ON3_bw0075']['passed']}",
        f"- ON4_safe07 passed: {stats['ON4_safe07']['passed']}",
        "",
        "## GPU",
        "",
        "- MAX_PARALLEL target: 5",
        f"- Peak memory.used: {fmt(monitor.get('peak_memory_mib') / 1024 if monitor.get('peak_memory_mib') is not None else None)} GiB",
        f"- Average memory.used: {fmt(monitor.get('avg_memory_mib') / 1024 if monitor.get('avg_memory_mib') is not None else None)} GiB",
        f"- Average GPU utilization: {fmt(monitor.get('avg_gpu_util'))}%",
        f"- Monitor samples: {monitor.get('samples', 'n/a')}",
        f"- OOM observed: {any_oom}",
        f"- NaN observed: {any_nan}",
        f"- Any failed run: {any_error}",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
