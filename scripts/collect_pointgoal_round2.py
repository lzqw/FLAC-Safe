#!/usr/bin/env python3
"""Collect PointGoal Round2 metrics and GPU scheduling summary."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pointgoal_round2"
REPORT = ROOT / "reports" / "pointgoal_round2" / "summary.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

RUNS = [
    ("R2_A", "r2_A_safe05_jvp00005_bw005", 0.5, 0.0005, 0.05),
    ("R2_B", "r2_B_safe05_jvp0001_bw005", 0.5, 0.001, 0.05),
    ("R2_C", "r2_C_safe05_jvp0002_bw005", 0.5, 0.002, 0.05),
    ("R2_D", "r2_D_safe05_jvp0005_bw005", 0.5, 0.005, 0.05),
    ("R2_E", "r2_E_safe05_jvp0001_bw010", 0.5, 0.001, 0.10),
    ("R2_F", "r2_F_safe03_jvp0002_bw005", 0.3, 0.002, 0.05),
]

EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def parse_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "evals": [], "batch": None, "updates": None, "hidden": None}
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in EVAL_RE.finditer(text)]
    has_error = ERR_RE.search(text) is not None
    if "out of memory" in text.lower() or "oom" in text.lower():
        status = "failed_oom"
    elif "nan" in text or "NaN" in text:
        status = "failed_nan"
    elif has_error:
        status = "failed_error"
    elif " END " in text and evals:
        status = "completed"
    elif evals:
        status = "partial"
    else:
        status = "no evals"
    batch = re.search(r"--batch_size\s+(\d+)", text)
    updates = re.search(r"--updates_per_step\s+(\d+)", text)
    hidden = re.search(r"--hidden_size\s+(\d+)", text)
    return {
        "status": status,
        "evals": evals,
        "batch": int(batch.group(1)) if batch else None,
        "updates": int(updates.group(1)) if updates else None,
        "hidden": int(hidden.group(1)) if hidden else None,
    }


def summarize(evals: list[tuple[float, float, float]]) -> dict[str, float | None]:
    if not evals:
        return dict(final_reward=None, final_cost=None, best_reward=None, best_cost=None, avg_last3_reward=None, avg_last3_cost=None)
    final_reward, final_cost, _ = evals[-1]
    best_reward, best_cost, _ = max(evals, key=lambda x: x[0])
    last3 = evals[-3:]
    return {
        "final_reward": final_reward,
        "final_cost": final_cost,
        "best_reward": best_reward,
        "best_cost": best_cost,
        "avg_last3_reward": sum(x[0] for x in last3) / len(last3),
        "avg_last3_cost": sum(x[1] for x in last3) / len(last3),
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


def main() -> None:
    rows = []
    any_oom = False
    any_nan = False
    for run, tag, lambda_safe, lambda_jvp, bandwidth in RUNS:
        path = LOG_DIR / f"{tag}.log"
        parsed = parse_log(path)
        metrics = summarize(parsed["evals"])  # type: ignore[arg-type]
        status = str(parsed["status"])
        any_oom = any_oom or status == "failed_oom"
        any_nan = any_nan or status == "failed_nan"
        rows.append((run, tag, lambda_safe, lambda_jvp, bandwidth, parsed, metrics, path))

    monitor = parse_monitor()
    completed = [row for row in rows if row[5]["status"] == "completed"]
    lowest_cost = min(completed, key=lambda r: r[6]["avg_last3_cost"] if r[6]["avg_last3_cost"] is not None else float("inf"), default=None)
    highest_reward = max(completed, key=lambda r: r[6]["final_reward"] if r[6]["final_reward"] is not None else float("-inf"), default=None)
    tradeoff = max(
        completed,
        key=lambda r: (r[6]["avg_last3_reward"] or -999) - 0.05 * (r[6]["avg_last3_cost"] or 999),
        default=None,
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PointGoal Round2 Summary",
        "",
        "| Run | lambda_safe | lambda_jvp | safe_bandwidth | batch_size | updates_per_step | hidden_size | parallel_N | Final Eval Reward | Final Eval Cost | Best Eval Reward | Best Eval Cost | Avg Last 3 Reward | Avg Last 3 Cost | Status | Log Path |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run, tag, lambda_safe, lambda_jvp, bandwidth, parsed, metrics, path in rows:
        lines.append(
            f"| {run} | {lambda_safe:g} | {lambda_jvp:g} | {bandwidth:g} | "
            f"{parsed['batch'] or 'n/a'} | {parsed['updates'] or 'n/a'} | {parsed['hidden'] or 'n/a'} | auto | "
            f"{fmt(metrics['final_reward'])} | {fmt(metrics['final_cost'])} | "
            f"{fmt(metrics['best_reward'])} | {fmt(metrics['best_cost'])} | "
            f"{fmt(metrics['avg_last3_reward'])} | {fmt(metrics['avg_last3_cost'])} | {parsed['status']} | {path} |"
        )

    lines += [
        "",
        "## GPU Parallel Scheduling",
        "",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        "- GPU memory fraction: configured in launch script",
        "- GPU reserve GB: configured in launch script",
        "- Estimated per-run memory GB: see launch script plan output",
        "- Computed N: see launch script plan output",
        "- Hard max parallel: configured in launch script",
        "- Actual max concurrent runs: infer from tmux/session schedule",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        f"- OOM observed: {any_oom}",
        f"- NaN observed: {any_nan}",
        "- CPU/env bottleneck observed: infer from utilization and wall-clock speed",
        "",
        "## Conclusion",
        "",
        f"- Most stable / lowest Avg Last 3 Cost: {lowest_cost[0] if lowest_cost else 'n/a'}",
        f"- Best Final Reward: {highest_reward[0] if highest_reward else 'n/a'}",
        f"- Best reward/cost tradeoff: {tradeoff[0] if tradeoff else 'n/a'}",
        "- Compare against C2/C4 after all Round2 runs complete.",
        "- Do not move to SafetyCarGoal1-v0 until PointGoal Round2 is reviewed.",
        "- Normalized JVP, forward JVP, and soft normal masking remain disabled for this round.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
