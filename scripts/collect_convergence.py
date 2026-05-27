#!/usr/bin/env python3
"""Collect SafetyPointGoal convergence metrics from log files."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "convergence"
REPORT = ROOT / "reports" / "convergence" / "pointgoal_convergence_summary.md"

RUNS = [
    ("C0_base", "C0_base.log", 0.0, 0.0, "none"),
    ("C1_weak_safety", "C1_weak_safety.log", 0.1, 0.0, "none"),
    ("C2_safety", "C2_safety.log", 0.5, 0.0, "none"),
    ("C3_weak_jvp", "C3_weak_jvp.log", 0.5, 0.001, "grad"),
    ("C4_slightly_stronger_jvp", "C4_slightly_stronger_jvp.log", 0.5, 0.005, "grad"),
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
        return {"status": "missing", "evals": [], "error": False}

    text = path.read_text(errors="replace")
    evals = [
        (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        for match in EVAL_RE.finditer(text)
    ]
    ended = " END " in text
    has_error = ERR_RE.search(text) is not None
    if has_error:
        status = "failed"
    elif ended and evals:
        status = "completed"
    elif evals:
        status = "partial"
    else:
        status = "no evals"
    return {"status": status, "evals": evals, "error": has_error}


def summarize(evals: list[tuple[float, float, float]]) -> dict[str, float | None]:
    if not evals:
        return {
            "final_reward": None,
            "final_cost": None,
            "best_reward": None,
            "best_cost": None,
            "avg_last3_reward": None,
            "avg_last3_cost": None,
        }
    final_reward, final_cost, _ = evals[-1]
    best_reward, best_cost, _ = max(evals, key=lambda item: item[0])
    last3 = evals[-3:]
    return {
        "final_reward": final_reward,
        "final_cost": final_cost,
        "best_reward": best_reward,
        "best_cost": best_cost,
        "avg_last3_reward": sum(item[0] for item in last3) / len(last3),
        "avg_last3_cost": sum(item[1] for item in last3) / len(last3),
    }


def main() -> None:
    rows = []
    for run, filename, lambda_safe, lambda_jvp, jvp_mode in RUNS:
        parsed = parse_log(LOG_DIR / filename)
        metrics = summarize(parsed["evals"])  # type: ignore[arg-type]
        rows.append((run, lambda_safe, lambda_jvp, jvp_mode, parsed["status"], metrics))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SafetyPointGoal1-v0 Convergence Summary",
        "",
        "This report is generated from `logs/convergence/*.log`.",
        "",
        "## Results",
        "",
        "| Run | lambda_safe | lambda_jvp | jvp_mode | Final Eval Reward | Final Eval Cost | Best Eval Reward | Best Eval Cost | Avg Last 3 Reward | Avg Last 3 Cost | Status |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run, lambda_safe, lambda_jvp, jvp_mode, status, metrics in rows:
        lines.append(
            f"| {run} | {lambda_safe:g} | {lambda_jvp:g} | {jvp_mode} | "
            f"{fmt(metrics['final_reward'])} | {fmt(metrics['final_cost'])} | "
            f"{fmt(metrics['best_reward'])} | {fmt(metrics['best_cost'])} | "
            f"{fmt(metrics['avg_last3_reward'])} | {fmt(metrics['avg_last3_cost'])} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- C0 disables actor safety loss and serves as the base flow check.",
            "- C1/C2 test safety-penalty-only settings.",
            "- C3/C4 enable grad-mode JVP-SCD with lambda_jvp 0.001 and 0.005.",
            "- No new experiments are launched by this collector.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
