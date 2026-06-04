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
    "explore/ref_grad_norm",
    "explore/qc_grad_norm",
    "explore/noise_total_norm",
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
    if "oom" in text.lower() or "out of memory" in text.lower():
        status = "failed_oom"
    elif "NaN" in text or "nan" in text:
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


def main() -> None:
    rows = [parse_log(group, seed) for group, cfg in GROUPS.items() for seed in cfg["seeds"]]
    lines = [
        "# PointGoal Directional Noise Summary",
        "",
        "| Group | Mode | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP | ref_grad_norm | qc_grad_norm | noise_total_norm | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group, cfg in GROUPS.items():
        complete = [r for r in rows if r["group"] == group and r["status"] == "completed"]
        fr = mean_std([r.get("final_reward") for r in complete])  # type: ignore[list-item]
        fc = mean_std([r.get("final_cost") for r in complete])  # type: ignore[list-item]
        ar = mean_std([r.get("avg_last3_reward") for r in complete])  # type: ignore[list-item]
        ac = mean_std([r.get("avg_last3_cost") for r in complete])  # type: ignore[list-item]
        jw = mean_std([r.get("loss/jvp_weighted") for r in complete])  # type: ignore[list-item]
        rg = mean_std([r.get("explore/ref_grad_norm") for r in complete])  # type: ignore[list-item]
        qg = mean_std([r.get("explore/qc_grad_norm") for r in complete])  # type: ignore[list-item]
        nt = mean_std([r.get("explore/noise_total_norm") for r in complete])  # type: ignore[list-item]
        decision = "pending" if len(complete) < len(cfg["seeds"]) else "completed"
        lines.append(
            f"| {group} | {cfg['mode']} | {len(complete)}/{len(cfg['seeds'])} | "
            f"{fmt(fr[0])} / {fmt(fr[1])} | {fmt(fc[0])} / {fmt(fc[1])} | "
            f"{fmt(ar[0])} / {fmt(ar[1])} | {fmt(ac[0])} / {fmt(ac[1])} | "
            f"{sci(jw[0])} | {sci(rg[0])} | {sci(qg[0])} | {sci(nt[0])} | {decision} |"
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
            f"| {row['group']} | {row['seed']} | {fmt(row.get('final_reward'))} | "
            f"{fmt(row.get('final_cost'))} | {fmt(row.get('avg_last3_reward'))} | "
            f"{fmt(row.get('avg_last3_cost'))} | {row['status']} | {row['path']} |"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
