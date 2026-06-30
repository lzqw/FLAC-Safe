#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY_TASKS = ["SafetyPointGoal1-v0", "SafetyCarButton1-v0"]
COLORS = {
    "SafetyPointGoal1-v0": "#4C78A8",
    "SafetyCarButton1-v0": "#F58518",
    "SafetyCarGoal1-v0": "#54A24B",
    "SafetyPointButton1-v0": "#B279A2",
}
AGE_ORDER = ["age=0", "age=1-5", "age=6-10", "age=11-20", "age>20"]


def fnum(value, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def mean_sem(values: list[float]) -> tuple[float, float]:
    vals = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def write_figure(rows: list[dict], out_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in rows if row.get("task") in PRIMARY_TASKS]
    if not primary:
        raise SystemExit("no primary mechanism rows; refusing to create Figure 2")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) Corridor risk lift vs reference age.
    ax = axes[0]
    width = 0.35
    x = np.arange(len(AGE_ORDER))
    for offset_idx, task in enumerate(PRIMARY_TASKS):
        vals_by_age = defaultdict(list)
        for row in primary:
            if row.get("task") == task:
                vals_by_age[row.get("reference_age_bin", "missing")].append(fnum(row.get("corridor_risk_lift")))
        means = []
        sems = []
        for age in AGE_ORDER:
            m, s = mean_sem(vals_by_age.get(age, []))
            means.append(0.0 if math.isnan(m) else m)
            sems.append(0.0 if math.isnan(s) else s)
        ax.bar(x + (offset_idx - 0.5) * width, means, width=width, yerr=sems, label=task.replace("Safety", "").replace("-v0", ""), color=COLORS[task], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Corridor risk lift")
    ax.set_title("(a) Lift by reference age")
    ax.grid(True, axis="y", alpha=0.25)

    # (b) rho_cor vs rho_cur by task.
    ax = axes[1]
    max_val = 0.0
    for task in PRIMARY_TASKS:
        xs = [fnum(row.get("rho_cur")) for row in primary if row.get("task") == task]
        ys = [fnum(row.get("rho_cor")) for row in primary if row.get("task") == task]
        pairs = [(a, b) for a, b in zip(xs, ys) if not math.isnan(a) and not math.isnan(b)]
        if not pairs:
            continue
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=18, alpha=0.35, color=COLORS[task], label=task.replace("Safety", "").replace("-v0", ""))
        max_val = max(max_val, max(max(p) for p in pairs))
    ax.plot([0, max_val], [0, max_val], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(r"$\rho_{cur}$")
    ax.set_ylabel(r"$\rho_{cor}$")
    ax.set_title(r"(b) Paired $\rho_{cor}$ vs $\rho_{cur}$")
    ax.grid(True, alpha=0.25)

    # (c) effective beta and shadow excess by reference age.
    ax = axes[2]
    vals_by_age = defaultdict(list)
    excess_by_age = defaultdict(list)
    for row in primary:
        vals_by_age[row.get("reference_age_bin", "missing")].append(fnum(row.get("effective_beta")))
        excess_by_age[row.get("reference_age_bin", "missing")].append(fnum(row.get("shadow_excess")))
    beta_means = [mean_sem(vals_by_age.get(age, []))[0] for age in AGE_ORDER]
    excess_means = [mean_sem(excess_by_age.get(age, []))[0] for age in AGE_ORDER]
    ax.plot(x, [0.0 if math.isnan(v) else v for v in beta_means], marker="o", color="#4C78A8", label="effective beta")
    ax.set_ylabel("Effective beta", color="#4C78A8")
    ax.tick_params(axis="y", labelcolor="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_ORDER, rotation=25, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, [0.0 if math.isnan(v) else v for v in excess_means], marker="s", color="#E45756", label="shadow excess")
    ax2.set_ylabel("Shadow excess", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    ax.set_title("(c) Beta / shadow excess")
    ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def update_caption(captions: Path) -> None:
    text = (
        "\n\n## Figure 2 Mechanism Validation\n\n"
        "Figure 2 uses training-time logged paired-audit diagnostics from final STAR-v2 runs. "
        "The primary comparison is corridor shadow risk versus equal-budget current-only samples under paired base noise. "
        "Final-run reference age is collapsed to the `age=1-5` bin, so age-dynamic claims are not made.\n"
    )
    old = captions.read_text(encoding="utf-8", errors="ignore") if captions.exists() else "# Figure Captions\n"
    marker = "## Figure 2 Mechanism Validation"
    if marker in old:
        old = old.split(marker)[0].rstrip() + "\n"
    captions.write_text(old.rstrip() + text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/star_v2_final/mechanism/corridor_mechanism_by_age.csv")
    parser.add_argument("--figure-dir", default="reports/star_v2_final/figures")
    args = parser.parse_args()
    rows = read_csv(Path(args.input))
    write_figure(rows, Path(args.figure_dir) / "fig_mechanism_validation")
    update_caption(Path(args.figure_dir) / "captions.md")
    print(f"wrote {Path(args.figure_dir) / 'fig_mechanism_validation.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
