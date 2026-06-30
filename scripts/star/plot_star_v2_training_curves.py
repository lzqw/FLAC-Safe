#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


MAIN_TASKS = ["SafetyPointGoal1-v0", "SafetyCarButton1-v0"]
ALL_TASKS = ["SafetyPointGoal1-v0", "SafetyCarButton1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"]
METHODS = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]
LABELS = {
    "pointwise_v2": "Pointwise",
    "current_only_v2": "Current-only",
    "sac_lag": "SAC-Lag",
    "star_v2": "STAR-v2",
}
COLORS = {
    "pointwise_v2": "#4C78A8",
    "current_only_v2": "#F58518",
    "sac_lag": "#54A24B",
    "star_v2": "#B279A2",
}


def fnum(value, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    out = np.empty_like(values, dtype=float)
    half = max(1, window // 2)
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out[idx] = np.nanmean(values[lo:hi])
    return out


def interpolate_seed(rows: list[dict], metric: str, grid: np.ndarray, smooth_window: int) -> np.ndarray | None:
    pairs = []
    for row in rows:
        step = fnum(row.get("step"))
        value = fnum(row.get(metric))
        if not math.isnan(step) and not math.isnan(value):
            pairs.append((step, value))
    if len(pairs) < 2:
        return None
    pairs = sorted(set(pairs))
    steps = np.array([p[0] for p in pairs], dtype=float)
    values = np.array([p[1] for p in pairs], dtype=float)
    values = rolling(values, smooth_window)
    valid_grid = grid[(grid >= steps.min()) & (grid <= steps.max())]
    if len(valid_grid) < 2:
        return None
    interp = np.full_like(grid, np.nan, dtype=float)
    mask = (grid >= steps.min()) & (grid <= steps.max())
    interp[mask] = np.interp(grid[mask], steps, values)
    return interp


def plot_panel(ax, grouped: dict, task: str, metric: str, ylabel: str, smooth_window: int) -> None:
    max_step = 0.0
    for method in METHODS:
        for seed_rows in grouped.get((task, method), {}).values():
            for row in seed_rows:
                max_step = max(max_step, fnum(row.get("step"), 0.0))
    if max_step <= 0:
        ax.text(0.5, 0.5, "No real curve logs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(task)
        ax.set_axis_off()
        return
    grid = np.linspace(0, max_step, 240)
    for method in METHODS:
        seed_arrays = []
        for seed, seed_rows in sorted(grouped.get((task, method), {}).items(), key=lambda item: int(item[0])):
            arr = interpolate_seed(seed_rows, metric, grid, smooth_window)
            if arr is not None:
                seed_arrays.append(arr)
        if not seed_arrays:
            continue
        data = np.vstack(seed_arrays)
        count = np.sum(~np.isnan(data), axis=0)
        total = np.nansum(data, axis=0)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        centered = np.where(~np.isnan(data), data - mean, np.nan)
        sq = np.nansum(centered * centered, axis=0)
        std = np.sqrt(np.divide(sq, count - 1, out=np.zeros_like(sq), where=count > 1))
        sem = np.divide(std, np.sqrt(np.maximum(count, 1)), out=np.zeros_like(std), where=count > 1)
        x = grid / 1000.0
        color = COLORS[method]
        ax.plot(x, mean, label=LABELS[method], color=color, linewidth=2.0)
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.16, linewidth=0)
    ax.set_title(task.replace("Safety", "").replace("-v0", ""))
    ax.set_xlabel("Environment steps (k)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.7)


def build_grouped(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        task = row.get("task", "")
        method = row.get("method", "")
        seed = row.get("seed", "")
        if task and method and seed:
            grouped[(task, method)][seed].append(row)
    return grouped


def write_figure(rows: list[dict], tasks: list[str], output_base: Path, smooth_window: int, *, appendix: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = build_grouped(rows)
    nrows = len(tasks)
    fig_width = 12.5
    fig_height = 3.2 * nrows
    fig, axes = plt.subplots(nrows, 2, figsize=(fig_width, fig_height), squeeze=False, sharex=False)
    for row_idx, task in enumerate(tasks):
        plot_panel(axes[row_idx, 0], grouped, task, "return_value", "Episode return", smooth_window)
        plot_panel(axes[row_idx, 1], grouped, task, "cumulative_cost", "Cumulative training cost", smooth_window)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dpi = 220 if appendix else 240
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def append_caption(path: Path, rows: list[dict]) -> None:
    kinds = sorted({row.get("source_kind", "") for row in rows if row.get("source_kind")})
    if "eval" in kinds and "train" not in kinds:
        first = "Figure 3 reports offline evaluation return and cumulative training cost over environment steps."
    elif "eval" in kinds:
        first = "Figure 3 reports available evaluation/training-log return and cumulative training cost over environment steps."
    else:
        first = "Figure 3 reports training-log episode return and cumulative training cost over environment steps."
    by_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("task") in MAIN_TASKS and row.get("method") in METHODS:
            by_seed[(row["task"], row["method"], row["seed"])].append(fnum(row.get("step")))
    partial = [
        key
        for key, steps in by_seed.items()
        if steps and min(step for step in steps if not math.isnan(step)) > 5000
    ]
    partial_text = ""
    if partial:
        partial_text = (
            " Some resumed seeds only have per-step resume logs after the 100k checkpoint; "
            "the mean and uncertainty bands use available seeds at each environment step."
        )
    text = (
        "\n\n"
        "## Figure 3 Training Curves\n\n"
        f"{first} Seed means are shown with 95% normal-approximation standard-error bands. "
        "No final-summary rows are used to draw the curves."
        f"{partial_text} Table 1 uses all completed final evaluations.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    marker = "## Figure 3 Training Curves"
    if marker in old:
        old = old.split(marker)[0].rstrip() + "\n"
    old_lines = [line for line in old.splitlines() if "`fig_training_curves`" not in line]
    old = "\n".join(old_lines).rstrip() + "\n"
    path.write_text(old + text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default="reports/star_v2_final/curves/training_curves_long.csv")
    parser.add_argument("--figure-dir", default="reports/star_v2_final/figures")
    parser.add_argument("--smooth-window", type=int, default=11)
    args = parser.parse_args()
    rows = read_rows(Path(args.curves))
    if not rows:
        raise SystemExit(f"no curve rows found in {args.curves}; refusing to create fake training curves")
    figure_dir = Path(args.figure_dir)
    write_figure(rows, MAIN_TASKS, figure_dir / "fig_training_curves", args.smooth_window)
    available_tasks = [task for task in ALL_TASKS if any(row.get("task") == task for row in rows)]
    if available_tasks:
        write_figure(rows, available_tasks, figure_dir / "appendix_training_curves_all_tasks", args.smooth_window, appendix=True)
    append_caption(figure_dir / "captions.md", rows)
    print(f"wrote {figure_dir / 'fig_training_curves.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
