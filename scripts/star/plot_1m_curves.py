#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO / "reports" / "star_1m_curves"
CURVE_CSV = REPORT_ROOT / "curves" / "training_curves_long.csv"
FIGURE_ROOT = REPORT_ROOT / "figures"

TASK_ORDER = ["PointGoal1", "CarGoal1", "PointPush1"]
METHOD_ORDER = ["PPO-Lag", "CPO", "CSPO", "SAC-Lag", "Safe Flow Q", "STAR"]
METHOD_ALIASES = {
    "ppo_lag": "PPO-Lag",
    "PPO-Lag": "PPO-Lag",
    "cpo": "CPO",
    "CPO": "CPO",
    "cspo": "CSPO",
    "CSPO": "CSPO",
    "sac_lag": "SAC-Lag",
    "SAC-Lag": "SAC-Lag",
    "safe_flow_q": "Safe Flow Q",
    "safe-flow-q": "Safe Flow Q",
    "Safe-Flow-Q": "Safe Flow Q",
    "Safe Flow Q": "Safe Flow Q",
    "star_v2": "STAR",
    "star": "STAR",
    "STAR": "STAR",
}
COLORS = {
    "PPO-Lag": "#777777",
    "CPO": "#54A24B",
    "CSPO": "#2CB1A4",
    "SAC-Lag": "#4C78A8",
    "Safe Flow Q": "#F58518",
    "STAR": "#7A4FA3",
}


def parse_methods(value: str | None, *, all_methods: bool = False, star_only: bool = False, core: bool = False) -> list[str]:
    if all_methods:
        return METHOD_ORDER[:]
    if star_only:
        return ["STAR"]
    if core:
        return ["STAR", "SAC-Lag", "Safe Flow Q"]
    if not value:
        return METHOD_ORDER[:]
    methods: list[str] = []
    for part in value.split(","):
        key = part.strip()
        if not key:
            continue
        methods.append(METHOD_ALIASES.get(key, key))
    return methods


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "savefig.dpi": 260,
        }
    )


def read_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"curve CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"curve CSV is empty: {path}")
    for column in ["step", "return_value", "cost_value", "cumulative_cost"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["task", "method", "seed", "step"])
    df["method"] = df["method"].map(lambda value: METHOD_ALIASES.get(str(value), str(value)))
    return df.sort_values(["task", "method", "seed", "step"])


def value_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(0, 1_000_001, 10_000)
    series = []
    for (_task, _seed), seed_df in df.groupby(["task", "seed"], dropna=False):
        seed_df = seed_df.sort_values("step")
        x = seed_df["step"].to_numpy(dtype=float)
        y = seed_df[value_col].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if len(np.unique(x)) < 2:
            continue
        uniq_x, uniq_idx = np.unique(x, return_index=True)
        uniq_y = y[uniq_idx]
        interp = np.interp(grid, uniq_x, uniq_y, left=np.nan, right=np.nan)
        series.append(interp)
    if not series:
        nan = np.full_like(grid, np.nan, dtype=float)
        return grid, nan, nan, np.zeros_like(grid, dtype=int)
    arr = np.vstack(series)
    counts = np.sum(np.isfinite(arr), axis=0)
    mean = np.full_like(grid, np.nan, dtype=float)
    std = np.full_like(grid, np.nan, dtype=float)
    valid = counts > 0
    mean[valid] = np.nanmean(arr[:, valid], axis=0)
    std[valid] = np.nanstd(arr[:, valid], axis=0)
    return grid, mean, std, counts


def format_steps(ax) -> None:
    ticks = [0, 200_000, 400_000, 600_000, 800_000, 1_000_000]
    labels = ["0", "200k", "400k", "600k", "800k", "1M"]
    ax.set_xlim(0, 1_000_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")


def plot_task_grid(df: pd.DataFrame, methods: list[str], base: Path, title: str | None = None) -> None:
    fig, axes = plt.subplots(len(TASK_ORDER), 2, figsize=(11.2, 8.2), sharex=True)
    if title:
        fig.suptitle(title, y=0.995, fontsize=11)
    for row_idx, task in enumerate(TASK_ORDER):
        task_df = df[df["task"] == task]
        for method in methods:
            method_df = task_df[task_df["method"] == method]
            if method_df.empty:
                continue
            for col_idx, (value_col, ylabel) in enumerate(
                [("return_value", "Episode return"), ("cumulative_cost", "Cumulative cost")]
            ):
                ax = axes[row_idx, col_idx]
                grid, mean, std, counts = value_grid(method_df, value_col)
                valid = np.isfinite(mean)
                if not valid.any():
                    continue
                color = COLORS.get(method, "#333333")
                ax.plot(grid[valid], mean[valid], color=color, lw=1.6, label=method)
                ax.fill_between(
                    grid[valid],
                    mean[valid] - std[valid],
                    mean[valid] + std[valid],
                    color=color,
                    alpha=0.16,
                    lw=0,
                )
        axes[row_idx, 0].set_ylabel(f"{task}\n{axes[row_idx, 0].get_ylabel() or 'Episode return'}")
        axes[row_idx, 1].set_ylabel("Cumulative cost")
        for ax in axes[row_idx, :]:
            ax.grid(True, alpha=0.22, lw=0.45)
            format_steps(ax)
    axes[-1, 0].set_xlabel("Environment steps")
    axes[-1, 1].set_xlabel("Environment steps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_all(fig, base)
    plt.close(fig)


def plot_summary(df: pd.DataFrame, methods: list[str], base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    for method in methods:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue
        for ax, value_col, ylabel in [
            (axes[0], "return_value", "Task-mean episode return"),
            (axes[1], "cumulative_cost", "Task-mean cumulative cost"),
        ]:
            grid, mean, std, counts = value_grid(method_df, value_col)
            valid = np.isfinite(mean)
            if not valid.any():
                continue
            color = COLORS.get(method, "#333333")
            ax.plot(grid[valid], mean[valid], color=color, lw=1.7, label=method)
            ax.fill_between(
                grid[valid],
                mean[valid] - std[valid],
                mean[valid] + std[valid],
                color=color,
                alpha=0.16,
                lw=0,
            )
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22, lw=0.45)
            format_steps(ax)
    for ax in axes:
        ax.set_xlabel("Environment steps")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_all(fig, base)
    plt.close(fig)


def write_captions(output_dir: Path) -> None:
    lines = [
        "# STAR 1M Curve Captions",
        "",
        "1M training curves on the three Table-1 tasks. Lines show seed means and shaded regions show one standard deviation. STAR is evaluated without the optional STAR+Exec candidate filter.",
        "",
        "Curves are drawn from normalized training logs in `reports/star_1m_curves/curves/training_curves_long.csv`; no final-summary rows are used.",
        "",
        "If a method is absent from a panel, no usable per-step training log for that task/method/seed set was available at plot time.",
    ]
    (output_dir / "captions.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CURVE_CSV)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--methods", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--star-only", action="store_true")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--tasks-methods", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    setup_style()
    df = read_curves(args.input)
    methods = parse_methods(args.methods, all_methods=args.all, star_only=args.star_only, core=args.core)

    explicit = bool(args.methods or args.all or args.star_only or args.core or args.tasks_methods or args.summary)
    if args.star_only or not explicit:
        plot_task_grid(df, ["STAR"], output_dir / "fig_1m_curves_star_only", "STAR 1M training curves")
    if args.core or not explicit:
        plot_task_grid(df, ["STAR", "SAC-Lag", "Safe Flow Q"], output_dir / "fig_1m_curves_core_methods")
    if args.all or args.tasks_methods or not explicit:
        plot_task_grid(df, METHOD_ORDER, output_dir / "fig_1m_curves_tasks_methods")
        plot_task_grid(df, METHOD_ORDER, output_dir / "fig_1m_curves_all_methods")
    if args.summary or args.all or not explicit:
        plot_summary(df, METHOD_ORDER if not args.methods else methods, output_dir / "fig_1m_curves_summary")
    if args.methods and not any([args.star_only, args.core, args.all, args.tasks_methods, args.summary]):
        safe = "_".join(m.lower().replace(" ", "_").replace("-", "_") for m in methods)
        plot_task_grid(df, methods, output_dir / f"fig_1m_curves_{safe}")
    write_captions(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
