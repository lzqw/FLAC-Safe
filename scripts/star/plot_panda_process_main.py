#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
PROCESS_ROOT = REPORT_ROOT / "process_viz"
FIG_ROOT = REPORT_ROOT / "figures"

STAR_COLOR = "#7A4FA3"
CURRENT_COLOR = "#3A78C2"
ACTOR_COLOR = "#1B8A3A"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.75,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def row_at_or_before(df: pd.DataFrame, step: int) -> pd.Series:
    eligible = df[df["step"] <= step].sort_values("step")
    if eligible.empty:
        return df.sort_values("step").iloc[0]
    return eligible.iloc[-1]


def geometry_from_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray, float, float]:
    obs = np.asarray([row["p_obs_x"], row["p_obs_y"]], dtype=float)
    goal = np.asarray([row["p_goal_x"], row["p_goal_y"]], dtype=float)
    return obs, goal, float(row["safe_margin"]), float(row["obstacle_radius"])


def add_geometry(ax, row: pd.Series, *, compact: bool = False) -> None:
    obs, goal, safe_margin, obstacle_radius = geometry_from_row(row)
    ax.add_patch(
        Circle(
            obs,
            safe_margin,
            facecolor="#F3B7B1",
            edgecolor="#C93A32",
            lw=0.75,
            alpha=0.23,
            zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            obs,
            obstacle_radius,
            facecolor="#C93A32",
            edgecolor="#8F1D18",
            lw=0.65,
            alpha=0.9,
            zorder=2,
        )
    )
    ax.scatter(goal[0], goal[1], marker="*", s=72 if compact else 105, color="#2F6FED", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.3, alpha=0.18)


def choose_process_steps(star: pd.DataFrame, audit_summary: pd.DataFrame) -> list[tuple[str, int]]:
    final = int(star["step"].max())
    approach_rows = audit_summary[audit_summary["role"] == "approach"]
    approach = int(approach_rows.iloc[0]["step"]) if not approach_rows.empty else max(1, min(14, final))
    audit = 16 if 16 <= final else final
    pass_step = min(final, max(audit + 2, approach + 2))
    if pass_step >= final and final > audit:
        pass_step = max(audit + 1, final - 1)
    stages = [
        ("Start", 0),
        ("Approach", approach),
        ("Audit", audit),
        ("Pass", pass_step),
        ("Goal", final),
    ]
    ordered: list[tuple[str, int]] = []
    last = -1
    for name, step in stages:
        step = int(max(0, min(step, final)))
        if name not in {"Start", "Goal"} and step <= last:
            step = min(final, last + 1)
        last = step
        ordered.append((name, step))
    return ordered


def draw_process_mini(ax, star: pd.DataFrame, step: int, label: str, *, audit: bool = False) -> None:
    star = star.sort_values("step")
    row = row_at_or_before(star, step)
    upto = star[star["step"] <= step]
    add_geometry(ax, row, compact=True)
    ax.plot(star["p_gripper_x"], star["p_gripper_y"], color="#C9A6C4", lw=0.8, alpha=0.38, zorder=3)
    ax.plot(upto["p_gripper_x"], upto["p_gripper_y"], color=STAR_COLOR, lw=1.7, zorder=4)
    ax.scatter(row["p_gripper_x"], row["p_gripper_y"], c=STAR_COLOR, s=38, zorder=10)
    ax.plot(
        [row["p_left_finger_x"], row["p_right_finger_x"]],
        [row["p_left_finger_y"], row["p_right_finger_y"]],
        c="#222222",
        lw=1.6,
        zorder=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label}\n$t={step}$", pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.35 if audit else 0.65)
        spine.set_edgecolor(STAR_COLOR if audit else "#444444")


def plot_process_strip(fig: plt.Figure, outer_spec, star: pd.DataFrame, audit_summary: pd.DataFrame) -> None:
    sub = outer_spec.subgridspec(1, 5, wspace=0.08)
    stages = choose_process_steps(star, audit_summary)
    for idx, (label, step) in enumerate(stages):
        ax = fig.add_subplot(sub[0, idx])
        draw_process_mini(ax, star, step, label, audit=(label == "Audit"))
    first_ax = fig.axes[-5]
    first_ax.text(
        -0.18,
        1.22,
        "(a) Gripper execution process",
        transform=first_ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
    )


def plot_audit_panel(ax, traj: pd.DataFrame, audit_rows: pd.DataFrame, audit_json: dict) -> None:
    step_rows = audit_rows[audit_rows["step"] == 16].copy()
    if step_rows.empty:
        raise RuntimeError("No audit_sequence_rows.csv entries for step 16")

    geom_row = traj[(traj["method"] == "STAR+Exec") & (traj["step"] <= 16)].sort_values("step").tail(1).iloc[0]
    add_geometry(ax, geom_row)

    start = step_rows.iloc[0]
    ax.scatter(start["start_x"], start["start_y"], c="#111111", s=28, zorder=9)

    current = step_rows[step_rows["candidate_type"] == "current_only"]
    corridor = step_rows[step_rows["candidate_type"] == "corridor_shadow"]
    actor = step_rows[step_rows["candidate_type"].isin(["actor_mean", "selected_exec"])]

    ax.scatter(
        current["endpoint_x"],
        current["endpoint_y"],
        s=20,
        color=CURRENT_COLOR,
        alpha=0.42,
        edgecolor="none",
        zorder=5,
    )
    sc = ax.scatter(
        corridor["endpoint_x"],
        corridor["endpoint_y"],
        c=corridor["predicted_QC"],
        cmap="coolwarm",
        vmin=0.0,
        vmax=max(0.8, float(corridor["predicted_QC"].max())),
        s=28,
        alpha=0.9,
        edgecolor="#333333",
        linewidth=0.2,
        zorder=6,
    )
    risky = corridor[corridor["is_high_risk"]]
    ax.scatter(
        risky["endpoint_x"],
        risky["endpoint_y"],
        s=62,
        facecolors="none",
        edgecolors="#B00020",
        linewidth=0.9,
        zorder=7,
    )
    if not actor.empty:
        mean = actor[actor["candidate_type"] == "actor_mean"].iloc[0]
        ax.annotate(
            "",
            xy=(mean["endpoint_x"], mean["endpoint_y"]),
            xytext=(mean["start_x"], mean["start_y"]),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=ACTOR_COLOR),
            zorder=8,
        )
        ax.scatter(mean["endpoint_x"], mean["endpoint_y"], marker="*", c=ACTOR_COLOR, s=95, zorder=10)

    ax.text(
        0.03,
        0.97,
        "$Q_{mean}=0.026 \\leq 0.05$\nmax shadow $Q=0.762$\nnot executed",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DDDDDD", lw=0.5, alpha=0.92),
    )
    ax.set_title("(b) Local STAR audit at $t=16$")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.48, 0.63)
    ax.set_ylim(0.10, 0.25)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label("$Q_C^+$", fontsize=8)
    return audit_json


def plot_clearance_panel(ax, traj: pd.DataFrame, audit_json: dict) -> None:
    star = traj[traj["method"] == "STAR+Exec"].sort_values("step")
    ax.plot(star["step"], star["clearance"], color=STAR_COLOR, lw=2.0, label="STAR+Exec")
    ax.axhline(0.0, color="#333333", ls="--", lw=0.85)
    ax.axvline(16, color="#555555", ls="--", lw=0.85)
    min_row = star.loc[star["clearance"].idxmin()]
    ax.scatter(min_row["step"], min_row["clearance"], c="#B00020", s=32, zorder=8)
    ax.annotate(
        f"min {min_row['clearance']:.3f} m",
        xy=(min_row["step"], min_row["clearance"]),
        xytext=(min_row["step"] + 1.6, min_row["clearance"] + 0.018),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"),
    )
    ax.text(
        0.58,
        0.12,
        f"lift={audit_json['corridor_lift']:.3f}\nsuccess, cost=0",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DDDDDD", lw=0.5, alpha=0.92),
    )
    ax.set_ylim(-0.012, float(star["clearance"].max()) + 0.018)
    ax.text(16.25, -0.006, "audit", fontsize=7.6, color="#444444")
    ax.set_title("(c) Clearance profile")
    ax.set_xlabel("step")
    ax.set_ylabel("clearance to keep-out (m)")
    ax.grid(True, lw=0.32, alpha=0.22)


def write_caption(audit_json: dict) -> None:
    caption = (
        "Panda gripper process visualization. "
        "(a) The gripper moves from start to target in a selected STAR+Exec episode; the audit state is highlighted. "
        "(b) At the audit state, the actor mean is predicted safe "
        f"($Q_C^+={audit_json['actor_mean_Q']:.4f} \\le d_\\mathrm{{aud}}={audit_json['d_aud']:.2f}$), "
        "while reference-to-current shadows include high-risk actions "
        f"(max $Q_C^+={audit_json['max_corridor_Q']:.4f}$) that are queried but not executed. "
        "(c) Clearance to the keep-out boundary remains positive after the audit step; the dashed line marks the unsafe boundary. "
        "This is a qualitative mechanism visualization, not a standalone Panda benchmark claim."
    )
    (FIG_ROOT / "fig_panda_process_main_caption.tex").write_text(caption + "\n")


def main() -> None:
    setup_style()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(PROCESS_ROOT / "gripper_trajectory_rows.csv")
    audit_rows = pd.read_csv(PROCESS_ROOT / "audit_sequence_rows.csv")
    audit_summary = pd.read_csv(PROCESS_ROOT / "audit_sequence_summary.csv")
    audit_json = json.loads((REPORT_ROOT / "qualitative" / "audit_summary.json").read_text())
    star = traj[traj["method"] == "STAR+Exec"].copy()
    if star.empty:
        raise RuntimeError("No STAR+Exec rows in gripper_trajectory_rows.csv")

    fig = plt.figure(figsize=(14.2, 6.4), dpi=250)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.82, 1.18], wspace=0.28, hspace=0.38)
    plot_process_strip(fig, gs[0, :], star, audit_summary)
    ax_audit = fig.add_subplot(gs[1, 0])
    plot_audit_panel(ax_audit, traj, audit_rows, audit_json)
    ax_clearance = fig.add_subplot(gs[1, 1])
    plot_clearance_panel(ax_clearance, traj, audit_json)

    base = FIG_ROOT / "fig_panda_process_main"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    write_caption(audit_json)
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
