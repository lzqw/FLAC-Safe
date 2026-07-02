#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
PROCESS_ROOT = REPORT_ROOT / "process_viz"
FIG_ROOT = REPORT_ROOT / "figures"

COLORS = {
    "SAC-Lag": "#4C78A8",
    "Current-only-N": "#F58518",
    "STAR": "#54A24B",
    "STAR+Exec": "#B279A2",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_geometry(ax, row, *, legend: bool = False) -> None:
    obs = np.asarray([row["p_obs_x"], row["p_obs_y"], row["p_obs_z"]], dtype=float)
    goal = np.asarray([row["p_goal_x"], row["p_goal_y"], row["p_goal_z"]], dtype=float)
    ax.add_patch(Circle(obs[:2], float(row["safe_margin"]), facecolor="#F3B7B1", edgecolor="#C93A32", lw=0.9, alpha=0.25, label="keep-out" if legend else None))
    ax.add_patch(Circle(obs[:2], float(row["obstacle_radius"]), facecolor="#C93A32", edgecolor="#8F1D18", lw=0.8, alpha=0.9, label="obstacle" if legend else None))
    ax.scatter(goal[0], goal[1], marker="*", s=95, color="#2F6FED", zorder=8, label="target" if legend else None)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.35, alpha=0.22)


def stage_steps(traj: pd.DataFrame, audit_sel: pd.DataFrame) -> list[tuple[str, int]]:
    star = traj[traj["method"] == "STAR+Exec"].sort_values("step")
    final = int(star["step"].max())
    near = int(star.sort_values("clearance").iloc[0]["step"])
    audit = 16
    approach_candidates = audit_sel[audit_sel["role"] == "approach"]
    approach = int(approach_candidates.iloc[0]["step"]) if not approach_candidates.empty else max(1, min(audit - 4, final))
    pass_step = min(final, max(audit + 2, near + 2))
    avoid = near
    stages = [
        ("Start", 0),
        ("Approach", approach),
        ("Audit", min(audit, final)),
        ("Avoid", avoid),
        ("Pass", pass_step),
        ("Goal", final),
    ]
    seen: set[int] = set()
    out = []
    for name, step in stages:
        step = int(max(0, min(step, final)))
        if name != "Audit" and step in seen:
            step = int(min(final, step + 1))
        seen.add(step)
        out.append((name, step))
    return out


def draw_process_frame(ax, traj: pd.DataFrame, step: int, title: str, audit_steps: set[int]) -> None:
    star = traj[traj["method"] == "STAR+Exec"].sort_values("step")
    row = star[star["step"] <= step].tail(1).iloc[0]
    upto = star[star["step"] <= step]
    add_geometry(ax, row)
    ax.plot(star["p_gripper_x"], star["p_gripper_y"], color="#C9A6C4", lw=0.9, alpha=0.35)
    ax.plot(upto["p_gripper_x"], upto["p_gripper_y"], color=COLORS["STAR+Exec"], lw=2.0)
    ax.scatter(star.iloc[0]["p_gripper_x"], star.iloc[0]["p_gripper_y"], c="black", s=28, zorder=8)
    ax.scatter(row["p_gripper_x"], row["p_gripper_y"], c=COLORS["STAR+Exec"], s=46, zorder=9)
    ax.plot([row["p_left_finger_x"], row["p_right_finger_x"]], [row["p_left_finger_y"], row["p_right_finger_y"]], c="#333333", lw=2.0, zorder=10)
    if step in audit_steps:
        ax.scatter(row["p_gripper_x"], row["p_gripper_y"], s=95, facecolors="none", edgecolors="#B00020", lw=1.2, zorder=11)
    ax.set_title(f"{title}  t={step}")
    ax.set_xticks([])
    ax.set_yticks([])


def save_frames(traj: pd.DataFrame, audit_sel: pd.DataFrame) -> list[Path]:
    frames_dir = PROCESS_ROOT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audit_steps = set(audit_sel["step"].astype(int))
    names = [
        "frame_01_start.png",
        "frame_02_approach.png",
        "frame_03_audit.png",
        "frame_04_avoid.png",
        "frame_05_pass.png",
        "frame_06_goal.png",
    ]
    paths = []
    for (title, step), filename in zip(stage_steps(traj, audit_sel), names):
        fig, ax = plt.subplots(figsize=(3.0, 2.7), dpi=180)
        draw_process_frame(ax, traj, step, title, audit_steps)
        fig.tight_layout()
        path = frames_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_trajectory(ax, traj: pd.DataFrame, audit_sel: pd.DataFrame) -> None:
    first = traj.iloc[0]
    add_geometry(ax, first, legend=True)
    for label, group in traj.groupby("method", sort=False):
        group = group.sort_values("step")
        color = COLORS.get(label)
        ax.plot(group["p_gripper_x"], group["p_gripper_y"], lw=1.9, color=color, label=label)
        bad = group[group["cost"] > 0]
        if not bad.empty:
            ax.scatter(bad["p_gripper_x"], bad["p_gripper_y"], marker="x", s=18, color="#B00020", lw=0.8)
    star = traj[traj["method"] == "STAR+Exec"]
    for _, row in audit_sel.iterrows():
        candidates = star[star["step"] == int(row["step"])]
        if not candidates.empty:
            r = candidates.iloc[0]
            ax.scatter(r["p_gripper_x"], r["p_gripper_y"], s=75, facecolors="none", edgecolors="#111111", lw=1.0, zorder=9)
            ax.text(r["p_gripper_x"], r["p_gripper_y"], str(row["role"]), fontsize=6.5)
    ax.set_title("(b) Top-down gripper trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", frameon=False)


def plot_audit_fans(ax, audit_rows: pd.DataFrame, traj: pd.DataFrame) -> None:
    star = traj[traj["method"] == "STAR+Exec"].sort_values("step")
    first_step = int(audit_rows.sort_values("step").iloc[0]["step"])
    geometry_rows = star[star["step"] <= first_step]
    first = geometry_rows.tail(1).iloc[0] if not geometry_rows.empty else star.iloc[0]
    add_geometry(ax, first)
    roles = list(audit_rows[["step", "role", "source"]].drop_duplicates().sort_values("step")["role"])
    offsets = np.linspace(-0.004, 0.004, max(1, len(roles)))
    for offset, (role, group) in zip(offsets, audit_rows.groupby("role", sort=False)):
        cor = group[group["candidate_type"] == "corridor_shadow"]
        cur = group[group["candidate_type"] == "current_only"]
        mean = group[group["candidate_type"].isin(["actor_mean", "selected_exec"])]
        ax.scatter(cur["endpoint_x"], cur["endpoint_y"] + offset, s=12, c="#3A78C2", alpha=0.28, edgecolor="none")
        sc = ax.scatter(cor["endpoint_x"], cor["endpoint_y"] + offset, c=cor["predicted_QC"], cmap="coolwarm", vmin=0.0, vmax=max(0.8, audit_rows["predicted_QC"].max()), s=18, alpha=0.85, edgecolor="#333333", linewidth=0.15)
        risky = cor[cor["is_high_risk"]]
        ax.scatter(risky["endpoint_x"], risky["endpoint_y"] + offset, s=42, facecolors="none", edgecolors="#B00020", lw=0.8)
        if not mean.empty:
            m = mean.iloc[0]
            ax.scatter(m["start_x"], m["start_y"] + offset, c="black", s=18, zorder=8)
            ax.scatter(m["endpoint_x"], m["endpoint_y"] + offset, marker="*", c="#1B8A3A", s=50, zorder=9)
            ax.text(m["start_x"], m["start_y"] + offset, role, fontsize=6.5)
    ax.set_title("(c) Multi-step shadow audit fans")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return sc


def plot_timeline(ax, traj: pd.DataFrame, summary: pd.DataFrame) -> None:
    star = traj[traj["method"] == "STAR+Exec"].sort_values("step")
    ax.plot(star["step"], star["clearance"], color=COLORS["STAR+Exec"], lw=2.0, label="STAR+Exec clearance")
    ax.axhline(0.0, color="#333333", ls="--", lw=0.8)
    ymin = float(min(star["clearance"].min() - 0.01, -0.04))
    ymax = float(max(star["clearance"].max() + 0.02, 0.08))
    ax.axhspan(ymin, 0.0, color="#F3B7B1", alpha=0.22, lw=0)
    for _, row in summary.iterrows():
        ax.axvline(int(row["step"]), color="#444444", lw=0.7, alpha=0.45)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("step")
    ax.set_ylabel("clearance (m)")
    ax.grid(True, lw=0.35, alpha=0.25)
    ax2 = ax.twinx()
    ax2.scatter(summary["step"], summary["max_Q_corridor"], marker="o", color="#B00020", s=28, label="max corridor Q")
    ax2.scatter(summary["step"], summary["Q_mean"], marker="*", color="#1B8A3A", s=55, label="actor mean Q")
    ax2.axhline(0.05, color="#777777", lw=0.7, ls=":")
    ax2.set_ylabel(r"audit $Q_C^+$")
    ax.set_title("(d) Clearance and audit-risk timeline")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)


def make_gif(frame_paths: list[Path]) -> None:
    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    out = PROCESS_ROOT / "panda_gripper_process.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=600, loop=0, optimize=True)


def write_text(summary: pd.DataFrame) -> None:
    audit = json.load(open(REPORT_ROOT / "qualitative" / "audit_summary.json"))
    caption = (
        "Multi-step Panda gripper process visualization. Process frames are generated from a real selected STAR+Exec episode, and the gripper/finger trajectory is logged from PyBullet link states. "
        f"The audit candidates are critic queries; at the selected audit state $Q_\\mathrm{{mean}}={audit['actor_mean_Q']:.4f}\\le d_\\mathrm{{aud}}={audit['d_aud']:.2f}$, max corridor $Q_C^+={audit['max_corridor_Q']:.4f}$, and corridor lift is {audit['corridor_lift']:.4f}. "
        f"Risky shadows are not executed; STAR+Exec succeeds with cost {audit['star_exec_cost']:.1f} and minimum clearance {audit['star_exec_min_clearance']:.3f} m. This is a qualitative mechanism visualization, not a standalone Panda benchmark claim."
    )
    paragraph = (
        "The gripper process figure shows the selected STAR+Exec episode as a sequence of actual PyBullet gripper states, together with multi-step shadow-audit fans. "
        "The fans expose candidate actions queried by the safety critic along the trajectory: actor-mean and selected actions are shown separately from current-only samples and reference-to-current corridor shadows. "
        "The selected audit state demonstrates the intended mechanism with a safe actor mean, high-risk nearby corridor shadows, positive corridor lift, and no execution of risky shadows."
    )
    (PROCESS_ROOT / "panda_gripper_process_caption.tex").write_text(caption + "\n")
    (PROCESS_ROOT / "panda_gripper_process_paragraph.tex").write_text(paragraph + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    setup_style()
    PROCESS_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(PROCESS_ROOT / "gripper_trajectory_rows.csv")
    audit_sel = pd.read_csv(PROCESS_ROOT / "audit_step_selection.csv")
    audit_rows = pd.read_csv(PROCESS_ROOT / "audit_sequence_rows.csv")
    audit_summary = pd.read_csv(PROCESS_ROOT / "audit_sequence_summary.csv")
    frame_paths = save_frames(traj, audit_sel)
    make_gif(frame_paths)

    fig = plt.figure(figsize=(14.2, 8.4), dpi=220)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.92, 1.15], wspace=0.38, hspace=0.36)
    strip_gs = gs[0, :].subgridspec(1, 6, wspace=0.08)
    audit_steps = set(audit_sel["step"].astype(int))
    for i, ((title, step), path) in enumerate(zip(stage_steps(traj, audit_sel), frame_paths)):
        ax = fig.add_subplot(strip_gs[0, i])
        draw_process_frame(ax, traj, step, title, audit_steps)
    fig.axes[0].text(-0.12, 1.12, "(a) Gripper execution process", transform=fig.axes[0].transAxes, fontsize=11, fontweight="bold")

    ax_traj = fig.add_subplot(gs[1, 0])
    plot_trajectory(ax_traj, traj, audit_sel)
    ax_audit = fig.add_subplot(gs[1, 1])
    sc = plot_audit_fans(ax_audit, audit_rows, traj)
    cbar = fig.colorbar(sc, ax=ax_audit, fraction=0.046, pad=0.015)
    cbar.set_label(r"predicted $Q_C^+$", fontsize=8)
    ax_time = fig.add_subplot(gs[1, 2])
    plot_timeline(ax_time, traj, audit_summary)

    base = FIG_ROOT / "fig_panda_gripper_process"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    write_text(audit_summary)
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
