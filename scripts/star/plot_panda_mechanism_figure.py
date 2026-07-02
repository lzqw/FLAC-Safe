#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO / "reports" / "star_arm_panda"
QUAL_ROOT = REPORT_ROOT / "qualitative"
FIG_ROOT = REPORT_ROOT / "figures"

COLORS = {
    "SAC-Lag": "#4C78A8",
    "Current-only-N": "#F58518",
    "STAR": "#54A24B",
    "STAR+Exec": "#B279A2",
}


def parse_vec(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    raise TypeError(value)


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
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_geometry(ax, start, goal, obstacle, radius, margin, *, legend: bool = False):
    ax.add_patch(Circle(obstacle[:2], margin, facecolor="#F3B7B1", edgecolor="#C93A32", lw=0.9, alpha=0.28, label="keep-out" if legend else None))
    ax.add_patch(Circle(obstacle[:2], radius, facecolor="#C93A32", edgecolor="#8F1D18", lw=0.8, alpha=0.9, label="obstacle" if legend else None))
    ax.scatter(start[0], start[1], s=38, c="black", marker="o", zorder=8, label="start" if legend else None)
    ax.scatter(goal[0], goal[1], s=95, c="#2F6FED", marker="*", zorder=8, label="target" if legend else None)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, lw=0.35, alpha=0.22)


def endpoint_frame(snapshot: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    row = snapshot.iloc[0]
    if {"p_ee_x", "p_goal_x", "p_obs_x"}.issubset(snapshot.columns):
        ee = np.asarray([row["p_ee_x"], row["p_ee_y"], row["p_ee_z"]], dtype=float)
        goal = np.asarray([row["p_goal_x"], row["p_goal_y"], row["p_goal_z"]], dtype=float)
        obstacle = np.asarray([row["p_obs_x"], row["p_obs_y"], row["p_obs_z"]], dtype=float)
    else:
        ee = np.asarray([row["ee_x"], row["ee_y"], row["ee_z"]], dtype=float)
        goal = parse_vec(row["goal_pos"])
        obstacle = parse_vec(row["obstacle_pos"])
    return ee, goal, obstacle, float(row.get("obstacle_radius", 0.07)), float(row["safe_margin"])


def plot_audit(ax, snapshot: pd.DataFrame):
    ee, goal, obstacle, radius, margin = endpoint_frame(snapshot)
    start = ee.copy()
    add_geometry(ax, start, goal, obstacle, radius, margin, legend=False)
    ax.scatter(ee[0], ee[1], s=45, marker="o", c="black", zorder=9, label="state")

    cur = snapshot[snapshot["candidate_type"] == "current_only"]
    cor = snapshot[snapshot["candidate_type"] == "corridor_shadow"]
    mean = snapshot[snapshot["candidate_type"] == "actor_mean"].iloc[0]
    q_col = "predicted_QC" if "predicted_QC" in snapshot.columns else "q_cost"
    threshold = float(mean["d_aud"] if "d_aud" in snapshot.columns else mean["threshold"])

    if not cur.empty:
        ax.scatter(cur["endpoint_x"], cur["endpoint_y"], s=20, c="#3A78C2", alpha=0.45, edgecolor="none", label="current-only samples")
    if not cor.empty:
        q = cor[q_col].to_numpy(float)
        vmax = max(threshold * 2.0, float(np.nanmax(q)) if len(q) else threshold)
        sc = ax.scatter(cor["endpoint_x"], cor["endpoint_y"], c=q, cmap="coolwarm", vmin=0.0, vmax=vmax, s=30, alpha=0.9, edgecolor="#333333", linewidth=0.25, label="STAR corridor shadows")
        risky = cor[cor[q_col] > threshold]
        if not risky.empty:
            ax.scatter(risky["endpoint_x"], risky["endpoint_y"], s=74, facecolors="none", edgecolors="#B00020", linewidth=1.1, label="high-risk queried")
        return_sc = sc
    else:
        return_sc = None

    ax.annotate(
        "",
        xy=(float(mean["endpoint_x"]), float(mean["endpoint_y"])),
        xytext=(ee[0], ee[1]),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1B8A3A"),
        zorder=12,
    )
    ax.scatter(float(mean["endpoint_x"]), float(mean["endpoint_y"]), marker="*", s=95, c="#1B8A3A", edgecolor="white", linewidth=0.5, zorder=13, label="actor mean executed")
    ax.set_title("(a) Local shadow audit")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", frameon=False, handlelength=1.4, borderpad=0.2)
    return return_sc


def plot_trajectories(ax, traj: pd.DataFrame):
    first = traj.iloc[0]
    if {"p_ee_x", "p_goal_x", "p_obs_x"}.issubset(traj.columns):
        start = traj.sort_values("step").iloc[0][["p_ee_x", "p_ee_y", "p_ee_z"]].to_numpy(float)
        goal = first[["p_goal_x", "p_goal_y", "p_goal_z"]].to_numpy(float)
        obstacle = first[["p_obs_x", "p_obs_y", "p_obs_z"]].to_numpy(float)
    else:
        start = parse_vec(first["start_pos"])
        goal = parse_vec(first["goal_pos"])
        obstacle = parse_vec(first["obstacle_pos"])
    radius = float(first.get("obstacle_radius", 0.07))
    margin = float(first["safe_margin"])
    add_geometry(ax, start, goal, obstacle, radius, margin, legend=True)
    method_col = "method_label" if "method_label" in traj.columns else "method"
    x_col = "ee_x" if "ee_x" in traj.columns else "p_ee_x"
    y_col = "ee_y" if "ee_y" in traj.columns else "p_ee_y"
    for label, group in traj.groupby(method_col, sort=False):
        group = group.sort_values("step")
        x = group[x_col].to_numpy(float)
        y = group[y_col].to_numpy(float)
        color = COLORS.get(label, None)
        ax.plot(x, y, lw=1.9, color=color, label=label)
        ax.scatter(x[-1], y[-1], s=30, color=color, zorder=8)
        bad = group[group["cost"] > 0]
        if not bad.empty:
            ax.scatter(bad[x_col], bad[y_col], marker="x", s=20, color="#B00020", lw=0.9, zorder=9)
    ax.set_title("(b) End-effector trajectories")
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.5)


def plot_clearance(ax, traj: pd.DataFrame):
    method_col = "method_label" if "method_label" in traj.columns else "method"
    clearance_col = "clearance_keepout" if "clearance_keepout" in traj.columns else "clearance"
    for label, group in traj.groupby(method_col, sort=False):
        group = group.sort_values("step")
        color = COLORS.get(label, None)
        ax.plot(group["step"], group[clearance_col], lw=1.8, color=color, label=label)
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ymin = float(np.nanmin(traj[clearance_col]))
    ymax = float(np.nanmax(traj[clearance_col]))
    ax.axhspan(min(ymin, -0.12), 0.0, color="#F3B7B1", alpha=0.25, lw=0)
    ax.set_ylim(min(ymin - 0.015, -0.04), max(ymax + 0.02, 0.04))
    ax.set_xlabel("step")
    ax.set_ylabel("clearance to keep-out (m)")
    ax.grid(True, lw=0.35, alpha=0.25)
    ax.set_title("(c) Clearance profile")
    ax.legend(loc="best", frameon=False, handlelength=1.5)


def write_text(selected: dict, methods: pd.DataFrame, snapshot: pd.DataFrame, out_dir: Path) -> None:
    state = selected["selected_state"]
    actor_safe = float(state["mean_action_risk"]) <= float(state["threshold"])
    risky = float(state["corridor_q_max"]) > float(state["threshold"])
    q_col = "predicted_QC" if "predicted_QC" in snapshot.columns else "q_cost"
    threshold_col = "d_aud" if "d_aud" in snapshot.columns else "threshold"
    risky_executed = bool((snapshot[(snapshot["candidate_type"] == "corridor_shadow") & (snapshot[q_col] > float(state["threshold"]))]["executed"] > 0).any()) if not snapshot.empty else False
    star_row = methods[(methods["method_label"] == "STAR")].iloc[0]
    current_row = methods[(methods["method_label"] == "Current-only-N")].iloc[0]
    sac_row = methods[(methods["method_label"] == "SAC-Lag")].iloc[0]
    exec_rows = methods[(methods["method_label"] == "STAR+Exec")]
    exec_row = exec_rows.iloc[0] if not exec_rows.empty else None
    star_exec_success = bool(exec_row is not None and float(exec_row["success"]) > 0.0)
    star_exec_safe = bool(exec_row is not None and float(exec_row["episode_cost"]) <= 0.0)
    positive_lift = float(state["corridor_lift"]) > 0.0

    caption = (
        "Panda obstacle-reaching STAR mechanism visualization. "
        "(a) At a real evaluation state, the STAR checkpoint queries local current-policy samples and reference-to-current corridor shadows; "
        "redder endpoints have higher predicted safety cost. High-risk shadow endpoints are critic queries and are not executed. "
        "(b) End-effector trajectories under the same seeded obstacle-reaching setup, including STAR+Exec on the same checkpoint. Red x markers indicate steps with nonzero keep-out cost. "
        "(c) Clearance to the keep-out boundary; values below zero indicate unsafe keep-out entry."
    )
    (out_dir / "fig_panda_mechanism_caption.txt").write_text(caption + "\n")

    lines = [
        "# Panda Mechanism Figure Text",
        "",
        "The figure is generated from real policy rollouts and critic evaluations, not hand-drawn or imputed data.",
        "",
        f"Selected STAR checkpoint: `{selected['selected_state']['checkpoint_path']}`",
        f"Selected eval seed / step: `{selected['selected_eval_seed']}` / `{selected['selected_step']}`",
        f"Actor mean predicted cost: `{float(state['mean_action_risk']):.6f}` with threshold `{float(state['threshold']):.6f}`; actor mean safe = `{actor_safe}`.",
        f"Max corridor shadow predicted cost: `{float(state['corridor_q_max']):.6f}`; high-risk corridor shadow found = `{risky}`.",
        f"Corridor lift over matched current-only samples: `{float(state['corridor_lift']):.6f}`.",
        f"Risky corridor shadows executed: `{risky_executed}`.",
        "",
        "Episode comparison on the selected seed:",
        f"- SAC-Lag: success={float(sac_row['success']):.3f}, cost={float(sac_row['episode_cost']):.3f}, min_clearance={float(sac_row['min_clearance']):.3f}",
        f"- Current-only-N: success={float(current_row['success']):.3f}, cost={float(current_row['episode_cost']):.3f}, min_clearance={float(current_row['min_clearance']):.3f}",
        f"- STAR: success={float(star_row['success']):.3f}, cost={float(star_row['episode_cost']):.3f}, min_clearance={float(star_row['min_clearance']):.3f}",
    ]
    if exec_row is not None:
        lines.append(f"- STAR+Exec: success={float(exec_row['success']):.3f}, cost={float(exec_row['episode_cost']):.3f}, min_clearance={float(exec_row['min_clearance']):.3f}")
    lines.append("")
    if star_exec_success and star_exec_safe:
        lines.append("Interpretation: the selected ref-corridor showcase checkpoint exposes high-risk nearby shadow actions; STAR+Exec on the same checkpoint reaches the target while avoiding keep-out cost. Raw STAR on this seed is not a clean safe-success trajectory and should be reported as a caveat.")
    else:
        lines.append("Interpretation caveat: the audit state is real and exposes high-risk nearby shadow actions, but the selected trajectory comparison is mixed and should not be described as a clean safe-success STAR rollout.")
    lines.append("This is a controlled showcase result for mechanism visualization, not a replacement for the main quantitative benchmark tables.")
    (out_dir / "panda_mechanism_text.md").write_text("\n".join(lines) + "\n")

    paper_ready = bool(actor_safe and risky and not risky_executed and positive_lift and star_exec_success and star_exec_safe)
    showcase_summary_path = REPORT_ROOT / "showcase" / "showcase_summary.csv"
    showcase_note = ""
    if showcase_summary_path.exists():
        showcase = pd.read_csv(showcase_summary_path)
        exec_summary = showcase[showcase["method_label"] == "STAR+Exec"]
        if not exec_summary.empty:
            row = exec_summary.iloc[0]
            showcase_note = (
                " Final 910000-910049 showcase evaluation is not a broad success result: "
                f"STAR+Exec success_mean={float(row['success_mean']):.3f}, "
                f"episode_cost_mean={float(row['episode_cost_mean']):.3f}, "
                f"min_clearance_mean={float(row['min_clearance_mean']):.3f}."
            )
    audit = [
        "# Panda Mechanism Final Audit",
        "",
        f"1. Is the figure generated from real logs? Yes: `{QUAL_ROOT / 'trajectory_rows.csv'}` and `{QUAL_ROOT / 'audit_snapshot.csv'}`.",
        f"2. Checkpoint/seed/episode/state: `{selected['selected_state']['checkpoint_path']}`, eval seed `{selected['selected_eval_seed']}`, step `{selected['selected_step']}`.",
        f"3. Actor mean predicted safe? `{actor_safe}`; q={float(state['mean_action_risk']):.6f}, threshold={float(state['threshold']):.6f}.",
        f"4. High-risk corridor shadows found? `{risky}`; max q={float(state['corridor_q_max']):.6f}.",
        f"5. Were risky shadows executed? `{risky_executed}`.",
        f"6. Corridor lift positive? `{float(state['corridor_lift']) > 0.0}`; lift={float(state['corridor_lift']):.6f}.",
        f"7. Did STAR reach target? `{float(star_row['success']) > 0.0}`.",
        f"8. Did STAR avoid keep-out zone? `{float(star_row['episode_cost']) <= 0.0}`; episode_cost={float(star_row['episode_cost']):.3f}, min_clearance={float(star_row['min_clearance']):.3f}.",
        f"9. Did Current-only or SAC-Lag provide contrast? Current-only cost={float(current_row['episode_cost']):.3f}; SAC-Lag cost={float(sac_row['episode_cost']):.3f}.",
        f"9b. Did STAR+Exec reach target and avoid keep-out zone? `{star_exec_success and star_exec_safe}`; success={float(exec_row['success']) if exec_row is not None else float('nan'):.3f}, episode_cost={float(exec_row['episode_cost']) if exec_row is not None else float('nan'):.3f}, min_clearance={float(exec_row['min_clearance']) if exec_row is not None else float('nan'):.3f}.",
        f"10. Is the result paper-worthy? `{paper_ready}` for a qualitative controlled ref-corridor mechanism figure; not as a standalone benchmark-win claim.",
        "11. Caveats: this is a targeted showcase checkpoint, not the main Panda benchmark table. Existing original final STAR checkpoints had reference=current at save time, so they could not support a reference-to-current corridor-lift claim. Raw STAR on the selected seed does not reach the target and enters the keep-out zone; STAR+Exec on the same checkpoint provides the clean safe-success trajectory for the selected episode." + showcase_note,
    ]
    (REPORT_ROOT / "final_mechanism_audit.md").write_text("\n".join(audit) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qual-dir", default=str(QUAL_ROOT))
    parser.add_argument("--out-dir", default=str(FIG_ROOT))
    args = parser.parse_args()
    qual_dir = Path(args.qual_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    selected = json.loads((qual_dir / "selected_episode.json").read_text())
    traj = pd.read_csv(qual_dir / "trajectory_rows.csv")
    snapshot = pd.read_csv(qual_dir / "audit_snapshot.csv")
    methods = pd.read_csv(qual_dir / "selected_episode_methods.csv")
    if snapshot.empty:
        raise RuntimeError(f"empty audit snapshot: {qual_dir / 'audit_snapshot.csv'}")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75), dpi=220)
    sc = plot_audit(axes[0], snapshot)
    plot_trajectories(axes[1], traj)
    plot_clearance(axes[2], traj)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.015)
        cbar.set_label(r"predicted $Q_C^+$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=1.0)
    base = out_dir / "fig_panda_mechanism"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    write_text(selected, methods, snapshot, out_dir)
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
