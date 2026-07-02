#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from main_star import reset_env, step_env
from scripts.star.find_panda_mechanism_episode import ee_pos, load_run_dir, seed_episode_rng, unwrap

try:
    import pybullet as p
except Exception:  # pragma: no cover
    p = None

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
QUAL_ROOT = REPORT_ROOT / "qualitative"
RENDER_ROOT = REPORT_ROOT / "rendered_keyframes"
FIG_ROOT = REPORT_ROOT / "figures"
LATEX_ROOT = REPORT_ROOT / "latex"

RUN_DIR = Path(
    "/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/"
    "SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30"
)

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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_visual_goal(env, goal: np.ndarray) -> int | None:
    if p is None:
        return None
    u = unwrap(env)
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.025,
        rgbaColor=(0.05, 0.25, 0.95, 0.85),
        physicsClientId=u.client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=np.asarray(goal, dtype=float).tolist(),
        physicsClientId=u.client_id,
    )


def frame_row(env, step: int, cost: float, success: float, label: str) -> dict[str, Any]:
    u = unwrap(env)
    ee = ee_pos(env)
    obs = np.asarray(u.obstacle_pos, dtype=float)
    goal = np.asarray(u.goal_pos, dtype=float)
    dist_obs = float(np.linalg.norm(ee - obs))
    return {
        "label": label,
        "step": int(step),
        "p_ee": ee.tolist(),
        "p_goal": goal.tolist(),
        "p_obs": obs.tolist(),
        "obstacle_radius": float(u.cfg.obstacle_radius),
        "safe_margin": float(u.cfg.safe_margin),
        "clearance": dist_obs - float(u.cfg.safe_margin),
        "distance_to_goal": float(np.linalg.norm(ee - goal)),
        "distance_to_obstacle": dist_obs,
        "success": float(success),
        "cost": float(cost),
    }


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def render_logged_audit_state(seed: int, audit_step: int, out_dir: Path) -> dict[str, Any]:
    audit = pd.read_csv(QUAL_ROOT / "audit_snapshot.csv")
    row = audit.iloc[0]
    ee = row[["p_ee_x", "p_ee_y", "p_ee_z"]].to_numpy(float)
    goal = row[["p_goal_x", "p_goal_y", "p_goal_z"]].to_numpy(float)
    obs = row[["p_obs_x", "p_obs_y", "p_obs_z"]].to_numpy(float)
    run = load_run_dir(RUN_DIR, label="STAR", mode="raw")
    reset_env(run.env, seed=seed)
    u = unwrap(run.env)
    u.debug_set_positions(ee_pos=ee, goal_pos=goal, obstacle_pos=obs)
    add_visual_goal(run.env, goal)
    rgb = run.env.render()
    save_rgb(out_dir / "render_audit_state.png", rgb)
    frame = frame_row(run.env, audit_step, 0.0, 0.0, "audit_state")
    frame["path"] = str(out_dir / "render_audit_state.png")
    frame["render_method"] = "reconstructed_from_logged_audit_state"
    frame["source"] = str(QUAL_ROOT / "audit_snapshot.csv")
    run.env.close()
    return frame


def replay_star_exec(seed: int, audit_step: int, out_dir: Path) -> dict[str, Any]:
    run = load_run_dir(RUN_DIR, label="STAR", mode="star_exec")
    seed_episode_rng(run, seed)
    state = reset_env(run.env, seed=seed)
    u = unwrap(run.env)
    add_visual_goal(run.env, np.asarray(u.goal_pos, dtype=float))

    traj = pd.read_csv(QUAL_ROOT / "trajectory_rows.csv")
    star_exec = traj[traj["method"] == "STAR+Exec"].sort_values("step")
    near_step = int(star_exec[star_exec["step"] > 0].sort_values("clearance").iloc[0]["step"])
    final_target_step = int(star_exec["step"].max())
    capture_steps = {0: "start", near_step: "near_obstacle"}
    frames: dict[str, dict[str, Any]] = {}
    rgb_by_label: dict[str, np.ndarray] = {}

    done = False
    step = 0
    last_cost = 0.0
    last_success = 0.0
    while not done and step < int(run.config.eval_numsteps):
        if step in capture_steps:
            label = capture_steps[step]
            rgb_by_label[label] = run.env.render()
            frames[label] = frame_row(run.env, step, last_cost, last_success, label)
        action = run.agent.select_action(
            state,
            evaluate=True,
            execution_mode=run.mode,
            total_numsteps=300000,
            diagnostics=True,
        )
        state, _reward, cost, terminated, truncated, info = step_env(run.env, action, run.config.safe_env)
        last_cost = float(cost)
        last_success = float(info.get("success", 0.0))
        done = bool(terminated or truncated)
        step += 1
    rgb_by_label["final"] = run.env.render()
    frames["final"] = frame_row(run.env, step, last_cost, last_success, "final")
    frames["audit_state"] = render_logged_audit_state(seed, audit_step, out_dir)

    file_map = {
        "start": "render_start.png",
        "near_obstacle": "render_near_obstacle.png",
        "final": "render_final.png",
    }
    for label, filename in file_map.items():
        save_rgb(out_dir / filename, rgb_by_label[label])
        frames[label]["path"] = str(out_dir / filename)

    metadata = {
        "render_method": "hybrid_true_replay_plus_logged_audit_state_reconstruction",
        "seed": seed,
        "episode": 0,
        "checkpoint": str(RUN_DIR / "checkpoint" / "final.torch"),
        "camera": {
            "source": "SafetyPandaReachObstacleEnv.render",
            "width": 640,
            "height": 480,
            "cameraEyePosition": [0.75, -0.75, 0.85],
            "cameraTargetPosition": [0.50, 0.0, 0.25],
            "cameraUpVector": [0.0, 0.0, 1.0],
            "fov": 50,
            "renderer": "pybullet.ER_TINY_RENDERER",
        },
        "frames": frames,
        "notes": "Start, near-obstacle, and final frames are captured from a replayed STAR+Exec evaluation episode with the selected checkpoint, seed, and deterministic episode RNG. The audit-state frame is rendered by setting the PyBullet robot to the logged audit_snapshot.csv end-effector, goal, and obstacle positions, because the exact stochastic audit query state is preserved in logs rather than recoverable from the later trajectory replay. A blue goal visual marker is added at the actual goal position for rendering only.",
    }
    (out_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    run.env.close()
    return metadata


def add_geometry(ax, goal: np.ndarray, obs: np.ndarray, radius: float, margin: float) -> None:
    ax.add_patch(Circle(obs[:2], margin, facecolor="#F3B7B1", edgecolor="#C93A32", lw=0.9, alpha=0.25, label="keep-out"))
    ax.add_patch(Circle(obs[:2], radius, facecolor="#C93A32", edgecolor="#8F1D18", lw=0.8, alpha=0.9, label="obstacle"))
    ax.scatter(goal[0], goal[1], s=95, c="#2F6FED", marker="*", zorder=8, label="target")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.35, alpha=0.22)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_topdown_audit(ax, audit: pd.DataFrame) -> None:
    row = audit.iloc[0]
    goal = row[["p_goal_x", "p_goal_y", "p_goal_z"]].to_numpy(float)
    obs = row[["p_obs_x", "p_obs_y", "p_obs_z"]].to_numpy(float)
    ee = row[["p_ee_x", "p_ee_y", "p_ee_z"]].to_numpy(float)
    add_geometry(ax, goal, obs, float(row["obstacle_radius"]), float(row["safe_margin"]))
    ax.scatter(ee[0], ee[1], s=42, c="black", zorder=9, label="state")
    cur = audit[audit["candidate_type"] == "current_only"]
    cor = audit[audit["candidate_type"] == "corridor_shadow"]
    mean = audit[audit["candidate_type"] == "actor_mean"].iloc[0]
    ax.scatter(cur["endpoint_x"], cur["endpoint_y"], s=20, c="#3A78C2", alpha=0.45, edgecolor="none", label="current-only")
    sc = ax.scatter(
        cor["endpoint_x"],
        cor["endpoint_y"],
        c=cor["predicted_QC"],
        cmap="coolwarm",
        vmin=0.0,
        vmax=max(float(cor["predicted_QC"].max()), float(mean["d_aud"]) * 2.0),
        s=30,
        alpha=0.9,
        edgecolor="#333333",
        linewidth=0.25,
        label="corridor shadows",
    )
    risky = cor[cor["is_high_risk"]]
    ax.scatter(risky["endpoint_x"], risky["endpoint_y"], s=74, facecolors="none", edgecolors="#B00020", linewidth=1.1, label="high-risk")
    ax.annotate(
        "",
        xy=(float(mean["endpoint_x"]), float(mean["endpoint_y"])),
        xytext=(ee[0], ee[1]),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1B8A3A"),
        zorder=12,
    )
    ax.scatter(mean["endpoint_x"], mean["endpoint_y"], marker="*", s=95, c="#1B8A3A", edgecolor="white", linewidth=0.5, zorder=13, label="executed")
    ax.set_title("audit candidates")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    return sc


def make_overlay(out_dir: Path) -> None:
    audit = pd.read_csv(QUAL_ROOT / "audit_snapshot.csv")
    render = Image.open(out_dir / "render_audit_state.png")
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=220)
    axes[0].imshow(render)
    axes[0].set_title("true PyBullet render")
    axes[0].axis("off")
    sc = plot_topdown_audit(axes[1], audit)
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.015)
    cbar.set_label(r"predicted $Q_C^+$", fontsize=8)
    fig.tight_layout(w_pad=0.8)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(overlay_dir / f"render_audit_state_overlay.{suffix}", bbox_inches="tight")
    plt.close(fig)


def make_render_strip(out_dir: Path) -> None:
    setup_style()
    frame_files = [
        ("render_start.png", "start"),
        ("render_audit_state.png", "audit state"),
        ("render_near_obstacle.png", "near obstacle"),
        ("render_final.png", "final"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), dpi=220)
    for ax, (filename, label) in zip(axes, frame_files):
        ax.imshow(Image.open(out_dir / filename))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.tight_layout(w_pad=0.3)
    base = out_dir / "fig_panda_render_process_strip"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def make_combined_rendered_figure(out_dir: Path) -> None:
    setup_style()
    traj = pd.read_csv(QUAL_ROOT / "trajectory_rows.csv")
    render_overlay = Image.open(out_dir / "overlays" / "render_audit_state_overlay.png")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.75), dpi=220, gridspec_kw={"width_ratios": [1.35, 1.0, 1.05]})
    axes[0].imshow(render_overlay)
    axes[0].set_title("(a) Rendered state and audit")
    axes[0].axis("off")

    first = traj.iloc[0]
    goal = first[["p_goal_x", "p_goal_y", "p_goal_z"]].to_numpy(float)
    obs = first[["p_obs_x", "p_obs_y", "p_obs_z"]].to_numpy(float)
    add_geometry(axes[1], goal, obs, float(first["obstacle_radius"]), float(first["safe_margin"]))
    for label, group in traj.groupby("method", sort=False):
        group = group.sort_values("step")
        color = COLORS.get(label)
        axes[1].plot(group["p_ee_x"], group["p_ee_y"], lw=1.8, color=color, label=label)
        bad = group[group["cost"] > 0]
        if not bad.empty:
            axes[1].scatter(bad["p_ee_x"], bad["p_ee_y"], marker="x", s=18, color="#B00020", lw=0.8)
    axes[1].set_title("(b) End-effector trajectories")
    axes[1].legend(loc="upper left", frameon=False, fontsize=7)

    for label, group in traj.groupby("method", sort=False):
        color = COLORS.get(label)
        axes[2].plot(group["step"], group["clearance"], lw=1.8, color=color, label=label)
    axes[2].axhline(0.0, color="#333333", lw=0.8, ls="--")
    ymin = float(traj["clearance"].min())
    ymax = float(traj["clearance"].max())
    axes[2].axhspan(min(ymin, -0.12), 0.0, color="#F3B7B1", alpha=0.25, lw=0)
    axes[2].set_ylim(min(ymin - 0.015, -0.04), max(ymax + 0.02, 0.04))
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("clearance to keep-out (m)")
    axes[2].set_title("(c) Clearance profile")
    axes[2].grid(True, lw=0.35, alpha=0.25)
    axes[2].legend(loc="best", frameon=False, fontsize=7)
    fig.tight_layout(w_pad=0.9)
    base = FIG_ROOT / "fig_panda_mechanism_rendered"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def write_render_text() -> None:
    LATEX_ROOT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((QUAL_ROOT / "audit_summary.json").read_text())
    caption = (
        "Rendered Panda obstacle-reaching mechanism visualization. The rendered audit frame uses the logged real evaluation audit state, rendered by setting PyBullet to the recorded end-effector, goal, and obstacle positions; the process frames are captured from the STAR+Exec replay. "
        f"The actor mean is predicted safe ($Q_C^+(s,a_\\mathrm{{mean}})={summary['actor_mean_Q']:.4f}\\le d_\\mathrm{{aud}}={summary['d_aud']:.2f}$), "
        f"the reference-to-current corridor contains high-risk shadows (max $Q_C^+={summary['max_corridor_Q']:.4f}$), and corridor lift is {summary['corridor_lift']:.4f}. "
        f"Risky shadows are queried but not executed. The selected STAR+Exec episode succeeds with cost {summary['star_exec_cost']:.1f} and minimum clearance {summary['star_exec_min_clearance']:.3f} m. "
        "This is a qualitative mechanism visualization, not a standalone Panda benchmark claim."
    )
    paragraph = (
        "The rendered Panda figure complements the top-down audit plot with PyBullet frames tied to the selected evaluation episode. "
        "The audit-state render is reconstructed from the logged real audit state to keep the robot pose aligned with the saved critic-query candidates, while the start, near-obstacle, and final frames are true STAR+Exec replay frames. "
        "The audit state was selected by fixed criteria requiring a safe actor mean, a high-risk reference-to-current shadow, positive corridor lift, and non-execution of risky shadows. "
        "The STAR+Exec replay reaches the target with zero keep-out cost in the selected episode, while broader Panda results remain mixed; we therefore use the figure only as qualitative mechanism evidence."
    )
    (LATEX_ROOT / "panda_rendered_mechanism_caption.tex").write_text(caption + "\n")
    (LATEX_ROOT / "panda_rendered_mechanism_paragraph.tex").write_text(paragraph + "\n")
    reproduce_path = QUAL_ROOT / "REPRODUCE_PANDA_FIGURE.md"
    if reproduce_path.exists():
        text = reproduce_path.read_text()
        block = """\n## Rendered Panda Keyframes\n\nRegenerate PyBullet rendered keyframes and the rendered combined figure:\n\n```bash\ncd /root/FLAC-Safe-star-v2\nexport PYTHONPATH=.\n/root/miniconda3/envs/flac/bin/python scripts/star/render_panda_selected_episode.py\n```\n\nThe script writes `reports/star_arm_panda/rendered_keyframes/render_metadata.json`. Start, near-obstacle, and final frames are true STAR+Exec replay frames. The audit-state frame is rendered from the logged real audit state in `audit_snapshot.csv` so the robot pose matches the saved critic-query candidates.\n"""
        if "## Rendered Panda Keyframes" not in text:
            reproduce_path.write_text(text.rstrip() + "\n" + block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=900078)
    parser.add_argument("--audit-step", type=int, default=16)
    parser.add_argument("--out-dir", default=str(RENDER_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = replay_star_exec(args.seed, args.audit_step, out_dir)
    make_overlay(out_dir)
    make_render_strip(out_dir)
    make_combined_rendered_figure(out_dir)
    write_render_text()
    print(json.dumps({"out_dir": str(out_dir), "render_method": metadata["render_method"], "frames": list(metadata["frames"])}, indent=2))


if __name__ == "__main__":
    main()
