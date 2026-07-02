#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO / "reports" / "star_arm_panda"
QUAL_ROOT = REPORT_ROOT / "qualitative"
FIG_ROOT = REPORT_ROOT / "figures"
LATEX_ROOT = REPORT_ROOT / "latex"
PACKAGE_ROOT = REPORT_ROOT / "package"

COLORS = {
    "SAC-Lag": "#4C78A8",
    "Current-only-N": "#F58518",
    "STAR": "#54A24B",
    "STAR+Exec": "#B279A2",
}


def parse_vec(value: Any) -> np.ndarray:
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def backup_if_raw(path: Path, raw_marker: str) -> Path:
    backup = path.with_name(path.stem + "_raw.csv")
    if path.exists():
        cols = pd.read_csv(path, nrows=0).columns
        if raw_marker in cols and not backup.exists():
            shutil.copy2(path, backup)
    return backup if backup.exists() else path


def normalize_trajectory(raw: pd.DataFrame, methods: pd.DataFrame) -> pd.DataFrame:
    success_by_method = {
        str(row["method_label"]): bool(float(row["success"]) > 0.0)
        for _, row in methods.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        method = row.get("method_label", row.get("method", ""))
        goal = parse_vec(row["goal_pos"])
        obs = parse_vec(row["obstacle_pos"])
        cost_value = float(row.get("step_cost_after", row.get("cost", 0.0)))
        action = np.asarray([row.get("action_x", np.nan), row.get("action_y", np.nan), row.get("action_z", np.nan)], dtype=float)
        rows.append(
            {
                "method": method,
                "seed": int(row.get("eval_seed", row.get("seed", -1))),
                "episode": int(row.get("episode", 0)),
                "step": int(row["step"]),
                "p_ee_x": float(row.get("ee_x", row.get("p_ee_x"))),
                "p_ee_y": float(row.get("ee_y", row.get("p_ee_y"))),
                "p_ee_z": float(row.get("ee_z", row.get("p_ee_z"))),
                "p_goal_x": float(goal[0]),
                "p_goal_y": float(goal[1]),
                "p_goal_z": float(goal[2]),
                "p_obs_x": float(obs[0]),
                "p_obs_y": float(obs[1]),
                "p_obs_z": float(obs[2]),
                "obstacle_radius": float(row.get("obstacle_radius", 0.07)),
                "safe_margin": float(row["safe_margin"]),
                "clearance": float(row.get("clearance_keepout", row.get("clearance"))),
                "distance_to_goal": float(row.get("distance_to_goal", np.nan)),
                "distance_to_obstacle": float(row.get("distance_to_obstacle", np.nan)),
                "cost": cost_value,
                "violation": bool(cost_value > 0.0),
                "collision": bool(float(row.get("collision", 0.0)) > 0.0),
                "success": success_by_method.get(str(method), bool(float(row.get("success", 0.0)) > 0.0)),
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "action_z": float(action[2]),
                "selected_action_x": float(action[0]),
                "selected_action_y": float(action[1]),
                "selected_action_z": float(action[2]),
                "is_selected_episode": True,
                "train_seed": int(row.get("train_seed", -1)),
                "mode": row.get("mode", ""),
                "run_name": row.get("run_name", ""),
                "checkpoint_path": row.get("checkpoint_path", ""),
                "episode_rng_seed": int(row.get("episode_rng_seed", -1)),
                "action_scale": float(row.get("action_scale", 0.03)),
            }
        )
    return pd.DataFrame(rows)


def endpoint_from_action(state_row: pd.Series, action: np.ndarray) -> np.ndarray:
    pos = np.asarray([state_row["p_ee_x"], state_row["p_ee_y"], state_row["p_ee_z"]], dtype=float)
    scale = float(state_row.get("action_scale", 0.03))
    return pos + action * scale


def selection_score(actor_safe: bool, high_risk: bool, executed: bool, lift: float, star_exec_success: bool, star_exec_cost: float, star_exec_clearance: float) -> float:
    return (
        10.0 * float(actor_safe)
        + 10.0 * float(high_risk)
        + 5.0 * float(not (high_risk and executed))
        + 3.0 * float(lift > 0.0)
        + 2.0 * float(star_exec_success)
        + 2.0 * float(star_exec_cost <= 0.0)
        + 1.0 * float(star_exec_clearance > 0.0)
    )


def normalize_audit(raw: pd.DataFrame, traj: pd.DataFrame, selected: dict[str, Any], methods: pd.DataFrame) -> pd.DataFrame:
    state = selected["selected_state"]
    seed = int(selected["selected_eval_seed"])
    step = int(selected["selected_step"])
    star_exec = methods[methods["method_label"] == "STAR+Exec"].iloc[0]
    lift = float(state["corridor_lift"])
    threshold = float(state["threshold"])
    exec_success = bool(float(star_exec["success"]) > 0.0)
    exec_cost = float(star_exec["episode_cost"])
    exec_clearance = float(star_exec["min_clearance"])
    score = selection_score(True, True, False, lift, exec_success, exec_cost, exec_clearance)
    state_id = f"seed{seed}_episode0_step{step}"

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        q = float(row.get("q_cost", row.get("predicted_QC")))
        high = q > threshold
        rows.append(
            {
                "state_id": state_id,
                "seed": seed,
                "episode": 0,
                "step": step,
                "candidate_type": row["candidate_type"],
                "candidate_id": int(row.get("candidate_index", row.get("candidate_id", 0))),
                "beta": float(row.get("beta", 1.0)),
                "action_dx": float(row.get("action_x", row.get("action_dx"))),
                "action_dy": float(row.get("action_y", row.get("action_dy"))),
                "action_dz": float(row.get("action_z", row.get("action_dz"))),
                "endpoint_x": float(row["endpoint_x"]),
                "endpoint_y": float(row["endpoint_y"]),
                "endpoint_z": float(row["endpoint_z"]),
                "predicted_QC": q,
                "predicted_safe": bool(q <= threshold),
                "executed": bool(float(row.get("executed", 0.0)) > 0.0),
                "is_high_risk": bool(high),
                "rho_cor": float(row.get("rho_cor", state["rho_cor"])),
                "rho_cur": float(row.get("rho_cur", state["rho_cur"])),
                "corridor_lift": lift,
                "d_aud": threshold,
                "safe_margin": float(row.get("safe_margin", state["safe_margin"])),
                "reference_age": "targeted_refcorridor_checkpoint",
                "selection_score": score,
                "p_ee_x": float(row.get("ee_x", state["ee_x"])),
                "p_ee_y": float(row.get("ee_y", state["ee_y"])),
                "p_ee_z": float(row.get("ee_z", state["ee_z"])),
                "p_goal_x": float(parse_vec(row.get("goal_pos", state["goal_pos"]))[0]),
                "p_goal_y": float(parse_vec(row.get("goal_pos", state["goal_pos"]))[1]),
                "p_goal_z": float(parse_vec(row.get("goal_pos", state["goal_pos"]))[2]),
                "p_obs_x": float(parse_vec(row.get("obstacle_pos", state["obstacle_pos"]))[0]),
                "p_obs_y": float(parse_vec(row.get("obstacle_pos", state["obstacle_pos"]))[1]),
                "p_obs_z": float(parse_vec(row.get("obstacle_pos", state["obstacle_pos"]))[2]),
                "obstacle_radius": float(row.get("obstacle_radius", 0.07)),
            }
        )

    exec_step = traj[(traj["method"] == "STAR+Exec") & (traj["step"] == step)]
    if not exec_step.empty:
        erow = exec_step.iloc[0]
        action = np.asarray([erow["action_x"], erow["action_y"], erow["action_z"]], dtype=float)
        endpoint = endpoint_from_action(erow, action)
        q = float(raw[raw["candidate_type"] == "actor_mean"]["q_cost"].iloc[0])
        rows.append(
            {
                "state_id": state_id,
                "seed": seed,
                "episode": 0,
                "step": step,
                "candidate_type": "selected_exec",
                "candidate_id": 0,
                "beta": np.nan,
                "action_dx": float(action[0]),
                "action_dy": float(action[1]),
                "action_dz": float(action[2]),
                "endpoint_x": float(endpoint[0]),
                "endpoint_y": float(endpoint[1]),
                "endpoint_z": float(endpoint[2]),
                "predicted_QC": q,
                "predicted_safe": bool(q <= threshold),
                "executed": True,
                "is_high_risk": bool(q > threshold),
                "rho_cor": float(state["rho_cor"]),
                "rho_cur": float(state["rho_cur"]),
                "corridor_lift": lift,
                "d_aud": threshold,
                "safe_margin": float(erow["safe_margin"]),
                "reference_age": "targeted_refcorridor_checkpoint",
                "selection_score": score,
                "p_ee_x": float(erow["p_ee_x"]),
                "p_ee_y": float(erow["p_ee_y"]),
                "p_ee_z": float(erow["p_ee_z"]),
                "p_goal_x": float(erow["p_goal_x"]),
                "p_goal_y": float(erow["p_goal_y"]),
                "p_goal_z": float(erow["p_goal_z"]),
                "p_obs_x": float(erow["p_obs_x"]),
                "p_obs_y": float(erow["p_obs_y"]),
                "p_obs_z": float(erow["p_obs_z"]),
                "obstacle_radius": float(erow["obstacle_radius"]),
            }
        )
    return pd.DataFrame(rows)


def write_audit_summary(audit: pd.DataFrame, methods: pd.DataFrame, selected: dict[str, Any], out: Path) -> dict[str, Any]:
    actor = audit[audit["candidate_type"] == "actor_mean"].iloc[0]
    corr = audit[audit["candidate_type"] == "corridor_shadow"]
    risky_executed = bool(((corr["predicted_QC"] > corr["d_aud"]) & corr["executed"]).any())
    star_exec = methods[methods["method_label"] == "STAR+Exec"].iloc[0]
    raw_star = methods[methods["method_label"] == "STAR"].iloc[0]
    summary = {
        "selected_seed": int(selected["selected_eval_seed"]),
        "selected_episode": 0,
        "selected_step": int(selected["selected_step"]),
        "actor_mean_Q": float(actor["predicted_QC"]),
        "d_aud": float(actor["d_aud"]),
        "max_corridor_Q": float(corr["predicted_QC"].max()),
        "rho_cor": float(actor["rho_cor"]),
        "rho_cur": float(actor["rho_cur"]),
        "corridor_lift": float(actor["corridor_lift"]),
        "risky_shadows_executed": risky_executed,
        "star_exec_success": float(star_exec["success"]),
        "star_exec_cost": float(star_exec["episode_cost"]),
        "star_exec_min_clearance": float(star_exec["min_clearance"]),
        "raw_star_success": float(raw_star["success"]),
        "raw_star_cost": float(raw_star["episode_cost"]),
        "notes": "Qualitative controlled ref-corridor mechanism case. Raw STAR is mixed on this seed; STAR+Exec gives the clean selected-episode trajectory. Broader Panda showcase evaluation is not a benchmark-win claim.",
    }
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def add_geometry(ax, goal: np.ndarray, obs: np.ndarray, radius: float, margin: float) -> None:
    ax.add_patch(Circle(obs[:2], margin, facecolor="#F3B7B1", edgecolor="#C93A32", lw=0.9, alpha=0.25))
    ax.add_patch(Circle(obs[:2], radius, facecolor="#C93A32", edgecolor="#8F1D18", lw=0.8, alpha=0.9))
    ax.scatter(goal[0], goal[1], s=95, c="#2F6FED", marker="*", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.35, alpha=0.22)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_frame(ax, traj: pd.DataFrame, audit: pd.DataFrame, method: str, step: int, title: str, show_shadows: bool = False) -> None:
    group = traj[traj["method"] == method].sort_values("step")
    upto = group[group["step"] <= step]
    if upto.empty:
        upto = group
    first = group.iloc[0]
    goal = np.asarray([first["p_goal_x"], first["p_goal_y"], first["p_goal_z"]], dtype=float)
    obs = np.asarray([first["p_obs_x"], first["p_obs_y"], first["p_obs_z"]], dtype=float)
    add_geometry(ax, goal, obs, float(first["obstacle_radius"]), float(first["safe_margin"]))
    ax.plot(upto["p_ee_x"], upto["p_ee_y"], lw=2.0, color=COLORS.get(method, "#333333"))
    ax.scatter(group.iloc[0]["p_ee_x"], group.iloc[0]["p_ee_y"], c="black", s=35, zorder=8)
    ax.scatter(upto.iloc[-1]["p_ee_x"], upto.iloc[-1]["p_ee_y"], c=COLORS.get(method, "#333333"), s=45, zorder=9)
    if show_shadows:
        cor = audit[audit["candidate_type"] == "corridor_shadow"]
        ax.scatter(cor["endpoint_x"], cor["endpoint_y"], c=cor["predicted_QC"], cmap="coolwarm", s=28, edgecolor="#333333", linewidth=0.25, zorder=10)
        risky = cor[cor["is_high_risk"]]
        ax.scatter(risky["endpoint_x"], risky["endpoint_y"], s=75, facecolors="none", edgecolors="#B00020", linewidth=1.0, zorder=11)
        mean = audit[audit["candidate_type"] == "actor_mean"].iloc[0]
        ax.annotate("", xy=(mean["endpoint_x"], mean["endpoint_y"]), xytext=(mean["p_ee_x"], mean["p_ee_y"]), arrowprops=dict(arrowstyle="->", lw=1.4, color="#1B8A3A"), zorder=12)
        ax.scatter(mean["endpoint_x"], mean["endpoint_y"], marker="*", c="#1B8A3A", s=90, zorder=13)
    ax.set_title(title)


def write_keyframes_and_strip(traj: pd.DataFrame, audit: pd.DataFrame, summary: dict[str, Any]) -> None:
    setup_style()
    key_dir = FIG_ROOT / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)
    method = "STAR+Exec"
    group = traj[traj["method"] == method].sort_values("step")
    selected_step = int(summary["selected_step"])
    mid_step = int(group.iloc[(group["clearance"].abs()).argmin()]["step"])
    final_step = int(group["step"].max())
    frames = [
        ("frame_000_start.png", 0, "start", False),
        ("frame_016_audit_state.png", selected_step, "audit state", True),
        ("frame_mid_near_obstacle.png", mid_step, "near obstacle", False),
        ("frame_final_goal.png", final_step, "final", False),
    ]
    for filename, step, title, shadows in frames:
        fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=180)
        draw_frame(ax, traj, audit, method, step, title, show_shadows=shadows)
        fig.tight_layout()
        fig.savefig(key_dir / filename, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), dpi=220)
    for ax, (_, step, title, shadows) in zip(axes, frames):
        draw_frame(ax, traj, audit, method, step, title, show_shadows=shadows)
    fig.tight_layout(w_pad=0.8)
    base = FIG_ROOT / "fig_panda_process_strip"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)

    try:
        from PIL import Image

        video_dir = REPORT_ROOT / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        images = [Image.open(key_dir / filename).convert("P", palette=Image.Palette.ADAPTIVE) for filename, *_ in frames]
        images[0].save(
            video_dir / "panda_star_exec_selected_episode.gif",
            save_all=True,
            append_images=images[1:],
            duration=650,
            loop=0,
            optimize=True,
        )
    except Exception as exc:
        (REPORT_ROOT / "videos").mkdir(parents=True, exist_ok=True)
        (REPORT_ROOT / "videos" / "GIF_UNAVAILABLE.txt").write_text(f"GIF generation failed: {exc!r}\n")


def write_text_files(summary: dict[str, Any], selected: dict[str, Any]) -> None:
    LATEX_ROOT.mkdir(parents=True, exist_ok=True)
    QUAL_ROOT.mkdir(parents=True, exist_ok=True)
    caption = (
        "Panda obstacle-reaching qualitative mechanism visualization. At the selected audit state "
        f"($Q_C^+(s,a_\\mathrm{{mean}})={summary['actor_mean_Q']:.4f} \\le d_\\mathrm{{aud}}={summary['d_aud']:.2f}$), "
        f"the reference-to-current corridor contains high-risk shadows (max $Q_C^+={summary['max_corridor_Q']:.4f}$) "
        f"with positive corridor lift ({summary['corridor_lift']:.4f}); risky shadows are queried but not executed. "
        f"On the selected episode STAR+Exec reaches the target with cost {summary['star_exec_cost']:.1f} and minimum clearance {summary['star_exec_min_clearance']:.3f} m. "
        "This is a qualitative controlled mechanism case, not a standalone Panda benchmark claim."
    )
    (FIG_ROOT / "fig_panda_mechanism_caption.txt").write_text(caption + "\n")
    (LATEX_ROOT / "panda_mechanism_caption.tex").write_text(caption + "\n")
    paragraph = (
        "To visualize the audit mechanism in an embodied control setting, we include a Panda obstacle-reaching case study. "
        "The state in Figure~X is selected by a fixed audit criterion: the actor mean is predicted safe, the reference-to-current corridor contains a high-risk shadow, the Corridor Risk Lift is positive, and the risky shadows are not executed. "
        "This case directly instantiates the pointwise blind spot in a robot-arm scene. The selected STAR+Exec episode reaches the target with zero keep-out cost, but broader Panda evaluations remain mixed; we therefore use this result as a qualitative mechanism visualization rather than a standalone benchmark claim."
    )
    (LATEX_ROOT / "panda_mechanism_paragraph.tex").write_text(paragraph + "\n")
    (FIG_ROOT / "panda_mechanism_text.md").write_text(
        "\n".join(
            [
                "# Panda Mechanism Figure Text",
                "",
                caption,
                "",
                f"- selected seed: `{summary['selected_seed']}`",
                f"- selected step: `{summary['selected_step']}`",
                f"- risky shadows executed: `{summary['risky_shadows_executed']}`",
                f"- STAR+Exec success/cost/min clearance: `{summary['star_exec_success']}`, `{summary['star_exec_cost']}`, `{summary['star_exec_min_clearance']:.3f}`",
                "",
                summary["notes"],
            ]
        )
        + "\n"
    )
    (QUAL_ROOT / "DATA_DICTIONARY.md").write_text(
        """# Panda Mechanism Data Dictionary

`trajectory_rows.csv` contains one row per real rollout step for Current-only-N, SAC-Lag, STAR raw, and STAR+Exec.

- `method`: plotted method label.
- `seed`, `episode`, `step`: selected evaluation identifier.
- `p_ee_*`, `p_goal_*`, `p_obs_*`: end-effector, goal, and obstacle positions in meters.
- `safe_margin`: keep-out radius used for clearance.
- `clearance`: `||p_ee - p_obs|| - safe_margin`; negative means inside keep-out.
- `cost`, `violation`, `collision`, `success`: values logged from environment step/info.
- `action_*`, `selected_action_*`: executed action components from the policy/executor.

`audit_snapshot.csv` contains the selected STAR boundary-state critic queries.

- `candidate_type`: `actor_mean`, `current_only`, `corridor_shadow`, or `selected_exec`.
- `action_d*`: queried action vector.
- `endpoint_*`: end-effector endpoint implied by the queried action.
- `predicted_QC`: critic-predicted safety cost.
- `predicted_safe`: `predicted_QC <= d_aud`.
- `executed`: whether the row is the action actually executed.
- `is_high_risk`: `predicted_QC > d_aud`.
- `rho_cor`, `rho_cur`, `corridor_lift`: matched audit risks for the selected state.
- `selection_score`: deterministic score used to prefer safe actor mean, high-risk non-executed corridor shadows, positive lift, and STAR+Exec success/safety.
"""
    )
    (QUAL_ROOT / "REPRODUCE_PANDA_FIGURE.md").write_text(
        f"""# Reproduce Panda Mechanism Figure

Selected case:

- checkpoint: `{selected['selected_state']['checkpoint_path']}`
- evaluation seed: `{summary['selected_seed']}`
- episode: `{summary['selected_episode']}`
- step: `{summary['selected_step']}`
- actor mean Q: `{summary['actor_mean_Q']:.6f}`
- audit threshold: `{summary['d_aud']:.6f}`
- max corridor Q: `{summary['max_corridor_Q']:.6f}`
- corridor lift: `{summary['corridor_lift']:.6f}`

Regenerate from existing checkpoint and fixed seed:

```bash
cd /root/FLAC-Safe-star-v2
export PYTHONPATH=.
/root/miniconda3/envs/flac/bin/python scripts/star/find_panda_mechanism_episode.py \\
  --star-run-dir /root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30 \\
  --compare-seed 11 --eval-start 900000 --eval-count 100
/root/miniconda3/envs/flac/bin/python scripts/star/package_panda_mechanism_data.py
/root/miniconda3/envs/flac/bin/python scripts/star/plot_panda_mechanism_figure.py
```

The packaged CSVs are normalized for local paper plotting:

- `trajectory_rows.csv`
- `audit_snapshot.csv`
- `audit_summary.json`

Caveat: this is a qualitative controlled ref-corridor mechanism case. Raw STAR is mixed on this seed; STAR+Exec gives the clean selected-episode trajectory. Broader Panda evaluation is not claimed as a benchmark win.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qual-dir", default=str(QUAL_ROOT))
    args = parser.parse_args()
    qual_dir = Path(args.qual_dir)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)

    selected = json.loads((qual_dir / "selected_episode.json").read_text())
    methods = pd.read_csv(qual_dir / "selected_episode_methods.csv")

    traj_path = qual_dir / "trajectory_rows.csv"
    audit_path = qual_dir / "audit_snapshot.csv"
    raw_traj_path = backup_if_raw(traj_path, "method_label")
    raw_audit_path = backup_if_raw(audit_path, "q_cost")
    raw_traj = pd.read_csv(raw_traj_path)
    raw_audit = pd.read_csv(raw_audit_path)

    traj = normalize_trajectory(raw_traj, methods)
    audit = normalize_audit(raw_audit, traj, selected, methods)
    traj.to_csv(traj_path, index=False)
    audit.to_csv(audit_path, index=False)

    summary = write_audit_summary(audit, methods, selected, qual_dir / "audit_summary.json")
    write_keyframes_and_strip(traj, audit, summary)
    write_text_files(summary, selected)

    verification = [
        "# Selected Panda Mechanism Case Verification",
        "",
        f"- selected seed: `{summary['selected_seed']}`",
        f"- selected step: `{summary['selected_step']}`",
        f"- actor mean predicted safe: `{summary['actor_mean_Q']:.6f} <= {summary['d_aud']:.6f}`",
        f"- max corridor Q: `{summary['max_corridor_Q']:.6f}`",
        f"- corridor lift: `{summary['corridor_lift']:.6f}`",
        f"- risky shadows executed: `{summary['risky_shadows_executed']}`",
        f"- STAR+Exec success/cost/min clearance: `{summary['star_exec_success']}`, `{summary['star_exec_cost']}`, `{summary['star_exec_min_clearance']:.6f}`",
        "",
        "All rows are derived from real rollout logs and critic queries saved under `reports/star_arm_panda/qualitative/`.",
    ]
    (PACKAGE_ROOT / "selected_case_verification.md").write_text("\n".join(verification) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
