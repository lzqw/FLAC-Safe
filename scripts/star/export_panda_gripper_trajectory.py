#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from main_star import reset_env, step_env
from scripts.star.find_panda_mechanism_episode import load_run, load_run_dir, seed_episode_rng, unwrap

try:
    import pybullet as p
except Exception:  # pragma: no cover
    p = None

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
PROCESS_ROOT = REPORT_ROOT / "process_viz"
RUN_DIR = Path(
    "/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/"
    "SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30"
)


def link_position(env, link: int) -> np.ndarray:
    u = unwrap(env)
    state = p.getLinkState(u.robot_id, link, computeLinkVelocity=0, physicsClientId=u.client_id)
    return np.asarray(state[4], dtype=float)


def link_orientation(env, link: int) -> np.ndarray:
    u = unwrap(env)
    state = p.getLinkState(u.robot_id, link, computeLinkVelocity=0, physicsClientId=u.client_id)
    quat_xyzw = np.asarray(state[5], dtype=float)
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def safety_row(env, info: dict[str, Any]) -> dict[str, float]:
    u = unwrap(env)
    p_ee = link_position(env, u.EE_LINK)
    p_gripper = link_position(env, 11)
    p_left = link_position(env, 9)
    p_right = link_position(env, 10)
    quat = link_orientation(env, u.EE_LINK)
    obs = np.asarray(u.obstacle_pos, dtype=float)
    goal = np.asarray(u.goal_pos, dtype=float)
    dist_obs = float(np.linalg.norm(p_ee - obs))
    dist_goal = float(np.linalg.norm(p_ee - goal))
    return {
        "p_gripper_x": float(p_gripper[0]),
        "p_gripper_y": float(p_gripper[1]),
        "p_gripper_z": float(p_gripper[2]),
        "p_left_finger_x": float(p_left[0]),
        "p_left_finger_y": float(p_left[1]),
        "p_left_finger_z": float(p_left[2]),
        "p_right_finger_x": float(p_right[0]),
        "p_right_finger_y": float(p_right[1]),
        "p_right_finger_z": float(p_right[2]),
        "p_ee_x": float(p_ee[0]),
        "p_ee_y": float(p_ee[1]),
        "p_ee_z": float(p_ee[2]),
        "quat_w": float(quat[0]),
        "quat_x": float(quat[1]),
        "quat_y": float(quat[2]),
        "quat_z": float(quat[3]),
        "p_goal_x": float(goal[0]),
        "p_goal_y": float(goal[1]),
        "p_goal_z": float(goal[2]),
        "p_obs_x": float(obs[0]),
        "p_obs_y": float(obs[1]),
        "p_obs_z": float(obs[2]),
        "safe_margin": float(u.cfg.safe_margin),
        "obstacle_radius": float(u.cfg.obstacle_radius),
        "clearance": dist_obs - float(u.cfg.safe_margin),
        "distance_to_goal": dist_goal,
        "distance_to_obstacle": dist_obs,
        "cost": float(info.get("cost", 0.0)),
        "violation": bool(float(info.get("cost", 0.0)) > 0.0),
        "collision": bool(float(info.get("collision", 0.0)) > 0.0),
        "success": bool(float(info.get("success", 0.0)) > 0.0),
    }


def replay(run, seed: int, *, episode_success: bool | None = None) -> list[dict[str, Any]]:
    seed_episode_rng(run, seed)
    state = reset_env(run.env, seed=seed)
    info = unwrap(run.env)._safety_info()
    rows: list[dict[str, Any]] = []
    done = False
    step = 0
    while not done and step < int(run.config.eval_numsteps):
        row = {
            "method": run.label,
            "seed": seed,
            "episode": 0,
            "step": step,
            **safety_row(run.env, info),
            "selected_action_x": np.nan,
            "selected_action_y": np.nan,
            "selected_action_z": np.nan,
            "is_selected_episode": True,
            "train_seed": run.train_seed,
            "mode": run.mode,
            "run_name": run.run_name,
            "checkpoint_path": run.checkpoint,
        }
        action = run.agent.select_action(state, evaluate=True, execution_mode=run.mode, total_numsteps=300000, diagnostics=True)
        row["selected_action_x"] = float(action[0])
        row["selected_action_y"] = float(action[1])
        row["selected_action_z"] = float(action[2])
        rows.append(row)
        state, _reward, _cost, terminated, truncated, info = step_env(run.env, action, run.config.safe_env)
        done = bool(terminated or truncated)
        step += 1
    if episode_success is not None:
        for row in rows:
            row["success"] = bool(episode_success)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=900078)
    parser.add_argument("--compare-seed", type=int, default=11)
    parser.add_argument("--output", default=str(PROCESS_ROOT / "gripper_trajectory_rows.csv"))
    args = parser.parse_args()

    PROCESS_ROOT.mkdir(parents=True, exist_ok=True)
    methods = pd.read_csv(REPORT_ROOT / "qualitative" / "selected_episode_methods.csv")
    success_by_label = {row["method_label"]: bool(float(row["success"]) > 0.0) for _, row in methods.iterrows()}
    runs = [
        load_run("sac_lag", args.compare_seed, mode="raw"),
        load_run("current_only_v2", args.compare_seed, mode="raw"),
        load_run_dir(RUN_DIR, label="STAR", mode="raw"),
        load_run_dir(RUN_DIR, label="STAR", mode="star_exec"),
    ]
    rows: list[dict[str, Any]] = []
    try:
        for run in runs:
            rows.extend(replay(run, args.seed, episode_success=success_by_label.get(run.label)))
    finally:
        for run in runs:
            run.env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    metadata = {
        "seed": args.seed,
        "episode": 0,
        "finger_links_available": True,
        "left_finger_link": 9,
        "right_finger_link": 10,
        "gripper_link": 11,
        "notes": "Gripper and finger positions are read from PyBullet link states during true policy replays.",
    }
    (PROCESS_ROOT / "gripper_trajectory_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
