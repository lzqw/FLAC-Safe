#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.star_agent import STARAgent
from main_star import make_env, reset_env, step_env
from scripts.star.goal_panda_arm import CONFIG_DIR, TASK, _config_from_spec, final_specs
from utilis.star_default_config import star_default_config

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
QUAL_ROOT = REPORT_ROOT / "qualitative"

METHOD_LABELS = {
    "sac_lag": "SAC-Lag",
    "current_only_v2": "Current-only-N",
    "star_v2": "STAR",
    "star_exec": "STAR+Exec",
}


@dataclass
class LoadedRun:
    label: str
    method: str
    train_seed: int
    run_name: str
    checkpoint: str
    agent: Any
    env: Any
    config: Any
    mode: str


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_selected_executor() -> dict[str, Any]:
    path = CONFIG_DIR / "star_arm_selected_executor.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"star_exec_candidates": 16, "star_exec_margin": 0.02}


def build_config_from_run_dir(run_dir: Path, *, mode: str = "raw"):
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing run metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    config = star_default_config.copy()
    config.update(metadata)
    config.cuda = bool(getattr(config, "cuda", False) and torch.cuda.is_available())
    config.device = 0
    config.eval_numsteps = 100
    config.star_exec_start_steps = 0
    if mode == "star_exec":
        selected = load_selected_executor()
        config.star_exec = True
        config.training_execution_mode = "raw"
        config.evaluation_execution_mode = "both"
        config.star_exec_candidates = int(selected.get("star_exec_candidates", 16))
        config.star_exec_margin = float(selected.get("star_exec_margin", 0.02))
    return config, metadata


def load_run_dir(run_dir: str | Path, *, label: str = "STAR", mode: str = "raw", checkpoint_name: str = "final.torch") -> LoadedRun:
    run_dir = Path(run_dir)
    config, metadata = build_config_from_run_dir(run_dir, mode=mode)
    checkpoint = run_dir / "checkpoint" / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
    agent.load_checkpoint(str(checkpoint))
    return LoadedRun(
        label if mode == "raw" else "STAR+Exec",
        str(metadata.get("method", "star_v2")),
        int(metadata.get("seed", -1)),
        str(metadata.get("run_name", run_dir.name)),
        str(checkpoint),
        agent,
        env,
        config,
        mode,
    )


def load_run(method: str, train_seed: int, *, mode: str = "raw") -> LoadedRun:
    spec = next(s for s in final_specs() if s.method == method and s.seed == train_seed)
    extra: dict[str, Any] = {"device": 0, "cuda": torch.cuda.is_available(), "eval_numsteps": 100, "star_exec_start_steps": 0}
    if mode == "star_exec":
        selected = load_selected_executor()
        extra.update(
            {
                "star_exec": True,
                "training_execution_mode": "raw",
                "evaluation_execution_mode": "both",
                "star_exec_candidates": int(selected.get("star_exec_candidates", 16)),
                "star_exec_margin": float(selected.get("star_exec_margin", 0.02)),
            }
        )
    config = _config_from_spec(spec, extra)
    env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
    agent.load_checkpoint(str(spec.final_checkpoint))
    label = METHOD_LABELS[method] if mode == "raw" else METHOD_LABELS["star_exec"]
    return LoadedRun(label, method, train_seed, spec.run_name, str(spec.final_checkpoint), agent, env, config, mode)


def close_runs(runs: list[LoadedRun]) -> None:
    for run in runs:
        try:
            run.env.close()
        except Exception:
            pass


def unwrap(env):
    return getattr(env, "unwrapped", env)


def ee_pos(env) -> np.ndarray:
    return np.asarray(unwrap(env)._ee_position(), dtype=np.float64)


def endpoint_for_action(env, action: np.ndarray) -> np.ndarray:
    u = unwrap(env)
    pos = ee_pos(env)
    target = pos + np.asarray(action, dtype=np.float64) * float(u.cfg.action_scale)
    low = np.asarray(u.cfg.workspace_low, dtype=np.float64)
    high = np.asarray(u.cfg.workspace_high, dtype=np.float64)
    return np.clip(target, low, high)


def geometry(env) -> dict[str, Any]:
    u = unwrap(env)
    return {
        "start_pos": np.asarray(u.start_pos, dtype=float).tolist(),
        "goal_pos": np.asarray(u.goal_pos, dtype=float).tolist(),
        "obstacle_pos": np.asarray(u.obstacle_pos, dtype=float).tolist(),
        "obstacle_radius": float(u.cfg.obstacle_radius),
        "safe_margin": float(u.cfg.safe_margin),
        "action_scale": float(u.cfg.action_scale),
    }


def obs_metrics(env, info: dict[str, Any]) -> dict[str, float]:
    u = unwrap(env)
    obstacle = np.asarray(u.obstacle_pos, dtype=np.float64)
    pos = ee_pos(env)
    dist_obs = float(np.linalg.norm(pos - obstacle))
    clearance_keepout = dist_obs - float(u.cfg.safe_margin)
    return {
        "ee_x": float(pos[0]),
        "ee_y": float(pos[1]),
        "ee_z": float(pos[2]),
        "distance_to_obstacle": dist_obs,
        "clearance_keepout": clearance_keepout,
        "clearance_obstacle": dist_obs - float(u.cfg.obstacle_radius),
        "distance_to_goal": float(info.get("distance_to_goal", np.nan)),
        "cost": float(info.get("cost", 0.0)),
        "success": float(info.get("success", 0.0)),
        "collision": float(info.get("collision", 0.0)),
    }


def deterministic_episode_seed(run: LoadedRun, eval_seed: int) -> int:
    key = f"{run.method}:{run.mode}:{run.train_seed}:{run.run_name}:{eval_seed}"
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def seed_episode_rng(run: LoadedRun, eval_seed: int) -> int:
    seed = deterministic_episode_seed(run, eval_seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@torch.no_grad()
def audit_state(run: LoadedRun, state: np.ndarray, env) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    agent = run.agent
    config = run.config
    device = agent.device
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    norm_state = agent.normalize_state(state_tensor)
    _, _, mean_action = agent.policy.sample(norm_state)
    mean_q = agent._cost_plus(norm_state, mean_action).view(-1)[0]
    matched = agent.audit.generate_matched_audits(agent.policy, agent.reference_policy, norm_state)
    q_cor = agent.audit.conservative_cost(agent.cost_critic, norm_state, matched.corridor.actions).view(-1)
    q_cur = agent.audit.conservative_cost(agent.cost_critic, norm_state, matched.current_only.actions).view(-1)
    rho_cor = agent.audit.shadow_risk(q_cor.view(1, -1)).view(-1)[0]
    rho_cur = agent.audit.shadow_risk(q_cur.view(1, -1)).view(-1)[0]
    threshold = float(config.star_risk_threshold)
    mean_np = mean_action.detach().cpu().numpy()[0]
    cor_np = matched.corridor.actions.detach().cpu().numpy()[0]
    cur_np = matched.current_only.actions.detach().cpu().numpy()[0]
    q_cor_np = q_cor.detach().cpu().numpy()
    q_cur_np = q_cur.detach().cpu().numpy()
    beta = matched.corridor.beta.detach().cpu().numpy().reshape(-1)
    stratum = matched.corridor.stratum_index.detach().cpu().numpy().reshape(-1)
    sample = matched.corridor.sample_index.detach().cpu().numpy().reshape(-1)
    mean_endpoint = endpoint_for_action(env, mean_np)
    items: list[dict[str, Any]] = []
    items.append(
        {
            "candidate_type": "actor_mean",
            "candidate_index": 0,
            "action_x": float(mean_np[0]),
            "action_y": float(mean_np[1]),
            "action_z": float(mean_np[2]),
            "endpoint_x": float(mean_endpoint[0]),
            "endpoint_y": float(mean_endpoint[1]),
            "endpoint_z": float(mean_endpoint[2]),
            "q_cost": float(mean_q.item()),
            "risky": float(mean_q.item() > threshold),
            "executed": 1.0,
            "beta": 1.0,
            "stratum": -1,
            "sample": -1,
        }
    )
    for i, (action, risk) in enumerate(zip(cur_np, q_cur_np)):
        ep = endpoint_for_action(env, action)
        items.append(
            {
                "candidate_type": "current_only",
                "candidate_index": int(i),
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "action_z": float(action[2]),
                "endpoint_x": float(ep[0]),
                "endpoint_y": float(ep[1]),
                "endpoint_z": float(ep[2]),
                "q_cost": float(risk),
                "risky": float(risk > threshold),
                "executed": 0.0,
                "beta": 1.0,
                "stratum": int(i),
                "sample": 0,
            }
        )
    for i, (action, risk) in enumerate(zip(cor_np, q_cor_np)):
        ep = endpoint_for_action(env, action)
        items.append(
            {
                "candidate_type": "corridor_shadow",
                "candidate_index": int(i),
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "action_z": float(action[2]),
                "endpoint_x": float(ep[0]),
                "endpoint_y": float(ep[1]),
                "endpoint_z": float(ep[2]),
                "q_cost": float(risk),
                "risky": float(risk > threshold),
                "executed": 0.0,
                "beta": float(beta[int(stratum[i])] if len(beta) else np.nan),
                "stratum": int(stratum[i]),
                "sample": int(sample[i]),
            }
        )
    row = {
        "mean_action_risk": float(mean_q.item()),
        "corridor_q_min": float(np.min(q_cor_np)),
        "corridor_q_mean": float(np.mean(q_cor_np)),
        "corridor_q_max": float(np.max(q_cor_np)),
        "current_q_min": float(np.min(q_cur_np)),
        "current_q_mean": float(np.mean(q_cur_np)),
        "current_q_max": float(np.max(q_cur_np)),
        "rho_cor": float(rho_cor.item()),
        "rho_cur": float(rho_cur.item()),
        "corridor_lift": float((rho_cor - rho_cur).item()),
        "hidden_risky_shadow": float(mean_q.item() <= threshold and np.max(q_cor_np) > threshold),
        "high_risk_shadow_not_executed": float(np.max(q_cor_np) > threshold),
        "threshold": threshold,
        "actor_mean_action_x": float(mean_np[0]),
        "actor_mean_action_y": float(mean_np[1]),
        "actor_mean_action_z": float(mean_np[2]),
        "actor_mean_endpoint_x": float(mean_endpoint[0]),
        "actor_mean_endpoint_y": float(mean_endpoint[1]),
        "actor_mean_endpoint_z": float(mean_endpoint[2]),
    }
    return row, items


def evaluate_episode(run: LoadedRun, eval_seed: int, *, capture_audit: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rng_seed = seed_episode_rng(run, eval_seed)
    state = reset_env(run.env, seed=eval_seed)
    info = unwrap(run.env)._safety_info() if hasattr(unwrap(run.env), "_safety_info") else {}
    geom = geometry(run.env)
    done = False
    step = 0
    reward_sum = 0.0
    cost_sum = 0.0
    violations = 0
    path_length = 0.0
    min_clearance = float("inf")
    traj_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    while not done and step < int(run.config.eval_numsteps):
        before = obs_metrics(run.env, info)
        base = {
            "method_label": run.label,
            "method": run.method,
            "mode": run.mode,
            "train_seed": run.train_seed,
            "run_name": run.run_name,
            "checkpoint_path": run.checkpoint,
            "eval_seed": eval_seed,
            "episode_rng_seed": episode_rng_seed,
            "step": step,
            **geom,
            **before,
        }
        action = run.agent.select_action(state, evaluate=True, execution_mode=run.mode, total_numsteps=300000, diagnostics=True)
        details = dict(getattr(run.agent, "last_action_info", {}) or {})
        for key, value in details.items():
            if isinstance(value, (bool, np.bool_)):
                base[key] = float(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                base[key] = float(value)
            else:
                base[key] = str(value)
        if capture_audit and run.method == "star_v2" and run.mode == "raw":
            audit_row, audit_items = audit_state(run, state, run.env)
            state_row = dict(base)
            state_row.update(audit_row)
            state_row["audit_items_json"] = json.dumps(audit_items, sort_keys=True)
            state_rows.append(state_row)
        next_state, reward, cost, terminated, truncated, info = step_env(run.env, action, run.config.safe_env)
        reward_sum += float(reward)
        cost_sum += float(cost)
        violations += int(float(cost) > 0.0)
        path_length += float(info.get("path_length_increment", 0.0))
        min_clearance = min(min_clearance, float(info.get("distance_to_obstacle", np.inf)) - float(geom["safe_margin"]))
        base.update(
            {
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "action_z": float(action[2]),
                "reward": float(reward),
                "step_cost_after": float(cost),
                "terminated": float(terminated),
                "truncated": float(truncated),
            }
        )
        traj_rows.append(base)
        done = bool(terminated or truncated)
        state = next_state
        step += 1
    episode = {
        "method_label": run.label,
        "method": run.method,
        "mode": run.mode,
        "train_seed": run.train_seed,
        "run_name": run.run_name,
        "checkpoint_path": run.checkpoint,
        "eval_seed": eval_seed,
        "episode_rng_seed": episode_rng_seed,
        "episode_return": reward_sum,
        "episode_cost": cost_sum,
        "episode_length": step,
        "success": float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0,
        "violation_rate": violations / max(1, step),
        "min_clearance": min_clearance if math.isfinite(min_clearance) else np.nan,
        "path_length": path_length,
        **geom,
    }
    return episode, traj_rows, state_rows


def rank_state(row: dict[str, Any]) -> tuple:
    return (
        float(row.get("hidden_risky_shadow", 0.0)),
        float(row.get("mean_action_risk", 1e9)) <= float(row.get("threshold", 0.0)),
        float(row.get("corridor_lift", -1e9)) > 0,
        float(row.get("corridor_lift", -1e9)),
        float(row.get("corridor_q_max", -1e9)),
        -abs(float(row.get("clearance_keepout", 0.0))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", default="10,11,12")
    parser.add_argument("--eval-start", type=int, default=900000)
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument("--output-dir", default=str(QUAL_ROOT))
    parser.add_argument("--star-run-dir", default="", help="Optional run_dir for a targeted STAR/showcase checkpoint.")
    parser.add_argument("--compare-seed", type=int, default=11, help="Final baseline train seed for SAC-Lag and Current-only-N comparisons when --star-run-dir is used.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = [int(x) for x in args.train_seeds.split(",") if x.strip()]
    eval_seeds = list(range(args.eval_start, args.eval_start + args.eval_count))

    all_candidate_states: list[dict[str, Any]] = []
    all_star_episodes: list[dict[str, Any]] = []
    runs: list[LoadedRun] = []
    try:
        if args.star_run_dir:
            search_runs = [load_run_dir(args.star_run_dir, label="STAR", mode="raw")]
        else:
            search_runs = [load_run("star_v2", train_seed, mode="raw") for train_seed in train_seeds]
        for run in search_runs:
            runs.append(run)
            for eval_seed in eval_seeds:
                episode, _traj, states = evaluate_episode(run, eval_seed, capture_audit=True)
                all_star_episodes.append(episode)
                for state_row in states:
                    state_row.update(
                        {
                            "episode_return": episode["episode_return"],
                            "episode_cost": episode["episode_cost"],
                            "episode_success": episode["success"],
                            "episode_min_clearance": episode["min_clearance"],
                        }
                    )
                    all_candidate_states.append(state_row)
        all_candidate_states.sort(key=rank_state, reverse=True)
        write_csv(out_dir / "candidate_states.csv", all_candidate_states)
        write_csv(out_dir / "candidate_episodes.csv", all_star_episodes)
        if not all_candidate_states:
            raise RuntimeError("no candidate states collected")
        selected = all_candidate_states[0]
        selected_seed = int(selected["eval_seed"])
        selected_train_seed = int(selected["train_seed"])
        selected_step = int(selected["step"])
    finally:
        close_runs(runs)

    baseline_seed = args.compare_seed if args.star_run_dir else selected_train_seed
    if args.star_run_dir:
        compare_runs = [
            load_run("sac_lag", baseline_seed, mode="raw"),
            load_run("current_only_v2", baseline_seed, mode="raw"),
            load_run_dir(args.star_run_dir, label="STAR", mode="raw"),
            load_run_dir(args.star_run_dir, label="STAR", mode="star_exec"),
        ]
    else:
        compare_runs = [
            load_run("sac_lag", selected_train_seed, mode="raw"),
            load_run("current_only_v2", selected_train_seed, mode="raw"),
            load_run("star_v2", selected_train_seed, mode="raw"),
            load_run("star_v2", selected_train_seed, mode="star_exec"),
        ]
    trajectory_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    audit_snapshot_rows: list[dict[str, Any]] = []
    try:
        for run in compare_runs:
            episode, traj, states = evaluate_episode(run, selected_seed, capture_audit=(run.method == "star_v2" and run.mode == "raw"))
            episode_rows.append(episode)
            trajectory_rows.extend(traj)
            # The selected audit state was already ranked from a concrete stochastic
            # shadow draw. Reuse that exact candidate set so panel (a), selection.md,
            # and selected_episode.json remain numerically identical.
            pass
    finally:
        close_runs(compare_runs)

    context_keys = [
        "method_label", "method", "mode", "train_seed", "run_name", "checkpoint_path",
        "eval_seed", "step", "ee_x", "ee_y", "ee_z", "goal_pos", "obstacle_pos",
        "obstacle_radius", "safe_margin", "action_scale", "threshold", "mean_action_risk",
        "corridor_q_max", "current_q_max", "rho_cor", "rho_cur", "corridor_lift",
        "hidden_risky_shadow", "clearance_keepout",
    ]
    audit_snapshot_rows = []
    for item in json.loads(selected.get("audit_items_json", "[]")):
        row = dict(item)
        for key in context_keys:
            row[key] = selected.get(key, "")
        audit_snapshot_rows.append(row)

    write_csv(out_dir / "selected_episode_trajectories.csv", trajectory_rows)
    write_csv(out_dir / "trajectory_rows.csv", trajectory_rows)
    write_csv(out_dir / "selected_episode_methods.csv", episode_rows)
    write_csv(out_dir / "audit_snapshot.csv", audit_snapshot_rows)

    selected_doc = {
        "selection_rule": "highest hidden_risky_shadow, actor-mean safe, positive corridor lift, high max corridor risk, near boundary",
        "selected_train_seed": selected_train_seed,
        "selected_eval_seed": selected_seed,
        "selected_step": selected_step,
        "selected_state": selected,
        "episode_comparison": episode_rows,
        "outputs": {
            "candidate_states_csv": str(out_dir / "candidate_states.csv"),
            "candidate_episodes_csv": str(out_dir / "candidate_episodes.csv"),
            "trajectory_rows_csv": str(out_dir / "trajectory_rows.csv"),
            "audit_snapshot_csv": str(out_dir / "audit_snapshot.csv"),
        },
    }
    (out_dir / "selected_episode.json").write_text(json.dumps(selected_doc, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Panda Mechanism Episode Selection",
        "",
        "Selection uses real STAR checkpoints and real policy/critic queries.",
        "",
        f"- selected train seed: `{selected_train_seed}`",
        f"- selected eval seed: `{selected_seed}`",
        f"- selected step: `{selected_step}`",
        f"- actor mean predicted cost: `{float(selected['mean_action_risk']):.6f}`",
        f"- audit threshold: `{float(selected['threshold']):.6f}`",
        f"- max corridor shadow predicted cost: `{float(selected['corridor_q_max']):.6f}`",
        f"- max current-only predicted cost: `{float(selected['current_q_max']):.6f}`",
        f"- corridor lift: `{float(selected['corridor_lift']):.6f}`",
        f"- hidden risky shadow: `{bool(float(selected['hidden_risky_shadow']))}`",
        f"- clearance to keep-out at state: `{float(selected['clearance_keepout']):.6f}`",
        "",
        "All candidate states are saved; failed/less suitable candidates are not hidden.",
    ]
    (out_dir / "selection.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"selected_eval_seed": selected_seed, "selected_step": selected_step, "states": len(all_candidate_states)}, indent=2))


if __name__ == "__main__":
    main()
