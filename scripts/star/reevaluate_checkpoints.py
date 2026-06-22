#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.star_agent import STARAgent
from main_star import EVAL_EPISODE_FIELDS, _success, append_csv, make_env, reset_env, step_env
from utilis.star_default_config import star_default_config


CORRECTED_FIELDS = EVAL_EPISODE_FIELDS + ["checkpoint_path", "checkpoint_name", "evaluation_only"]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_config(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return json.load(f)


def build_config(metadata: dict):
    config = star_default_config.copy()
    config.update(metadata)
    config.cuda = bool(config.cuda and torch.cuda.is_available())
    return config


def checkpoint_files(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoint"
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.rglob("*.torch"))


def evaluate_checkpoint_mode(agent, env, config, mode: str, seeds: list[int], checkpoint: Path, run_dir: Path) -> None:
    run_name = run_dir.name
    for seed in seeds:
        state = reset_env(env, seed=seed)
        done = False
        episode_reward = 0.0
        episode_cost = 0.0
        steps = 0
        violations = 0
        boundary_safe = 0
        boundary_total = 0
        fallbacks = 0
        safe_fraction_sum = 0.0
        risk_sum = 0.0
        reward_pred_sum = 0.0
        found_count = 0
        info = {}
        while not done and steps < int(config.eval_numsteps):
            action = agent.select_action(state, evaluate=True, execution_mode=mode, total_numsteps=int(config.num_steps))
            next_state, reward, cost, terminated, truncated, info = step_env(env, action, config.safe_env)
            done = terminated or truncated
            details = agent.last_action_info
            risk = float(details.get("selected_predicted_risk", 0.0))
            near_boundary = abs(risk - float(config.star_risk_threshold)) <= float(config.boundary_epsilon)
            if near_boundary:
                boundary_total += 1
                if cost <= 0:
                    boundary_safe += 1
            found = (
                bool(details.get("any_shadow_predicted_unsafe", False))
                and risk <= float(config.star_risk_threshold)
                and cost <= 0
            )
            found_count += int(found)
            fallbacks += int(bool(details.get("execution_fallback", False)))
            safe_fraction_sum += float(details.get("safe_candidate_fraction", 0.0))
            risk_sum += risk
            reward_pred_sum += float(details.get("selected_predicted_reward", 0.0))
            violations += int(cost > 0)
            episode_reward += reward
            episode_cost += cost
            steps += 1
            state = next_state
        row = {
            "run_name": run_name,
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "env_cost_limit": float(config.env_cost_limit),
            "global_step": int(config.num_steps),
            "mode": mode,
            "eval_seed": seed,
            "episode_reward": episode_reward,
            "episode_cost": episode_cost,
            "episode_length": steps,
            "cost_rate": episode_cost / max(1, steps),
            "violation_rate": violations / max(1, steps),
            "success": _success(info),
            "constraint_satisfied": float(episode_cost <= float(config.env_cost_limit)),
            "boundary_safe_coverage": boundary_safe / max(1, steps),
            "boundary_safety_rate": boundary_safe / max(1, boundary_total),
            "execution_fallback_rate": fallbacks / max(1, steps),
            "safe_candidate_fraction": safe_fraction_sum / max(1, steps),
            "executed_predicted_risk": risk_sum / max(1, steps),
            "selected_predicted_reward": reward_pred_sum / max(1, steps),
            "found_but_not_executed_rate": found_count / max(1, steps),
            "checkpoint_path": str(checkpoint),
            "checkpoint_name": checkpoint.stem,
            "evaluation_only": True,
        }
        append_csv(run_dir / "corrected_eval_episodes.csv", row, CORRECTED_FIELDS)


def reevaluate_run(run_dir: Path, seeds: list[int], checkpoint_selector: str) -> int:
    metadata = load_config(run_dir)
    if not metadata:
        return 0
    config = build_config(metadata)
    checkpoints = checkpoint_files(run_dir)
    if checkpoint_selector == "best":
        checkpoints = [path for path in checkpoints if path.stem == "best"]
    count = 0
    for checkpoint in checkpoints:
        raw_env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
        exec_env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
        try:
            agent = STARAgent(raw_env.observation_space.shape[0], raw_env.action_space, config)
            agent.load_checkpoint(str(checkpoint))
            evaluate_checkpoint_mode(agent, raw_env, config, "raw", seeds, checkpoint, run_dir)
            evaluate_checkpoint_mode(agent, exec_env, config, "star_exec", seeds, checkpoint, run_dir)
            count += 1
        finally:
            raw_env.close()
            exec_env.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/star_aaai")
    parser.add_argument("--eval-seeds", default="100000,100001,100002,100003,100004,100005,100006,100007,100008,100009")
    parser.add_argument("--checkpoint-selector", choices=["all", "best"], default="best")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    root = Path(args.root)
    seeds = parse_seed_list(args.eval_seeds)
    total = 0
    for meta in sorted(root.rglob("run_metadata.json")):
        total += reevaluate_run(meta.parent, seeds, args.checkpoint_selector)
    print(f"Reevaluated {total} checkpoints under {root}")


if __name__ == "__main__":
    main()
