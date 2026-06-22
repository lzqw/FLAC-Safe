from __future__ import annotations

import csv
import datetime
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import wandb
except ImportError:  # pragma: no cover
    class _WandbFallback:
        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _WandbFallback()

import gymnasium as gym
import numpy as np
import torch

from agents.star_agent import STARAgent
from utilis.SafeReplaybuffer import SafeReplayMemory
from utilis.config import ARGConfig
from utilis.star_default_config import star_default_config


TRAIN_EPISODE_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "env_cost_limit",
    "episode",
    "start_step",
    "end_step",
    "episode_reward",
    "episode_cost",
    "episode_length",
    "train_total_cost",
    "train_total_cost_rate",
    "replay_size",
    "wall_clock_time",
]

EVAL_EPISODE_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "env_cost_limit",
    "global_step",
    "mode",
    "eval_seed",
    "episode_reward",
    "episode_cost",
    "episode_length",
    "cost_rate",
    "violation_rate",
    "success",
    "constraint_satisfied",
    "boundary_safe_coverage",
    "boundary_safety_rate",
    "execution_fallback_rate",
    "safe_candidate_fraction",
    "executed_predicted_risk",
    "selected_predicted_reward",
    "found_but_not_executed_rate",
]

MECHANISM_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "env_cost_limit",
    "step",
    "update",
    "pSVR",
    "any_unsafe_shadow_rate",
    "hidden_unsafe_rate",
    "shadow_risk_mean",
    "shadow_risk_batch_max",
    "shadow_risk_max_mean",
    "shadow_q_mean",
    "shadow_q_std",
    "actor_mean_action_risk",
    "kl_mean",
    "kl_max",
    "kl_exceed_rate",
    "candidate_spread",
    "shadow_penalty",
    "lagrange_residual",
    "lagrange_value",
    "lagrange_mean_qc",
    "shadow_k",
    "shadow_temperature",
    "shadow_aggregation",
    "shadow_reference_mode",
    "safe_candidate_fraction",
    "fallback_rate",
    "executed_predicted_risk",
    "found_but_not_executed_rate",
    "boundary_safe_coverage",
    "boundary_safety_rate",
    "cost_critic_forward_calls",
    "wall_clock_time",
]

EFFICIENCY_FIELDS = [
    "run_name",
    "task",
    "method",
    "seed",
    "env_cost_limit",
    "step",
    "episode",
    "wall_clock_time",
    "env_time",
    "update_time",
    "eval_time",
    "env_steps_per_second",
    "updates_per_second",
    "gpu_memory_peak_mb",
    "cost_critic_forward_calls",
    "replay_size",
]


def reset_env(env, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action, safe_env=False):
    if safe_env:
        next_state, reward, cost, terminated, truncated, info = env.step(action)
        return next_state, float(reward), float(cost), terminated, truncated, info
    next_state, reward, terminated, truncated, info = env.step(action)
    return next_state, float(reward), float(info.get("cost", 0.0)), terminated, truncated, info


def make_env(task, *, safe_env=False, train=True, binary_cost=True):
    if safe_env:
        from envs.safety_gym_wrapper import make_safe_env

        return make_safe_env(task, train=train, binary_cost=binary_cost)
    return gym.make(task)


def execution_mode_for_config(config) -> str:
    return "star_exec" if bool(config.star_exec) and config.method in ("star", "star_exec") else "raw"


def _success(info: dict) -> float:
    if "solved" in info:
        return float(info["solved"])
    if "success" in info:
        return float(info["success"])
    return 0.0


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "run"


def unique_run_dir(config) -> tuple[Path, str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(config.run_name).strip()
    if not run_name:
        run_name = f"{config.task}_{config.method}_seed{config.seed}_{timestamp}"
    run_name = _safe_name(run_name)
    root = Path(config.output_root) / _safe_name(config.task) / _safe_name(config.method)
    run_dir = root / run_name
    if run_dir.exists():
        run_dir = root / f"{run_name}_{timestamp}"
        run_name = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def serializable_config(config) -> dict:
    return {key: config[key] for key in config if not str(key).startswith("_")}


def gpu_memory_peak_mb(agent: STARAgent) -> float:
    if agent.device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(agent.device) / (1024.0 * 1024.0))


def eval_episode_row(
    *,
    run_name: str,
    config,
    mode: str,
    total_steps: int,
    eval_seed: int,
    episode_reward: float,
    episode_cost: float,
    steps: int,
    violations: int,
    success: float,
    boundary_safe: int,
    boundary_total: int,
    fallbacks: int,
    safe_fraction_sum: float,
    risk_sum: float,
    reward_pred_sum: float,
    found_count: int,
) -> dict:
    return {
        "run_name": run_name,
        "task": config.task,
        "method": config.method,
        "seed": config.seed,
        "env_cost_limit": float(config.env_cost_limit),
        "global_step": total_steps,
        "mode": mode,
        "eval_seed": eval_seed,
        "episode_reward": episode_reward,
        "episode_cost": episode_cost,
        "episode_length": steps,
        "cost_rate": episode_cost / max(1, steps),
        "violation_rate": violations / max(1, steps),
        "success": success,
        "constraint_satisfied": float(episode_cost <= float(config.env_cost_limit)),
        "boundary_safe_coverage": boundary_safe / max(1, steps),
        "boundary_safety_rate": boundary_safe / max(1, boundary_total),
        "execution_fallback_rate": fallbacks / max(1, steps),
        "safe_candidate_fraction": safe_fraction_sum / max(1, steps),
        "executed_predicted_risk": risk_sum / max(1, steps),
        "selected_predicted_reward": reward_pred_sum / max(1, steps),
        "found_but_not_executed_rate": found_count / max(1, steps),
    }


def evaluate_mode(
    agent,
    env,
    config,
    mode: str,
    total_steps: int,
    seeds: list[int],
    run_dir: Path,
    run_name: str,
) -> Dict[str, float]:
    rows = []
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
            action = agent.select_action(
                state,
                evaluate=True,
                execution_mode=mode,
                total_numsteps=total_steps,
            )
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
        row = eval_episode_row(
            run_name=run_name,
            config=config,
            mode=mode,
            total_steps=total_steps,
            eval_seed=seed,
            episode_reward=episode_reward,
            episode_cost=episode_cost,
            steps=steps,
            violations=violations,
            success=_success(info),
            boundary_safe=boundary_safe,
            boundary_total=boundary_total,
            fallbacks=fallbacks,
            safe_fraction_sum=safe_fraction_sum,
            risk_sum=risk_sum,
            reward_pred_sum=reward_pred_sum,
            found_count=found_count,
        )
        append_csv(run_dir / "eval_episodes.csv", row, EVAL_EPISODE_FIELDS)
        rows.append(row)

    totals = {}
    for key in [
        "episode_reward",
        "episode_cost",
        "cost_rate",
        "violation_rate",
        "success",
        "constraint_satisfied",
        "boundary_safe_coverage",
        "execution_fallback_rate",
        "boundary_safety_rate",
        "safe_candidate_fraction",
        "executed_predicted_risk",
        "found_but_not_executed_rate",
    ]:
        totals[key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    return totals


def evaluate(agent, config, total_steps: int, run_dir: Path, run_name: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    seeds = [int(config.seed) + 100000 + i for i in range(int(config.eval_times))]
    raw_env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    exec_env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    try:
        raw = evaluate_mode(agent, raw_env, config, "raw", total_steps, seeds, run_dir, run_name)
        star_exec = evaluate_mode(agent, exec_env, config, "star_exec", total_steps, seeds, run_dir, run_name)
    finally:
        raw_env.close()
        exec_env.close()
    payload = {}
    for prefix, metrics in (("eval/raw", raw), ("eval/star_exec", star_exec)):
        for key, value in metrics.items():
            payload[f"{prefix}/{key}"] = value
    wandb.log(payload, step=total_steps)
    print(
        "STAR_EVAL step={} eval/raw/reward={:.6g} eval/raw/episode_cost={:.6g} "
        "eval/raw/violation_rate={:.6g} eval/star_exec/reward={:.6g} "
        "eval/star_exec/episode_cost={:.6g} eval/star_exec/violation_rate={:.6g}".format(
            total_steps,
            raw["episode_reward"],
            raw["episode_cost"],
            raw["violation_rate"],
            star_exec["episode_reward"],
            star_exec["episode_cost"],
            star_exec["violation_rate"],
        ),
        flush=True,
    )
    return raw, star_exec


def env_interaction_metrics(info: Dict[str, float | bool | str], cost: float, config) -> Dict[str, float]:
    risk = float(info.get("selected_predicted_risk", 0.0))
    any_unsafe = bool(info.get("any_shadow_predicted_unsafe", False))
    found_but_not_executed = (
        any_unsafe
        and risk <= float(config.star_risk_threshold)
        and float(cost) <= 0.0
    )
    near_boundary = abs(risk - float(config.star_risk_threshold)) <= float(config.boundary_epsilon)
    boundary_safe = near_boundary and float(cost) <= 0.0
    return {
        "star/executed_violation_rate": float(cost > 0),
        "star/execution_fallback_rate": float(bool(info.get("execution_fallback", False))),
        "star/execution_safe_candidate_fraction": float(info.get("safe_candidate_fraction", 0.0)),
        "star/executed_predicted_risk": risk,
        "star/found_but_not_executed_rate": float(found_but_not_executed),
        "star/boundary_safe_coverage": float(boundary_safe),
        "star/boundary_safety_rate": float(boundary_safe) if near_boundary else 0.0,
        "star/selected_predicted_reward": float(info.get("selected_predicted_reward", 0.0)),
        "star/shadow_predicted_unsafe_fraction": float(info.get("shadow_predicted_unsafe_fraction", 0.0)),
        "star/action_candidate_spread": float(info.get("action_candidate_spread", 0.0)),
    }


def mechanism_row(run_name: str, config, step: int, update: int, log: dict, interaction: dict, wall_time: float) -> dict:
    return {
        "run_name": run_name,
        "task": config.task,
        "method": config.method,
        "seed": config.seed,
        "env_cost_limit": float(config.env_cost_limit),
        "step": step,
        "update": update,
        "pSVR": log.get("star/pSVR", 0.0),
        "any_unsafe_shadow_rate": log.get("star/any_unsafe_shadow_rate", 0.0),
        "hidden_unsafe_rate": log.get("star/hidden_unsafe_rate", 0.0),
        "shadow_risk_mean": log.get("star/shadow_risk_mean", 0.0),
        "shadow_risk_batch_max": log.get("star/shadow_risk_batch_max", 0.0),
        "shadow_risk_max_mean": log.get("star/shadow_risk_max_mean", 0.0),
        "shadow_q_mean": log.get("star/shadow_q_mean", 0.0),
        "shadow_q_std": log.get("star/shadow_q_std", 0.0),
        "actor_mean_action_risk": log.get("star/actor_mean_action_risk", 0.0),
        "kl_mean": log.get("star/kl_mean", 0.0),
        "kl_max": log.get("star/kl_max", 0.0),
        "kl_exceed_rate": log.get("star/kl_exceed_rate", 0.0),
        "candidate_spread": log.get("star/action_spread", 0.0),
        "shadow_penalty": log.get("star/shadow_penalty", 0.0),
        "lagrange_residual": log.get("star/lagrange_residual", 0.0),
        "lagrange_value": log.get("star/lagrange_value", 0.0),
        "lagrange_mean_qc": log.get("star/lagrange_mean_qc", 0.0),
        "shadow_k": log.get("star/shadow_k", getattr(config, "shadow_k", 0)),
        "shadow_temperature": log.get("star/shadow_temperature", getattr(config, "shadow_temperature", 0)),
        "shadow_aggregation": log.get("star/shadow_aggregation", getattr(config, "shadow_aggregation", "")),
        "shadow_reference_mode": log.get("star/shadow_reference_mode", getattr(config, "shadow_reference_mode", "")),
        "safe_candidate_fraction": interaction.get("star/execution_safe_candidate_fraction", 0.0),
        "fallback_rate": interaction.get("star/execution_fallback_rate", 0.0),
        "executed_predicted_risk": interaction.get("star/executed_predicted_risk", 0.0),
        "found_but_not_executed_rate": interaction.get("star/found_but_not_executed_rate", 0.0),
        "boundary_safe_coverage": interaction.get("star/boundary_safe_coverage", 0.0),
        "boundary_safety_rate": interaction.get("star/boundary_safety_rate", 0.0),
        "cost_critic_forward_calls": log.get("efficiency/cost_critic_forward_calls", 0.0),
        "wall_clock_time": wall_time,
    }


def train_loop(config) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    env = make_env(config.task, safe_env=config.safe_env, train=True, binary_cost=config.binary_cost)
    env.action_space.seed(config.seed)
    agent = STARAgent(env.observation_space.shape[0], env.action_space, config)

    run_dir, run_name = unique_run_dir(config)
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.log").open("w") as f:
        f.write(str(config))
    with (run_dir / "run_metadata.json").open("w") as f:
        metadata = serializable_config(config)
        metadata.update({"run_name": run_name, "run_dir": str(run_dir), "start_time": datetime.datetime.now().isoformat()})
        json.dump(metadata, f, indent=2, sort_keys=True)

    memory = SafeReplayMemory(config.replay_size, config.seed, recent_fraction=config.recent_fraction)
    total_steps = 0
    updates = 0
    train_total_cost = 0.0
    best_raw_reward = -1e9
    eval_interval = max(1, int(config.eval_interval_steps or config.eval_numsteps))
    save_interval = int(config.save_interval_steps)
    metric_interval = max(1, int(config.metric_log_interval_steps))
    last_metric_step = -metric_interval
    start_time = time.perf_counter()
    env_time = 0.0
    update_time = 0.0
    eval_time = 0.0
    last_interaction_log: Dict[str, float] = {}

    # Evaluation uses unscaled raw environment rewards; training may use wrapper reward scaling.
    print(
        "STAR_CONFIG method={} run_name={} run_dir={} eval_interval_steps={} eval_times={} "
        "save_interval_steps={} recent_fraction={} shadow_aggregation={} shadow_reference_mode={} "
            "cost_critic_reduce={}".format(
            config.method,
            run_name,
            run_dir,
            eval_interval,
            config.eval_times,
            save_interval,
            config.recent_fraction,
            config.shadow_aggregation,
            config.shadow_reference_mode,
            config.cost_critic_reduce,
        ),
        flush=True,
    )

    for episode in itertools.count(1):
        episode_start = total_steps
        state = reset_env(env, seed=config.seed if episode == 1 else None)
        agent.observe(state)
        done = False
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        while not done:
            if total_steps < config.start_steps:
                action = env.action_space.sample()
                agent.last_action_info = {
                    "execution_mode": "random",
                    "candidate_count": 1,
                    "safe_candidate_fraction": 0.0,
                    "execution_fallback": False,
                    "selected_predicted_risk": 0.0,
                    "selected_predicted_reward": 0.0,
                    "mean_action_predicted_risk": 0.0,
                    "any_shadow_predicted_unsafe": False,
                    "shadow_predicted_unsafe_fraction": 0.0,
                    "selected_from_mean": False,
                    "action_candidate_spread": 0.0,
                }
            else:
                action = agent.select_action(
                    state,
                    evaluate=False,
                    execution_mode=execution_mode_for_config(config),
                    total_numsteps=total_steps,
                )

            if total_steps >= config.start_steps and len(memory) >= config.batch_size:
                for _ in range(int(config.updates_per_step)):
                    t0 = time.perf_counter()
                    log = agent.update_parameters(memory, config.batch_size, updates, total_steps)
                    update_time += time.perf_counter() - t0
                    if log:
                        wandb.log(log, step=total_steps)
                        if total_steps - last_metric_step >= metric_interval:
                            append_csv(
                                run_dir / "mechanism.csv",
                                mechanism_row(
                                    run_name,
                                    config,
                                    total_steps,
                                    updates,
                                    log,
                                    last_interaction_log,
                                    time.perf_counter() - start_time,
                                ),
                                MECHANISM_FIELDS,
                            )
                            last_metric_step = total_steps
                    updates += 1

            t0 = time.perf_counter()
            next_state, reward, cost, terminated, truncated, env_info = step_env(env, action, config.safe_env)
            env_time += time.perf_counter() - t0
            done = terminated or truncated
            mask = 0.0 if terminated else 1.0
            memory.push(state, action, reward, cost, next_state, mask)
            agent.observe(next_state)

            total_steps += 1
            episode_steps += 1
            episode_reward += reward
            episode_cost += cost
            train_total_cost += cost
            last_interaction_log = env_interaction_metrics(agent.last_action_info, cost, config)
            wandb.log(last_interaction_log, step=total_steps)
            state = next_state

            if config.eval and total_steps % eval_interval == 0:
                t0 = time.perf_counter()
                raw_eval, exec_eval = evaluate(agent, config, total_steps, run_dir, run_name)
                eval_time += time.perf_counter() - t0
                if config.save and raw_eval["episode_reward"] >= best_raw_reward:
                    best_raw_reward = raw_eval["episode_reward"]
                    agent.save_checkpoint(str(checkpoint_dir), "best")

            if config.save and save_interval > 0 and total_steps % save_interval == 0:
                agent.save_checkpoint(str(checkpoint_dir), f"step_{total_steps}")

            if total_steps >= config.num_steps:
                break

        wall_time = time.perf_counter() - start_time
        train_cost_rate = train_total_cost / max(1, total_steps)
        train_row = {
            "run_name": run_name,
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "env_cost_limit": float(config.env_cost_limit),
            "episode": episode,
            "start_step": episode_start,
            "end_step": total_steps,
            "episode_reward": episode_reward,
            "episode_cost": episode_cost,
            "episode_length": episode_steps,
            "train_total_cost": train_total_cost,
            "train_total_cost_rate": train_cost_rate,
            "replay_size": len(memory),
            "wall_clock_time": wall_time,
        }
        append_csv(run_dir / "train_episodes.csv", train_row, TRAIN_EPISODE_FIELDS)
        append_csv(
            run_dir / "efficiency.csv",
            {
                "run_name": run_name,
                "task": config.task,
                "method": config.method,
                "seed": config.seed,
                "env_cost_limit": float(config.env_cost_limit),
                "step": total_steps,
                "episode": episode,
                "wall_clock_time": wall_time,
                "env_time": env_time,
                "update_time": update_time,
                "eval_time": eval_time,
                "env_steps_per_second": total_steps / max(1e-9, wall_time),
                "updates_per_second": updates / max(1e-9, wall_time),
                "gpu_memory_peak_mb": gpu_memory_peak_mb(agent),
                "cost_critic_forward_calls": agent.cost_critic_forward_calls,
                "replay_size": len(memory),
            },
            EFFICIENCY_FIELDS,
        )
        wandb.log(
            {
                "train/reward": episode_reward,
                "train/cost": episode_cost,
                "train/cost_rate": episode_cost / max(1, episode_steps),
                "train/total_cost": train_total_cost,
                "train/total_env_steps": total_steps,
                "train/total_cost_rate": train_cost_rate,
                "train/replay_size": len(memory),
                "efficiency/wall_clock_time": wall_time,
                "efficiency/env_steps_per_second": total_steps / max(1e-9, wall_time),
                "efficiency/updates_per_second": updates / max(1e-9, wall_time),
                "efficiency/gpu_memory_peak_mb": gpu_memory_peak_mb(agent),
                "efficiency/cost_critic_forward_calls": float(agent.cost_critic_forward_calls),
            },
            step=total_steps,
        )
        print(
            "STAR_TRAIN episode={} step={} reward={:.6g} cost={:.6g} train_cost_rate={:.8g} "
            "replay_size={} wall_clock_time={:.3f} env_steps_per_second={:.6g}".format(
                episode,
                total_steps,
                episode_reward,
                episode_cost,
                train_cost_rate,
                len(memory),
                wall_time,
                total_steps / max(1e-9, wall_time),
            ),
            flush=True,
        )
        if total_steps >= config.num_steps:
            break
    env.close()
    wandb.finish()


def build_arg_config() -> ARGConfig:
    arg = ARGConfig()
    for key, value in star_default_config.items():
        arg.add_arg(key, value)
    return arg


if __name__ == "__main__":
    arg = build_arg_config()
    arg.parser("STAR: Safety-Shadow Trust-Region Actor-Critic")
    config = star_default_config.copy()
    config.update(arg)
    wandb.init(project=config.algo, name=f"{config.task}_{config.method}_seed{config.seed}", config=config)
    print(f">>>> Training STAR method={config.method} on {config.task}", flush=True)
    train_loop(config)
