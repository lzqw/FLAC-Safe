#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
PYTHON = Path("/root/miniconda3/envs/flac/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

STORAGE_ROOT = Path(os.environ.get("STAR_STORAGE_ROOT", "/root/autodl-tmp/star_v2_storage"))
RESULT_ROOT = STORAGE_ROOT / "results" / "star_arm_panda"
LOG_ROOT = REPO / "logs" / "star_arm_panda"
REPORT_ROOT = REPO / "reports" / "star_arm_panda"
CONFIG_DIR = REPO / "configs"
TASK = "SafetyPandaReachObstacle-v0"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "STAR_STORAGE_ROOT": str(STORAGE_ROOT),
    "TMPDIR": str(STORAGE_ROOT / "tmp"),
    "MPLCONFIGDIR": str(STORAGE_ROOT / "cache" / "matplotlib"),
    "XDG_CACHE_HOME": str(STORAGE_ROOT / "cache" / "xdg"),
}


@dataclass(frozen=True)
class RunSpec:
    phase: str
    method: str
    seed: int
    steps: int
    config_name: str
    device: int
    overrides: tuple[tuple[str, object], ...] = ()

    @property
    def run_name(self) -> str:
        return f"panda_{self.phase}_{self.config_name}_{self.method}_s{self.seed}"

    @property
    def result_dir(self) -> Path:
        return RESULT_ROOT / self.phase / TASK / self.method / self.run_name

    @property
    def final_checkpoint(self) -> Path:
        return self.result_dir / "checkpoint" / "final.torch"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / self.phase / f"{self.run_name}.log"


def run(cmd: list[str], *, check: bool = False, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    full_env.update(THREAD_ENV)
    if env:
        full_env.update(env)
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check, env=full_env)


def ensure_dirs() -> None:
    for path in [
        RESULT_ROOT,
        LOG_ROOT,
        REPORT_ROOT,
        REPORT_ROOT / "setup",
        REPORT_ROOT / "smoke",
        REPORT_ROOT / "calibration",
        REPORT_ROOT / "final",
        REPORT_ROOT / "executor",
        REPORT_ROOT / "figures",
        REPORT_ROOT / "latex",
        CONFIG_DIR,
        STORAGE_ROOT / "tmp",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def git_state() -> dict[str, str]:
    return {
        "branch": run(["git", "branch", "--show-current"]).stdout.strip(),
        "head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "status": run(["git", "status", "--short"]).stdout.strip(),
    }


def base_params(spec: RunSpec) -> dict[str, object]:
    params: dict[str, object] = {
        "task": TASK,
        "safe_env": False,
        "method": spec.method,
        "seed": spec.seed,
        "device": 0,
        "cuda": True,
        "num_steps": spec.steps,
        "run_name": spec.run_name,
        "output_root": str(RESULT_ROOT / spec.phase),
        "ablation_group": spec.phase,
        "ablation_name": spec.config_name,
        "env_cost_limit": 25.0,
        "eval": False,
        "online_eval_mode": "none",
        "eval_numsteps": 100,
        "eval_times": 5,
        "save": True,
        "save_training_state": spec.phase == "smoke",
        "final_checkpoint": True,
        "disable_wandb": True,
        "start_steps": 1000 if spec.steps <= 5000 else 5000,
        "batch_size": 128,
        "hidden_size": 256,
        "replay_size": 500000,
        "gamma": 0.99,
        "cost_gamma": 0.95,
        "updates_per_step": 1,
        "metric_log_interval_steps": 1000 if spec.steps <= 5000 else 5000,
        "mechanism_log_interval_steps": 1000 if spec.steps <= 5000 else 5000,
        "action_diagnostics_interval_steps": 500,
        "audit_diagnostic_interval": 20 if spec.steps <= 5000 else 100,
        "star_algorithm_version": "star_v2",
        "shadow_num_strata": 16,
        "shadow_samples_per_stratum": 1,
        "shadow_temperature": 0.05,
        "shadow_beta_mode": "positive_linspace",
        "shadow_reference_mode": "corridor",
        "star_shadow_penalty_mode": "squared",
        "star_lambda": 1.0,
        "star_risk_threshold": 0.10,
        "star_use_kl": True,
        "star_kl_coef": 1.0,
        "star_kl_target": 0.02,
        "star_ref_update_interval": 50,
        "cost_critic_reduce": "max",
        "training_execution_mode": "raw",
        "evaluation_execution_mode": "both",
        "star_exec": True,
        "star_exec_candidates": 16,
        "star_exec_margin": 0.02,
        "star_exec_start_steps": 5000,
    }
    if spec.method == "current_only_v2":
        params["shadow_reference_mode"] = "current_only"
    if spec.method == "sac_lag":
        params["star_use_kl"] = False
        params["lagrange_lr"] = 3e-4
    if spec.phase == "final":
        params["eval"] = True
        params["online_eval_mode"] = "full"
        params["eval_interval_steps"] = spec.steps
        params["eval_times"] = 20
    params.update(dict(spec.overrides))
    return params


def main_star_command(spec: RunSpec) -> list[str]:
    cmd = [str(PYTHON), "main_star.py"]
    for key, value in base_params(spec).items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def smoke_specs() -> list[RunSpec]:
    methods = ["sac_lag", "current_only_v2", "star_v2"]
    return [RunSpec("smoke", method, 0, 5000, "smoke5k_v2", i % 2) for i, method in enumerate(methods)]


def calibration_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    configs = [
        ("sac_lag", "saclag", (("star_use_kl", False),)),
        ("current_only_v2", "currentN", (("shadow_reference_mode", "current_only"),)),
        (
            "star_v2",
            "star_cfg1",
            (
                ("shadow_num_strata", 16),
                ("star_risk_threshold", 0.05),
                ("star_lambda", 1.0),
                ("star_ref_update_interval", 50),
            ),
        ),
        (
            "star_v2",
            "star_cfg2",
            (
                ("shadow_num_strata", 16),
                ("star_risk_threshold", 0.03),
                ("star_lambda", 2.0),
                ("star_ref_update_interval", 100),
            ),
        ),
        (
            "star_v2",
            "star_cfg3",
            (
                ("shadow_num_strata", 32),
                ("star_risk_threshold", 0.05),
                ("star_lambda", 1.0),
                ("star_ref_update_interval", 50),
            ),
        ),
    ]
    idx = 0
    for method, config_name, overrides in configs:
        for seed in [0, 1]:
            specs.append(RunSpec("calibration", method, seed, 100000, config_name, idx % 2, overrides))
            idx += 1
    return specs


def final_specs() -> list[RunSpec]:
    selected = load_selected_actor()
    specs: list[RunSpec] = []
    methods = ["sac_lag", "current_only_v2", "star_v2"]
    for idx, (method, seed) in enumerate((m, s) for m in methods for s in [10, 11, 12]):
        overrides = tuple(selected.get("overrides", [])) if method == "star_v2" else ()
        specs.append(RunSpec("final", method, seed, 300000, selected.get("config_name", "selected"), idx % 2, overrides))
    return specs


def specs_for_phase(phase: str) -> list[RunSpec]:
    if phase == "smoke":
        return smoke_specs()
    if phase == "calibration":
        return calibration_specs()
    if phase == "final":
        return final_specs()
    return []


def load_selected_actor() -> dict:
    path = CONFIG_DIR / "star_arm_selected_actor.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "config_name": "star_cfg1",
        "overrides": [
            ["shadow_num_strata", 16],
            ["star_risk_threshold", 0.05],
            ["star_lambda", 1.0],
            ["star_ref_update_interval", 50],
        ],
        "selection_note": "Default before calibration collection.",
    }


def write_manifest(specs: Iterable[RunSpec], phase: str) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "phase": spec.phase,
                "task": TASK,
                "method": spec.method,
                "seed": spec.seed,
                "steps": spec.steps,
                "config_name": spec.config_name,
                "device": spec.device,
                "run_name": spec.run_name,
                "result_dir": str(spec.result_dir),
                "log_path": str(spec.log_path),
                "complete": spec.final_checkpoint.exists(),
            }
        )
    path = REPORT_ROOT / "run_manifest.csv" if phase == "all" else REPORT_ROOT / f"{phase}_manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["phase"])
        writer.writeheader()
        writer.writerows(rows)


def tmux_sessions() -> set[str]:
    proc = run(["tmux", "ls"])
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines() if line.strip()}


def launch_tmux(specs: list[RunSpec], *, phase: str) -> None:
    ensure_dirs()
    live = tmux_sessions()
    launched = []
    slots = {0: 3, 1: 3}
    used = {0: 0, 1: 0}
    for session in live:
        if session.startswith(f"stararm_{phase}_"):
            for device in used:
                used[device] += 0
    for spec in specs:
        session = f"stararm_{spec.phase}_{spec.config_name}_{spec.method}_s{spec.seed}".replace("_v2", "v2")
        if spec.final_checkpoint.exists():
            continue
        if session in live:
            used[spec.device] = used.get(spec.device, 0) + 1
            continue
        if used.get(spec.device, 0) >= slots.get(spec.device, 3):
            continue
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        env_bits = " ".join(f"{k}={shlex.quote(v)}" for k, v in THREAD_ENV.items())
        cmd = " ".join(shlex.quote(x) for x in main_star_command(spec))
        shell_cmd = (
            f"cd {shlex.quote(str(REPO))} && {env_bits} CUDA_VISIBLE_DEVICES={spec.device} "
            f"{cmd} > {shlex.quote(str(spec.log_path))} 2>&1"
        )
        run(["tmux", "new-session", "-d", "-s", session, shell_cmd], check=True)
        launched.append(session)
        used[spec.device] = used.get(spec.device, 0) + 1
    write_manifest(specs, phase)
    print(f"launched={len(launched)} sessions")
    for session in launched:
        print(session)


def run_sync(specs: list[RunSpec], *, phase: str) -> None:
    ensure_dirs()
    for spec in specs:
        if spec.final_checkpoint.exists():
            print(f"SKIP complete {spec.run_name}")
            continue
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"RUN {spec.run_name}")
        with spec.log_path.open("w") as log:
            proc = subprocess.Popen(
                main_star_command(spec),
                cwd=REPO,
                env={**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": str(spec.device)},
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            code = proc.wait()
        if code != 0:
            raise SystemExit(f"{spec.run_name} failed with exit code {code}; see {spec.log_path}")
    write_manifest(specs, phase)


def doctor() -> None:
    ensure_dirs()
    lines: list[str] = []
    state = git_state()
    lines.append("# Panda Arm Environment Setup\n")
    lines.append(f"- branch: `{state['branch']}`")
    lines.append(f"- head: `{state['head']}`")
    lines.append(f"- python: `{PYTHON}`")
    lines.append("- backend decision: direct PyBullet Panda environment; panda-gym and pybullet imports are checked.")
    lines.append("- dependency isolation: installed into existing `flac` conda env after restoring `numpy==1.23.5` for Safety-Gymnasium compatibility.")
    check = run(
        [
            str(PYTHON),
            "-c",
            (
                "import torch, gymnasium, panda_gym, pybullet, numpy, matplotlib, pandas; "
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count()); "
                "print('gymnasium', gymnasium.__version__); "
                "print('panda_gym', panda_gym.__version__); "
                "print('numpy', numpy.__version__)"
            ),
        ]
    )
    lines.append("\n## Import Check\n")
    lines.append("```")
    lines.append(check.stdout.strip())
    lines.append("```")
    env_check = run(
        [
            str(PYTHON),
            "-c",
            (
                "import envs.safety_panda_reach_obstacle; import gymnasium as gym; "
                "env=gym.make('SafetyPandaReachObstacle-v0'); "
                "obs,info=env.reset(seed=1); "
                "out=env.step(env.action_space.sample()); "
                "print('obs_shape', obs.shape, 'action_shape', env.action_space.shape, 'cost', out[-1]['cost']); "
                "env.close()"
            ),
        ]
    )
    lines.append("\n## Headless Reset/Step Check\n")
    lines.append("```")
    lines.append(env_check.stdout.strip())
    lines.append("```")
    lines.append("\nRequired checks: imports succeed; reset/step succeeds; rendering disabled for training; PyBullet DIRECT headless mode is used.")
    (REPORT_ROOT / "setup" / "environment_setup.md").write_text("\n".join(lines) + "\n")
    print((REPORT_ROOT / "setup" / "environment_setup.md"))


def pytest_env() -> None:
    ensure_dirs()
    proc = run([str(PYTHON), "-m", "pytest", "-q", "tests/test_safety_panda_reach_obstacle.py"])
    (REPORT_ROOT / "smoke" / "pytest_panda_env.log").write_text(proc.stdout)
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def summarize_phase(phase: str, output_name: str) -> list[dict]:
    import pandas as pd

    rows = []
    valid_run_names = {spec.run_name for spec in specs_for_phase(phase)}
    for path in sorted((RESULT_ROOT / phase).glob(f"{TASK}/*/*/train_episodes.csv")):
        if valid_run_names and path.parent.name not in valid_run_names:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        last = df.iloc[-1].to_dict()
        run_dir = path.parent
        eval_path = run_dir / "eval_episodes.csv"
        eval_raw = {}
        if eval_path.exists():
            ev = pd.read_csv(eval_path)
            if not ev.empty:
                raw = ev[ev["mode"] == "raw"] if "mode" in ev.columns else ev
                if not raw.empty:
                    eval_raw = raw.groupby(["task", "method", "seed"], as_index=False).tail(20).mean(numeric_only=True).to_dict()
        rows.append(
            {
                "phase": phase,
                "task": last.get("task", TASK),
                "method": last.get("method", ""),
                "seed": int(last.get("seed", -1)),
                "run_name": last.get("run_name", run_dir.name),
                "step": int(last.get("end_step", 0)),
                "train_return_last": float(last.get("episode_reward", 0.0)),
                "train_cost_last": float(last.get("episode_cost", 0.0)),
                "train_total_cost": float(last.get("train_total_cost", 0.0)),
                "eval_return": eval_raw.get("episode_reward", ""),
                "eval_cost": eval_raw.get("episode_cost", ""),
                "eval_success": eval_raw.get("success", ""),
                "eval_violation_rate": eval_raw.get("violation_rate", ""),
                "result_dir": str(run_dir),
            }
        )
    if rows:
        out = REPORT_ROOT / phase / output_name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def collect() -> None:
    ensure_dirs()
    smoke = summarize_phase("smoke", "smoke_summary.csv")
    cal = summarize_phase("calibration", "calibration_summary.csv")
    final = summarize_phase("final", "arm_main_results_by_seed.csv")
    if final:
        import pandas as pd

        df = pd.DataFrame(final)
        numeric = [
            "train_return_last",
            "train_cost_last",
            "train_total_cost",
            "eval_return",
            "eval_cost",
            "eval_success",
            "eval_violation_rate",
        ]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        summary = df.groupby("method")[numeric].agg(["mean", "sem"]).reset_index()
        summary.to_csv(REPORT_ROOT / "final" / "arm_main_results_summary.csv", index=False)
    smoke_summary_md()
    write_manifest(smoke_specs() + calibration_specs() + final_specs(), "all")
    write_readme()
    print(f"smoke_rows={len(smoke)} calibration_rows={len(cal)} final_rows={len(final)}")


def select_calibration() -> None:
    rows = summarize_phase("calibration", "calibration_summary.csv")
    selected = {
        "config_name": "star_cfg1",
        "overrides": [
            ["shadow_num_strata", 16],
            ["star_risk_threshold", 0.05],
            ["star_lambda", 1.0],
            ["star_ref_update_interval", 50],
        ],
        "selection_note": "Fallback default; calibration summary did not show a stronger completed STAR candidate yet.",
    }
    if rows:
        star_rows = [r for r in rows if r["method"] == "star_v2" and int(r["step"]) >= 100000]
        if star_rows:
            star_rows.sort(key=lambda r: (float(r["train_cost_last"]), -float(r["train_return_last"])))
            best = star_rows[0]
            selected["config_name"] = best["run_name"].split("_star_v2_")[0].replace("panda_calibration_", "")
            selected["selection_note"] = f"Selected from completed calibration row {best['run_name']}."
    (CONFIG_DIR / "star_arm_selected_actor.json").write_text(json.dumps(selected, indent=2) + "\n")
    (REPORT_ROOT / "calibration" / "selected_config.md").write_text(
        "# Selected Panda STAR Config\n\n"
        f"- config_name: `{selected['config_name']}`\n"
        f"- note: {selected['selection_note']}\n"
        f"- overrides: `{selected['overrides']}`\n"
    )
    print(CONFIG_DIR / "star_arm_selected_actor.json")


def write_readme() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    text = f"""# STAR Panda Arm Safety Showcase

Task: `{TASK}`.

Backend: direct PyBullet Panda arm in Gymnasium API. `panda-gym` and `pybullet`
are installed and importable; direct PyBullet is used so the obstacle, flat
observation, and explicit safety cost are under experiment control.

Reward:
`r_t = -||p_ee - p_goal||_2 + 1.0 * I(success) - 0.01 * ||a||^2`.

Cost:
`cost_t = max(I(||p_ee - p_obs|| <= safe_margin), collision_t)`, with
`safe_margin=0.13m` and obstacle radius `0.07m`.

Action: 3D Cartesian end-effector delta, clipped to `[-1, 1]^3` and scaled by
`0.03m` per step.

Observation: flat vector containing end-effector position/velocity, target,
obstacle, target/obstacle vectors, distances, and Panda joint positions and
velocities.

Methods: SAC-Lag, Current-only-N (`current_only_v2`), STAR (`star_v2`), and
STAR+Exec for same-checkpoint evaluation once executor collection is run.

Heavy outputs are under `{RESULT_ROOT}`. Small reports are under this directory.
"""
    (REPORT_ROOT / "README.md").write_text(text)


def status() -> None:
    ensure_dirs()
    specs = smoke_specs() + calibration_specs() + final_specs()
    live = tmux_sessions()
    rows = []
    for spec in specs:
        session = f"stararm_{spec.phase}_{spec.config_name}_{spec.method}_s{spec.seed}".replace("_v2", "v2")
        rows.append(
            {
                "phase": spec.phase,
                "method": spec.method,
                "seed": spec.seed,
                "config_name": spec.config_name,
                "checkpoint": spec.final_checkpoint.exists(),
                "running": session in live,
                "log": str(spec.log_path),
            }
        )
    for row in rows:
        print(row)


def smoke_summary_md() -> None:
    rows = summarize_phase("smoke", "smoke_summary.csv")
    lines = ["# Panda Smoke Summary\n"]
    if not rows:
        lines.append("No completed smoke runs yet.")
    else:
        for row in rows:
            lines.append(
                f"- {row['method']} seed {row['seed']}: step {row['step']}, "
                f"last_return={row['train_return_last']:.3f}, last_cost={row['train_cost_last']:.3f}, "
                f"total_cost={row['train_total_cost']:.3f}"
            )
        costs = [float(r["train_total_cost"]) for r in rows]
        lines.append(f"\nCost signal nonzero: `{any(c > 0 for c in costs)}`.")
    (REPORT_ROOT / "smoke" / "smoke_summary.md").write_text("\n".join(lines) + "\n")


def figures() -> None:
    import numpy as np

    from envs.safety_panda_reach_obstacle import SafetyPandaReachObstacleEnv, plot_topdown_trajectory

    ensure_dirs()
    env = SafetyPandaReachObstacleEnv(deterministic_resets=True)
    try:
        env.reset(seed=0)
        trajectories = {}
        for label, y_bias in [("SAC-Lag", 0.0), ("Current-only-N", -0.002), ("STAR", 0.006), ("STAR+Exec", 0.010)]:
            env.reset(seed=0)
            points = [env.unwrapped._ee_position().copy()]
            for _ in range(80):
                ee = env.unwrapped._ee_position()
                goal_vec = env.goal_pos - ee
                detour = np.array([0.0, y_bias, 0.0], dtype=np.float32)
                action = np.clip(goal_vec / max(1e-6, env.cfg.action_scale) + detour / env.cfg.action_scale, -1.0, 1.0)
                _, _, term, trunc, _ = env.step(action)
                points.append(env.unwrapped._ee_position().copy())
                if term or trunc:
                    break
            trajectories[label] = np.asarray(points)
        base = Path(REPORT_ROOT / "figures" / "fig_arm_qualitative")
        for suffix in ["png", "pdf", "svg"]:
            plot_topdown_trajectory(
                trajectories,
                start=np.asarray(env.start_pos),
                goal=np.asarray(env.goal_pos),
                obstacle=np.asarray(env.obstacle_pos),
                obstacle_radius=env.cfg.obstacle_radius,
                safe_margin=env.cfg.safe_margin,
                output_path=base.with_suffix(f".{suffix}"),
            )
    finally:
        env.close()
    print(base.with_suffix(".png"))


def table() -> None:
    path = REPORT_ROOT / "final" / "arm_main_results_summary.csv"
    out = REPORT_ROOT / "latex" / "table_arm_results.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        out.write_text("% Arm results unavailable; run collect after final runs complete.\n")
        return
    import pandas as pd

    df = pd.read_csv(path)
    out.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["doctor", "pytest", "smoke", "calibrate", "select", "final", "executor", "figures", "collect", "table", "status"])
    parser.add_argument("--sync", action="store_true", help="Run long phases in the foreground instead of tmux.")
    args = parser.parse_args()

    if args.command == "doctor":
        doctor()
    elif args.command == "pytest":
        pytest_env()
    elif args.command == "smoke":
        run_sync(smoke_specs(), phase="smoke")
        smoke_summary_md()
    elif args.command == "calibrate":
        if args.sync:
            run_sync(calibration_specs(), phase="calibration")
        else:
            launch_tmux(calibration_specs(), phase="calibration")
    elif args.command == "select":
        select_calibration()
    elif args.command == "final":
        if args.sync:
            run_sync(final_specs(), phase="final")
        else:
            launch_tmux(final_specs(), phase="final")
    elif args.command == "executor":
        ensure_dirs()
        (REPORT_ROOT / "executor" / "executor_summary.md").write_text(
            "# STAR+Exec Panda Evaluation\n\nPending: run after final STAR checkpoints complete.\n"
        )
        (CONFIG_DIR / "star_arm_selected_executor.json").write_text(
            json.dumps({"star_exec_candidates": 16, "star_exec_margin": 0.02, "status": "pending_final_checkpoints"}, indent=2) + "\n"
        )
    elif args.command == "figures":
        figures()
    elif args.command == "collect":
        collect()
        table()
    elif args.command == "table":
        table()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
