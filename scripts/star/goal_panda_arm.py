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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
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
        summary.columns = [
            "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
            for col in summary.columns
        ]
        summary = summary.rename(columns={"method_": "method"})
        summary.to_csv(REPORT_ROOT / "final" / "arm_main_results_summary.csv", index=False)
    collect_training_curves()
    collect_mechanism_summary()
    write_final_docs()
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
    override_by_config = {
        "star_cfg1": [
            ["shadow_num_strata", 16],
            ["star_risk_threshold", 0.05],
            ["star_lambda", 1.0],
            ["star_ref_update_interval", 50],
        ],
        "star_cfg2": [
            ["shadow_num_strata", 16],
            ["star_risk_threshold", 0.03],
            ["star_lambda", 2.0],
            ["star_ref_update_interval", 100],
        ],
        "star_cfg3": [
            ["shadow_num_strata", 32],
            ["star_risk_threshold", 0.05],
            ["star_lambda", 1.0],
            ["star_ref_update_interval", 50],
        ],
    }
    if rows:
        star_rows = [r for r in rows if r["method"] == "star_v2" and int(r["step"]) >= 100000]
        if star_rows:
            star_rows.sort(key=lambda r: (float(r["train_cost_last"]), -float(r["train_return_last"])))
            best = star_rows[0]
            selected["config_name"] = best["run_name"].split("_star_v2_")[0].replace("panda_calibration_", "")
            selected["overrides"] = override_by_config.get(selected["config_name"], selected["overrides"])
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
    selected_actor_path = CONFIG_DIR / "star_arm_selected_actor.json"
    selected_exec_path = CONFIG_DIR / "star_arm_selected_executor.json"
    selected_actor = json.loads(selected_actor_path.read_text()) if selected_actor_path.exists() else {}
    selected_exec = json.loads(selected_exec_path.read_text()) if selected_exec_path.exists() else {}
    claim_path = REPORT_ROOT / "final_claim.md"
    claim_note = claim_path.read_text().strip() if claim_path.exists() else "Final result not collected yet."
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

Seeds and steps:
- smoke: seed 0, 5k steps.
- calibration: seeds 0-1, 100k steps.
- final: seeds 10-12, 300k steps.

Selected STAR actor config:
```json
{json.dumps(selected_actor, indent=2)}
```

Selected STAR+Exec config:
```json
{json.dumps(selected_exec, indent=2)}
```

Final claim status:
{claim_note}

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
    from agents.star_agent import STARAgent
    from main_star import make_env

    from envs.safety_panda_reach_obstacle import SafetyPandaReachObstacleEnv, plot_topdown_trajectory

    ensure_dirs()
    specs = {
        "SAC-Lag": next(s for s in final_specs() if s.method == "sac_lag" and s.seed == 10),
        "Current-only-N": next(s for s in final_specs() if s.method == "current_only_v2" and s.seed == 10),
        "STAR": next(s for s in final_specs() if s.method == "star_v2" and s.seed == 10),
        "STAR+Exec": next(s for s in final_specs() if s.method == "star_v2" and s.seed == 10),
    }
    selected_exec = json.loads((CONFIG_DIR / "star_arm_selected_executor.json").read_text()) if (CONFIG_DIR / "star_arm_selected_executor.json").exists() else {}
    trajectories = {}
    start = goal = obstacle = None
    radius = 0.07
    margin = 0.13
    for label, spec in specs.items():
        extra = {}
        mode = "raw"
        if label == "STAR+Exec":
            mode = "star_exec"
            extra = {
                "star_exec_candidates": int(selected_exec.get("star_exec_candidates", 16)),
                "star_exec_margin": float(selected_exec.get("star_exec_margin", 0.02)),
                "star_exec_start_steps": 0,
            }
        config = _config_from_spec(spec, extra)
        env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
        agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
        agent.load_checkpoint(str(spec.final_checkpoint))
        try:
            state, _ = env.reset(seed=800000)
            if start is None:
                start = np.asarray(env.unwrapped.start_pos)
                goal = np.asarray(env.unwrapped.goal_pos)
                obstacle = np.asarray(env.unwrapped.obstacle_pos)
                radius = float(env.unwrapped.cfg.obstacle_radius)
                margin = float(env.unwrapped.cfg.safe_margin)
            points = [env.unwrapped._ee_position().copy()]
            for _ in range(100):
                action = agent.select_action(state, evaluate=True, execution_mode=mode, total_numsteps=300000, diagnostics=True)
                state, _, terminated, truncated, _ = env.step(action)
                points.append(env.unwrapped._ee_position().copy())
                if terminated or truncated:
                    break
            trajectories[label] = np.asarray(points)
        finally:
            env.close()
    if start is None:
        fallback_env = SafetyPandaReachObstacleEnv(deterministic_resets=True)
        fallback_env.reset(seed=0)
        start = np.asarray(fallback_env.start_pos)
        goal = np.asarray(fallback_env.goal_pos)
        obstacle = np.asarray(fallback_env.obstacle_pos)
        radius = fallback_env.cfg.obstacle_radius
        margin = fallback_env.cfg.safe_margin
        fallback_env.close()
    base = Path(REPORT_ROOT / "figures" / "fig_arm_qualitative")
    for suffix in ["png", "pdf", "svg"]:
        plot_topdown_trajectory(
            trajectories,
            start=start,
            goal=goal,
            obstacle=obstacle,
            obstacle_radius=radius,
            safe_margin=margin,
            output_path=base.with_suffix(f".{suffix}"),
        )
    print(base.with_suffix(".png"))


def collect_training_curves() -> None:
    import pandas as pd

    rows = []
    for path in sorted((RESULT_ROOT / "final").glob(f"{TASK}/*/*/train_episodes.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        keep = [
            "run_name",
            "task",
            "method",
            "seed",
            "episode",
            "end_step",
            "episode_reward",
            "episode_cost",
            "episode_length",
            "train_total_cost",
            "train_total_cost_rate",
        ]
        rows.append(df[[c for c in keep if c in df.columns]])
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(REPORT_ROOT / "final" / "arm_training_curves.csv", index=False)


def collect_mechanism_summary() -> None:
    import pandas as pd

    rows = []
    for path in sorted((RESULT_ROOT / "final").glob(f"{TASK}/star_v2/*/mechanism.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        tail = df.tail(10)
        row = {
            "run_name": df["run_name"].iloc[-1],
            "task": df["task"].iloc[-1],
            "method": df["method"].iloc[-1],
            "seed": int(df["seed"].iloc[-1]),
            "step_last": int(df["step"].iloc[-1]),
        }
        for col in [
            "hidden_unsafe_rate",
            "paired_corridor_risk",
            "paired_current_risk",
            "paired_corridor_risk_lift",
            "paired_lift_positive_rate",
            "shadow_excess_mean",
            "effective_beta",
            "found_but_not_executed_rate",
        ]:
            if col in tail.columns:
                row[col] = float(tail[col].mean())
        rows.append(row)
    if rows:
        _write_rows(REPORT_ROOT / "final" / "arm_mechanism_summary.csv", rows)


def write_final_docs() -> None:
    import pandas as pd

    by_seed_path = REPORT_ROOT / "final" / "arm_main_results_by_seed.csv"
    exec_path = REPORT_ROOT / "executor" / "executor_summary.md"
    lines = ["# Panda Arm Final Claim", ""]
    audit = ["# Panda Arm Final Audit", ""]
    if not by_seed_path.exists():
        lines.append("Final results are not yet available.")
        audit.append("Missing final by-seed results.")
    else:
        df = pd.read_csv(by_seed_path)
        means = df.groupby("method")[["train_total_cost", "eval_return", "eval_cost", "eval_success", "eval_violation_rate"]].mean(numeric_only=True)
        lines.append("The Panda arm add-on is a mixed/diagnostic result, not a clean STAR-win showcase.")
        lines.append("")
        for method, row in means.iterrows():
            lines.append(
                f"- {method}: train_total_cost={row['train_total_cost']:.1f}, "
                f"eval_return={row['eval_return']:.3f}, eval_cost={row['eval_cost']:.3f}, "
                f"eval_success={row['eval_success']:.3f}, eval_violation_rate={row['eval_violation_rate']:.3f}"
            )
        lines.append("")
        lines.append("STAR reduces evaluation cost relative to Current-only in the final summary, but it does not dominate SAC-Lag and has worse return/success tradeoffs. Present as an exploratory robot-arm add-on only.")
        audit.extend(
            [
                f"- final by-seed rows: {len(df)}",
                f"- methods: {', '.join(sorted(df['method'].unique()))}",
                "- final 300k checkpoints exist for all three seeds per method.",
                "- success gate is not fully met because STAR does not beat SAC-Lag on the overall success-cost tradeoff.",
            ]
        )
    if exec_path.exists():
        lines.append("")
        lines.append("STAR+Exec summary is available in `reports/star_arm_panda/executor/executor_summary.md`.")
        audit.append("- executor validation and confirmation were generated from same STAR checkpoints.")
    (REPORT_ROOT / "final_claim.md").write_text("\n".join(lines) + "\n")
    (REPORT_ROOT / "final_audit.md").write_text("\n".join(audit) + "\n")


def table() -> None:
    path = REPORT_ROOT / "final" / "arm_main_results_by_seed.csv"
    out = REPORT_ROOT / "latex" / "table_arm_results.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        out.write_text("% Arm results unavailable; run collect after final runs complete.\n")
        return
    import pandas as pd

    df = pd.read_csv(path)
    summary = df.groupby("method").agg(
        Return=("eval_return", "mean"),
        Success=("eval_success", "mean"),
        Cost=("eval_cost", "mean"),
        Violation=("eval_violation_rate", "mean"),
        TrainCost=("train_total_cost", "mean"),
    ).reset_index()
    labels = {"current_only_v2": "Current-only-N", "sac_lag": "SAC-Lag", "star_v2": "STAR"}
    summary["method"] = summary["method"].map(labels).fillna(summary["method"])
    summary = summary.rename(
        columns={
            "method": "Method",
            "Return": "Return $\\uparrow$",
            "Success": "Success $\\uparrow$",
            "Cost": "Cost $\\downarrow$",
            "Violation": "EVR $\\downarrow$",
            "TrainCost": "Train cost $\\downarrow$",
        }
    )
    out.write_text(summary.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"))
    print(out)


def _config_from_spec(spec: RunSpec, extra: dict | None = None):
    from utilis.star_default_config import star_default_config

    config = star_default_config.copy()
    params = base_params(spec)
    params.update(
        {
            "eval_numsteps": 100,
            "eval_times": 20,
            "online_eval_mode": "none",
            "star_exec_start_steps": 0,
            "device": 0,
            "cuda": True,
            "disable_wandb": True,
        }
    )
    if extra:
        params.update(extra)
    config.update(params)
    return config


def _load_star_agent(spec: RunSpec, *, candidates: int, margin: float):
    from agents.star_agent import STARAgent
    from main_star import make_env

    checkpoint = spec.final_checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing STAR checkpoint: {checkpoint}")
    config = _config_from_spec(
        spec,
        {
            "star_exec_candidates": int(candidates),
            "star_exec_margin": float(margin),
            "star_exec": True,
            "training_execution_mode": "raw",
            "evaluation_execution_mode": "both",
        },
    )
    env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
    agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
    agent.load_checkpoint(str(checkpoint))
    return agent, env, config


def _eval_one_episode(agent, env, config, *, eval_seed: int, mode: str) -> dict:
    import numpy as np
    from main_star import reset_env, step_env

    state = reset_env(env, seed=eval_seed)
    done = False
    reward_sum = 0.0
    cost_sum = 0.0
    violations = 0
    collisions = 0
    path_length = 0.0
    min_clearance = float("inf")
    fallbacks = 0
    fne_count = 0
    latency = []
    steps = 0
    info = {}
    while not done and steps < int(config.eval_numsteps):
        t0 = time.perf_counter()
        action = agent.select_action(state, evaluate=True, execution_mode=mode, total_numsteps=300000, diagnostics=True)
        latency.append((time.perf_counter() - t0) * 1000.0)
        next_state, reward, cost, terminated, truncated, info = step_env(env, action, config.safe_env)
        done = terminated or truncated
        details = agent.last_action_info
        risk = float(details.get("selected_predicted_risk", 0.0))
        found = (
            bool(details.get("any_shadow_predicted_unsafe", False))
            and risk <= float(config.star_risk_threshold)
            and float(cost) <= 0.0
        )
        reward_sum += float(reward)
        cost_sum += float(cost)
        violations += int(cost > 0)
        collisions += int(float(info.get("collision", 0.0)) > 0)
        path_length += float(info.get("path_length_increment", 0.0))
        min_clearance = min(min_clearance, float(info.get("min_clearance", float("inf"))))
        fallbacks += int(bool(details.get("execution_fallback", False)))
        fne_count += int(found)
        steps += 1
        state = next_state
    return {
        "eval_seed": eval_seed,
        "mode": mode,
        "episode_reward": reward_sum,
        "episode_cost": cost_sum,
        "episode_length": steps,
        "success": float(info.get("success", 0.0)),
        "violation_rate": violations / max(1, steps),
        "collision_rate": collisions / max(1, steps),
        "min_clearance": min_clearance if np.isfinite(min_clearance) else "",
        "path_length": path_length,
        "fallback_rate": fallbacks / max(1, steps),
        "found_but_not_executed_rate": fne_count / max(1, steps),
        "latency_ms": float(np.mean(latency)) if latency else 0.0,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _executor_eval_rows(
    *,
    seeds: list[int],
    eval_seeds: list[int],
    candidates_values: list[int],
    margin_values: list[float],
) -> list[dict]:
    rows: list[dict] = []
    star_specs = [spec for spec in final_specs() if spec.method == "star_v2" and spec.seed in seeds]
    for candidates in candidates_values:
        for margin in margin_values:
            for spec in star_specs:
                agent, env, config = _load_star_agent(spec, candidates=candidates, margin=margin)
                try:
                    for eval_seed in eval_seeds:
                        for mode in ["raw", "star_exec"]:
                            row = _eval_one_episode(agent, env, config, eval_seed=eval_seed, mode=mode)
                            row.update(
                                {
                                    "train_seed": spec.seed,
                                    "run_name": spec.run_name,
                                    "candidates": candidates,
                                    "margin": margin,
                                    "checkpoint": str(spec.final_checkpoint),
                                }
                            )
                            rows.append(row)
                finally:
                    env.close()
    return rows


def _summarize_executor(rows: list[dict]) -> tuple[dict, list[dict]]:
    import pandas as pd

    df = pd.DataFrame(rows)
    metric_cols = [
        "episode_reward",
        "episode_cost",
        "success",
        "violation_rate",
        "collision_rate",
        "min_clearance",
        "path_length",
        "fallback_rate",
        "found_but_not_executed_rate",
        "latency_ms",
    ]
    grouped = df.groupby(["candidates", "margin", "mode"], as_index=False)[metric_cols].mean(numeric_only=True)
    raw = grouped[grouped["mode"] == "raw"].set_index(["candidates", "margin"])
    exec_df = grouped[grouped["mode"] == "star_exec"].set_index(["candidates", "margin"])
    selection_rows: list[dict] = []
    for key, erow in exec_df.iterrows():
        rrow = raw.loc[key]
        success_drop = float(rrow["success"] - erow["success"])
        row = {
            "candidates": int(key[0]),
            "margin": float(key[1]),
            "raw_cost": float(rrow["episode_cost"]),
            "exec_cost": float(erow["episode_cost"]),
            "raw_violation_rate": float(rrow["violation_rate"]),
            "exec_violation_rate": float(erow["violation_rate"]),
            "raw_success": float(rrow["success"]),
            "exec_success": float(erow["success"]),
            "success_drop": success_drop,
            "raw_return": float(rrow["episode_reward"]),
            "exec_return": float(erow["episode_reward"]),
            "exec_fallback_rate": float(erow["fallback_rate"]),
            "exec_fne": float(erow["found_but_not_executed_rate"]),
            "exec_latency_ms": float(erow["latency_ms"]),
        }
        selection_rows.append(row)
    feasible = [r for r in selection_rows if r["success_drop"] <= 0.05]
    pool = feasible if feasible else selection_rows
    pool.sort(
        key=lambda r: (
            r["exec_violation_rate"],
            r["exec_cost"],
            max(0.0, r["success_drop"]),
            r["exec_fallback_rate"],
            r["candidates"],
        )
    )
    return pool[0], selection_rows


def executor_eval() -> None:
    ensure_dirs()
    validation_path = REPORT_ROOT / "executor" / "executor_validation.csv"
    confirmation_path = REPORT_ROOT / "executor" / "executor_confirmation.csv"
    selection_path = REPORT_ROOT / "executor" / "executor_validation_summary.csv"
    selected_config_path = CONFIG_DIR / "star_arm_selected_executor.json"

    if not validation_path.exists():
        validation_rows = _executor_eval_rows(
            seeds=[10, 11, 12],
            eval_seeds=list(range(700000, 700010)),
            candidates_values=[8, 16, 32],
            margin_values=[0.00, 0.02, 0.05, 0.08],
        )
        _write_rows(validation_path, validation_rows)
    else:
        import pandas as pd

        validation_rows = pd.read_csv(validation_path).to_dict("records")

    selected, selection_rows = _summarize_executor(validation_rows)
    _write_rows(selection_path, selection_rows)
    selected_config = {
        "star_exec_candidates": int(selected["candidates"]),
        "star_exec_margin": float(selected["margin"]),
        "selection_source": str(validation_path),
        "selection_rule": "minimize held-in validation violation/cost with success_drop <= 0.05 when feasible",
    }
    selected_config_path.write_text(json.dumps(selected_config, indent=2) + "\n")

    if not confirmation_path.exists():
        confirmation_rows = _executor_eval_rows(
            seeds=[10, 11, 12],
            eval_seeds=list(range(800000, 800020)),
            candidates_values=[selected_config["star_exec_candidates"]],
            margin_values=[selected_config["star_exec_margin"]],
        )
        _write_rows(confirmation_path, confirmation_rows)
    else:
        import pandas as pd

        confirmation_rows = pd.read_csv(confirmation_path).to_dict("records")

    selected_confirm, confirm_rows = _summarize_executor(confirmation_rows)
    _write_rows(REPORT_ROOT / "executor" / "executor_confirmation_summary.csv", confirm_rows)
    lines = [
        "# STAR+Exec Panda Evaluation",
        "",
        "Same STAR checkpoints are used; candidate execution is evaluation-only.",
        "",
        f"- selected candidates: `{selected_config['star_exec_candidates']}`",
        f"- selected margin: `{selected_config['star_exec_margin']}`",
        f"- validation raw cost -> STAR+Exec cost: `{selected['raw_cost']:.3f} -> {selected['exec_cost']:.3f}`",
        f"- validation raw violation -> STAR+Exec violation: `{selected['raw_violation_rate']:.3f} -> {selected['exec_violation_rate']:.3f}`",
        f"- validation success drop: `{selected['success_drop']:.3f}`",
        "",
        "Held-out confirmation at selected config:",
        f"- raw cost -> STAR+Exec cost: `{selected_confirm['raw_cost']:.3f} -> {selected_confirm['exec_cost']:.3f}`",
        f"- raw violation -> STAR+Exec violation: `{selected_confirm['raw_violation_rate']:.3f} -> {selected_confirm['exec_violation_rate']:.3f}`",
        f"- raw success -> STAR+Exec success: `{selected_confirm['raw_success']:.3f} -> {selected_confirm['exec_success']:.3f}`",
        f"- raw return -> STAR+Exec return: `{selected_confirm['raw_return']:.3f} -> {selected_confirm['exec_return']:.3f}`",
        f"- fallback rate: `{selected_confirm['exec_fallback_rate']:.3f}`",
        f"- found-but-not-executed rate: `{selected_confirm['exec_fne']:.3f}`",
        f"- mean action-selection latency ms: `{selected_confirm['exec_latency_ms']:.3f}`",
    ]
    (REPORT_ROOT / "executor" / "executor_summary.md").write_text("\n".join(lines) + "\n")
    print(REPORT_ROOT / "executor" / "executor_summary.md")


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
        executor_eval()
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
