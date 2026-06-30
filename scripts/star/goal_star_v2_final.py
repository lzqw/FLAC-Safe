#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO / "reports" / "star_v2_final"
LOG_ROOT = REPO / "logs" / "star_v2_final"
RESULT_ROOT = REPO / "results" / "star_v2_final"
STATUS_PATH = REPORT_ROOT / "scheduler_status.csv"
PYTHON = sys.executable

GPU_SLOTS = {0: 3, 1: 3}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
    "MUJOCO_GL": "egl",
    "STAR_STORAGE_ROOT": "/root/autodl-tmp/star_v2_storage",
    "TMPDIR": "/root/autodl-tmp/star_v2_storage/tmp",
    "TORCH_HOME": "/root/autodl-tmp/star_v2_storage/cache/torch",
    "MPLCONFIGDIR": "/root/autodl-tmp/star_v2_storage/cache/matplotlib",
    "XDG_CACHE_HOME": "/root/autodl-tmp/star_v2_storage/cache/xdg",
}


@dataclass(frozen=True)
class RunSpec:
    phase: str
    task: str
    method: str
    seed: int
    steps: int
    config_name: str
    device: int
    overrides: tuple[tuple[str, object], ...] = ()

    @property
    def run_name(self) -> str:
        safe_task = self.task.replace("-", "_")
        safe_config = self.config_name.replace("-", "_")
        return f"starv2_{self.phase}_{safe_task}_{safe_config}_{self.method}_s{self.seed}"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / self.phase / f"{self.run_name}.log"

    @property
    def result_dir(self) -> Path:
        # Resume-300k runs for seeds 10-12 intentionally continue in the
        # original core-100k run directory under the resume_300k root.  The
        # tmux/log run_name remains the resume spec name, so completion checks
        # must use resume_run_dir when present; otherwise the scheduler will
        # relaunch an already completed resumed run.
        override_map = dict(self.overrides)
        resume_run_dir = override_map.get("resume_run_dir")
        if resume_run_dir:
            return Path(str(resume_run_dir))
        return RESULT_ROOT / self.phase / self.task / self.method / self.run_name

    @property
    def final_checkpoint(self) -> Path:
        return self.result_dir / "checkpoint" / "final.torch"


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def ensure_dirs() -> None:
    for path in [
        REPORT_ROOT,
        REPORT_ROOT / "provenance",
        REPORT_ROOT / "configs",
        REPORT_ROOT / "tables",
        REPORT_ROOT / "figures",
        REPORT_ROOT / "latex",
        LOG_ROOT,
        RESULT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def git_sha() -> str:
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def load_json(path: str) -> dict:
    with (REPO / path).open() as handle:
        return json.load(handle)


def flatten_overrides(data: dict) -> tuple[tuple[str, object], ...]:
    ignored = {
        "schema",
        "algorithm_git_sha",
        "method",
        "methods",
        "shared",
        "selection_source",
        "selected_calibration_name",
    }
    return tuple((key, value) for key, value in data.items() if key not in ignored)


def write_status(rows: Iterable[dict]) -> None:
    rows = list(rows)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phase", "task", "method", "seed", "steps", "device", "run_name", "pid", "status", "log_path"]
    tmp = STATUS_PATH.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(STATUS_PATH)


def tmux_sessions() -> list[str]:
    proc = run(["tmux", "ls"])
    if proc.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in proc.stdout.splitlines() if line.strip()]


def build_main_star_command(spec: RunSpec, extra: dict | None = None) -> list[str]:
    actor = load_json("configs/star_v2_selected_actor.json")
    spec_overrides = dict(spec.overrides)
    params = {
        "task": spec.task,
        "method": spec.method,
        "seed": spec.seed,
        # When CUDA_VISIBLE_DEVICES is set per process, PyTorch sees the assigned
        # physical GPU as local cuda:0.
        "device": 0,
        "num_steps": spec.steps,
        "run_name": spec.run_name,
        "output_root": str(RESULT_ROOT / spec.phase),
        "ablation_group": spec.phase,
        "ablation_name": spec.config_name,
        "star_algorithm_version": "star_v2",
        "shadow_beta_mode": actor["shadow_beta_mode"],
        "shadow_num_strata": actor["shadow_num_strata"],
        "shadow_samples_per_stratum": actor["shadow_samples_per_stratum"],
        "shadow_temperature": actor["shadow_temperature"],
        "shadow_reference_mode": "current_only" if spec.method == "current_only_v2" else actor["shadow_reference_mode"],
        "star_shadow_penalty_mode": actor["star_shadow_penalty_mode"],
        "star_risk_threshold": actor["star_risk_threshold"],
        "star_lambda": actor["star_lambda"],
        "star_use_kl": actor["star_use_kl"],
        "star_kl_coef": actor["star_kl_coef"],
        "star_kl_target": actor["star_kl_target"],
        "star_ref_update_interval": actor["star_ref_update_interval"],
        "cost_gamma": actor["cost_gamma"],
        "cost_critic_reduce": actor["cost_critic_reduce"],
        "training_execution_mode": "raw",
        "eval": False,
        "save": True,
        "save_training_state": True,
        "save_interval_steps": 0,
        "disable_wandb": True,
        "online_eval_mode": "none",
    }
    if spec.phase.startswith("smoke"):
        params.update(
            {
                "start_steps": 200,
                "batch_size": 64,
                "hidden_size": 128,
                "metric_log_interval_steps": 500,
                "mechanism_log_interval_steps": 500,
                "audit_diagnostic_interval": 10,
            }
        )
    if spec.method == "pointwise_v2":
        params["shadow_reference_mode"] = "corridor"
    if spec.method == "sac_lag":
        params["star_use_kl"] = False
    params.update(spec_overrides)
    if extra:
        params.update(extra)
    cmd = [PYTHON, "main_star.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def smoke_specs() -> list[RunSpec]:
    tasks = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
    methods = ["pointwise_v2", "current_only_v2", "star_v2", "sac_lag"]
    specs = []
    for index, (task, method) in enumerate((t, m) for t in tasks for m in methods):
        device = 0 if index < 4 else 1
        specs.append(RunSpec("smoke5k_train", task, method, 0, 5000, "star_v2_smoke", device))
    return specs


def calibration_star_specs() -> list[RunSpec]:
    tasks = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
    seeds = [0, 1]
    configs = [
        ("starA_thr045_lam05", 0.45, 0.5),
        ("starB_thr050_lam05", 0.50, 0.5),
        ("starC_thr050_lam10", 0.50, 1.0),
        ("starD_thr055_lam10", 0.55, 1.0),
    ]
    specs: list[RunSpec] = []
    index = 0
    for config_name, threshold, lam in configs:
        for task in tasks:
            for seed in seeds:
                device = index % 2
                specs.append(
                    RunSpec(
                        "calibration_star100k",
                        task,
                        "star_v2",
                        seed,
                        100000,
                        config_name,
                        device,
                        overrides=(
                            ("star_risk_threshold", threshold),
                            ("star_lambda", lam),
                            ("shadow_beta_mode", "positive_linspace"),
                            ("shadow_reference_mode", "corridor"),
                            ("star_shadow_penalty_mode", "squared"),
                            ("cost_gamma", 0.95),
                            ("save_interval_steps", 0),
                            ("mechanism_log_interval_steps", 5000),
                            ("metric_log_interval_steps", 5000),
                            ("audit_diagnostic_interval", 100),
                        ),
                    )
                )
                index += 1
    return specs


def calibration_baseline_specs() -> list[RunSpec]:
    tasks = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
    specs: list[RunSpec] = []
    configs = [
        ("pointwise_lam05", "pointwise_v2", (("star_lambda", 0.5),)),
        ("pointwise_lam10", "pointwise_v2", (("star_lambda", 1.0),)),
        ("saclag_lr3e4", "sac_lag", (("lagrange_lr", 3e-4), ("star_use_kl", False))),
        ("saclag_lr1e3", "sac_lag", (("lagrange_lr", 1e-3), ("star_use_kl", False))),
    ]
    index = 0
    for config_name, method, extra in configs:
        for task in tasks:
            specs.append(
                RunSpec(
                    "calibration_baseline100k",
                    task,
                    method,
                    0,
                    100000,
                    config_name,
                    index % 2,
                    overrides=(
                        ("star_risk_threshold", 0.50),
                        ("star_lambda", 0.5),
                        ("star_shadow_penalty_mode", "squared"),
                        ("cost_gamma", 0.95),
                        ("save_interval_steps", 0),
                        ("mechanism_log_interval_steps", 5000),
                        ("metric_log_interval_steps", 5000),
                        ("audit_diagnostic_interval", 100),
                    )
                    + extra,
                )
            )
            index += 1
    return specs


def final_tasks() -> list[str]:
    full = load_json("configs/star_v2_selected_full.json")
    if full.get("selected_final_tasks"):
        return [str(task) for task in full["selected_final_tasks"]]
    return [str(task) for task in full.get("preferred_final_tasks", [])]


def baseline_method_overrides(method: str) -> tuple[tuple[str, object], ...]:
    data = load_json("configs/star_v2_selected_baselines.json")
    shared = dict(data.get("shared", {}))
    method_cfg = dict(data.get("methods", {}).get(method, {}))
    merged = {**shared, **method_cfg}
    return flatten_overrides(merged)


def actor_overrides() -> tuple[tuple[str, object], ...]:
    return flatten_overrides(load_json("configs/star_v2_selected_actor.json"))


def development_tasks() -> list[str]:
    full = load_json("configs/star_v2_selected_full.json")
    tasks = full.get("development_tasks") or final_tasks()[:2]
    return [str(task) for task in tasks]


def core_100k_specs() -> list[RunSpec]:
    tasks = final_tasks()
    seeds = [10, 11, 12]
    methods = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]
    specs: list[RunSpec] = []
    index = 0
    for task in tasks:
        for method in methods:
            for seed in seeds:
                if method == "star_v2":
                    overrides = actor_overrides()
                    config_name = "selected_star_v2"
                else:
                    overrides = baseline_method_overrides(method)
                    config_name = f"selected_{method}"
                specs.append(
                    RunSpec(
                        "core_100k",
                        task,
                        method,
                        seed,
                        100000,
                        config_name,
                        index % 2,
                        overrides=overrides
                        + (
                            ("save_interval_steps", 0),
                            ("mechanism_log_interval_steps", 5000),
                            ("metric_log_interval_steps", 5000),
                            ("audit_diagnostic_interval", 100),
                        ),
                    )
                )
                index += 1
    return specs


def resume_300k_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    # Seeds 10-12 continue from the core-100k checkpoints. Seeds 13-14 are
    # independent fresh 300k runs, matching the final STAR-v2 protocol.
    for core in core_100k_specs():
        checkpoint = core.final_checkpoint
        specs.append(
            RunSpec(
                "resume_300k",
                core.task,
                core.method,
                core.seed,
                300000,
                core.config_name,
                core.device,
                overrides=core.overrides
                + (
                    ("resume_checkpoint", str(checkpoint)),
                    ("resume_run_dir", str(RESULT_ROOT / "resume_300k" / core.task / core.method / core.run_name)),
                    ("save_training_state", True),
                    ("save_interval_steps", 0),
                ),
            )
        )
    tasks = final_tasks()
    methods = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]
    index = len(specs)
    for task in tasks:
        for method in methods:
            for seed in [13, 14]:
                if method == "star_v2":
                    overrides = actor_overrides()
                    config_name = "selected_star_v2"
                else:
                    overrides = baseline_method_overrides(method)
                    config_name = f"selected_{method}"
                specs.append(
                    RunSpec(
                        "resume_300k",
                        task,
                        method,
                        seed,
                        300000,
                        config_name,
                        index % 2,
                        overrides=overrides
                        + (
                            ("save_training_state", True),
                            ("save_interval_steps", 0),
                            ("mechanism_log_interval_steps", 5000),
                            ("metric_log_interval_steps", 5000),
                            ("audit_diagnostic_interval", 100),
                        ),
                    )
                )
                index += 1
    return specs


def ablation_specs() -> list[RunSpec]:
    """Compact 2x2 STAR-v2 design ablation.

    The final protocol compares the endpoint-grid fix and squared-penalty fix
    only, on development tasks with seeds 20/21/22.
    """

    tasks = development_tasks()
    seeds = [20, 21, 22]
    configs: list[tuple[str, tuple[tuple[str, object], ...]]] = [
        (
            "original_legacy_linear",
            (("shadow_beta_mode", "legacy_endpoints"), ("star_shadow_penalty_mode", "linear")),
        ),
        (
            "endpoint_fix_positive_linear",
            (("shadow_beta_mode", "positive_linspace"), ("star_shadow_penalty_mode", "linear")),
        ),
        (
            "penalty_fix_legacy_squared",
            (("shadow_beta_mode", "legacy_endpoints"), ("star_shadow_penalty_mode", "squared")),
        ),
        (
            "star_v2_positive_squared",
            (("shadow_beta_mode", "positive_linspace"), ("star_shadow_penalty_mode", "squared")),
        ),
    ]
    specs: list[RunSpec] = []
    index = 0
    for task in tasks:
        for seed in seeds:
            for config_name, overrides in configs:
                method = str(dict(overrides).get("method", "star_v2"))
                merged = actor_overrides() + overrides
                specs.append(
                    RunSpec(
                        "ablation_100k",
                        task,
                        method,
                        seed,
                        100000,
                        config_name,
                        index % 2,
                        overrides=merged
                        + (
                            ("ablation_group", "ablation_100k"),
                            ("ablation_name", config_name),
                            ("save_interval_steps", 0),
                            ("mechanism_log_interval_steps", 5000),
                            ("metric_log_interval_steps", 5000),
                            ("audit_diagnostic_interval", 100),
                        ),
                    )
                )
                index += 1
    return specs


def doctor(args: argparse.Namespace) -> int:
    ensure_dirs()
    print(f"repo={REPO}")
    print(f"git_sha={git_sha()}")
    print(f"branch={run(['git', 'branch', '--show-current']).stdout.strip()}")
    print(run(["git", "status", "--short"]).stdout.strip() or "git_status=clean")
    print(run([PYTHON, "--version"]).stdout.strip())
    print(run([PYTHON, "-m", "py_compile", "agents/shadow_audit.py", "agents/star_agent.py", "main_star.py", "utilis/star_default_config.py"]).stdout.strip())
    print(run(["nvidia-smi", "-L"]).stdout.strip())
    for path in [
        "configs/star_v2_selected_actor.json",
        "configs/star_v2_selected_baselines.json",
        "configs/star_v2_selected_executor.json",
        "configs/star_v2_selected_full.json",
    ]:
        data = load_json(path)
        print(f"config={path} schema={data.get('schema')}")
    return 0


def status(args: argparse.Namespace) -> int:
    ensure_dirs()
    sessions = set(tmux_sessions())
    rows = []
    for phase_dir in sorted(LOG_ROOT.glob("*")):
        if not phase_dir.is_dir():
            continue
        for log_path in sorted(phase_dir.glob("*.log")):
            run_name = log_path.stem
            rows.append(
                {
                    "phase": phase_dir.name,
                    "run_name": run_name,
                    "status": "running" if run_name in sessions else "log_only",
                    "log_path": str(log_path),
                }
            )
    write_status(rows)
    print(f"tmux_sessions={len(sessions)}")
    for session in sorted(sessions):
        if session.startswith("starv2_"):
            print(f"active={session}")
    print(f"status_csv={STATUS_PATH}")
    return 0


def launch_specs(specs: list[RunSpec], *, dry_run: bool, max_submit: int) -> int:
    ensure_dirs()
    rows = []
    sessions = set(tmux_sessions())
    # Count only sessions for the run specs managed by this invocation against
    # the experiment parallelism. Supervisor, evaluator, and recovery tmux
    # sessions may also use the starv2_ prefix, but they are not training runs
    # from this specs list and should not permanently consume training slots.
    # Stale tmux sessions for already completed runs should not consume
    # training slots.  This matters for resume_300k because completed sessions
    # can remain attached after final.torch is written.
    incomplete_run_names = {spec.run_name for spec in specs if not spec.final_checkpoint.exists()}
    starv2_running = [name for name in sessions if name in incomplete_run_names]
    remaining_slots = max(0, int(max_submit) - len(starv2_running))
    submitted = 0
    for spec in specs:
        cmd = build_main_star_command(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        command_text = " ".join(shlex.quote(part) for part in cmd)
        already_running = spec.run_name in sessions
        completed = spec.final_checkpoint.exists()
        can_submit = dry_run or completed or already_running or submitted < remaining_slots
        status_value = (
            "completed"
            if completed
            else "planned"
            if dry_run or not can_submit
            else ("running" if already_running else "submitted")
        )
        rows.append(
            {
                "phase": spec.phase,
                "task": spec.task,
                "method": spec.method,
                "seed": spec.seed,
                "steps": spec.steps,
                "device": spec.device,
                "run_name": spec.run_name,
                "status": status_value,
                "log_path": str(spec.log_path),
            }
        )
        print(f"{spec.run_name}: CUDA_VISIBLE_DEVICES={spec.device} {command_text}")
        if dry_run:
            continue
        if completed:
            print(f"skip_completed={spec.run_name}")
            continue
        if not can_submit:
            print(f"defer_no_slot={spec.run_name}")
            continue
        if spec.run_name in sessions:
            print(f"skip_existing_session={spec.run_name}")
            continue
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in THREAD_ENV.items())
        tmux_cmd = (
            f"cd {shlex.quote(str(REPO))} && "
            f"source /root/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
            f"export CUDA_VISIBLE_DEVICES={spec.device} {env_prefix} && "
            f"{command_text} > {shlex.quote(str(spec.log_path))} 2>&1"
        )
        subprocess.run(["tmux", "new", "-d", "-s", spec.run_name, tmux_cmd], cwd=REPO, check=True)
        submitted += 1
    write_status(rows)
    return 0


def smoke(args: argparse.Namespace) -> int:
    return launch_specs(smoke_specs(), dry_run=bool(args.dry_run), max_submit=int(args.max_parallel))


def calibrate(args: argparse.Namespace) -> int:
    grid = str(args.grid)
    specs: list[RunSpec] = []
    if grid in ("star", "all"):
        specs.extend(calibration_star_specs())
    if grid in ("baseline", "all"):
        specs.extend(calibration_baseline_specs())
    if grid not in ("star", "baseline", "all"):
        raise ValueError(f"unknown calibration grid: {grid}")
    return launch_specs(specs, dry_run=bool(args.dry_run), max_submit=int(args.max_parallel))


def core_100k(args: argparse.Namespace) -> int:
    return launch_specs(core_100k_specs(), dry_run=bool(args.dry_run), max_submit=int(args.max_parallel))


def plan_core_100k(args: argparse.Namespace) -> int:
    ensure_dirs()
    specs = core_100k_specs()
    output = Path(getattr(args, "output", REPORT_ROOT / "recovery" / "core100k_resume_plan.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    sessions = set(tmux_sessions())
    fields = [
        "phase", "task", "method", "seed", "run_dir", "status",
        "checkpoint_exists", "eval_exists", "log_exists", "has_error", "recommended_action",
    ]
    rows = []
    for spec in specs:
        log_exists = spec.log_path.exists()
        has_error = False
        if log_exists:
            text = spec.log_path.read_text(errors="ignore")
            error_terms = [
                "Traceback", "RuntimeError", "CUDA out of memory", "out of memory",
                "NaN", "nan", "Segmentation fault", "MuJoCo", "Exception", "Error",
                "No space left",
            ]
            has_error = any(term in text for term in error_terms)
        eval_path = spec.result_dir / "corrected_eval_episodes.csv"
        checkpoint_exists = spec.final_checkpoint.exists()
        eval_exists = eval_path.exists()
        if spec.run_name in sessions:
            status_value = "RUNNING"
            action = "skip"
        elif checkpoint_exists and eval_exists and not has_error:
            status_value = "COMPLETED_TRAINING_AND_EVAL"
            action = "skip"
        elif checkpoint_exists and not has_error:
            status_value = "COMPLETED_TRAINING_NEEDS_EVAL"
            action = "eval_only"
        elif has_error:
            status_value = "FAILED"
            action = "manual_investigation"
        else:
            status_value = "PENDING"
            action = "relaunch_training"
        rows.append({
            "phase": spec.phase,
            "task": spec.task,
            "method": spec.method,
            "seed": spec.seed,
            "run_dir": str(spec.result_dir),
            "status": status_value,
            "checkpoint_exists": checkpoint_exists,
            "eval_exists": eval_exists,
            "log_exists": log_exists,
            "has_error": has_error,
            "recommended_action": action,
        })
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)
    print(f"plan_rows={len(rows)} output={output}")
    return 0


def _spec_plan_rows(specs: list[RunSpec], sessions: set[str]) -> list[dict]:
    rows: list[dict] = []
    for spec in specs:
        log_exists = spec.log_path.exists()
        has_error = False
        if log_exists:
            log_text = spec.log_path.read_text(errors="ignore")
            error_terms = [
                "Traceback",
                "RuntimeError",
                "CUDA out of memory",
                "out of memory",
                "Segmentation fault",
                "No space left",
                "unrecognized arguments",
            ]
            has_error = any(term in log_text for term in error_terms)
        checkpoint_exists = spec.final_checkpoint.exists()
        eval_exists = (spec.result_dir / "corrected_eval_episodes.csv").exists()
        if spec.run_name in sessions:
            status_value = "RUNNING"
            action = "skip"
        elif checkpoint_exists and eval_exists and not has_error:
            status_value = "COMPLETED_TRAINING_AND_EVAL"
            action = "skip"
        elif checkpoint_exists and not has_error:
            status_value = "COMPLETED_TRAINING_NEEDS_EVAL"
            action = "eval_only"
        elif has_error:
            status_value = "FAILED"
            action = "manual_investigation"
        else:
            status_value = "PENDING"
            action = "resume_training"
        rows.append(
            {
                "phase": spec.phase,
                "task": spec.task,
                "method": spec.method,
                "seed": spec.seed,
                "run_dir": str(spec.result_dir),
                "status": status_value,
                "checkpoint_exists": checkpoint_exists,
                "eval_exists": eval_exists,
                "log_exists": log_exists,
                "has_error": has_error,
                "recommended_action": action,
            }
        )
    return rows


def _artifact_row(phase: str, path: Path, *, required: bool = True) -> dict:
    exists = path.exists()
    return {
        "phase": phase,
        "task": "",
        "method": "",
        "seed": "",
        "run_dir": str(path),
        "status": "COMPLETED_TRAINING" if exists else ("PENDING" if required else "NOT_YET_REACHED"),
        "checkpoint_exists": "",
        "eval_exists": exists,
        "log_exists": "",
        "has_error": False,
        "recommended_action": "skip" if exists else "run_phase",
    }


def plan_all(args: argparse.Namespace) -> int:
    ensure_dirs()
    output = Path(getattr(args, "output", REPORT_ROOT / "recovery" / "full_resume_plan.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    sessions = set(tmux_sessions())
    fields = [
        "phase",
        "task",
        "method",
        "seed",
        "run_dir",
        "status",
        "checkpoint_exists",
        "eval_exists",
        "log_exists",
        "has_error",
        "recommended_action",
    ]
    rows: list[dict] = []
    rows.extend(_spec_plan_rows(core_100k_specs(), sessions))
    rows.append(_artifact_row("core_gate", REPORT_ROOT / "core_100k" / "gate.md"))
    rows.extend(_spec_plan_rows(resume_300k_specs(), sessions))
    rows.append(_artifact_row("final_collect", REPORT_ROOT / "main_results_summary.csv"))
    for rel in [
        "mechanism/corridor_mechanism.csv",
        "mechanism/reference_age_summary.csv",
        "mechanism/mechanism_summary.md",
        "oracle/oracle_status.md",
    ]:
        rows.append(_artifact_row(rel.split("/", 1)[0], REPORT_ROOT / rel))
    rows.extend(_spec_plan_rows(ablation_specs(), sessions))
    for rel in [
        "ablation/ablation_by_seed.csv",
        "ablation/ablation_summary.csv",
        "ablation/ablation_summary.md",
        "executor/executor_grid_validation.csv",
        "executor/executor_confirmation.csv",
        "executor/selection.md",
        "executor/confirmation.md",
        "latex/table_main_results.tex",
        "latex/table_ablation_executor.tex",
        "figures/fig_mechanism_validation.pdf",
        "figures/fig_mechanism_validation.png",
        "figures/fig_mechanism_validation.svg",
        "figures/fig_training_curves.pdf",
        "figures/fig_training_curves.png",
        "figures/fig_training_curves.svg",
        "figures/captions.md",
        "latex/result_macros.tex",
        "final_claims.md",
        "final_audit.md",
        "README.md",
    ]:
        rows.append(_artifact_row("paper_artifact", REPORT_ROOT / rel))
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["status"])
        counts[key] = counts.get(key, 0) + 1
    print(f"plan_rows={len(rows)} output={output} status_counts={counts}")
    return 0


def resume_300k(args: argparse.Namespace) -> int:
    return launch_specs(resume_300k_specs(), dry_run=bool(args.dry_run), max_submit=int(args.max_parallel))


def oracle(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        PYTHON,
        "scripts/star/run_shadow_oracle.py",
        "--root",
        str(args.root),
        "--report-dir",
        str(REPORT_ROOT),
        "--eval-seeds",
        str(args.eval_seeds),
        "--horizons",
        str(args.horizons),
        "--max-states-per-run",
        str(args.max_states_per_run),
        "--methods",
        str(args.methods),
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=REPO)
    status_path = REPORT_ROOT / "oracle" / "oracle_status.md"
    rows = []
    summary_path = REPORT_ROOT / "oracle" / "oracle_summary.csv"
    if summary_path.exists():
        rows.append(f"- summary: `{summary_path}`")
    rows.append(f"- root: `{args.root}`")
    rows.append(f"- methods: `{args.methods}`")
    rows.append(f"- eval_seeds: `{args.eval_seeds}`")
    rows.append(f"- horizons: `{args.horizons}`")
    rows.append(f"- exit_code: `{proc.returncode}`")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("# STAR-v2 Oracle Status\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return int(proc.returncode)


def collect(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        PYTHON,
        "scripts/star/collect_star_v2_results.py",
        "--root",
        str(RESULT_ROOT),
        "--log-root",
        str(LOG_ROOT),
        "--report-dir",
        str(REPORT_ROOT),
        "--phase",
        str(args.phase),
    ]
    if args.strict:
        cmd.append("--strict")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=REPO).returncode)


def gate_core_100k(args: argparse.Namespace) -> int:
    """Write the non-blocking core-100k gate report.

    The storage-override protocol treats this as a reporting checkpoint only;
    downstream phases must continue unless there is a real technical failure.
    """

    ensure_dirs()
    specs = core_100k_specs()
    out_dir = REPORT_ROOT / "core_100k"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = set(tmux_sessions())
    missing_rows = []
    completed = 0
    eval_ready = 0
    running = 0
    failed = 0
    for spec in specs:
        checkpoint_exists = spec.final_checkpoint.exists()
        eval_exists = (spec.result_dir / "corrected_eval_episodes.csv").exists()
        log_exists = spec.log_path.exists()
        has_error = False
        if log_exists:
            text = spec.log_path.read_text(errors="ignore")
            error_terms = [
                "Traceback",
                "RuntimeError",
                "CUDA out of memory",
                "out of memory",
                "NaN",
                "nan",
                "Segmentation fault",
                "MuJoCo",
                "No space left",
                "unrecognized arguments",
            ]
            has_error = any(term in text for term in error_terms)
        if checkpoint_exists:
            completed += 1
        if checkpoint_exists and eval_exists and not has_error:
            eval_ready += 1
        if spec.run_name in sessions:
            running += 1
        if has_error:
            failed += 1
        if not (checkpoint_exists and eval_exists and not has_error):
            missing_rows.append(
                {
                    "task": spec.task,
                    "method": spec.method,
                    "seed": spec.seed,
                    "checkpoint": checkpoint_exists,
                    "eval": eval_exists,
                    "running": spec.run_name in sessions,
                    "has_error": has_error,
                    "run_name": spec.run_name,
                }
            )
    expected = len(specs)
    if failed:
        label = "TECHNICAL_ISSUES"
    elif eval_ready == expected:
        label = "PASS"
    elif completed == expected:
        label = "TRAINING_COMPLETE_EVAL_PENDING"
    else:
        label = "INCOMPLETE"
    gate_path = out_dir / "gate.md"
    gate_path.write_text(
        "\n".join(
            [
                "# STAR-v2 Core-100k Gate",
                "",
                f"- label: `{label}`",
                f"- expected_runs: `{expected}`",
                f"- completed_training: `{completed}`",
                f"- completed_eval: `{eval_ready}`",
                f"- running: `{running}`",
                f"- failed_or_error_logs: `{failed}`",
                "",
                "This gate is a reporting checkpoint only under the storage override.",
                "The pipeline should continue unless there is a fatal technical failure.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    missing_path = out_dir / "missing_results.md"
    lines = ["# STAR-v2 Core-100k Missing Results", ""]
    if not missing_rows:
        lines.append("No missing core-100k results.")
    else:
        lines.append("| Task | Method | Seed | Checkpoint | Eval | Running | Error | Run |")
        lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
        for row in missing_rows:
            lines.append(
                "| {task} | {method} | {seed} | {checkpoint} | {eval} | {running} | {has_error} | `{run_name}` |".format(
                    **row
                )
            )
    missing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"gate_label={label} gate={gate_path} missing={missing_path}")
    return 0


def eval_core(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        PYTHON,
        "scripts/star/reevaluate_checkpoints.py",
        "--root",
        str(RESULT_ROOT / "core_100k"),
        "--eval-seeds",
        str(args.eval_seeds),
        "--checkpoint-selector",
        "final",
        "--modes",
        "raw",
    ]
    if args.overwrite_derived:
        cmd.append("--overwrite-derived")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=REPO).returncode)


def eval_final_300k(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        PYTHON,
        "scripts/star/reevaluate_checkpoints.py",
        "--root",
        str(RESULT_ROOT / "resume_300k"),
        "--eval-seeds",
        str(args.eval_seeds),
        "--checkpoint-selector",
        "final",
        "--modes",
        "raw",
    ]
    if args.overwrite_derived:
        cmd.append("--overwrite-derived")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=REPO).returncode)


def ablation(args: argparse.Namespace) -> int:
    return launch_specs(ablation_specs(), dry_run=bool(args.dry_run), max_submit=int(args.max_parallel))


def mechanism(args: argparse.Namespace) -> int:
    """Summarize paired corridor-vs-current-only diagnostics from mechanism.csv.

    Training already records the paired audit statistics with shared Gaussian
    base noise. This command turns those per-run traces into the final report
    artifacts required by the STAR-v2 pipeline.
    """

    ensure_dirs()
    root = Path(args.root)
    out_dir = REPORT_ROOT / "mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    age_rows: list[dict] = []
    for mech_path in sorted(root.glob("*/*/*/mechanism.csv")):
        rel = mech_path.relative_to(root)
        task, method, run_name = rel.parts[:3]
        with mech_path.open(newline="", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            mech_rows = list(reader)
        if not mech_rows:
            continue
        for row in mech_rows:
            lift = row.get("paired_corridor_risk_lift", row.get("star/paired_corridor_risk_lift", ""))
            try:
                lift_value = float(lift)
            except (TypeError, ValueError):
                lift_value = 0.0
            out = {
                "task": task,
                "method": method,
                "run_name": run_name,
                "step": row.get("step", ""),
                "rho_corridor": row.get("paired_corridor_risk", row.get("star/paired_corridor_risk", "")),
                "rho_current": row.get("paired_current_risk", row.get("star/paired_current_risk", "")),
                "corridor_risk_lift": lift,
                "lift_positive_rate": 1.0 if lift_value > 0 else 0.0,
                "effective_beta": row.get("effective_beta", ""),
                "shadow_excess": row.get("shadow_excess_mean", row.get("star/shadow_excess_mean", "")),
                "reference_age": row.get("reference_age", row.get("star/reference_age", "")),
            }
            rows.append(out)
        final = mech_rows[-1]
        age_rows.append(
            {
                "task": task,
                "method": method,
                "run_name": run_name,
                "final_step": final.get("step", ""),
                "reference_age": final.get("reference_age", final.get("star/reference_age", "")),
                "reference_age_pre_update": final.get(
                    "reference_age_pre_update", final.get("star/reference_age_pre_update", "")
                ),
                "reference_age_post_update": final.get(
                    "reference_age_post_update", final.get("star/reference_age_post_update", "")
                ),
            }
        )
    fields = [
        "task",
        "method",
        "run_name",
        "step",
        "rho_corridor",
        "rho_current",
        "corridor_risk_lift",
        "lift_positive_rate",
        "effective_beta",
        "shadow_excess",
        "reference_age",
    ]
    with (out_dir / "corridor_mechanism.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    age_fields = [
        "task",
        "method",
        "run_name",
        "final_step",
        "reference_age",
        "reference_age_pre_update",
        "reference_age_post_update",
    ]
    with (out_dir / "reference_age_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=age_fields)
        writer.writeheader()
        writer.writerows(age_rows)
    positive = [float(row["lift_positive_rate"]) for row in rows]
    lift_values = []
    for row in rows:
        try:
            lift_values.append(float(row["corridor_risk_lift"]))
        except (TypeError, ValueError):
            pass
    mean_lift = sum(lift_values) / len(lift_values) if lift_values else 0.0
    mean_positive = sum(positive) / len(positive) if positive else 0.0
    (out_dir / "mechanism_summary.md").write_text(
        "\n".join(
            [
                "# STAR-v2 Mechanism Summary",
                "",
                f"- root: `{root}`",
                f"- mechanism_rows: `{len(rows)}`",
                f"- runs_with_mechanism: `{len(age_rows)}`",
                f"- mean_corridor_risk_lift: `{mean_lift:.6g}`",
                f"- lift_positive_rate: `{mean_positive:.6g}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"mechanism_rows={len(rows)} out_dir={out_dir}")
    return 0


def executor(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        PYTHON,
        "scripts/star/evaluate_executor_grid.py",
        "--root",
        str(args.root),
        "--eval-seeds",
        str(args.eval_seeds),
        "--candidates",
        str(args.candidates),
        "--margins",
        str(args.margins),
        "--methods",
        str(args.methods),
        "--report-dir",
        str(REPORT_ROOT / "executor"),
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=REPO).returncode)


def paper(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        PYTHON,
        "scripts/star/build_star_v2_paper_artifacts.py",
        "--report-dir",
        str(REPORT_ROOT),
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=REPO).returncode)


def not_yet(command: str) -> int:
    ensure_dirs()
    print(f"{command}: not implemented yet in this scaffold. Complete prior gates before using this phase.")
    return 2


def add_storage_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage-policy",
        choices=["unblocked", "warn-only", "default"],
        default="unblocked",
        help="Compatibility option; unblocked/warn-only storage policy never stops scheduling on soft free-space thresholds.",
    )
    parser.add_argument(
        "--ignore-storage-gate",
        action="store_true",
        help="Compatibility option; bypasses legacy free-space gates. Actual write failures still surface normally.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STAR-v2 final experiment orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--dry-run", action="store_true")
    smoke_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    add_storage_policy_args(smoke_parser)
    calibrate_parser = sub.add_parser("calibrate")
    calibrate_parser.add_argument("--grid", choices=["star", "baseline", "all"], default="all")
    calibrate_parser.add_argument("--dry-run", action="store_true")
    calibrate_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    add_storage_policy_args(calibrate_parser)
    core_parser = sub.add_parser("core-100k")
    core_parser.add_argument("--dry-run", action="store_true")
    core_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    core_parser.add_argument("--resume", action="store_true", help="compatibility no-op; completed runs are always skipped")
    add_storage_policy_args(core_parser)
    plan_core_parser = sub.add_parser("plan-core-100k")
    plan_core_parser.add_argument("--output", type=Path, default=REPORT_ROOT / "recovery" / "core100k_resume_plan.csv")
    add_storage_policy_args(plan_core_parser)
    plan_all_parser = sub.add_parser("plan-all")
    plan_all_parser.add_argument("--output", type=Path, default=REPORT_ROOT / "recovery" / "full_resume_plan.csv")
    add_storage_policy_args(plan_all_parser)
    resume_parser = sub.add_parser("resume-300k")
    resume_parser.add_argument("--dry-run", action="store_true")
    resume_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    resume_parser.add_argument("--resume", action="store_true", help="compatibility no-op; completed runs are always skipped")
    add_storage_policy_args(resume_parser)
    final_parser = sub.add_parser("final-300k")
    final_parser.add_argument("--dry-run", action="store_true")
    final_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    final_parser.add_argument("--resume", action="store_true", help="compatibility no-op; completed runs are always skipped")
    add_storage_policy_args(final_parser)
    status_parser = sub.add_parser("status")
    add_storage_policy_args(status_parser)
    oracle_parser = sub.add_parser("oracle")
    oracle_parser.add_argument("--root", type=Path, default=RESULT_ROOT / "resume_300k")
    oracle_parser.add_argument("--eval-seeds", default="900000,900001,900002,900003,900004")
    oracle_parser.add_argument("--horizons", default="1,5")
    oracle_parser.add_argument("--max-states-per-run", type=int, default=200)
    oracle_parser.add_argument("--methods", default="star_v2")
    oracle_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(oracle_parser)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--phase", choices=["core_100k", "resume_300k", "ablation_100k", "all"], default="all")
    collect_parser.add_argument("--strict", action="store_true")
    collect_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(collect_parser)
    eval_core_parser = sub.add_parser("eval-core")
    eval_core_parser.add_argument(
        "--eval-seeds",
        default="500000,500001,500002,500003,500004,500005,500006,500007,500008,500009",
    )
    eval_core_parser.add_argument("--overwrite-derived", action="store_true")
    eval_core_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(eval_core_parser)
    eval_core_alias_parser = sub.add_parser("eval-core-100k")
    eval_core_alias_parser.add_argument(
        "--eval-seeds",
        default="500000,500001,500002,500003,500004,500005,500006,500007,500008,500009",
    )
    eval_core_alias_parser.add_argument("--overwrite-derived", action="store_true")
    eval_core_alias_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(eval_core_alias_parser)
    eval_final_parser = sub.add_parser("eval-final-300k")
    eval_final_parser.add_argument(
        "--eval-seeds",
        default="600000,600001,600002,600003,600004,600005,600006,600007,600008,600009,600010,600011,600012,600013,600014,600015,600016,600017,600018,600019",
    )
    eval_final_parser.add_argument("--overwrite-derived", action="store_true")
    eval_final_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(eval_final_parser)
    gate_parser = sub.add_parser("gate-core-100k")
    add_storage_policy_args(gate_parser)
    ablation_parser = sub.add_parser("ablation")
    ablation_parser.add_argument("--dry-run", action="store_true")
    ablation_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    add_storage_policy_args(ablation_parser)
    mechanism_parser = sub.add_parser("mechanism")
    mechanism_parser.add_argument("--root", type=Path, default=RESULT_ROOT / "resume_300k")
    add_storage_policy_args(mechanism_parser)
    executor_parser = sub.add_parser("executor")
    executor_parser.add_argument("--root", type=Path, default=RESULT_ROOT / "resume_300k")
    executor_parser.add_argument("--eval-seeds", default="700000,700001,700002,700003,700004,700005,700006,700007,700008,700009")
    executor_parser.add_argument("--candidates", default="8,16")
    executor_parser.add_argument("--margins", default="0.00,0.02,0.05")
    executor_parser.add_argument("--methods", default="star_v2,star_collect_v2")
    executor_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(executor_parser)
    paper_parser = sub.add_parser("paper")
    paper_parser.add_argument("--dry-run", action="store_true")
    add_storage_policy_args(paper_parser)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        ignored = " ".join(unknown)
        print(f"warning: ignoring compatibility arguments: {ignored}")
    if args.command == "doctor":
        return doctor(args)
    if args.command == "status":
        return status(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "core-100k":
        return core_100k(args)
    if args.command == "plan-core-100k":
        return plan_core_100k(args)
    if args.command == "plan-all":
        return plan_all(args)
    if args.command in ("resume-300k", "final-300k"):
        return resume_300k(args)
    if args.command == "oracle":
        return oracle(args)
    if args.command == "collect":
        return collect(args)
    if args.command in ("eval-core", "eval-core-100k"):
        return eval_core(args)
    if args.command == "eval-final-300k":
        return eval_final_300k(args)
    if args.command == "gate-core-100k":
        return gate_core_100k(args)
    if args.command == "ablation":
        return ablation(args)
    if args.command == "mechanism":
        return mechanism(args)
    if args.command == "executor":
        return executor(args)
    if args.command == "paper":
        return paper(args)
    return not_yet(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
