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

GPU_SLOTS = {0: 3, 1: 3}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "WANDB_MODE": "offline",
    "MUJOCO_GL": "egl",
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
    ignored = {"schema", "algorithm_git_sha", "method", "methods", "shared"}
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
        "save_interval_steps": 50000 if spec.steps >= 50000 else spec.steps,
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
    cmd = ["python", "main_star.py"]
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
                            ("save_interval_steps", 50000),
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
                        ("save_interval_steps", 50000),
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
                            ("save_interval_steps", 50000),
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
                    ("save_interval_steps", 50000),
                ),
            )
        )
    return specs


def doctor(args: argparse.Namespace) -> int:
    ensure_dirs()
    print(f"repo={REPO}")
    print(f"git_sha={git_sha()}")
    print(f"branch={run(['git', 'branch', '--show-current']).stdout.strip()}")
    print(run(["git", "status", "--short"]).stdout.strip() or "git_status=clean")
    print(run(["python", "--version"]).stdout.strip())
    print(run(["python", "-m", "py_compile", "agents/shadow_audit.py", "agents/star_agent.py", "main_star.py", "utilis/star_default_config.py"]).stdout.strip())
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
    starv2_running = [name for name in sessions if name.startswith("starv2_")]
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


def resume_300k(args: argparse.Namespace) -> int:
    return launch_specs(resume_300k_specs(), dry_run=bool(args.dry_run), max_submit=int(args.max_parallel))


def oracle(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        "python",
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
        "python",
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


def eval_core(args: argparse.Namespace) -> int:
    ensure_dirs()
    cmd = [
        "python",
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


def not_yet(command: str) -> int:
    ensure_dirs()
    print(f"{command}: not implemented yet in this scaffold. Complete prior gates before using this phase.")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STAR-v2 final experiment orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--dry-run", action="store_true")
    smoke_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    calibrate_parser = sub.add_parser("calibrate")
    calibrate_parser.add_argument("--grid", choices=["star", "baseline", "all"], default="all")
    calibrate_parser.add_argument("--dry-run", action="store_true")
    calibrate_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    core_parser = sub.add_parser("core-100k")
    core_parser.add_argument("--dry-run", action="store_true")
    core_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    resume_parser = sub.add_parser("resume-300k")
    resume_parser.add_argument("--dry-run", action="store_true")
    resume_parser.add_argument("--max-parallel", type=int, default=sum(GPU_SLOTS.values()))
    sub.add_parser("status")
    oracle_parser = sub.add_parser("oracle")
    oracle_parser.add_argument("--root", type=Path, default=RESULT_ROOT / "resume_300k")
    oracle_parser.add_argument("--eval-seeds", default="900000,900001,900002,900003,900004")
    oracle_parser.add_argument("--horizons", default="1,5")
    oracle_parser.add_argument("--max-states-per-run", type=int, default=200)
    oracle_parser.add_argument("--methods", default="star_v2")
    oracle_parser.add_argument("--dry-run", action="store_true")
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--phase", choices=["core_100k", "resume_300k", "all"], default="all")
    collect_parser.add_argument("--strict", action="store_true")
    collect_parser.add_argument("--dry-run", action="store_true")
    eval_core_parser = sub.add_parser("eval-core")
    eval_core_parser.add_argument(
        "--eval-seeds",
        default="500000,500001,500002,500003,500004,500005,500006,500007,500008,500009",
    )
    eval_core_parser.add_argument("--overwrite-derived", action="store_true")
    eval_core_parser.add_argument("--dry-run", action="store_true")
    for name in ["ablation", "executor", "paper"]:
        sub.add_parser(name)
    args = parser.parse_args(argv)
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
    if args.command == "resume-300k":
        return resume_300k(args)
    if args.command == "oracle":
        return oracle(args)
    if args.command == "collect":
        return collect(args)
    if args.command == "eval-core":
        return eval_core(args)
    return not_yet(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
