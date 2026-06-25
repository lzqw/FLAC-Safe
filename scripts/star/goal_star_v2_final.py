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

    @property
    def run_name(self) -> str:
        safe_task = self.task.replace("-", "_")
        return f"starv2_{self.phase}_{safe_task}_{self.method}_s{self.seed}"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / self.phase / f"{self.run_name}.log"


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
        "save_interval_steps": 50000 if spec.steps >= 50000 else spec.steps,
        "disable_wandb": True,
    }
    if spec.method == "pointwise_v2":
        params["shadow_reference_mode"] = "corridor"
    if spec.method == "sac_lag":
        params["star_use_kl"] = False
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
        specs.append(RunSpec("smoke5k", task, method, 0, 5000, "star_v2_smoke", device))
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
        can_submit = dry_run or already_running or submitted < remaining_slots
        status_value = "planned" if dry_run or not can_submit else ("running" if already_running else "submitted")
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
    sub.add_parser("status")
    for name in ["calibrate", "core-100k", "resume-300k", "oracle", "ablation", "executor", "collect", "paper"]:
        sub.add_parser(name)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args)
    if args.command == "status":
        return status(args)
    if args.command == "smoke":
        return smoke(args)
    return not_yet(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
