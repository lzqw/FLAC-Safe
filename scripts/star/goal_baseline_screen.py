#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPORT = Path("reports/star_goal")
LOG_ROOT = Path("logs/star_goal/baseline_screen")
TASKS = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
SEED = 10
STEPS = 100000
MAX_TOTAL_RUNS = 6
RUNS_PER_GPU = 3
GPU_IDS = [0, 1]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    method: str
    extra: tuple[str, ...]


SPECS = [
    BaselineSpec("pointwise_lambda050", "pointwise", ("--star_lambda", "0.5")),
    BaselineSpec("pointwise_lambda100", "pointwise", ("--star_lambda", "1.0")),
    BaselineSpec("pointwise_lambda200", "pointwise", ("--star_lambda", "2.0")),
    BaselineSpec("sac_lag_lr0001", "sac_lag", ("--lagrange_lr", "0.0001")),
    BaselineSpec("sac_lag_lr0003", "sac_lag", ("--lagrange_lr", "0.0003")),
    BaselineSpec("sac_lag_lr0010", "sac_lag", ("--lagrange_lr", "0.001")),
]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def git_dirty() -> str:
    return "1" if subprocess.run(["git", "diff", "--quiet"]).returncode != 0 else "0"


def safe(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def result_root() -> str:
    return f"results/star_baseline_screen_{git_sha()}"


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def all_jobs() -> list[dict]:
    jobs = []
    idx = 0
    for spec in SPECS:
        for task in TASKS:
            short = "pg1" if "Point" in task else "cg1"
            jobs.append({
                "index": idx,
                "config": spec.name,
                "method": spec.method,
                "extra": spec.extra,
                "task": task,
                "short": short,
                "seed": SEED,
                "steps": STEPS,
            })
            idx += 1
    return jobs


def run_name(job: dict) -> str:
    return f"baseline_screen_{job['config']}_{job['short']}_s{job['seed']}"


def run_dir(job: dict) -> Path:
    return Path(result_root()) / safe(job["task"]) / safe(job["method"]) / run_name(job)


def is_complete(job: dict) -> bool:
    eff = run_dir(job) / "efficiency.csv"
    if not eff.exists():
        return False
    try:
        rows = list(csv.DictReader(eff.open()))
        return bool(rows) and int(float(rows[-1].get("step", 0))) >= STEPS
    except Exception:
        return False


def has_error(job: dict) -> bool:
    log = LOG_ROOT / f"{run_name(job)}.log"
    if not log.exists():
        return False
    text = log.read_text(errors="ignore")
    return any(item in text for item in ["Traceback", "RuntimeError", "NaN", "nan", "OOM", "out of memory", "CUDA error", "No space left"])


def assigned_gpu(running_counts: dict[int, int]) -> int:
    return sorted(GPU_IDS, key=lambda gpu: (running_counts.get(gpu, 0), gpu))[0]


def command_for(job: dict, gpu: int) -> str:
    args = [
        "python", "main_star.py",
        "--task", job["task"],
        "--safe_env", "True",
        "--method", job["method"],
        "--star_exec", "False",
        "--seed", str(job["seed"]),
        "--device", "0",
        "--cuda", "True",
        "--num_steps", str(job["steps"]),
        "--start_steps", "5000",
        "--batch_size", "256",
        "--hidden_size", "256",
        "--updates_per_step", "1",
        "--shadow_k", "16",
        "--shadow_temperature", "0.05",
        "--shadow_aggregation", "log_mean_exp",
        "--shadow_reference_mode", "corridor",
        "--star_risk_threshold", "0.10",
        "--cost_gamma", "0.99",
        "--cost_critic_reduce", "max",
        "--recent_fraction", "0.0",
        "--binary_cost", "True",
        "--eval", "False",
        "--save", "True",
        "--save_interval_steps", "0",
        "--final_checkpoint", "True",
        "--disable_wandb", "True",
        "--online_eval_mode", "none",
        "--wandb_log_interval_steps", "1000",
        "--mechanism_log_interval_steps", "1000",
        "--action_diagnostics_interval_steps", "1000",
        "--output_root", result_root(),
        "--run_name", run_name(job),
        "--ablation_group", "baseline_screen",
        "--ablation_name", job["config"],
    ] + list(job["extra"])
    env = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
        f"export CUDA_VISIBLE_DEVICES={gpu} WANDB_MODE=disabled "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl && "
    )
    return "cd /root/FLAC-Safe && " + env + " ".join(shlex.quote(arg) for arg in args)


def append_manifest(row: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "baseline_screen_manifest.csv"
    exists = path.exists()
    fields = ["time", "status", "config", "method", "task", "seed", "gpu", "session", "run_name", "run_dir", "log_path", "command", "git_commit", "git_dirty", "hostname"]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def active_sessions() -> set[str]:
    return {session for session in tmux_sessions() if session.startswith("star_goal_baseline_screen_run_")}


def scheduler_loop(poll: int) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    while True:
        sessions = active_sessions()
        running_counts = {gpu: 0 for gpu in GPU_IDS}
        for session in sessions:
            for part in session.split("_"):
                if part.startswith("g") and part[1:].isdigit():
                    running_counts[int(part[1:])] = running_counts.get(int(part[1:]), 0) + 1
        pending = [job for job in jobs if not is_complete(job) and not has_error(job)]
        if not pending and not sessions:
            break
        capacity = MAX_TOTAL_RUNS - len(sessions)
        for job in pending:
            if capacity <= 0:
                break
            if any(run_name(job) in session for session in sessions):
                continue
            gpu = assigned_gpu(running_counts)
            if running_counts.get(gpu, 0) >= RUNS_PER_GPU:
                continue
            session = f"star_goal_baseline_screen_run_g{gpu}_{run_name(job)}"
            log_path = LOG_ROOT / f"{run_name(job)}.log"
            cmd = command_for(job, gpu)
            subprocess.check_call(["tmux", "new", "-d", "-s", session, f"{cmd} 2>&1 | tee {shlex.quote(str(log_path))}"])
            append_manifest({
                "time": datetime.now().isoformat(),
                "status": "launched",
                "config": job["config"],
                "method": job["method"],
                "task": job["task"],
                "seed": job["seed"],
                "gpu": gpu,
                "session": session,
                "run_name": run_name(job),
                "run_dir": str(run_dir(job)),
                "log_path": str(log_path),
                "command": cmd,
                "git_commit": git_sha(),
                "git_dirty": git_dirty(),
                "hostname": socket.gethostname(),
            })
            running_counts[gpu] += 1
            sessions.add(session)
            capacity -= 1
        time.sleep(poll)


def launch() -> None:
    scheduler = "star_goal_baseline_screen_scheduler"
    if scheduler in tmux_sessions():
        print(f"scheduler already running: {scheduler}")
        return
    cmd = "cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/goal_baseline_screen.py --scheduler-loop"
    subprocess.check_call(["tmux", "new", "-d", "-s", scheduler, cmd])
    print(f"launched {scheduler}")


def status() -> None:
    print("Baseline screen sessions:")
    for session in sorted(s for s in tmux_sessions() if s.startswith("star_goal_baseline_screen")):
        print(f"  {session}")
    jobs = all_jobs()
    print(f"completed={sum(is_complete(j) for j in jobs)}/{len(jobs)} failed={sum(has_error(j) for j in jobs)}")
    for job in jobs:
        if is_complete(job):
            st = "completed"
        elif has_error(job):
            st = "failed"
        elif any(run_name(job) in s for s in tmux_sessions()):
            st = "running"
        else:
            st = "pending"
        print(f"{st:10s} {job['config']:22s} {job['task']} seed={job['seed']}")


def plan() -> None:
    print("| Config | Method | Task | Seed | Steps |")
    print("| --- | --- | --- | ---: | ---: |")
    for job in all_jobs():
        print(f"| {job['config']} | {job['method']} | {job['task']} | {job['seed']} | {job['steps']} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--scheduler-loop", action="store_true")
    parser.add_argument("--poll", type=int, default=20)
    args = parser.parse_args()
    if args.scheduler_loop:
        scheduler_loop(args.poll)
    elif args.launch:
        launch()
    elif args.status:
        status()
    else:
        plan()


if __name__ == "__main__":
    main()
