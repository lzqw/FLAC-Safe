#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPORT = Path("reports/star_goal")
LOG_ROOT = Path("logs/star_goal/paper_support_300k")
TASKS = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
METHODS = ["pointwise", "sac_lag", "star_actor", "star"]
SEEDS = [10, 11, 12]
STEPS = 300000
MAX_TOTAL_RUNS = 6
RUNS_PER_GPU = 3
GPU_IDS = [0, 1]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def git_dirty() -> str:
    return "1" if subprocess.run(["git", "diff", "--quiet"]).returncode != 0 else "0"


def safe(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def result_root() -> str:
    return f"results/star_paper_support_300k_{git_sha()}"


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return dict(default)


def actor_config(report_dir: Path) -> dict:
    return load_json(report_dir / "selected_actor_config.json", {
        "cost_gamma": 0.99,
        "star_risk_threshold": 0.10,
        "star_lambda": 1.0,
        "shadow_k": 16,
        "shadow_temperature": 0.05,
        "shadow_aggregation": "log_mean_exp",
        "shadow_reference_mode": "corridor",
        "star_ref_update_interval": 20,
        "star_kl_coef": 1.0,
        "star_kl_target": 0.01,
        "cost_critic_reduce": "max",
    })


def baseline_config(report_dir: Path) -> dict:
    return load_json(report_dir / "selected_baseline_config.json", {
        "pointwise_star_lambda": 1.0,
        "sac_lag_lagrange_lr": 0.0003,
    })


def all_jobs() -> list[dict]:
    jobs = []
    idx = 0
    for method in METHODS:
        for task in TASKS:
            short = "pg1" if "Point" in task else "cg1"
            for seed in SEEDS:
                jobs.append({
                    "index": idx,
                    "method": method,
                    "task": task,
                    "short": short,
                    "seed": seed,
                    "steps": STEPS,
                })
                idx += 1
    return jobs


def run_name(job: dict) -> str:
    return f"paper300k_{job['method']}_{job['short']}_s{job['seed']}"


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


def common_args(job: dict, report_dir: Path) -> list[str]:
    cfg = actor_config(report_dir)
    base = baseline_config(report_dir)
    star_exec = "True" if job["method"] == "star" else "False"
    args = [
        "python", "main_star.py",
        "--task", job["task"],
        "--safe_env", "True",
        "--method", job["method"],
        "--star_exec", star_exec,
        "--seed", str(job["seed"]),
        "--device", "0",
        "--cuda", "True",
        "--num_steps", str(job["steps"]),
        "--start_steps", "5000",
        "--batch_size", "256",
        "--hidden_size", "256",
        "--updates_per_step", "1",
        "--shadow_k", str(int(cfg.get("shadow_k", 16))),
        "--shadow_temperature", str(float(cfg.get("shadow_temperature", 0.05))),
        "--shadow_aggregation", str(cfg.get("shadow_aggregation", "log_mean_exp")),
        "--shadow_reference_mode", str(cfg.get("shadow_reference_mode", "corridor")),
        "--star_risk_threshold", str(float(cfg.get("star_risk_threshold", 0.10))),
        "--star_lambda", str(float(cfg.get("star_lambda", 1.0))),
        "--star_ref_update_interval", str(int(cfg.get("star_ref_update_interval", 20))),
        "--star_kl_coef", str(float(cfg.get("star_kl_coef", 1.0))),
        "--star_kl_target", str(float(cfg.get("star_kl_target", 0.01))),
        "--cost_gamma", str(float(cfg.get("cost_gamma", 0.99))),
        "--cost_critic_reduce", str(cfg.get("cost_critic_reduce", "max")),
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
        "--ablation_group", "paper_support_300k",
        "--ablation_name", job["method"],
    ]
    if job["method"] == "pointwise":
        args.extend(["--star_lambda", str(float(base.get("pointwise_star_lambda", cfg.get("star_lambda", 1.0))))])
    if job["method"] == "sac_lag":
        args.extend(["--lagrange_lr", str(float(base.get("sac_lag_lagrange_lr", 0.0003)))])
    return args


def command_for(job: dict, gpu: int, report_dir: Path) -> str:
    args = common_args(job, report_dir)
    env = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
        f"export CUDA_VISIBLE_DEVICES={gpu} WANDB_MODE=disabled "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl && "
    )
    return "cd /root/FLAC-Safe && " + env + " ".join(shlex.quote(arg) for arg in args)


def append_manifest(row: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "paper_support_300k_manifest.csv"
    exists = path.exists()
    fields = ["time", "status", "method", "task", "seed", "gpu", "session", "run_name", "run_dir", "log_path", "command", "git_commit", "git_dirty", "hostname"]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def active_sessions() -> set[str]:
    return {session for session in tmux_sessions() if session.startswith("star_goal_paper300k_run_")}


def scheduler_loop(report_dir: Path, poll: int) -> None:
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
            session = f"star_goal_paper300k_run_g{gpu}_{run_name(job)}"
            log_path = LOG_ROOT / f"{run_name(job)}.log"
            cmd = command_for(job, gpu, report_dir)
            subprocess.check_call(["tmux", "new", "-d", "-s", session, f"{cmd} 2>&1 | tee {shlex.quote(str(log_path))}"])
            append_manifest({
                "time": datetime.now().isoformat(),
                "status": "launched",
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


def launch(report_dir: Path) -> None:
    scheduler = "star_goal_paper300k_scheduler"
    if scheduler in tmux_sessions():
        print(f"scheduler already running: {scheduler}")
        return
    cmd = f"cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/goal_paper_support_300k.py --scheduler-loop --report-dir {shlex.quote(str(report_dir))}"
    subprocess.check_call(["tmux", "new", "-d", "-s", scheduler, cmd])
    print(f"launched {scheduler}")


def status() -> None:
    print("Paper 300k sessions:")
    for session in sorted(s for s in tmux_sessions() if s.startswith("star_goal_paper300k")):
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
        print(f"{st:10s} {job['method']:12s} {job['task']} seed={job['seed']}")


def plan(report_dir: Path) -> None:
    print("| Method | Task | Seed | Steps |")
    print("| --- | --- | ---: | ---: |")
    for job in all_jobs():
        print(f"| {job['method']} | {job['task']} | {job['seed']} | {job['steps']} |")
    print(f"actor_config={report_dir / 'selected_actor_config.json'}")
    print(f"baseline_config={report_dir / 'selected_baseline_config.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--scheduler-loop", action="store_true")
    parser.add_argument("--poll", type=int, default=20)
    args = parser.parse_args()
    if args.scheduler_loop:
        scheduler_loop(args.report_dir, args.poll)
    elif args.launch:
        launch(args.report_dir)
    elif args.status:
        status()
    else:
        plan(args.report_dir)


if __name__ == "__main__":
    main()
