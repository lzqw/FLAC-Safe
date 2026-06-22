#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPORT = Path("reports/star_goal")
LOG_ROOT = Path("logs/star_goal/risk_scale_calibration")
TASKS = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
SEED = 10
STEPS = 100000
MAX_TOTAL_RUNS = 6
RUNS_PER_GPU = 3
GPU_IDS = [0, 1]


@dataclass(frozen=True)
class RiskScaleSpec:
    name: str
    cost_gamma: float
    threshold_key: str


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def git_dirty() -> str:
    return "1" if subprocess.run(["git", "diff", "--quiet"]).returncode != 0 else "0"


def safe(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def result_root() -> str:
    return f"results/star_risk_scale_calibration_{git_sha()}"


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def read_thresholds(report_dir: Path) -> tuple[float, float]:
    decision = report_dir / "stage1_decision_thresholds.json"
    if decision.exists():
        data = json.loads(decision.read_text())
        return float(data["T_mid"]), float(data["T_high"])
    grid = report_dir / "shadow_threshold_grid.csv"
    if not grid.exists():
        raise SystemExit("missing shadow_threshold_grid.csv; run diagnose_shadow_risk first")
    pooled: dict[float, list[float]] = {}
    with grid.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                threshold = round(float(row["threshold"]), 2)
                rate = float(row["rho_active_rate"])
            except (KeyError, ValueError):
                continue
            pooled.setdefault(threshold, []).append(rate)
    if not pooled:
        raise SystemExit("shadow_threshold_grid.csv has no usable rho_active_rate rows")
    rates = {threshold: sum(vals) / len(vals) for threshold, vals in pooled.items()}
    t_mid = min(rates, key=lambda th: abs(rates[th] - 0.50))
    t_high = min(rates, key=lambda th: abs(rates[th] - 0.25))
    decision.write_text(json.dumps({"T_mid": t_mid, "T_high": t_high, "source": str(grid)}, indent=2) + "\n")
    return t_mid, t_high


def specs(report_dir: Path, include_gamma090: bool = False) -> list[RiskScaleSpec]:
    t_mid, t_high = read_thresholds(report_dir)
    out = [
        RiskScaleSpec(f"cg097_thr{t_mid:.2f}".replace(".", "p"), 0.97, "mid"),
        RiskScaleSpec(f"cg097_thr{t_high:.2f}".replace(".", "p"), 0.97, "high"),
        RiskScaleSpec(f"cg095_thr{t_mid:.2f}".replace(".", "p"), 0.95, "mid"),
        RiskScaleSpec(f"cg095_thr{t_high:.2f}".replace(".", "p"), 0.95, "high"),
    ]
    if include_gamma090:
        out.append(RiskScaleSpec(f"cg090_thr{t_high:.2f}".replace(".", "p"), 0.90, "high"))
    return out


def threshold_for(spec: RiskScaleSpec, report_dir: Path) -> float:
    t_mid, t_high = read_thresholds(report_dir)
    return t_mid if spec.threshold_key == "mid" else t_high


def all_jobs(report_dir: Path, include_gamma090: bool = False) -> list[dict]:
    jobs = []
    idx = 0
    for spec in specs(report_dir, include_gamma090=include_gamma090):
        for task in TASKS:
            short = "pg1" if "Point" in task else "cg1"
            jobs.append({
                "index": idx,
                "config": spec.name,
                "cost_gamma": spec.cost_gamma,
                "threshold": threshold_for(spec, report_dir),
                "task": task,
                "short": short,
                "seed": SEED,
                "steps": STEPS,
            })
            idx += 1
    return jobs


def run_name(job: dict) -> str:
    return f"risk_scale_{job['config']}_{job['short']}_s{job['seed']}"


def run_dir(job: dict) -> Path:
    return Path(result_root()) / safe(job["task"]) / "star_actor" / run_name(job)


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
    needles = ["Traceback", "RuntimeError", "NaN", "nan", "OOM", "out of memory", "CUDA error", "No space left"]
    return any(item in text for item in needles)


def assigned_gpu(running_counts: dict[int, int]) -> int:
    return sorted(GPU_IDS, key=lambda gpu: (running_counts.get(gpu, 0), gpu))[0]


def command_for(job: dict, gpu: int) -> str:
    args = [
        "python", "main_star.py",
        "--task", job["task"],
        "--safe_env", "True",
        "--method", "star_actor",
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
        "--star_risk_threshold", f"{job['threshold']:.2f}",
        "--star_lambda", "1.0",
        "--star_ref_update_interval", "20",
        "--cost_gamma", f"{job['cost_gamma']:.2f}",
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
        "--ablation_group", "risk_scale_calibration",
        "--ablation_name", job["config"],
    ]
    env = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
        f"export CUDA_VISIBLE_DEVICES={gpu} WANDB_MODE=disabled "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl && "
    )
    return "cd /root/FLAC-Safe && " + env + " ".join(shlex.quote(arg) for arg in args)


def append_manifest(row: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "risk_scale_calibration_manifest.csv"
    exists = path.exists()
    fields = [
        "time", "status", "config", "task", "seed", "cost_gamma", "threshold",
        "gpu", "session", "run_name", "run_dir", "log_path", "command",
        "git_commit", "git_dirty", "hostname",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def active_sessions() -> set[str]:
    return {session for session in tmux_sessions() if session.startswith("star_goal_risk_scale_run_")}


def scheduler_loop(report_dir: Path, include_gamma090: bool, poll: int) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs(report_dir, include_gamma090=include_gamma090)
    while True:
        sessions = active_sessions()
        running_counts = {gpu: 0 for gpu in GPU_IDS}
        for session in sessions:
            parts = session.split("_")
            for part in parts:
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
            session = f"star_goal_risk_scale_run_g{gpu}_{run_name(job)}"
            log_path = LOG_ROOT / f"{run_name(job)}.log"
            cmd = command_for(job, gpu)
            subprocess.check_call(["tmux", "new", "-d", "-s", session, f"{cmd} 2>&1 | tee {shlex.quote(str(log_path))}"])
            append_manifest({
                "time": datetime.now().isoformat(),
                "status": "launched",
                "config": job["config"],
                "task": job["task"],
                "seed": job["seed"],
                "cost_gamma": job["cost_gamma"],
                "threshold": job["threshold"],
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
            sessions.add(session)
            running_counts[gpu] += 1
            capacity -= 1
        time.sleep(poll)


def launch(report_dir: Path, include_gamma090: bool) -> None:
    scheduler = "star_goal_risk_scale_scheduler"
    if scheduler in tmux_sessions():
        print(f"scheduler already running: {scheduler}")
        return
    gamma_arg = " --include-gamma090" if include_gamma090 else ""
    cmd = f"cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/goal_risk_scale_calibration.py --scheduler-loop --report-dir {shlex.quote(str(report_dir))}{gamma_arg}"
    subprocess.check_call(["tmux", "new", "-d", "-s", scheduler, cmd])
    print(f"launched {scheduler}")


def status(report_dir: Path, include_gamma090: bool) -> None:
    print("Risk-scale sessions:")
    for session in sorted(s for s in tmux_sessions() if s.startswith("star_goal_risk_scale")):
        print(f"  {session}")
    jobs = all_jobs(report_dir, include_gamma090=include_gamma090)
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
        print(f"{st:10s} {job['config']:18s} {job['task']} seed={job['seed']} threshold={job['threshold']:.2f} gamma={job['cost_gamma']:.2f}")


def plan(report_dir: Path, include_gamma090: bool) -> None:
    print("| Config | Task | Seed | Cost gamma | Threshold | Steps |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for job in all_jobs(report_dir, include_gamma090=include_gamma090):
        print(f"| {job['config']} | {job['task']} | {job['seed']} | {job['cost_gamma']:.2f} | {job['threshold']:.2f} | {job['steps']} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--include-gamma090", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--scheduler-loop", action="store_true")
    parser.add_argument("--poll", type=int, default=20)
    args = parser.parse_args()
    if args.scheduler_loop:
        scheduler_loop(args.report_dir, args.include_gamma090, args.poll)
    elif args.launch:
        launch(args.report_dir, args.include_gamma090)
    elif args.status:
        status(args.report_dir, args.include_gamma090)
    else:
        plan(args.report_dir, args.include_gamma090)


if __name__ == "__main__":
    main()
