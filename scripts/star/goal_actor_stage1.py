
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
LOG_ROOT = Path("logs/star_goal/actor_stage1")
MANIFEST = REPORT / "actor_stage1_manifest.csv"
RESULT_PREFIX = "star_tune_actor_stage1"
TASKS = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
SEED = 10
STEPS = 100000
MAX_TOTAL_RUNS = 6
RUNS_PER_GPU = 3
GPU_IDS = [0, 1]


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    extra: tuple[str, ...]


CONFIGS = [
    ConfigSpec("default", ()),
    ConfigSpec("threshold_005", ("--star_risk_threshold", "0.05")),
    ConfigSpec("threshold_020", ("--star_risk_threshold", "0.20")),
    ConfigSpec("lambda_050", ("--star_lambda", "0.5")),
    ConfigSpec("lambda_200", ("--star_lambda", "2.0")),
    ConfigSpec("k8", ("--shadow_k", "8")),
    ConfigSpec("k32", ("--shadow_k", "32")),
    ConfigSpec("temperature_003", ("--shadow_temperature", "0.03")),
    ConfigSpec("temperature_010", ("--shadow_temperature", "0.10")),
    ConfigSpec("ref_interval10", ("--star_ref_update_interval", "10")),
    ConfigSpec("ref_interval50", ("--star_ref_update_interval", "50")),
    ConfigSpec("kl_off", ("--star_kl_coef", "0.0", "--star_kl_target", "0.0")),
]


def safe(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def git_dirty() -> str:
    return "1" if subprocess.run(["git", "diff", "--quiet"]).returncode != 0 else "0"


def output_root() -> str:
    return f"results/{RESULT_PREFIX}_{git_sha()}"


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def all_jobs() -> list[dict]:
    jobs = []
    idx = 0
    for cfg in CONFIGS:
        for task in TASKS:
            short = "pg1" if "Point" in task else "cg1"
            jobs.append({
                "index": idx,
                "config": cfg.name,
                "extra": cfg.extra,
                "task": task,
                "short": short,
                "seed": SEED,
                "steps": STEPS,
            })
            idx += 1
    return jobs


def run_name(job: dict) -> str:
    return f"actor_stage1_{job['config']}_{job['short']}_s{job['seed']}"


def run_dir(job: dict) -> Path:
    return Path(output_root()) / safe(job["task"]) / "star_actor" / run_name(job)


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
    needles = ["Traceback", "RuntimeError", "NaN", "nan", "OOM", "out of memory", "CUDA error"]
    return any(n in text for n in needles)


def assigned_gpu(running_counts: dict[int, int]) -> int:
    candidates = sorted(GPU_IDS, key=lambda g: (running_counts.get(g, 0), g))
    return candidates[0]


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
        "--star_risk_threshold", "0.10",
        "--star_lambda", "1.0",
        "--star_ref_update_interval", "20",
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
        "--output_root", output_root(),
        "--run_name", run_name(job),
        "--ablation_group", "actor_stage1",
        "--ablation_name", job["config"],
    ] + list(job["extra"])
    env = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
        f"export CUDA_VISIBLE_DEVICES={gpu} WANDB_MODE=disabled "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl && "
    )
    return "cd /root/FLAC-Safe && " + env + " ".join(shlex.quote(a) for a in args)


def append_manifest(row: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST.exists()
    fields = [
        "time", "stage", "status", "config", "task", "seed", "gpu", "session",
        "run_name", "run_dir", "log_path", "command", "git_commit", "git_dirty", "hostname",
    ]
    with MANIFEST.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def active_actor_sessions() -> set[str]:
    return {s for s in tmux_sessions() if s.startswith("star_goal_actor_stage1_run_")}


def scheduler_loop(poll: int = 20) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    while True:
        sessions = active_actor_sessions()
        running_counts = {g: 0 for g in GPU_IDS}
        for session in sessions:
            parts = session.split("_")
            for part in parts:
                if part.startswith("g") and part[1:].isdigit():
                    running_counts[int(part[1:])] = running_counts.get(int(part[1:]), 0) + 1
        pending = [j for j in jobs if not is_complete(j) and not has_error(j)]
        if not pending and not sessions:
            break
        capacity = MAX_TOTAL_RUNS - len(sessions)
        for job in pending:
            if capacity <= 0:
                break
            session = f"star_goal_actor_stage1_run_gX_{run_name(job)}"
            if any(run_name(job) in s for s in sessions):
                continue
            gpu = assigned_gpu(running_counts)
            if running_counts.get(gpu, 0) >= RUNS_PER_GPU:
                continue
            session = f"star_goal_actor_stage1_run_g{gpu}_{run_name(job)}"
            log_path = LOG_ROOT / f"{run_name(job)}.log"
            cmd = command_for(job, gpu)
            tmux_cmd = f"{cmd} 2>&1 | tee {shlex.quote(str(log_path))}"
            subprocess.check_call(["tmux", "new", "-d", "-s", session, tmux_cmd])
            append_manifest({
                "time": datetime.now().isoformat(),
                "stage": "actor_stage1",
                "status": "launched",
                "config": job["config"],
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
    scheduler = "star_goal_actor_stage1_scheduler"
    if scheduler in tmux_sessions():
        print(f"scheduler already running: {scheduler}")
        return
    cmd = "cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/goal_actor_stage1.py --scheduler-loop"
    subprocess.check_call(["tmux", "new", "-d", "-s", scheduler, cmd])
    mon = "star_goal_actor_stage1_monitor"
    subprocess.run(["tmux", "kill-session", "-t", mon], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mon_cmd = "cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/monitor_resources.py --output reports/star_goal/actor_stage1_resource_monitor.csv --interval 5 --prefix star_goal_actor_stage1"
    subprocess.check_call(["tmux", "new", "-d", "-s", mon, mon_cmd])
    print("launched actor stage1 scheduler")


def status() -> None:
    jobs = all_jobs()
    sessions = tmux_sessions()
    print("Actor stage1 sessions:")
    for s in sorted(x for x in sessions if x.startswith("star_goal_actor_stage1")):
        print(f"  {s}")
    done = sum(is_complete(j) for j in jobs)
    failed = sum(has_error(j) for j in jobs)
    print(f"completed={done}/{len(jobs)} failed={failed}")
    for job in jobs:
        if is_complete(job):
            st = "completed"
        elif has_error(job):
            st = "failed"
        elif any(run_name(job) in s for s in sessions):
            st = "running"
        else:
            st = "pending"
        print(f"{st:10s} {job['config']:18s} {job['task']} seed={job['seed']} run={run_name(job)}")


def plan() -> None:
    print("| Config | Task | Seed | Steps |")
    print("| --- | --- | ---: | ---: |")
    for job in all_jobs():
        print(f"| {job['config']} | {job['task']} | {job['seed']} | {job['steps']} |")


def collect() -> None:
    rows = []
    for job in all_jobs():
        d = run_dir(job)
        eff = d / "efficiency.csv"
        train = d / "train_episodes.csv"
        row = {
            "config": job["config"],
            "task": job["task"],
            "seed": job["seed"],
            "run_name": run_name(job),
            "status": "completed" if is_complete(job) else ("failed" if has_error(job) else "pending"),
            "final_step": "",
            "wall_time": "",
            "transitions_per_sec": "",
            "train_cost_rate": "",
            "checkpoint": str(d / "checkpoint" / "final.torch") if (d / "checkpoint" / "final.torch").exists() else "",
        }
        if eff.exists():
            ers = list(csv.DictReader(eff.open()))
            if ers:
                row["final_step"] = ers[-1].get("step", "")
                row["wall_time"] = ers[-1].get("wall_clock_time", "")
                row["transitions_per_sec"] = ers[-1].get("env_steps_per_second", "")
        if train.exists():
            trs = list(csv.DictReader(train.open()))
            if trs:
                row["train_cost_rate"] = trs[-1].get("train_total_cost_rate", "")
        rows.append(row)
    out = REPORT / "actor_stage1.csv"
    with out.open("w", newline="") as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Actor Stage1 Status", "", "| Config | Task | Seed | Status | Step | t/s | Cost rate | Checkpoint |", "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for r in rows:
        md.append(f"| {r['config']} | {r['task']} | {r['seed']} | {r['status']} | {r['final_step']} | {r['transitions_per_sec']} | {r['train_cost_rate']} | {bool(r['checkpoint'])} |")
    (REPORT / "actor_stage1.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--scheduler-loop", action="store_true")
    args = parser.parse_args()
    if args.scheduler_loop:
        scheduler_loop()
    elif args.launch:
        launch()
    elif args.status:
        status()
    elif args.collect:
        collect()
    else:
        plan()


if __name__ == "__main__":
    main()
