from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shlex
import socket
import subprocess
from pathlib import Path


REPORT = Path("reports/star_goal")
LOG_ROOT = Path("logs/star_goal")
MANIFEST = REPORT / "experiment_manifest.csv"


def git(cmd: list[str]) -> str:
    return subprocess.check_output(["git"] + cmd, text=True).strip()


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def append_manifest(row: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST.exists()
    fields = [
        "start_time", "status", "suite", "session", "pid", "gpu", "task", "method", "seed",
        "steps", "runs_per_gpu", "cpu_cores", "git_commit", "git_dirty", "hostname",
        "command", "log_path", "output_root", "run_name",
    ]
    with MANIFEST.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def benchmark_jobs(runs_per_gpu: int, steps: int, seed_base: int, gpus: list[int]) -> list[dict]:
    jobs = []
    tasks = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
    idx = 0
    for gpu in gpus:
        for slot in range(runs_per_gpu):
            task = tasks[idx % len(tasks)]
            short = "pg1" if "Point" in task else "cg1"
            seed = seed_base + idx
            jobs.append({
                "gpu": gpu,
                "slot": slot,
                "task": task,
                "short": short,
                "seed": seed,
                "steps": steps,
                "method": "star",
            })
            idx += 1
    return jobs


def command_for(job: dict, output_root: str, run_name: str) -> str:
    args = [
        "python", "main_star.py",
        "--task", job["task"],
        "--safe_env", "True",
        "--method", job["method"],
        "--seed", str(job["seed"]),
        "--device", "0",
        "--cuda", "True",
        "--num_steps", str(job["steps"]),
        "--start_steps", "5000",
        "--batch_size", "256",
        "--hidden_size", "256",
        "--updates_per_step", "1",
        "--shadow_k", "16",
        "--star_exec", "True",
        "--eval", "False",
        "--save", "False",
        "--final_checkpoint", "False",
        "--disable_wandb", "True",
        "--online_eval_mode", "none",
        "--wandb_log_interval_steps", "1000",
        "--mechanism_log_interval_steps", "1000",
        "--action_diagnostics_interval_steps", "1000",
        "--output_root", output_root,
        "--run_name", run_name,
    ]
    env = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
        f"export CUDA_VISIBLE_DEVICES={job['gpu']} WANDB_MODE=disabled "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl && "
    )
    return "cd /root/FLAC-Safe && " + env + " ".join(shlex.quote(a) for a in args)


def launch(args: argparse.Namespace) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    git_commit = git(["rev-parse", "--short", "HEAD"])
    git_dirty = "1" if subprocess.run(["git", "diff", "--quiet"]).returncode != 0 else "0"
    output_root = f"results/star_parallel_bench_{git_commit}/rpg{args.runs_per_gpu}"
    jobs = benchmark_jobs(args.runs_per_gpu, args.steps, args.seed_base, gpus)
    sessions = tmux_sessions()
    rows = []
    for job in jobs:
        run_name = f"parallel_rpg{args.runs_per_gpu}_{job['short']}_g{job['gpu']}_slot{job['slot']}_s{job['seed']}"
        session = f"star_goal_rpg{args.runs_per_gpu}_g{job['gpu']}_s{job['slot']}_{job['short']}"
        log_path = LOG_ROOT / f"{run_name}.log"
        cmd = command_for(job, output_root, run_name)
        rows.append((job, session, run_name, log_path, cmd))
        if args.plan or args.dry_run:
            print(f"{session}: {cmd} > {log_path}")
            continue
        if session in sessions:
            print(f"skip running session {session}")
            continue
        tmux_cmd = f"{cmd} 2>&1 | tee {shlex.quote(str(log_path))}"
        subprocess.check_call(["tmux", "new", "-d", "-s", session, tmux_cmd])
        append_manifest({
            "start_time": dt.datetime.now().isoformat(),
            "status": "launched",
            "suite": "parallel_benchmark",
            "session": session,
            "pid": "",
            "gpu": job["gpu"],
            "task": job["task"],
            "method": job["method"],
            "seed": job["seed"],
            "steps": job["steps"],
            "runs_per_gpu": args.runs_per_gpu,
            "cpu_cores": "unbound",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "hostname": socket.gethostname(),
            "command": cmd,
            "log_path": str(log_path),
            "output_root": output_root,
            "run_name": run_name,
        })
    if not args.plan and not args.dry_run:
        monitor = f"star_goal_monitor_rpg{args.runs_per_gpu}"
        subprocess.run(["tmux", "kill-session", "-t", monitor], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mon_cmd = (
            "cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && "
            f"python scripts/star/monitor_resources.py --output reports/star_goal/resource_monitor_rpg{args.runs_per_gpu}.csv "
            "--interval 5 --prefix star_goal_rpg"
        )
        subprocess.check_call(["tmux", "new", "-d", "-s", monitor, mon_cmd])
        print(f"launched {len(rows)} jobs, output_root={output_root}")


def stop() -> None:
    for session in sorted(s for s in tmux_sessions() if s.startswith("star_goal_")):
        print(f"killing {session}")
        subprocess.run(["tmux", "kill-session", "-t", session])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--retry-infrastructure-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-per-gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed-base", type=int, default=9300)
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "0,1"))
    args = parser.parse_args()
    if args.status:
        subprocess.call(["python", "scripts/star/goal_status.py"])
    elif args.stop:
        stop()
    else:
        launch(args)


if __name__ == "__main__":
    main()
