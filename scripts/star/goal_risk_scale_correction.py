#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPORT = Path("reports/star_goal")
LOG_ROOT = Path("logs/star_goal/risk_scale_correction")
CORRECTION_REPORT = REPORT / "risk_scale_correction"
CONFIG_DIR = Path("configs")
TASKS = ["SafetyPointGoal1-v0", "SafetyCarGoal1-v0"]
SEED = 10
STEPS = 100000
MAX_TOTAL_RUNS = 6
RUNS_PER_GPU = 3
GPU_IDS = [0, 1]

# One allowed directed correction after the initial risk-scale calibration failed.
# Direction: adjust only cost_gamma / risk threshold. No STAR core changes.
CORRECTION_SPECS = [
    ("cg097_thr0p55", 0.97, 0.55),
    ("cg097_thr0p60", 0.97, 0.60),
    ("cg095_thr0p45", 0.95, 0.45),
    ("cg095_thr0p50", 0.95, 0.50),
]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def git_dirty() -> str:
    return "1" if subprocess.run(["git", "diff", "--quiet"]).returncode != 0 else "0"


def safe(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def result_root() -> str:
    return f"results/star_risk_scale_correction_{git_sha()}"


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def all_jobs() -> list[dict]:
    jobs = []
    idx = 0
    for name, cost_gamma, threshold in CORRECTION_SPECS:
        for task in TASKS:
            short = "pg1" if "Point" in task else "cg1"
            jobs.append({
                "index": idx,
                "config": name,
                "cost_gamma": cost_gamma,
                "threshold": threshold,
                "task": task,
                "short": short,
                "seed": SEED,
                "steps": STEPS,
            })
            idx += 1
    return jobs


def run_name(job: dict) -> str:
    return f"risk_correction_{job['config']}_{job['short']}_s{job['seed']}"


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
    needles = ["Traceback", "RuntimeError", "NaN", "nan", "OOM", "out of memory", "CUDA error", "No space left", "invalid loss"]
    return any(item in text for item in needles)


def active_sessions() -> set[str]:
    return {session for session in tmux_sessions() if session.startswith("star_goal_risk_correction_run_")}


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
        "--ablation_group", "risk_scale_correction",
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
    path = REPORT / "risk_scale_correction_manifest.csv"
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
            session = f"star_goal_risk_correction_run_g{gpu}_{run_name(job)}"
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
            running_counts[gpu] = running_counts.get(gpu, 0) + 1
            capacity -= 1
        time.sleep(poll)


def print_status() -> None:
    sessions = active_sessions()
    if sessions:
        print("Risk-scale correction sessions:")
        for session in sorted(sessions):
            print(f"  {session}")
    completed = sum(is_complete(job) for job in all_jobs())
    failed = sum(has_error(job) for job in all_jobs())
    print(f"completed={completed}/{len(all_jobs())} failed={failed}")
    for job in all_jobs():
        status = "completed" if is_complete(job) else "failed" if has_error(job) else "running" if any(run_name(job) in s for s in sessions) else "pending"
        print(f"{status:10s} {job['config']:16s} {job['task']} seed={job['seed']} threshold={job['threshold']:.2f} gamma={job['cost_gamma']:.2f}")


def all_complete() -> bool:
    jobs = all_jobs()
    return all(is_complete(job) for job in jobs) and not any(has_error(job) for job in jobs)


def sh(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.check_call(cmd)


def write_selected_markdown(selected_path: Path) -> None:
    selected_report = REPORT / "selected_actor_config.md"
    if not selected_path.exists():
        selected_report.write_text(
            "# Selected STAR Actor Config\n\n"
            "No valid risk-scale correction actor configuration has been selected yet.\n"
        )
        return
    data = json.loads(selected_path.read_text())
    lines = [
        "# Selected STAR Actor Config",
        "",
        "Selected from the directed risk-scale correction stage.",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in [
        "selected_config",
        "cost_gamma",
        "star_risk_threshold",
        "star_lambda",
        "shadow_k",
        "shadow_temperature",
        "shadow_aggregation",
        "shadow_reference_mode",
        "star_ref_update_interval",
        "star_kl_coef",
        "star_kl_target",
        "cost_critic_reduce",
        "star_exec_candidates",
        "star_exec_margin",
    ]:
        lines.append(f"| {key} | `{data.get(key, '')}` |")
    selected_report.write_text("\n".join(lines) + "\n")


def collect() -> None:
    root = Path(result_root())
    REPORT.mkdir(parents=True, exist_ok=True)
    CORRECTION_REPORT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in [
        REPORT / "selected_actor_config.json",
        REPORT / "selected_actor_config.md",
        CONFIG_DIR / "star_selected_actor.json",
    ]:
        if stale.exists():
            stale.unlink()

    completed = sum(is_complete(job) for job in all_jobs())
    failed = sum(has_error(job) for job in all_jobs())
    total = len(all_jobs())
    if completed < total or failed:
        print(f"warning: collecting partial correction results completed={completed}/{total} failed={failed}", flush=True)

    sh([
        "python",
        "scripts/star/collect_risk_scale_results.py",
        "--root",
        str(root),
        "--report-dir",
        str(CORRECTION_REPORT),
    ])

    runs = CORRECTION_REPORT / "risk_scale_runs.csv"
    candidates = CORRECTION_REPORT / "risk_scale_candidates.csv"
    selection = CORRECTION_REPORT / "risk_scale_selection.md"
    if runs.exists():
        shutil.copyfile(runs, REPORT / "risk_scale_correction_summary.csv")
    if candidates.exists():
        shutil.copyfile(candidates, REPORT / "risk_scale_correction_candidates.csv")
    if selection.exists():
        shutil.copyfile(selection, REPORT / "risk_scale_correction_selection.md")

    selected_path = REPORT / "selected_actor_config.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        selected["selected_from_stage"] = "risk_scale_correction"
        selected_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        config_path = CONFIG_DIR / "star_selected_actor.json"
        config_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        write_selected_markdown(config_path)
        print(f"wrote {config_path}")
    else:
        write_selected_markdown(selected_path)

    status_lines = [
        "# Risk-Scale Correction Status",
        "",
        f"- Updated: {datetime.now().isoformat()}",
        f"- Completed: {completed}/{total}",
        f"- Failed: {failed}",
        f"- Result root: `{root}`",
        f"- Summary: `{REPORT / 'risk_scale_correction_summary.csv'}`",
        f"- Selection: `{REPORT / 'risk_scale_correction_selection.md'}`",
    ]
    if selected_path.exists():
        status_lines.append(f"- Selected actor config: `{CONFIG_DIR / 'star_selected_actor.json'}`")
    else:
        status_lines.append("- Selected actor config: not selected")
    (REPORT / "risk_scale_correction_status.md").write_text("\n".join(status_lines) + "\n")


def launch() -> None:
    session = "star_goal_risk_correction_scheduler"
    if session in tmux_sessions():
        print(f"scheduler already running: {session}")
        return
    cmd = "cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/goal_risk_scale_correction.py --scheduler-loop"
    subprocess.check_call(["tmux", "new", "-d", "-s", session, cmd])
    print(f"launched {session}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=["plan", "launch", "status", "scheduler-loop", "collect"])
    parser.add_argument("--status", action="store_true", help="Compatibility alias for: status")
    parser.add_argument("--collect", action="store_true", help="Compatibility alias for: collect")
    parser.add_argument("--scheduler-loop", action="store_true", help="Compatibility alias for: scheduler-loop")
    parser.add_argument("--poll", type=int, default=30)
    args = parser.parse_args()
    command = args.command
    if args.status:
        command = "status"
    if args.collect:
        command = "collect"
    if args.scheduler_loop:
        command = "scheduler-loop"
    if command is None:
        parser.error("one command is required")
    if command == "plan":
        print(f"result_root={result_root()}")
        for job in all_jobs():
            print(f"{job['config']} {job['task']} seed={job['seed']} threshold={job['threshold']:.2f} gamma={job['cost_gamma']:.2f} run={run_name(job)}")
    elif command == "launch":
        launch()
    elif command == "status":
        print_status()
    elif command == "scheduler-loop":
        scheduler_loop(args.poll)
    elif command == "collect":
        collect()


if __name__ == "__main__":
    main()
