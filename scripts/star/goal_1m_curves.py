#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO / "reports" / "star_1m_curves"
LOG_ROOT = REPO / "logs" / "star_1m_curves"
RESULT_ROOT = REPO / "results" / "star_1m_curves"
SESSION = "star_1m_curves_scheduler"
PYTHON = sys.executable

TASKS = {
    "PointGoal1": "SafetyPointGoal1-v0",
    "CarGoal1": "SafetyCarGoal1-v0",
    "PointPush1": "SafetyPointPush1-v0",
}
AVAILABLE_METHODS = {
    "star_v2": "STAR",
    "sac_lag": "SAC-Lag",
}
UNAVAILABLE_METHODS = {
    "safe_flow_q": "Safe Flow Q",
    "ppo_lag": "PPO-Lag",
    "cpo": "CPO",
    "cspo": "CSPO",
}
GPU_SLOTS = {0: 3, 1: 3}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "MUJOCO_GL": "egl",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "STAR_STORAGE_ROOT": "/dev/shm/star_v2_storage",
    "TMPDIR": "/dev/shm/star_v2_storage/tmp",
    "TORCH_HOME": "/root/autodl-tmp/star_v2_storage/cache/torch",
    "MPLCONFIGDIR": "/root/autodl-tmp/star_v2_storage/cache/matplotlib",
    "XDG_CACHE_HOME": "/root/autodl-tmp/star_v2_storage/cache/xdg",
}
ERROR_PATTERNS = ("Traceback", "RuntimeError", "CUDA out of memory", "No space left", "nan", "NaN")


@dataclass(frozen=True)
class RunSpec:
    stage: str
    task_name: str
    env_id: str
    method: str
    seed: int
    steps: int = 1_000_000

    @property
    def run_name(self) -> str:
        safe_env = self.env_id.replace("-", "_")
        return f"curves1m_{self.stage}_{safe_env}_{self.method}_s{self.seed}"

    @property
    def result_dir(self) -> Path:
        return RESULT_ROOT / self.stage / self.env_id / self.method / self.run_name

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / self.stage / f"{self.run_name}.log"

    @property
    def final_checkpoint(self) -> Path:
        return self.result_dir / "checkpoint" / "final.torch"


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def ensure_dirs() -> None:
    for path in [
        REPORT_ROOT / "setup",
        REPORT_ROOT / "curves",
        REPORT_ROOT / "figures",
        REPORT_ROOT / "status",
        LOG_ROOT,
        RESULT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    storage = Path(os.environ.get("STAR_STORAGE_ROOT", "/dev/shm/star_v2_storage"))
    for name in ("results", "logs", "tmp", "cache"):
        (storage / name).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    with (REPO / path).open() as handle:
        return json.load(handle)


def specs_for_stage(stage: str) -> list[RunSpec]:
    if stage == "stage_a_star":
        return [
            RunSpec(stage, task_name, env_id, "star_v2", seed)
            for task_name, env_id in TASKS.items()
            for seed in range(5)
        ]
    if stage == "stage_b_baselines1":
        return [
            RunSpec(stage, task_name, env_id, "sac_lag", seed)
            for task_name, env_id in TASKS.items()
            for seed in range(3)
        ]
    return []


def all_specs() -> list[RunSpec]:
    return specs_for_stage("stage_a_star") + specs_for_stage("stage_b_baselines1")


def latest_resume_checkpoint(spec: RunSpec) -> Path | None:
    checkpoint_dir = spec.result_dir / "checkpoint"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("step_*.torch"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def max_train_step(spec: RunSpec) -> int:
    path = spec.result_dir / "train_episodes.csv"
    if not path.exists():
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(path, usecols=["end_step"])
        if df.empty:
            return 0
        return int(df["end_step"].max())
    except Exception:
        return 0


def train_progress_stats(spec: RunSpec) -> dict[str, object]:
    path = spec.result_dir / "train_episodes.csv"
    empty = {
        "max_step": 0,
        "episodes": 0,
        "wall_clock_time": 0.0,
        "overall_steps_per_sec": 0.0,
        "recent_steps_per_sec": 0.0,
        "eta_hours": "",
        "train_total_cost": "",
        "train_cost_rate": "",
    }
    if not path.exists():
        return empty
    try:
        import pandas as pd

        df = pd.read_csv(path)
    except Exception:
        return empty
    if df.empty or "end_step" not in df.columns:
        return empty
    df = df.sort_values("end_step")
    max_step = int(df["end_step"].max())
    episodes = int(len(df))
    wall = float(df["wall_clock_time"].dropna().iloc[-1]) if "wall_clock_time" in df.columns and df["wall_clock_time"].notna().any() else 0.0
    overall = float(max_step / wall) if wall > 0 else 0.0
    recent = 0.0
    if "wall_clock_time" in df.columns and len(df) >= 2:
        tail = df.dropna(subset=["end_step", "wall_clock_time"]).tail(6)
        if len(tail) >= 2:
            step_delta = float(tail["end_step"].iloc[-1] - tail["end_step"].iloc[0])
            time_delta = float(tail["wall_clock_time"].iloc[-1] - tail["wall_clock_time"].iloc[0])
            if time_delta > 0:
                recent = step_delta / time_delta
    speed = recent or overall
    eta_hours: float | str = ""
    if speed > 0 and max_step < spec.steps:
        eta_hours = (spec.steps - max_step) / speed / 3600.0
    total_cost: float | str = ""
    cost_rate: float | str = ""
    if "train_total_cost" in df.columns and df["train_total_cost"].notna().any():
        total_cost = float(df["train_total_cost"].dropna().iloc[-1])
    if "train_total_cost_rate" in df.columns and df["train_total_cost_rate"].notna().any():
        cost_rate = float(df["train_total_cost_rate"].dropna().iloc[-1])
    return {
        "max_step": max_step,
        "episodes": episodes,
        "wall_clock_time": wall,
        "overall_steps_per_sec": overall,
        "recent_steps_per_sec": recent,
        "eta_hours": eta_hours,
        "train_total_cost": total_cost,
        "train_cost_rate": cost_rate,
    }


def is_complete(spec: RunSpec) -> bool:
    return spec.final_checkpoint.exists() or max_train_step(spec) >= spec.steps


def ps_output() -> str:
    return run(["bash", "-lc", "ps -eo pid,args | grep '[m]ain_star.py' || true"]).stdout


def is_running(spec: RunSpec, ps_text: str | None = None) -> bool:
    ps_text = ps_output() if ps_text is None else ps_text
    return spec.run_name in ps_text


def status_for_spec(spec: RunSpec, ps_text: str | None = None) -> str:
    if is_complete(spec):
        return "completed"
    if is_running(spec, ps_text):
        return "running"
    if latest_resume_checkpoint(spec) is not None:
        return "partial"
    if max_train_step(spec) > 0:
        return "partial_no_checkpoint"
    return "pending"


def build_main_star_command(spec: RunSpec, device: int) -> list[str]:
    actor = load_json("configs/star_v2_selected_actor.json")
    baselines = load_json("configs/star_v2_selected_baselines.json")
    shared = baselines.get("shared", {})
    params: dict[str, object] = {
        "task": spec.env_id,
        "method": spec.method,
        "seed": spec.seed,
        "device": 0,
        "num_steps": spec.steps,
        "run_name": spec.run_name,
        "output_root": str(RESULT_ROOT / spec.stage),
        "ablation_group": spec.stage,
        "ablation_name": "1m_training_curve",
        "safe_env": True,
        "normalize_obs": True,
        "star_algorithm_version": "star_v2",
        "shadow_beta_mode": actor["shadow_beta_mode"],
        "shadow_num_strata": actor["shadow_num_strata"],
        "shadow_samples_per_stratum": actor["shadow_samples_per_stratum"],
        "shadow_temperature": actor["shadow_temperature"],
        "shadow_reference_mode": actor["shadow_reference_mode"],
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
        "evaluation_execution_mode": "raw",
        "eval": False,
        "online_eval_mode": "none",
        "disable_wandb": True,
        "save": True,
        "save_training_state": False,
        "save_interval_steps": 0,
        "final_checkpoint": True,
        "metric_log_interval_steps": 10000,
        "mechanism_log_interval_steps": 10000,
        "wandb_log_interval_steps": 10000,
        "action_diagnostics_interval_steps": 10000,
        "audit_diagnostic_interval": 100,
    }
    if spec.method == "sac_lag":
        params.update(shared)
        params.update(baselines.get("methods", {}).get("sac_lag", {}))
        params["method"] = "sac_lag"
        params["star_use_kl"] = False
        params["training_execution_mode"] = "raw"
        params["evaluation_execution_mode"] = "raw"
    resume = latest_resume_checkpoint(spec)
    if resume and not is_complete(spec):
        params["resume_checkpoint"] = str(resume)
        params["resume_run_dir"] = str(spec.result_dir)
    cmd = [PYTHON, "main_star.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def task_mapping_md() -> None:
    lines = ["# Task Mapping", ""]
    for name, env_id in TASKS.items():
        lines.append(f"- `{name}` -> `{env_id}`")
    (REPORT_ROOT / "setup" / "task_mapping.md").write_text("\n".join(lines) + "\n")


def method_unavailable_md() -> None:
    lines = [
        "# Method Availability",
        "",
        "Available in the current repo through `main_star.py` / `agents.star_agent.STARAgent`:",
    ]
    for key, label in AVAILABLE_METHODS.items():
        lines.append(f"- `{label}` (`{key}`)")
    lines.extend(
        [
            "",
            "Not launched because no matching implementation/training entry point was found in this checkout:",
        ]
    )
    for key, label in UNAVAILABLE_METHODS.items():
        lines.append(f"- `{label}` (`{key}`)")
    lines.extend(
        [
            "",
            "Evidence: `STARAgent.valid_methods` accepts `sac`, `pointwise`, `sac_lag`, `star_actor`, "
            "`star_exec`, `star`, `pointwise_v2`, `current_only_v2`, `star_v2`, and `star_collect_v2`; "
            "grep over configs/scripts/agents did not find runnable Safe Flow Q, PPO-Lag, CPO, or CSPO trainers.",
        ]
    )
    (REPORT_ROOT / "setup" / "method_unavailable.md").write_text("\n".join(lines) + "\n")


def doctor() -> None:
    ensure_dirs()
    task_mapping_md()
    method_unavailable_md()
    state = run(
        [
            "bash",
            "-lc",
            "{ echo \"doctor $(date)\"; hostname; git branch --show-current; git rev-parse HEAD; "
            "git status --short; /root/miniconda3/envs/flac/bin/python --version; nvidia-smi || true; df -h; }",
        ]
    ).stdout
    (REPORT_ROOT / "setup" / "doctor_state.txt").write_text(state)
    smoke = run(
        [
            "bash",
            "-lc",
            f"export PYTHONPATH={shlex.quote(str(REPO))} MUJOCO_GL=egl; {shlex.quote(PYTHON)} - <<'PY'\n"
            "import safety_gymnasium\n"
            f"tasks = {list(TASKS.values())!r}\n"
            "for task in tasks:\n"
            "    env = safety_gymnasium.make(task)\n"
            "    obs, info = env.reset(seed=0)\n"
            "    print(task, 'OK', 'obs', getattr(env.observation_space, 'shape', None), 'act', getattr(env.action_space, 'shape', None))\n"
            "    env.close()\n"
            "PY",
        ]
    ).stdout
    (REPORT_ROOT / "setup" / "task_smoke.txt").write_text(smoke)
    write_manifest()
    print(state)
    print(smoke)


def write_manifest() -> None:
    ensure_dirs()
    ps_text = ps_output()
    path = REPORT_ROOT / "run_manifest.csv"
    fields = [
        "stage",
        "task",
        "env_id",
        "method",
        "display_method",
        "seed",
        "steps",
        "status",
        "max_train_step",
        "run_name",
        "result_dir",
        "log_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in all_specs():
            writer.writerow(
                {
                    "stage": spec.stage,
                    "task": spec.task_name,
                    "env_id": spec.env_id,
                    "method": spec.method,
                    "display_method": AVAILABLE_METHODS.get(spec.method, spec.method),
                    "seed": spec.seed,
                    "steps": spec.steps,
                    "status": status_for_spec(spec, ps_text),
                    "max_train_step": max_train_step(spec),
                    "run_name": spec.run_name,
                    "result_dir": spec.result_dir,
                    "log_path": spec.log_path,
                }
            )


def tmux(args: list[str]) -> subprocess.CompletedProcess:
    return run(["tmux", *args])


def ensure_tmux() -> None:
    if tmux(["has-session", "-t", SESSION]).returncode != 0:
        tmux(["new-session", "-d", "-s", SESSION, "-n", "controller", "bash"])


def active_counts(ps_text: str) -> tuple[int, dict[int, int]]:
    total = 0
    by_gpu = {0: 0, 1: 0}
    for line in ps_text.splitlines():
        if "main_star.py" not in line:
            continue
        parts = line.strip().split(None, 1)
        pid = parts[0] if parts else ""
        cmd = parts[1] if len(parts) > 1 else ""
        if cmd.startswith(("bash ", "sh ", "/bin/bash ", "/bin/sh ")):
            continue
        total += 1
        gpu = 0
        env_path = Path("/proc") / pid / "environ"
        if env_path.exists():
            try:
                env = env_path.read_bytes().decode(errors="ignore").replace("\x00", "\n")
                if "CUDA_VISIBLE_DEVICES=1" in env:
                    gpu = 1
            except OSError:
                gpu = 0
        by_gpu[gpu] += 1
    return total, by_gpu


def launch_one(spec: RunSpec, gpu: int) -> None:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_main_star_command(spec, gpu)
    env_bits = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in {**THREAD_ENV, "CUDA_VISIBLE_DEVICES": gpu}.items())
    shell = (
        f"cd {shlex.quote(str(REPO))}; "
        f"{env_bits} PYTHONPATH={shlex.quote(str(REPO))} "
        f"{' '.join(shlex.quote(part) for part in cmd)} "
        f">> {shlex.quote(str(spec.log_path))} 2>&1"
    )
    window = spec.run_name[-80:]
    tmux(["new-window", "-t", SESSION, "-n", window, shell])


def launch_pending(stages: set[str], max_new: int = 6) -> int:
    ensure_dirs()
    ensure_tmux()
    ps_text = ps_output()
    total, by_gpu = active_counts(ps_text)
    capacity = sum(GPU_SLOTS.values())
    launched = 0
    for spec in all_specs():
        if spec.stage not in stages:
            continue
        if launched >= max_new:
            break
        if total + launched >= capacity:
            break
        spec_status = status_for_spec(spec, ps_text)
        if spec_status not in {"pending", "partial"}:
            continue
        gpu = min(GPU_SLOTS, key=lambda item: by_gpu[item])
        if by_gpu[gpu] >= GPU_SLOTS[gpu]:
            other = 1 - gpu
            if by_gpu[other] < GPU_SLOTS[other]:
                gpu = other
            else:
                break
        launch_one(spec, gpu)
        by_gpu[gpu] += 1
        launched += 1
        time.sleep(1)
    write_manifest()
    write_status_md()
    return launched


def stage_done(stages: set[str]) -> bool:
    ps_text = ps_output()
    for spec in all_specs():
        if spec.stage in stages and status_for_spec(spec, ps_text) != "completed":
            return False
    return True


def start_supervisor(stages: list[str], name: str) -> None:
    ensure_dirs()
    ensure_tmux()
    stage_arg = ",".join(stages)
    shell = (
        f"cd {shlex.quote(str(REPO))}; "
        f"PYTHONPATH={shlex.quote(str(REPO))} {shlex.quote(PYTHON)} "
        f"scripts/star/goal_1m_curves.py resume --stages {shlex.quote(stage_arg)} --loop "
        f">> {shlex.quote(str(LOG_ROOT / (name + '.log')))} 2>&1"
    )
    existing = tmux(["list-windows", "-t", SESSION, "-F", "#{window_name}"]).stdout
    if name not in existing:
        tmux(["new-window", "-t", SESSION, "-n", name, shell])


def resume(args: argparse.Namespace) -> None:
    stages = set(args.stages.split(",")) if args.stages else {"stage_a_star", "stage_b_baselines1"}
    while True:
        if "stage_b_baselines1" in stages and "stage_a_star" not in stages and not stage_done({"stage_a_star"}):
            write_status_md()
            if not args.loop:
                print("stage_b_baselines1 is waiting for stage_a_star to complete")
                break
            time.sleep(args.interval)
            continue
        launch_pending(stages, max_new=args.max_new)
        if not args.loop or stage_done(stages):
            break
        time.sleep(args.interval)


def scan_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        tail = path.read_text(errors="replace")[-12000:]
    except Exception:
        return ["unreadable_log"]
    return [pattern for pattern in ERROR_PATTERNS if pattern in tail]


def write_status_md() -> None:
    ensure_dirs()
    write_manifest()
    ps_text = ps_output()
    lines = ["# STAR 1M Curves Status", "", f"Updated: `{time.strftime('%F %T')}`", ""]
    rows = []
    progress_rows = []
    for spec in all_specs():
        status = status_for_spec(spec, ps_text)
        stats = train_progress_stats(spec)
        rows.append((spec, status, stats, scan_log(spec.log_path)))
        progress_rows.append(
            {
                "stage": spec.stage,
                "task": spec.task_name,
                "env_id": spec.env_id,
                "method": spec.method,
                "display_method": AVAILABLE_METHODS.get(spec.method, spec.method),
                "seed": spec.seed,
                "status": status,
                **stats,
            }
        )
    progress_path = REPORT_ROOT / "status" / "progress_summary.csv"
    progress_fields = [
        "stage",
        "task",
        "env_id",
        "method",
        "display_method",
        "seed",
        "status",
        "max_step",
        "episodes",
        "wall_clock_time",
        "overall_steps_per_sec",
        "recent_steps_per_sec",
        "eta_hours",
        "train_total_cost",
        "train_cost_rate",
    ]
    with progress_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=progress_fields)
        writer.writeheader()
        for row in progress_rows:
            writer.writerow(row)
    loadavg = run(["env", "LC_ALL=C", "LANG=C", "bash", "-lc", "cat /proc/loadavg; nproc"]).stdout.strip().splitlines()
    if loadavg:
        cpu_line = loadavg[0]
        nproc = loadavg[1] if len(loadavg) > 1 else "unknown"
        lines.extend(["## Host Load", "", f"- CPU loadavg: `{cpu_line}`", f"- nproc: `{nproc}`", ""])
    for stage in ("stage_a_star", "stage_b_baselines1"):
        subset = [(s, st, stats, errs) for s, st, stats, errs in rows if s.stage == stage]
        lines.append(f"## {stage}")
        counts: dict[str, int] = {}
        for _s, st, _stats, _errs in subset:
            counts[st] = counts.get(st, 0) + 1
        lines.append(" ".join(f"`{k}`={v}" for k, v in sorted(counts.items())) or "no specs")
        lines.append("")
        lines.append("| task | method | seed | status | max_step | recent steps/s | ETA h | errors |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---|")
        for spec, st, stats, errs in subset:
            recent = float(stats["recent_steps_per_sec"] or 0.0)
            overall = float(stats["overall_steps_per_sec"] or 0.0)
            eta = stats["eta_hours"]
            eta_text = f"{float(eta):.2f}" if isinstance(eta, float) else ""
            speed_text = f"{(recent or overall):.2f}" if (recent or overall) else ""
            lines.append(
                f"| {spec.task_name} | {AVAILABLE_METHODS.get(spec.method, spec.method)} | {spec.seed} | {st} | {stats['max_step']} | {speed_text} | {eta_text} | {', '.join(errs)} |"
            )
        lines.append("")
    resources = run(["bash", "-lc", "nvidia-smi || true; df -h"]).stdout
    lines.extend(["## Resources", "", "```", resources.strip(), "```", ""])
    (REPORT_ROOT / "status" / "status.md").write_text("\n".join(lines))


def status(_args: argparse.Namespace | None = None) -> None:
    write_status_md()
    print((REPORT_ROOT / "status" / "status.md").read_text())


def plan(_args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    method_unavailable_md()
    task_mapping_md()
    write_manifest()
    lines = [
        "# STAR 1M Training-Curve Plan",
        "",
        "Stage A launches STAR (`star_v2`) for 3 tasks x 5 seeds = 15 runs.",
        "Stage B launches SAC-Lag for 3 tasks x 3 seeds = 9 runs.",
        "Safe Flow Q, PPO-Lag, CPO, and CSPO are documented as unavailable in this checkout and are not faked.",
        "",
        "Use `python scripts/star/goal_1m_curves.py launch-star` first.",
        "Use `python scripts/star/goal_1m_curves.py launch-baselines1` after or alongside Stage A if resources allow.",
    ]
    (REPORT_ROOT / "setup" / "execution_plan.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def collect(_args: argparse.Namespace | None = None) -> None:
    proc = run([PYTHON, "scripts/star/collect_1m_curves.py"])
    print(proc.stdout)


def plot(_args: argparse.Namespace | None = None) -> None:
    collect(None)
    cmd = [PYTHON, "scripts/star/plot_1m_curves.py"]
    if _args is not None:
        if getattr(_args, "methods", ""):
            cmd.extend(["--methods", str(_args.methods)])
        if getattr(_args, "all", False):
            cmd.append("--all")
        if getattr(_args, "star_only", False):
            cmd.append("--star-only")
        if getattr(_args, "core", False):
            cmd.append("--core")
    proc = run(cmd)
    print(proc.stdout)


def package(_args: argparse.Namespace | None = None) -> None:
    collect(None)
    archive = Path("/root/star_1m_curves_package.tar.gz")
    include = [
        REPORT_ROOT / "run_manifest.csv",
        REPORT_ROOT / "setup",
        REPORT_ROOT / "status",
        REPORT_ROOT / "curves",
        REPORT_ROOT / "final",
        REPORT_ROOT / "figures",
        REPO / "scripts" / "star" / "goal_1m_curves.py",
        REPO / "scripts" / "star" / "collect_1m_curves.py",
        REPO / "scripts" / "star" / "plot_1m_curves.py",
        REPO / "scripts" / "star" / "finalize_1m_curves.py",
    ]
    with tarfile.open(archive, "w:gz") as tar:
        for path in include:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(REPO)))
    print(archive)


def eval_final(args: argparse.Namespace) -> None:
    cmd = [PYTHON, "scripts/star/finalize_1m_curves.py"]
    if getattr(args, "run_eval", False):
        cmd.append("--run-eval")
    if getattr(args, "overwrite_derived", False):
        cmd.append("--overwrite-derived")
    if getattr(args, "eval_seeds", ""):
        cmd.extend(["--eval-seeds", str(args.eval_seeds)])
    proc = run(cmd)
    print(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("plan")
    sub.add_parser("status")
    sub.add_parser("collect")
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--methods", default="")
    plot_parser.add_argument("--all", action="store_true")
    plot_parser.add_argument("--star-only", action="store_true")
    plot_parser.add_argument("--core", action="store_true")
    sub.add_parser("package")
    eval_parser = sub.add_parser("eval-final")
    eval_parser.add_argument("--run-eval", action="store_true")
    eval_parser.add_argument("--overwrite-derived", action="store_true")
    eval_parser.add_argument(
        "--eval-seeds",
        default="100000,100001,100002,100003,100004,100005,100006,100007,100008,100009",
    )
    launch_star_parser = sub.add_parser("launch-star")
    launch_star_parser.add_argument("--resume", action="store_true")
    launch_baselines1_parser = sub.add_parser("launch-baselines1")
    launch_baselines1_parser.add_argument("--resume", action="store_true")
    launch_baselines2_parser = sub.add_parser("launch-baselines2")
    launch_baselines2_parser.add_argument("--resume", action="store_true")
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--stages", default="stage_a_star,stage_b_baselines1")
    resume_parser.add_argument("--loop", action="store_true")
    resume_parser.add_argument("--interval", type=int, default=60)
    resume_parser.add_argument("--max-new", type=int, default=6)
    args = parser.parse_args()

    if args.command == "doctor":
        doctor()
    elif args.command == "plan":
        plan(args)
    elif args.command == "launch-star":
        start_supervisor(["stage_a_star"], "supervisor_stage_a")
        status(args)
    elif args.command == "launch-baselines1":
        start_supervisor(["stage_b_baselines1"], "supervisor_stage_b")
        status(args)
    elif args.command == "launch-baselines2":
        method_unavailable_md()
        print(REPORT_ROOT / "setup" / "method_unavailable.md")
    elif args.command == "resume":
        resume(args)
    elif args.command == "status":
        status(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "plot":
        plot(args)
    elif args.command == "package":
        package(args)
    elif args.command == "eval-final":
        eval_final(args)


if __name__ == "__main__":
    main()
