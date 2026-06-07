#!/usr/bin/env python3
"""Short G4_fixed_main transfer smoke runs for Safety-Gymnasium env names."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import safety_gymnasium  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "transfer_g4_fixed"
STATUS_FILE = LOG_DIR / "smoke_status.tsv"

SMOKE_STEPS = int(os.environ.get("SMOKE_NUM_STEPS", "3000"))
SMOKE_START_STEPS = int(os.environ.get("SMOKE_START_STEPS", "1000"))


def env_registered(env_id: str) -> bool:
    return env_id in gym.registry


def resolve_swimmer() -> str | None:
    if env_registered("SafetySwimmerVelocity-v1"):
        return "SafetySwimmerVelocity-v1"
    if env_registered("SafetySwimmerVelocity-v0"):
        return "SafetySwimmerVelocity-v0"
    return None


def run_smoke(label: str, task: str) -> tuple[str, int]:
    log_path = LOG_DIR / f"smoke_{label}.log"
    cmd = [
        sys.executable,
        "main.py",
        "--task",
        task,
        "--safe_env",
        "True",
        "--safe_policy_loss",
        "True",
        "--safety_critic_mode",
        "cdf",
        "--qc_geom_mode",
        "mean",
        "--safe_threshold",
        "0.05",
        "--lambda_safe",
        "0.7",
        "--lambda_jvp",
        "0.003",
        "--safe_bandwidth",
        "0.05",
        "--normalize_jvp",
        "True",
        "--jvp_norm_mode",
        "exact",
        "--jvp_mode",
        "grad",
        "--cdf_binarize_cost",
        "True",
        "--cdf_target_clip",
        "True",
        "--batch_size",
        "512",
        "--updates_per_step",
        "1",
        "--hidden_size",
        "512",
        "--num_steps",
        str(SMOKE_STEPS),
        "--start_steps",
        str(SMOKE_START_STEPS),
        "--eval",
        "False",
        "--distributional_critic",
        "False",
        "--compile_model",
        "False",
        "--soft_normal_masking",
        "False",
        "--directional_ref_noise",
        "False",
        "--epsilon",
        "0.0",
        "--save",
        "False",
        "--steps",
        "1",
        "--seed",
        "0",
        "--algo",
        "MF_SCTD_G4_Fixed_TransferSmoke",
        "--tag",
        f"smoke_{label}_G4_fixed_main",
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write(f"===== START smoke {label} task={task} =====\n")
        handle.write("Command: " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        rc = proc.wait()
        handle.write(f"===== END smoke {label} task={task} rc={rc} =====\n")
    return str(log_path), rc


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, str | None, bool]] = [
        ("PointGoal2", "SafetyPointGoal2-v0", False),
        ("CarGoal1", "SafetyCarGoal1-v0", False),
        ("CarGoal2", "SafetyCarGoal2-v0", True),
        ("SwimmerVelocity", resolve_swimmer(), True),
    ]
    rows: list[tuple[str, str, str, str]] = []
    failed = False
    for label, task, optional in specs:
        if task is None or not env_registered(task):
            status = "skipped_missing_optional" if optional else "missing_required"
            rows.append((label, task or "n/a", status, "n/a"))
            if not optional:
                failed = True
            continue
        log_path, rc = run_smoke(label, task)
        status = "passed" if rc == 0 else "failed"
        rows.append((label, task, status, log_path))
        if rc != 0:
            failed = True

    with STATUS_FILE.open("w") as handle:
        handle.write("label\ttask\tstatus\tlog\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")

    for label, task, status, log_path in rows:
        print(f"{label}\t{task}\t{status}\t{log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
