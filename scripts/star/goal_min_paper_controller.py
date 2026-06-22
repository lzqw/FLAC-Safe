#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPORT = Path("reports/star_goal")
STAGE1_DONE = REPORT / "stage1_postprocess.done"
STATUS = REPORT / "paper_support_status.md"


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def write_status(stage: str, note: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# STAR Minimum Paper Evidence Status",
        "",
        f"- Updated: {datetime.now().isoformat()}",
        f"- Stage: {stage}",
        f"- Stage1 postprocess done: {STAGE1_DONE.exists()}",
        "",
        "## Current Gate State",
        "",
        "- Gate A Shadow audit: pending or derived from `shadow_risk_summary.csv`",
        "- Gate B Raw actor safer: pending 300k decisive comparison",
        "- Gate C Executor benefit: pending same-checkpoint executor grid",
        "- Gate D Corridor mechanism: pending corridor-vs-current-only evidence",
        "",
        "## Notes",
        "",
        note,
        "",
    ]
    STATUS.write_text("\n".join(lines))


def sh(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def stage1_requires_risk_scale() -> bool:
    threshold_file = REPORT / "stage1_decision_thresholds.json"
    decision_md = REPORT / "stage1_decision.md"
    if threshold_file.exists():
        data = json.loads(threshold_file.read_text())
        return bool(data.get("pSVR_rho_active_saturated", False))
    if decision_md.exists():
        text = decision_md.read_text(errors="ignore")
        return "Risk-scale calibration is required" in text
    raise FileNotFoundError("missing stage1 decision outputs")


def launch_risk_scale_if_needed() -> None:
    sessions = tmux_sessions()
    if any(session.startswith("star_goal_risk_scale") for session in sessions):
        write_status("risk_scale_running", "Risk-scale calibration is already running.")
        return
    sh(["python", "scripts/star/goal_risk_scale_calibration.py", "--launch"])
    write_status("risk_scale_launched", "Stage1 indicated saturated risk scale; launched risk-scale calibration.")


def launch_baseline_if_needed() -> None:
    if not (REPORT / "selected_actor_config.json").exists():
        write_status("waiting_selected_actor_config", "Stage1 did not produce `selected_actor_config.json`; manual review is required before baseline screen.")
        return
    sessions = tmux_sessions()
    if any(session.startswith("star_goal_baseline_screen") for session in sessions):
        write_status("baseline_screen_running", "Baseline screen or its postprocess watcher is already running.")
        return
    sh(["python", "scripts/star/goal_baseline_screen.py", "--launch"])
    sh([
        "tmux",
        "new",
        "-d",
        "-s",
        "star_goal_baseline_screen_postprocess",
        "cd /root/FLAC-Safe && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac && python scripts/star/goal_baseline_screen_postprocess.py --poll-seconds 300 2>&1 | tee logs/star_goal/baseline_screen_postprocess.log",
    ])
    write_status("baseline_screen_launched", "Stage1 selected an actor config; launched baseline screen and postprocess watcher.")


def step_once() -> bool:
    if not STAGE1_DONE.exists():
        write_status("waiting_stage1", "Controller is waiting for Stage1 postprocess to complete.")
        return False
    if stage1_requires_risk_scale():
        launch_risk_scale_if_needed()
    else:
        launch_baseline_if_needed()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            done = step_once()
        except Exception as exc:
            write_status("controller_error", f"Controller error: {exc}")
            raise
        if args.once or done:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
