#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from goal_baseline_screen import all_jobs, has_error, is_complete, result_root


REPORT = Path("reports/star_goal")
DONE_MARKER = REPORT / "baseline_screen_postprocess.done"
EVAL_SEEDS = "200000,200001,200002,200003,200004,200005,200006,200007,200008,200009"


def sh(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def counts() -> tuple[int, int, int]:
    jobs = all_jobs()
    return sum(is_complete(job) for job in jobs), sum(has_error(job) for job in jobs), len(jobs)


def write_status(stage: str, note: str = "") -> None:
    completed, failed, total = counts()
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "baseline_screen_status.md"
    lines = [
        "# Baseline Screen Status",
        "",
        f"- Updated: {datetime.now().isoformat()}",
        f"- Stage: {stage}",
        f"- Completed: {completed}/{total}",
        f"- Failed: {failed}",
        f"- Result root: `{result_root()}`",
    ]
    if note:
        lines.extend(["", "## Note", "", note])
    path.write_text("\n".join(lines) + "\n")


def run_postprocess() -> None:
    write_status("postprocess_running", "Running raw-only reevaluation and baseline selector.")
    sh([
        "python",
        "scripts/star/reevaluate_checkpoints.py",
        "--root",
        result_root(),
        "--eval-seeds",
        EVAL_SEEDS,
        "--checkpoint-selector",
        "final",
        "--modes",
        "raw",
        "--overwrite-derived",
    ])
    sh([
        "python",
        "scripts/star/collect_baseline_screen_results.py",
        "--root",
        result_root(),
        "--report-dir",
        str(REPORT),
    ])
    DONE_MARKER.write_text(datetime.now().isoformat() + "\n")
    write_status("postprocess_complete", "Baseline configs selected for paper-support 300k if both methods have valid candidates.")


def wait_loop(poll_seconds: int) -> None:
    write_status("running", "Waiting for all baseline-screen runs to complete.")
    while True:
        if DONE_MARKER.exists():
            print(f"done marker exists: {DONE_MARKER}", flush=True)
            return
        completed, failed, total = counts()
        print(f"[{datetime.now().isoformat()}] completed={completed}/{total} failed={failed}", flush=True)
        if failed:
            write_status("failed", "At least one baseline-screen run failed. Postprocess did not run.")
            return
        if completed >= total:
            run_postprocess()
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--run-now", action="store_true")
    args = parser.parse_args()
    if args.run_now:
        completed, failed, total = counts()
        if failed:
            raise SystemExit(f"refusing postprocess: failed={failed}")
        if completed < total:
            raise SystemExit(f"refusing postprocess: completed={completed}/{total}")
        run_postprocess()
    else:
        wait_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
