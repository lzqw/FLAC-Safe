#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from goal_actor_stage1 import all_jobs, has_error, is_complete


ROOT = Path("results/star_tune_actor_stage1_3393af8")
REPORT = Path("reports/star_goal")
DONE_MARKER = REPORT / "stage1_postprocess.done"
STATUS_FILE = REPORT / "paper_support_status.md"
EVAL_SEEDS_10 = "200000,200001,200002,200003,200004,200005,200006,200007,200008,200009"
EVAL_SEEDS_5 = "200000,200001,200002,200003,200004"


def sh(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def counts() -> tuple[int, int, int]:
    jobs = all_jobs()
    completed = sum(is_complete(job) for job in jobs)
    failed = sum(has_error(job) for job in jobs)
    return completed, failed, len(jobs)


def write_status(stage: str, extra: str = "") -> None:
    completed, failed, total = counts()
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# STAR Minimum Paper Evidence Status",
        "",
        f"- Updated: {datetime.now().isoformat()}",
        f"- Stage: {stage}",
        f"- Stage1 completed runs: {completed}/{total}",
        f"- Stage1 failed runs: {failed}",
        f"- Result root: `{ROOT}`",
        f"- Report root: `{REPORT}`",
        "",
        "## Current Gate State",
        "",
        "- Gate A Shadow audit: pending shadow-risk diagnostics",
        "- Gate B Raw actor safer: pending 300k decisive comparison",
        "- Gate C Executor benefit: pending same-checkpoint executor grid",
        "- Gate D Corridor mechanism: pending corridor-vs-current-only evidence",
        "",
    ]
    if extra:
        lines.extend(["## Notes", "", extra, ""])
    STATUS_FILE.write_text("\n".join(lines))


def run_postprocess() -> None:
    write_status("stage1_postprocess_running", "Running raw-only final reevaluation, shadow risk diagnostics, and actor stage1 collection.")
    sh(["python", "scripts/star/goal_actor_stage1.py", "--collect"])
    sh([
        "python",
        "scripts/star/reevaluate_checkpoints.py",
        "--root",
        str(ROOT),
        "--eval-seeds",
        EVAL_SEEDS_10,
        "--checkpoint-selector",
        "final",
        "--modes",
        "raw",
        "--overwrite-derived",
    ])
    sh([
        "python",
        "scripts/star/diagnose_shadow_risk.py",
        "--root",
        str(ROOT),
        "--eval-seeds",
        EVAL_SEEDS_5,
        "--max-states-per-run",
        "1000",
        "--report-dir",
        str(REPORT),
    ])
    sh([
        "python",
        "scripts/star/collect_actor_stage1_results.py",
        "--root",
        str(ROOT),
        "--report-dir",
        str(REPORT),
    ])
    DONE_MARKER.write_text(datetime.now().isoformat() + "\n")
    write_status("stage1_postprocess_complete", "Stage1 offline raw evaluation and shadow-risk diagnostics completed. Next step is candidate filtering and possible risk-scale calibration.")


def wait_loop(poll_seconds: int) -> None:
    write_status("stage1_running", "Postprocess watcher is active and waiting for all 24 Stage1 runs to finish naturally.")
    while True:
        if DONE_MARKER.exists():
            print(f"done marker exists: {DONE_MARKER}", flush=True)
            return
        completed, failed, total = counts()
        print(f"[{datetime.now().isoformat()}] completed={completed}/{total} failed={failed}", flush=True)
        if failed:
            write_status("stage1_failed", "At least one Stage1 run has an error. Postprocess did not run.")
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
