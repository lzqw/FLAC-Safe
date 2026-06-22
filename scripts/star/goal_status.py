from __future__ import annotations

import csv
import subprocess
from pathlib import Path


MANIFEST = Path("reports/star_goal/experiment_manifest.csv")


def running_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in proc.stdout.splitlines()}


def main() -> None:
    sessions = running_sessions()
    print("Active star_goal sessions:")
    for session in sorted(s for s in sessions if s.startswith("star_goal_")):
        print(f"  {session}")
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}")
        return
    print("\nManifest:")
    for row in csv.DictReader(MANIFEST.open()):
        status = "running" if row.get("session") in sessions else row.get("status", "launched")
        print(
            f"{status:10s} gpu={row.get('gpu')} task={row.get('task')} method={row.get('method')} "
            f"seed={row.get('seed')} session={row.get('session')} log={row.get('log_path')}"
        )


if __name__ == "__main__":
    main()
