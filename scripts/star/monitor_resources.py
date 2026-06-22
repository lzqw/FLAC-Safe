from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def tmux_sessions(prefix: str) -> list[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True)
    if proc.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in proc.stdout.splitlines() if line.startswith(prefix)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/star_goal/resource_monitor.csv")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--prefix", default="star_goal_")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["wall_time", "index", "name", "util_gpu", "memory_used", "memory_total", "power", "temperature"])
        while tmux_sessions(args.prefix):
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
            )
            now = time.time()
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    writer.writerow([now] + [part.strip() for part in line.split(",")])
                f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
