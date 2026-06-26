#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def disk_row(path: str) -> tuple[str, bool, float, str]:
    p = Path(path)
    if not p.exists():
        return path, False, -1.0, "missing"
    try:
        st = os.statvfs(str(p))
        free_gb = st.f_bavail * st.f_frsize / 1024**3
        return path, True, free_gb, "ok"
    except Exception as exc:
        return path, False, -1.0, f"{type(exc).__name__}: {exc}"


def write_test(path: str) -> bool:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".starv2_write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception as exc:
        print(f"write_probe_failed path={path} error={type(exc).__name__}:{exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Warn-only STAR-v2 disk monitor")
    parser.add_argument("--paths", nargs="*", default=["/root", "/root/autodl-tmp"])
    parser.add_argument("--storage-root", default="/root/autodl-tmp/star_v2_storage")
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--fatal-on-write-failure", action="store_true")
    args = parser.parse_args()

    print("path,exists,free_gb,status")
    for path in args.paths:
        row = disk_row(path)
        print(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]}")
    ok = write_test(args.storage_root)
    try:
        st = os.statvfs(args.storage_root)
        free_gb = st.f_bavail * st.f_frsize / 1024**3
    except Exception:
        free_gb = -1.0
    print(f"storage_root={args.storage_root}")
    print(f"storage_root_write_ok={ok}")
    print(f"storage_root_free_gb={free_gb:.2f}")
    if args.fatal_on_write_failure and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
