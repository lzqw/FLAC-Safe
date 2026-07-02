#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / "results" / "star_1m_curves"
REPORT_ROOT = REPO / "reports" / "star_1m_curves"
CURVE_ROOT = REPORT_ROOT / "curves"

TASKS = {
    "PointGoal1": "SafetyPointGoal1-v0",
    "CarGoal1": "SafetyCarGoal1-v0",
    "PointPush1": "SafetyPointPush1-v0",
}
EXPECTED = [
    ("stage_a_star", env_id, "star_v2", seed)
    for env_id in TASKS.values()
    for seed in range(5)
] + [
    ("stage_b_baselines1", env_id, "sac_lag", seed)
    for env_id in TASKS.values()
    for seed in range(3)
]
METHOD_LABEL = {
    "star_v2": "STAR",
    "sac_lag": "SAC-Lag",
}


def read_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def infer_from_path(path: Path) -> dict:
    parts = path.parts
    data: dict[str, object] = {}
    for env_id in TASKS.values():
        if env_id in parts:
            data["task"] = env_id
    for method in METHOD_LABEL:
        if method in parts:
            data["method"] = method
    match = re.search(r"_s(\d+)(?:_|$)", str(path))
    if match:
        data["seed"] = int(match.group(1))
    return data


def inventory_row(path: Path, usable: bool, reason: str = "") -> dict:
    run_dir = path.parent
    meta = read_metadata(run_dir)
    inferred = infer_from_path(path)
    task = str(meta.get("task") or inferred.get("task") or "")
    method = str(meta.get("method") or inferred.get("method") or "")
    seed = meta.get("seed", inferred.get("seed", ""))
    columns = []
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # noqa: BLE001
        usable = False
        reason = reason or f"read_error:{exc}"
    return {
        "file_path": str(path),
        "run_dir": str(run_dir),
        "stage": str(meta.get("ablation_group") or inferred.get("stage") or ""),
        "task": task,
        "method": method,
        "seed": seed,
        "source_type": "csv",
        "step_column": "end_step" if "end_step" in columns else "step" if "step" in columns else "",
        "reward_columns": ";".join([c for c in columns if c in {"episode_reward", "train_return", "return_value"}]),
        "cost_columns": ";".join([c for c in columns if c in {"episode_cost", "train_total_cost", "cost_value", "cumulative_cost"}]),
        "usable": usable,
        "reason": reason,
    }


def collect_train_file(path: Path) -> tuple[list[dict], dict]:
    inv = inventory_row(path, usable=False)
    required = {"end_step", "episode_reward", "episode_cost"}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        inv["reason"] = f"read_error:{exc}"
        return [], inv
    if not required.issubset(df.columns):
        inv["reason"] = f"missing_columns:{sorted(required - set(df.columns))}"
        return [], inv
    meta = read_metadata(path.parent)
    inferred = infer_from_path(path)
    env_id = str(meta.get("task") or inferred.get("task") or "")
    method = str(meta.get("method") or inferred.get("method") or "")
    seed = int(meta.get("seed", inferred.get("seed", -1)))
    rows: list[dict] = []
    cumulative = 0.0
    for _, row in df.sort_values("end_step").iterrows():
        if "train_total_cost" in df.columns and pd.notna(row.get("train_total_cost")):
            cumulative = float(row["train_total_cost"])
        else:
            cumulative += float(row["episode_cost"])
        rows.append(
            {
                "task": next((name for name, candidate in TASKS.items() if candidate == env_id), env_id),
                "env_id": env_id,
                "method": METHOD_LABEL.get(method, method),
                "method_key": method,
                "seed": seed,
                "step": int(row["end_step"]),
                "return_value": float(row["episode_reward"]),
                "cost_value": float(row["episode_cost"]),
                "cumulative_cost": cumulative,
                "success": row.get("success", ""),
                "source_file": str(path),
                "source_kind": "train_episode",
            }
        )
    inv["usable"] = bool(rows)
    inv["reason"] = "" if rows else "empty_train_file"
    return rows, inv


def write_missing(curves: pd.DataFrame) -> None:
    lines = ["# Missing 1M Curve Sources", ""]
    available = set()
    if not curves.empty:
        for _, row in curves[["env_id", "method_key", "seed"]].drop_duplicates().iterrows():
            available.add((str(row["env_id"]), str(row["method_key"]), int(row["seed"])))
    for _stage, env_id, method, seed in EXPECTED:
        if (env_id, method, seed) not in available:
            lines.append(f"- `{env_id}` `{METHOD_LABEL.get(method, method)}` seed `{seed}`: no usable `train_episodes.csv` yet.")
    unavailable = REPORT_ROOT / "setup" / "method_unavailable.md"
    if unavailable.exists():
        lines.extend(["", "## Methods not launched by this repo", "", unavailable.read_text()])
    (CURVE_ROOT / "missing_curve_sources.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    CURVE_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    inventory: list[dict] = []
    for path in sorted(RESULT_ROOT.rglob("train_episodes.csv")):
        rows, inv = collect_train_file(path)
        all_rows.extend(rows)
        inventory.append(inv)
    curves = pd.DataFrame(all_rows)
    if not curves.empty:
        curves = curves.sort_values(["env_id", "method", "seed", "step"])
    columns = [
        "task",
        "env_id",
        "method",
        "method_key",
        "seed",
        "step",
        "return_value",
        "cost_value",
        "cumulative_cost",
        "success",
        "source_file",
        "source_kind",
    ]
    curves.to_csv(CURVE_ROOT / "training_curves_long.csv", index=False, columns=columns)
    inv_columns = [
        "file_path",
        "run_dir",
        "stage",
        "task",
        "method",
        "seed",
        "source_type",
        "step_column",
        "reward_columns",
        "cost_columns",
        "usable",
        "reason",
    ]
    with (CURVE_ROOT / "curve_source_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=inv_columns)
        writer.writeheader()
        for row in inventory:
            writer.writerow({key: row.get(key, "") for key in inv_columns})
    write_missing(curves)
    print(f"wrote {len(curves)} curve rows from {len(inventory)} train files")


if __name__ == "__main__":
    main()
