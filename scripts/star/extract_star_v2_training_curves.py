#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


TASKS = ["SafetyPointGoal1-v0", "SafetyCarButton1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"]
METHODS = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]
FINAL_SEEDS = [10, 11, 12, 13, 14]
PREFERRED_REWARD = [
    "eval_return",
    "raw_return",
    "train_episode_reward",
    "train_return",
    "episode_reward",
    "reward",
    "return",
    "EpRet",
]
PREFERRED_COST = [
    "episode_cost",
    "train_episode_cost",
    "train_cost",
    "eval_cost",
    "raw_cost",
    "cost",
    "EpCost",
]
CUMULATIVE_COST = ["train_total_cost", "cumulative_cost", "total_cost"]


def fnum(value, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def split_cols(value: str) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def choose(cols: list[str], preferred: list[str]) -> str:
    for name in preferred:
        if name in cols:
            return name
    return cols[0] if cols else ""


def extract_csv_curve(inv: dict) -> list[dict]:
    path = Path(inv["file_path"])
    rows = read_csv_rows(path)
    if not rows:
        return []
    reward_cols = split_cols(inv.get("reward_columns", ""))
    cost_cols = split_cols(inv.get("cost_columns", ""))
    step_col = inv.get("step_column", "")
    reward_col = choose(reward_cols, PREFERRED_REWARD)
    cost_col = choose([c for c in cost_cols if c not in CUMULATIVE_COST], PREFERRED_COST)
    cumulative_col = choose([c for c in cost_cols if c in CUMULATIVE_COST], CUMULATIVE_COST)
    out = []
    running_cost = 0.0
    source_kind = "eval" if "eval" in path.name or reward_col.startswith(("eval_", "raw_")) else "train"
    for row in rows:
        step = fnum(row.get(step_col))
        if math.isnan(step):
            continue
        ret = fnum(row.get(reward_col)) if reward_col else math.nan
        cost = fnum(row.get(cost_col)) if cost_col else math.nan
        if cumulative_col:
            cumulative = fnum(row.get(cumulative_col))
        else:
            if not math.isnan(cost):
                running_cost += cost
            cumulative = running_cost if running_cost else math.nan
        out.append(
            {
                "phase": inv["phase"],
                "task": inv["task"],
                "method": inv["method"],
                "seed": inv["seed"],
                "step": int(step),
                "return_value": ret,
                "cost_value": cost,
                "cumulative_cost": cumulative,
                "source_file": str(path),
                "source_kind": source_kind,
            }
        )
    return out


def extract_log_curve(inv: dict) -> list[dict]:
    path = Path(inv["file_path"])
    out = []
    running_cost = 0.0
    pattern = re.compile(r"(?P<key>step|global_step|env_step|end_step|episode_reward|reward|return|episode_cost|cost|train_total_cost)=(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
    for line in path.read_text(errors="ignore").splitlines():
        vals = {m.group("key").lower(): fnum(m.group("value")) for m in pattern.finditer(line)}
        step = vals.get("step", vals.get("global_step", vals.get("env_step", vals.get("end_step", math.nan))))
        if math.isnan(step):
            continue
        ret = vals.get("episode_reward", vals.get("reward", vals.get("return", math.nan)))
        cost = vals.get("episode_cost", vals.get("cost", math.nan))
        cumulative = vals.get("train_total_cost", math.nan)
        if math.isnan(cumulative):
            if not math.isnan(cost):
                running_cost += cost
            cumulative = running_cost if running_cost else math.nan
        out.append(
            {
                "phase": inv["phase"],
                "task": inv["task"],
                "method": inv["method"],
                "seed": inv["seed"],
                "step": int(step),
                "return_value": ret,
                "cost_value": cost,
                "cumulative_cost": cumulative,
                "source_file": str(path),
                "source_kind": "train",
            }
        )
    return out


def extract_tensorboard_curve(inv: dict) -> list[dict]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return []
    path = Path(inv["file_path"])
    ea = event_accumulator.EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    reward_tag = choose(split_cols(inv.get("reward_columns", "")), PREFERRED_REWARD)
    cost_tag = choose(split_cols(inv.get("cost_columns", "")), PREFERRED_COST + CUMULATIVE_COST)
    rewards = {event.step: event.value for event in ea.Scalars(reward_tag)} if reward_tag else {}
    costs = {event.step: event.value for event in ea.Scalars(cost_tag)} if cost_tag else {}
    out = []
    running_cost = 0.0
    for step in sorted(set(rewards) | set(costs)):
        cost = costs.get(step, math.nan)
        if cost_tag in CUMULATIVE_COST:
            cumulative = cost
        else:
            if not math.isnan(cost):
                running_cost += cost
            cumulative = running_cost if running_cost else math.nan
        out.append(
            {
                "phase": inv["phase"],
                "task": inv["task"],
                "method": inv["method"],
                "seed": inv["seed"],
                "step": int(step),
                "return_value": rewards.get(step, math.nan),
                "cost_value": cost,
                "cumulative_cost": cumulative,
                "source_file": str(path),
                "source_kind": "train",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="reports/star_v2_final/curve_audit/curve_source_inventory.csv")
    parser.add_argument("--output", default="reports/star_v2_final/curves/training_curves_long.csv")
    parser.add_argument("--missing-output", default="reports/star_v2_final/curves/missing_curve_sources.md")
    parser.add_argument("--summary-output", default="reports/star_v2_final/curve_audit/curve_extraction_summary.md")
    parser.add_argument("--phase", default="resume_300k")
    args = parser.parse_args()

    inventory = read_csv_rows(Path(args.inventory)) if Path(args.inventory).exists() else []
    selected_by_run: dict[tuple[str, str, str], dict] = {}
    for inv in inventory:
        if inv.get("usable") != "True":
            continue
        if inv.get("phase") != args.phase:
            continue
        if inv.get("task") not in TASKS or inv.get("method") not in METHODS or not inv.get("seed"):
            continue
        key = (inv["task"], inv["method"], inv["seed"])
        current = selected_by_run.get(key)
        if current is None:
            selected_by_run[key] = inv
            continue
        # Prefer train_episodes.csv, then CSV, then anything else.
        old_rank = (Path(current["file_path"]).name != "train_episodes.csv", current.get("source_type") != "csv")
        new_rank = (Path(inv["file_path"]).name != "train_episodes.csv", inv.get("source_type") != "csv")
        if new_rank < old_rank:
            selected_by_run[key] = inv

    rows = []
    failures = []
    for key, inv in sorted(selected_by_run.items()):
        try:
            source_type = inv.get("source_type", "")
            if source_type == "csv":
                extracted = extract_csv_curve(inv)
            elif source_type == "tensorboard":
                extracted = extract_tensorboard_curve(inv)
            else:
                extracted = extract_log_curve(inv)
            if not extracted:
                failures.append((*key, "extracted_zero_rows"))
            rows.extend(extracted)
        except Exception as exc:
            failures.append((*key, f"extract_error:{exc}"))

    fields = [
        "phase",
        "task",
        "method",
        "seed",
        "step",
        "return_value",
        "cost_value",
        "cumulative_cost",
        "source_file",
        "source_kind",
    ]
    rows.sort(key=lambda r: (r["task"], r["method"], int(r["seed"]), int(r["step"])))
    write_csv(Path(args.output), rows, fields)

    present = {(task, method, int(seed)) for task, method, seed in selected_by_run}
    missing_lines = ["# Missing STAR-v2 Training Curve Sources", ""]
    for task in TASKS:
        for method in METHODS:
            for seed in FINAL_SEEDS:
                if (task, method, seed) not in present:
                    missing_lines.append(f"- {args.phase} {task} {method} seed={seed}: missing usable curve source")
    for task, method, seed, reason in failures:
        missing_lines.append(f"- {args.phase} {task} {method} seed={seed}: {reason}")
    if len(missing_lines) == 2:
        missing_lines.append("No missing final-300k curve sources detected for expected tasks/methods/seeds.")
    Path(args.missing_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.missing_output).write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

    grouped = defaultdict(int)
    source_kind = defaultdict(int)
    for row in rows:
        grouped[(row["task"], row["method"])] += 1
        source_kind[row["source_kind"]] += 1
    summary = [
        "# STAR-v2 Curve Extraction Summary",
        "",
        f"- phase: `{args.phase}`",
        f"- selected_sources: `{len(selected_by_run)}`",
        f"- extracted_rows: `{len(rows)}`",
        f"- source_kind_counts: `{dict(source_kind)}`",
        "",
        "## Rows By Task/Method",
        "",
    ]
    for (task, method), count in sorted(grouped.items()):
        summary.append(f"- {task} {method}: {count}")
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {args.output} rows={len(rows)} sources={len(selected_by_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
