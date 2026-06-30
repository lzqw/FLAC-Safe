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


def rank_train_source(inv: dict) -> tuple[int, int, str]:
    name = Path(inv["file_path"]).name
    return (
        0 if name == "train_episodes.csv" else 1,
        0 if inv.get("source_type") == "csv" else 1,
        inv["file_path"],
    )


def rank_eval_source(inv: dict) -> tuple[int, int, str]:
    name = Path(inv["file_path"]).name
    return (
        0 if name in {"eval_episodes.csv", "corrected_eval_episodes.csv"} else 1,
        0 if inv.get("source_type") == "csv" else 1,
        inv["file_path"],
    )


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


def extract_training_csv(inv: dict) -> list[dict]:
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
    for row in rows:
        step = fnum(row.get(step_col))
        if math.isnan(step):
            continue
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
                "return_value": fnum(row.get(reward_col)) if reward_col else math.nan,
                "cost_value": cost,
                "cumulative_cost": cumulative,
                "source_file": str(path),
                "source_kind": "train",
            }
        )
    return out


def extract_eval_csv(inv: dict, train_rows: list[dict]) -> list[dict]:
    path = Path(inv["file_path"])
    rows = read_csv_rows(path)
    if not rows:
        return []
    step_col = inv.get("step_column", "")
    reward_col = choose(split_cols(inv.get("reward_columns", "")), PREFERRED_REWARD)
    cost_col = choose(split_cols(inv.get("cost_columns", "")), PREFERRED_COST)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        step = fnum(row.get(step_col))
        if not math.isnan(step):
            grouped[int(step)].append(row)
    if len(grouped) < 2:
        return []

    cost_by_step = []
    for row in train_rows:
        step = fnum(row.get("step"))
        cumulative = fnum(row.get("cumulative_cost"))
        cost = fnum(row.get("cost_value"))
        if not math.isnan(step):
            cost_by_step.append((int(step), cumulative, cost))
    cost_by_step.sort(key=lambda item: item[0])

    def nearest_training_cost(step: int) -> tuple[float, float]:
        cumulative = math.nan
        cost = math.nan
        for train_step, train_cumulative, train_cost in cost_by_step:
            if train_step > step:
                break
            cumulative = train_cumulative
            cost = train_cost
        return cumulative, cost

    out = []
    for step, step_rows in sorted(grouped.items()):
        returns = [fnum(row.get(reward_col)) for row in step_rows] if reward_col else []
        returns = [value for value in returns if not math.isnan(value)]
        costs = [fnum(row.get(cost_col)) for row in step_rows] if cost_col else []
        costs = [value for value in costs if not math.isnan(value)]
        cumulative, train_cost = nearest_training_cost(step)
        out.append(
            {
                "phase": inv["phase"],
                "task": inv["task"],
                "method": inv["method"],
                "seed": inv["seed"],
                "step": step,
                "return_value": sum(returns) / len(returns) if returns else math.nan,
                "cost_value": sum(costs) / len(costs) if costs else train_cost,
                "cumulative_cost": cumulative,
                "source_file": f"{path};training_cost={train_rows[0]['source_file'] if train_rows else ''}",
                "source_kind": "eval",
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
    sources_by_run: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for inv in inventory:
        if inv.get("usable") != "True":
            continue
        if inv.get("phase") != args.phase:
            continue
        if inv.get("task") not in TASKS or inv.get("method") not in METHODS or not inv.get("seed"):
            continue
        key = (inv["task"], inv["method"], inv["seed"])
        sources_by_run[key].append(inv)

    rows = []
    failures = []
    selected_sources = []
    source_notes = []
    for key, sources in sorted(sources_by_run.items()):
        try:
            train_candidates = [
                inv
                for inv in sources
                if Path(inv["file_path"]).name == "train_episodes.csv"
                or (inv.get("source_type") != "csv" and "train" in Path(inv["file_path"]).name.lower())
            ]
            eval_candidates = [
                inv
                for inv in sources
                if Path(inv["file_path"]).name in {"eval_episodes.csv", "corrected_eval_episodes.csv"}
            ]
            train_inv = sorted(train_candidates or sources, key=rank_train_source)[0]
            selected_sources.append(train_inv)
            if train_inv.get("source_type") == "csv":
                train_rows = extract_training_csv(train_inv)
            elif train_inv.get("source_type") == "tensorboard":
                train_rows = extract_tensorboard_curve(train_inv)
            else:
                train_rows = extract_log_curve(train_inv)
            if not train_rows:
                failures.append((*key, "extracted_zero_rows"))
                continue

            periodic_eval_rows = []
            if eval_candidates:
                eval_inv = sorted(eval_candidates, key=rank_eval_source)[0]
                selected_sources.append(eval_inv)
                if eval_inv.get("source_type") == "csv":
                    periodic_eval_rows = extract_eval_csv(eval_inv, train_rows)
                if periodic_eval_rows:
                    rows.extend(periodic_eval_rows)
                    source_notes.append((*key, "periodic_eval_return"))
                else:
                    rows.extend(train_rows)
                    source_notes.append((*key, "training_return_eval_file_not_periodic"))
            else:
                rows.extend(train_rows)
                source_notes.append((*key, "training_return_no_eval_curve"))
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

    present = {(task, method, int(seed)) for task, method, seed in sources_by_run}
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
    ranges: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"])] += 1
        source_kind[row["source_kind"]] += 1
        ranges[(row["task"], row["method"], row["seed"])].append(int(row["step"]))
    summary = [
        "# STAR-v2 Curve Extraction Summary",
        "",
        f"- phase: `{args.phase}`",
        f"- selected_sources: `{len(selected_sources)}`",
        f"- extracted_rows: `{len(rows)}`",
        f"- source_kind_counts: `{dict(source_kind)}`",
        "",
        "## Source Notes",
        "",
    ]
    for task, method, seed, note in sorted(source_notes):
        summary.append(f"- {task} {method} seed={seed}: {note}")
    summary.extend([
        "",
        "## Step Ranges",
        "",
    ])
    for (task, method, seed), steps in sorted(ranges.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        summary.append(f"- {task} {method} seed={seed}: {min(steps)}-{max(steps)} ({len(steps)} rows)")
    summary.extend([
        "",
        "## Rows By Task/Method",
        "",
    ])
    for (task, method), count in sorted(grouped.items()):
        summary.append(f"- {task} {method}: {count}")
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {args.output} rows={len(rows)} sources={len(sources_by_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
