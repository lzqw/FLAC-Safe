#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPORT = Path("reports/star_goal")
ERROR_RE = re.compile(
    r"No space left|Traceback|RuntimeError|OOM|out of memory|CUDA error|Segmentation fault|KeyboardInterrupt|invalid loss",
    re.IGNORECASE,
)


def latest_root() -> Path:
    roots = sorted(Path("results").glob("star_baseline_screen_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not roots:
        return Path("results/star_baseline_screen_missing")
    return roots[-1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=math.nan) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def finite(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def avg(values: list[float]) -> float:
    vals = finite(values)
    return mean(vals) if vals else math.nan


def fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def latest_by_step(rows: list[dict], step_key: str) -> list[dict]:
    if not rows:
        return []
    step = max(to_int(row.get(step_key)) for row in rows)
    return [row for row in rows if to_int(row.get(step_key)) == step]


def tail_avg(rows: list[dict], key: str, n: int = 3) -> float:
    return avg([to_float(row.get(key)) for row in rows[-n:]])


def scan_error(run_dir: Path, log_root: Path | None, run_name: str) -> tuple[bool, str]:
    paths = list(run_dir.rglob("*.log"))
    if log_root and log_root.exists():
        paths.extend(path for path in log_root.rglob("*") if path.is_file() and run_name in path.name)
    for path in sorted(set(paths)):
        try:
            for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
                if ERROR_RE.search(line):
                    return True, f"{path}:{lineno}:{line[:180]}"
        except OSError:
            continue
    return False, ""


def eval_summary(rows: list[dict], mode: str = "raw") -> dict:
    selected = latest_by_step([row for row in rows if row.get("mode") == mode], "global_step")
    if not selected:
        return {}
    return {
        "raw_return": avg([to_float(row.get("episode_reward", row.get("episode_return"))) for row in selected]),
        "raw_cost": avg([to_float(row.get("episode_cost")) for row in selected]),
        "raw_evr": avg([to_float(row.get("violation_rate")) for row in selected]),
        "raw_constraint": avg([to_float(row.get("constraint_satisfied")) for row in selected]),
        "raw_eval_episodes": len(selected),
    }


def run_summary(run_dir: Path, log_root: Path | None) -> dict:
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run_name = meta.get("run_name", run_dir.name)
    train = read_csv(run_dir / "train_episodes.csv")
    eff = read_csv(run_dir / "efficiency.csv")
    eval_path = run_dir / "corrected_eval_episodes.csv"
    if not eval_path.exists():
        eval_path = run_dir / "eval_episodes.csv"
    eval_rows = read_csv(eval_path)
    has_error, error_detail = scan_error(run_dir, log_root, run_name)
    final_step = 0
    if eff:
        final_step = max(final_step, to_int(eff[-1].get("step")))
    if train:
        final_step = max(final_step, to_int(train[-1].get("end_step")))
    target_steps = to_int(meta.get("num_steps"), 100000)
    completed = target_steps > 0 and final_step >= target_steps
    row = {
        "config": meta.get("ablation_name", ""),
        "method": meta.get("method", ""),
        "task": meta.get("task", ""),
        "seed": meta.get("seed", ""),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "completed": int(completed),
        "has_error": int(has_error),
        "error_detail": error_detail,
        "target_steps": target_steps,
        "final_step": final_step,
        "checkpoint": str(run_dir / "checkpoint" / "final.torch") if (run_dir / "checkpoint" / "final.torch").exists() else "",
        "eval_source": str(eval_path) if eval_path.exists() else "",
        "raw_eval_episodes": 0,
        "raw_return": math.nan,
        "raw_cost": math.nan,
        "raw_evr": math.nan,
        "raw_constraint": math.nan,
        "train_avg_last3_reward": tail_avg(train, "episode_reward"),
        "train_avg_last3_cost": tail_avg(train, "episode_cost"),
        "train_cost_rate": to_float(train[-1].get("train_total_cost_rate")) if train else math.nan,
        "env_steps_per_second": to_float(eff[-1].get("env_steps_per_second")) if eff else math.nan,
    }
    row.update(eval_summary(eval_rows))
    return row


def metric_return(row: dict) -> float:
    value = to_float(row.get("raw_return"))
    return value if not math.isnan(value) else to_float(row.get("train_avg_last3_reward"))


def metric_cost(row: dict) -> float:
    value = to_float(row.get("raw_cost"))
    return value if not math.isnan(value) else to_float(row.get("train_avg_last3_cost"))


def method_selection(rows: list[dict], method: str) -> tuple[dict | None, list[dict]]:
    candidates = []
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            by_config[row["config"]].append(row)
    for config, items in sorted(by_config.items()):
        tasks = {item["task"] for item in items}
        usable = [
            item for item in items
            if to_int(item.get("completed")) == 1 and to_int(item.get("has_error")) == 0
        ]
        reasons = []
        if len(tasks) < 2:
            reasons.append("missing_task")
        if len(usable) < len(items) or len(usable) < 2:
            reasons.append("incomplete_or_error")
        for item in usable:
            if metric_return(item) < 5:
                reasons.append(f"{item['task']}:return_collapse")
            if metric_cost(item) > 150:
                reasons.append(f"{item['task']}:cost_exploded")
        mean_return = avg([metric_return(item) for item in usable])
        mean_cost = avg([metric_cost(item) for item in usable])
        mean_evr = avg([to_float(item.get("raw_evr")) for item in usable])
        mean_cost_rate = avg([to_float(item.get("train_cost_rate")) for item in usable])
        score = mean_return - 0.05 * mean_cost
        candidates.append({
            "config": config,
            "method": method,
            "tasks": ",".join(sorted(tasks)),
            "usable_runs": len(usable),
            "mean_return": mean_return,
            "mean_cost": mean_cost,
            "mean_evr": mean_evr,
            "mean_train_cost_rate": mean_cost_rate,
            "score": score,
            "decision": "candidate" if not reasons else "filtered",
            "reasons": ";".join(sorted(set(reasons))),
        })
    selected = None
    valid = [row for row in candidates if row["decision"] == "candidate"]
    if valid:
        selected = sorted(valid, key=lambda row: (-to_float(row["score"], -1e9), row["config"]))[0]
    return selected, sorted(candidates, key=lambda row: (row["decision"] != "candidate", -to_float(row["score"], -1e9), row["config"]))


def config_value_from_name(config: str) -> float:
    if config.endswith("lambda050"):
        return 0.5
    if config.endswith("lambda100"):
        return 1.0
    if config.endswith("lambda200"):
        return 2.0
    if config.endswith("lr0001"):
        return 0.0001
    if config.endswith("lr0003"):
        return 0.0003
    if config.endswith("lr0010"):
        return 0.001
    return math.nan


def write_markdown(report_dir: Path, rows: list[dict], selected_pointwise: dict | None, selected_lag: dict | None) -> None:
    lines = [
        "# Baseline Screen Selection",
        "",
        "This screen is used only to choose one global Pointwise and one global SAC-Lag-local setting before the decisive 300k runs.",
        "",
        "## Candidates",
        "",
        "| Method | Config | Decision | Return | Cost | EVR | Train Cost Rate | Reasons |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['config']} | {row['decision']} | {fmt(row['mean_return'])} | "
            f"{fmt(row['mean_cost'])} | {fmt(row['mean_evr'])} | {fmt(row['mean_train_cost_rate'], 4)} | {row['reasons']} |"
        )
    lines.extend(["", "## Selected", ""])
    if selected_pointwise:
        lines.append(f"- Pointwise: `{selected_pointwise['config']}`")
    else:
        lines.append("- Pointwise: not selected")
    if selected_lag:
        lines.append(f"- SAC-Lag-local: `{selected_lag['config']}`")
    else:
        lines.append("- SAC-Lag-local: not selected")
    (report_dir / "baseline_screen_selection.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--log-root", type=Path, default=Path("logs/star_goal/baseline_screen"))
    args = parser.parse_args()
    root = args.root or latest_root()
    rows = [run_summary(meta.parent, args.log_root) for meta in sorted(root.rglob("run_metadata.json"))]
    fields = [
        "config", "method", "task", "seed", "run_name", "completed", "has_error", "target_steps", "final_step",
        "raw_eval_episodes", "raw_return", "raw_cost", "raw_evr", "raw_constraint",
        "train_avg_last3_reward", "train_avg_last3_cost", "train_cost_rate", "env_steps_per_second",
        "checkpoint", "eval_source", "run_dir", "error_detail",
    ]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "baseline_screen_runs.csv", rows, fields)
    selected_pointwise, pointwise_rows = method_selection(rows, "pointwise")
    selected_lag, lag_rows = method_selection(rows, "sac_lag")
    candidate_rows = pointwise_rows + lag_rows
    candidate_fields = [
        "method", "config", "tasks", "usable_runs", "mean_return", "mean_cost", "mean_evr",
        "mean_train_cost_rate", "score", "decision", "reasons",
    ]
    write_csv(args.report_dir / "baseline_screen_candidates.csv", candidate_rows, candidate_fields)
    write_markdown(args.report_dir, candidate_rows, selected_pointwise, selected_lag)
    selected = {}
    if selected_pointwise:
        selected["pointwise_star_lambda"] = config_value_from_name(selected_pointwise["config"])
        selected["pointwise_config"] = selected_pointwise["config"]
    if selected_lag:
        selected["sac_lag_lagrange_lr"] = config_value_from_name(selected_lag["config"])
        selected["sac_lag_config"] = selected_lag["config"]
    selected["source_root"] = str(root)
    if selected_pointwise and selected_lag:
        (args.report_dir / "selected_baseline_config.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.report_dir / 'baseline_screen_runs.csv'}")
    print(f"wrote {args.report_dir / 'baseline_screen_candidates.csv'}")
    print(f"wrote {args.report_dir / 'baseline_screen_selection.md'}")


if __name__ == "__main__":
    main()
