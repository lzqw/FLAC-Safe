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


ERROR_RE = re.compile(
    r"No space left|Traceback|RuntimeError|OOM|out of memory|CUDA error|Segmentation fault|KeyboardInterrupt|invalid loss",
    re.IGNORECASE,
)

DEFAULT_ROOT = Path("results/star_tune_actor_stage1_3393af8")
DEFAULT_REPORT = Path("reports/star_goal")
TARGET_STEPS = 100000


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


def fmt(value, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def avg(values: list[float]) -> float:
    vals = finite(values)
    return mean(vals) if vals else math.nan


def latest_by_step(rows: list[dict], key: str) -> list[dict]:
    if not rows:
        return []
    step = max(to_int(row.get(key)) for row in rows)
    return [row for row in rows if to_int(row.get(key)) == step]


def tail_avg(rows: list[dict], key: str, n: int = 3) -> float:
    vals = [to_float(row.get(key)) for row in rows[-n:]]
    return avg(vals)


def scan_error(run_dir: Path, log_root: Path | None = None, run_name: str = "") -> tuple[bool, str]:
    paths = list(run_dir.rglob("*.log"))
    if log_root and log_root.exists() and run_name:
        paths.extend(path for path in log_root.rglob("*") if path.is_file() and run_name in path.name)
    for path in sorted(set(paths)):
        try:
            for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
                if ERROR_RE.search(line):
                    return True, f"{path}:{lineno}:{line[:180]}"
        except OSError:
            continue
    return False, ""


def eval_summary(rows: list[dict], mode: str) -> dict:
    selected = latest_by_step([row for row in rows if row.get("mode") == mode], "global_step")
    if not selected:
        return {}
    return {
        f"{mode}_return": avg([to_float(row.get("episode_reward", row.get("episode_return"))) for row in selected]),
        f"{mode}_cost": avg([to_float(row.get("episode_cost")) for row in selected]),
        f"{mode}_evr": avg([to_float(row.get("violation_rate")) for row in selected]),
        f"{mode}_success": avg([to_float(row.get("success")) for row in selected]),
        f"{mode}_constraint": avg([to_float(row.get("constraint_satisfied")) for row in selected]),
        f"{mode}_episodes": len(selected),
        f"{mode}_eval_step": max(to_int(row.get("global_step")) for row in selected),
    }


def run_summary(run_dir: Path, log_root: Path | None) -> dict:
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run_name = meta.get("run_name", run_dir.name)
    train = read_csv(run_dir / "train_episodes.csv")
    eff = read_csv(run_dir / "efficiency.csv")
    mech = read_csv(run_dir / "mechanism.csv")
    eval_path = run_dir / "corrected_eval_episodes.csv"
    eval_source = eval_path
    if not eval_path.exists():
        eval_path = run_dir / "eval_episodes.csv"
        eval_source = eval_path
    eval_rows = read_csv(eval_path)
    has_error, error_detail = scan_error(run_dir, log_root, run_name)

    final_step = 0
    if eff:
        final_step = max(final_step, to_int(eff[-1].get("step")))
    if train:
        final_step = max(final_step, to_int(train[-1].get("end_step")))
    target_steps = to_int(meta.get("num_steps"), TARGET_STEPS)
    completed = final_step >= target_steps and target_steps > 0
    final_checkpoint = run_dir / "checkpoint" / "final.torch"
    status = "completed" if completed else "running_or_partial"
    if has_error:
        status = "error"

    row = {
        "config": meta.get("ablation_name", ""),
        "task": meta.get("task", ""),
        "seed": meta.get("seed", ""),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": status,
        "completed": int(completed),
        "has_error": int(has_error),
        "error_detail": error_detail,
        "target_steps": target_steps,
        "final_step": final_step,
        "checkpoint": str(final_checkpoint) if final_checkpoint.exists() else "",
        "eval_source": str(eval_source) if eval_source.exists() else "",
        "raw_eval_episodes": 0,
        "raw_return": math.nan,
        "raw_cost": math.nan,
        "raw_evr": math.nan,
        "raw_constraint": math.nan,
        "star_exec_return": math.nan,
        "star_exec_cost": math.nan,
        "star_exec_evr": math.nan,
        "star_exec_constraint": math.nan,
        "train_last_reward": train[-1].get("episode_reward", "") if train else "",
        "train_last_cost": train[-1].get("episode_cost", "") if train else "",
        "train_avg_last3_reward": tail_avg(train, "episode_reward"),
        "train_avg_last3_cost": tail_avg(train, "episode_cost"),
        "train_cost_rate": to_float(train[-1].get("train_total_cost_rate")) if train else math.nan,
        "env_steps_per_second": to_float(eff[-1].get("env_steps_per_second")) if eff else math.nan,
        "wall_clock_time": to_float(eff[-1].get("wall_clock_time")) if eff else math.nan,
        "gpu_memory_peak_mb": to_float(eff[-1].get("gpu_memory_peak_mb")) if eff else math.nan,
        "pSVR": math.nan,
        "any_unsafe_shadow_rate": math.nan,
        "hidden_unsafe_rate": math.nan,
        "shadow_risk_mean": math.nan,
        "shadow_risk_max_mean": math.nan,
        "actor_mean_action_risk": math.nan,
        "kl_mean": math.nan,
        "candidate_spread": math.nan,
        "shadow_penalty": math.nan,
        "shadow_k": meta.get("shadow_k", ""),
        "shadow_temperature": meta.get("shadow_temperature", ""),
        "star_lambda": meta.get("star_lambda", ""),
        "star_risk_threshold": meta.get("star_risk_threshold", ""),
        "star_kl_coef": meta.get("star_kl_coef", ""),
        "star_ref_update_interval": meta.get("star_ref_update_interval", ""),
        "debug_score": math.nan,
        "decision": "",
    }

    raw = eval_summary(eval_rows, "raw")
    if raw:
        row.update({
            "raw_return": raw.get("raw_return", math.nan),
            "raw_cost": raw.get("raw_cost", math.nan),
            "raw_evr": raw.get("raw_evr", math.nan),
            "raw_constraint": raw.get("raw_constraint", math.nan),
            "raw_eval_episodes": raw.get("raw_episodes", 0),
        })
    star_exec = eval_summary(eval_rows, "star_exec")
    if star_exec:
        row.update({
            "star_exec_return": star_exec.get("star_exec_return", math.nan),
            "star_exec_cost": star_exec.get("star_exec_cost", math.nan),
            "star_exec_evr": star_exec.get("star_exec_evr", math.nan),
            "star_exec_constraint": star_exec.get("star_exec_constraint", math.nan),
        })

    final_mech = latest_by_step(mech, "step")
    if final_mech:
        for key in (
            "pSVR",
            "any_unsafe_shadow_rate",
            "hidden_unsafe_rate",
            "shadow_risk_mean",
            "shadow_risk_max_mean",
            "actor_mean_action_risk",
            "kl_mean",
            "candidate_spread",
            "shadow_penalty",
        ):
            row[key] = avg([to_float(item.get(key)) for item in final_mech])

    reward_metric = row["raw_return"] if not math.isnan(to_float(row["raw_return"])) else to_float(row["train_avg_last3_reward"])
    cost_metric = row["raw_cost"] if not math.isnan(to_float(row["raw_cost"])) else to_float(row["train_avg_last3_cost"])
    psvr = to_float(row["pSVR"])
    spread = to_float(row["candidate_spread"])
    speed = to_float(row["env_steps_per_second"])
    debug_score = reward_metric - 0.08 * cost_metric
    if not math.isnan(psvr):
        debug_score -= 2.0 * abs(psvr - 0.35)
    if not math.isnan(spread):
        debug_score += min(spread, 2.0)
    if not math.isnan(speed):
        debug_score += 0.01 * min(speed, 80.0)
    if has_error:
        debug_score = -1e9
    row["debug_score"] = debug_score

    if has_error:
        row["decision"] = "reject_error"
    elif not completed:
        row["decision"] = "pending"
    elif not eval_rows:
        row["decision"] = "needs_offline_eval"
    elif reward_metric < 5:
        row["decision"] = "reject_reward_collapse"
    elif cost_metric > 100:
        row["decision"] = "reject_cost_high"
    elif not math.isnan(psvr) and (psvr < 0.02 or psvr > 0.98):
        row["decision"] = "mechanism_extreme_watch"
    else:
        row["decision"] = "candidate"
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["task"])].append(row)
    out = []
    for (config, task), items in sorted(groups.items()):
        out.append({
            "config": config,
            "task": task,
            "runs": len(items),
            "completed": sum(to_int(item["completed"]) for item in items),
            "errors": sum(to_int(item["has_error"]) for item in items),
            "mean_debug_score": avg([to_float(item["debug_score"]) for item in items]),
            "mean_raw_return": avg([to_float(item["raw_return"]) for item in items]),
            "mean_raw_cost": avg([to_float(item["raw_cost"]) for item in items]),
            "mean_train_reward_last3": avg([to_float(item["train_avg_last3_reward"]) for item in items]),
            "mean_train_cost_last3": avg([to_float(item["train_avg_last3_cost"]) for item in items]),
            "mean_train_cost_rate": avg([to_float(item["train_cost_rate"]) for item in items]),
            "mean_pSVR": avg([to_float(item["pSVR"]) for item in items]),
            "mean_hidden_unsafe_rate": avg([to_float(item["hidden_unsafe_rate"]) for item in items]),
            "mean_candidate_spread": avg([to_float(item["candidate_spread"]) for item in items]),
            "mean_speed": avg([to_float(item["env_steps_per_second"]) for item in items]),
        })
    return sorted(out, key=lambda row: (row["task"], row["config"]))


def render_markdown(rows: list[dict], groups: list[dict], report_dir: Path) -> None:
    lines = [
        "# Actor Stage1 Ranking",
        "",
        "This report is generated from selected actor-stage1 run directories. It is compatible with partial training results and switches to corrected offline evaluation rows when they exist.",
        "",
        "## Group Ranking",
        "",
        "| Config | Task | Completed | Debug Score | Raw Return | Raw Cost | Train Last3 Reward | Train Last3 Cost | pSVR | Hidden Unsafe | Spread | Speed |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        lines.append(
            f"| {row['config']} | {row['task']} | {row['completed']}/{row['runs']} | "
            f"{fmt(row['mean_debug_score'])} | {fmt(row['mean_raw_return'])} | {fmt(row['mean_raw_cost'])} | "
            f"{fmt(row['mean_train_reward_last3'])} | {fmt(row['mean_train_cost_last3'])} | "
            f"{fmt(row['mean_pSVR'])} | {fmt(row['mean_hidden_unsafe_rate'])} | "
            f"{fmt(row['mean_candidate_spread'])} | {fmt(row['mean_speed'])} |"
        )
    lines.extend([
        "",
        "## Run Details",
        "",
        "| Config | Task | Seed | Status | Step | Decision | Debug Score | Raw Return | Raw Cost | Train Last3 Reward | Train Last3 Cost | Cost Rate | pSVR | Hidden Unsafe | Checkpoint |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in sorted(rows, key=lambda x: (x["task"], x["config"], str(x["seed"]))):
        lines.append(
            f"| {row['config']} | {row['task']} | {row['seed']} | {row['status']} | {row['final_step']} | "
            f"{row['decision']} | {fmt(row['debug_score'])} | {fmt(row['raw_return'])} | {fmt(row['raw_cost'])} | "
            f"{fmt(row['train_avg_last3_reward'])} | {fmt(row['train_avg_last3_cost'])} | "
            f"{fmt(row['train_cost_rate'], 4)} | {fmt(row['pSVR'])} | {fmt(row['hidden_unsafe_rate'])} | {bool(row['checkpoint'])} |"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "actor_stage1_ranking.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-root", type=Path, default=Path("logs/star_goal/actor_stage1"))
    args = parser.parse_args()

    rows = []
    for meta_path in sorted(args.root.rglob("run_metadata.json")):
        rows.append(run_summary(meta_path.parent, args.log_root))
    fields = [
        "config", "task", "seed", "run_name", "status", "completed", "has_error", "target_steps", "final_step",
        "raw_eval_episodes", "raw_return", "raw_cost", "raw_evr", "raw_constraint",
        "star_exec_return", "star_exec_cost", "star_exec_evr", "star_exec_constraint",
        "train_last_reward", "train_last_cost", "train_avg_last3_reward", "train_avg_last3_cost", "train_cost_rate",
        "env_steps_per_second", "wall_clock_time", "gpu_memory_peak_mb",
        "pSVR", "any_unsafe_shadow_rate", "hidden_unsafe_rate", "shadow_risk_mean", "shadow_risk_max_mean",
        "actor_mean_action_risk", "kl_mean", "candidate_spread", "shadow_penalty",
        "shadow_k", "shadow_temperature", "star_lambda", "star_risk_threshold", "star_kl_coef", "star_ref_update_interval",
        "debug_score", "decision", "checkpoint", "eval_source", "run_dir", "error_detail",
    ]
    group_rows = aggregate(rows)
    group_fields = [
        "config", "task", "runs", "completed", "errors", "mean_debug_score", "mean_raw_return", "mean_raw_cost",
        "mean_train_reward_last3", "mean_train_cost_last3", "mean_train_cost_rate",
        "mean_pSVR", "mean_hidden_unsafe_rate", "mean_candidate_spread", "mean_speed",
    ]
    write_csv(args.report_dir / "actor_stage1_eval_summary.csv", rows, fields)
    write_csv(args.report_dir / "actor_stage1_group_ranking.csv", group_rows, group_fields)
    render_markdown(rows, group_rows, args.report_dir)
    print(f"wrote {args.report_dir / 'actor_stage1_eval_summary.csv'}")
    print(f"wrote {args.report_dir / 'actor_stage1_group_ranking.csv'}")
    print(f"wrote {args.report_dir / 'actor_stage1_ranking.md'}")


if __name__ == "__main__":
    main()
