#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPORT = Path("reports/star_goal")


def latest_root() -> Path:
    roots = sorted(Path("results").glob("star_risk_scale_calibration_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return roots[-1] if roots else Path("results/star_risk_scale_calibration_missing")


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


def avg(values: list[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return mean(vals) if vals else math.nan


def fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def latest_by_step(rows: list[dict], key: str) -> list[dict]:
    if not rows:
        return []
    step = max(to_int(row.get(key)) for row in rows)
    return [row for row in rows if to_int(row.get(key)) == step]


def tail_avg(rows: list[dict], key: str, n: int = 3) -> float:
    return avg([to_float(row.get(key)) for row in rows[-n:]])


def parse_config(config: str) -> tuple[float, float]:
    # Examples: cg097_thr0p50, cg095_thr0p25
    gamma = 0.99
    threshold = math.nan
    if config.startswith("cg097"):
        gamma = 0.97
    elif config.startswith("cg095"):
        gamma = 0.95
    elif config.startswith("cg090"):
        gamma = 0.90
    if "_thr" in config:
        threshold = float(config.split("_thr", 1)[1].replace("p", "."))
    return gamma, threshold


def run_summary(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    train = read_csv(run_dir / "train_episodes.csv")
    eff = read_csv(run_dir / "efficiency.csv")
    eval_rows = read_csv(run_dir / "corrected_eval_episodes.csv")
    raw = latest_by_step([row for row in eval_rows if row.get("mode") == "raw"], "global_step")
    final_step = 0
    if eff:
        final_step = max(final_step, to_int(eff[-1].get("step")))
    if train:
        final_step = max(final_step, to_int(train[-1].get("end_step")))
    target_steps = to_int(meta.get("num_steps"), 100000)
    config = meta.get("ablation_name", "")
    gamma, threshold = parse_config(config)
    return {
        "config": config,
        "task": meta.get("task", ""),
        "seed": meta.get("seed", ""),
        "run_name": meta.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
        "cost_gamma": gamma,
        "star_risk_threshold": threshold,
        "completed": int(final_step >= target_steps and target_steps > 0),
        "final_step": final_step,
        "checkpoint": str(run_dir / "checkpoint" / "final.torch") if (run_dir / "checkpoint" / "final.torch").exists() else "",
        "raw_eval_episodes": len(raw),
        "raw_return": avg([to_float(row.get("episode_reward", row.get("episode_return"))) for row in raw]),
        "raw_cost": avg([to_float(row.get("episode_cost")) for row in raw]),
        "raw_evr": avg([to_float(row.get("violation_rate")) for row in raw]),
        "train_avg_last3_reward": tail_avg(train, "episode_reward"),
        "train_avg_last3_cost": tail_avg(train, "episode_cost"),
        "train_cost_rate": to_float(train[-1].get("train_total_cost_rate")) if train else math.nan,
        "speed": to_float(eff[-1].get("env_steps_per_second")) if eff else math.nan,
    }


def load_risk_summary(report_dir: Path) -> dict[str, dict]:
    out = {}
    for row in read_csv(report_dir / "shadow_risk_summary.csv"):
        out[row.get("run_name", "")] = row
    return out


def load_threshold_grid(report_dir: Path) -> dict[str, dict]:
    rows = read_csv(report_dir / "shadow_threshold_grid.csv")
    by_run_threshold = {(row.get("run_name", ""), round(to_float(row.get("threshold")), 2)): row for row in rows}
    return {f"{run}:{threshold:.2f}": row for (run, threshold), row in by_run_threshold.items()}


def merge(rows: list[dict], report_dir: Path) -> list[dict]:
    risk = load_risk_summary(report_dir)
    grid = load_threshold_grid(report_dir)
    out = []
    for row in rows:
        merged = dict(row)
        diag = risk.get(row["run_name"], {})
        merged["audit_gap_mean"] = to_float(diag.get("audit_gap_mean"))
        merged["audit_gap_positive_rate"] = to_float(diag.get("audit_gap_positive_rate"))
        threshold = round(to_float(row.get("star_risk_threshold")), 2)
        grid_row = grid.get(f"{row['run_name']}:{threshold:.2f}", {})
        merged["rho_active_rate"] = to_float(grid_row.get("rho_active_rate"))
        merged["hidden_unsafe_rate"] = to_float(grid_row.get("hidden_unsafe_rate"))
        merged["pSVR"] = to_float(grid_row.get("pSVR"))
        out.append(merged)
    return out


def select(rows: list[dict]) -> tuple[dict | None, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    candidates = []
    for config, items in sorted(grouped.items()):
        usable = [row for row in items if to_int(row["completed"]) == 1]
        tasks = {row["task"] for row in usable}
        reasons = []
        if len(tasks) < 2:
            reasons.append("missing_task")
        for row in usable:
            if to_float(row.get("audit_gap_mean")) <= 0:
                reasons.append(f"{row['task']}:audit_gap_nonpositive")
            rho = to_float(row.get("rho_active_rate"))
            if not math.isnan(rho) and (rho < 0.1 or rho > 0.9):
                reasons.append(f"{row['task']}:rho_active_outside_target")
            if to_float(row.get("raw_return"), to_float(row.get("train_avg_last3_reward"))) < 5:
                reasons.append(f"{row['task']}:return_collapse")
        mean_return = avg([to_float(row.get("raw_return"), to_float(row.get("train_avg_last3_reward"))) for row in usable])
        mean_cost = avg([to_float(row.get("raw_cost"), to_float(row.get("train_avg_last3_cost"))) for row in usable])
        mean_rho = avg([to_float(row.get("rho_active_rate")) for row in usable])
        mean_gap = avg([to_float(row.get("audit_gap_mean")) for row in usable])
        score = mean_return - 0.05 * mean_cost + min(max(mean_gap, -1.0), 1.0) - abs(mean_rho - 0.5)
        gamma, threshold = parse_config(config)
        candidates.append({
            "config": config,
            "cost_gamma": gamma,
            "star_risk_threshold": threshold,
            "tasks": ",".join(sorted(tasks)),
            "mean_return": mean_return,
            "mean_cost": mean_cost,
            "mean_rho_active_rate": mean_rho,
            "mean_audit_gap": mean_gap,
            "score": score,
            "decision": "candidate" if not reasons else "filtered",
            "reasons": ";".join(sorted(set(reasons))),
        })
    # The goal-mode selection contract intentionally prioritizes safety over
    # the scalar score: lower raw cost first, then higher return, then a
    # stronger audit gap, with cost_gamma=0.97 preferred only when comparable.
    candidates = sorted(candidates, key=lambda row: (
        row["decision"] != "candidate",
        to_float(row["mean_cost"], 1e9),
        -to_float(row["mean_return"], -1e9),
        -to_float(row["mean_audit_gap"], -1e9),
        0 if abs(to_float(row["cost_gamma"], 0.0) - 0.97) < 1e-9 else 1,
        row["config"],
    ))
    selected = next((row for row in candidates if row["decision"] == "candidate"), None)
    return selected, candidates


def write_selected(report_dir: Path, selected: dict | None) -> None:
    if selected is None:
        return
    data = {
        "cost_gamma": selected["cost_gamma"],
        "star_risk_threshold": selected["star_risk_threshold"],
        "star_lambda": 1.0,
        "shadow_k": 16,
        "shadow_temperature": 0.05,
        "shadow_aggregation": "log_mean_exp",
        "shadow_reference_mode": "corridor",
        "star_ref_update_interval": 20,
        "star_kl_coef": 1.0,
        "star_kl_target": 0.01,
        "cost_critic_reduce": "max",
        "star_exec_candidates": 16,
        "star_exec_margin": 0.02,
        "selected_from_stage": "risk_scale_calibration",
        "selected_config": selected["config"],
    }
    (report_dir / "selected_actor_config.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_md(report_dir: Path, candidates: list[dict], selected: dict | None) -> None:
    lines = [
        "# Risk-Scale Calibration Selection",
        "",
        "| Config | Decision | Return | Cost | Rho Active | Audit Gap | Score | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidates:
        lines.append(
            f"| {row['config']} | {row['decision']} | {fmt(row['mean_return'])} | {fmt(row['mean_cost'])} | "
            f"{fmt(row['mean_rho_active_rate'])} | {fmt(row['mean_audit_gap'])} | {fmt(row['score'])} | {row['reasons']} |"
        )
    lines.extend(["", "## Selected", ""])
    lines.append(f"`{selected['config']}`" if selected else "No valid risk-scale calibration candidate selected.")
    (report_dir / "risk_scale_selection.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=REPORT / "risk_scale_calibration")
    args = parser.parse_args()
    root = args.root or latest_root()
    rows = [run_summary(meta.parent) for meta in sorted(root.rglob("run_metadata.json"))]
    rows = merge(rows, args.report_dir)
    fields = [
        "config", "task", "seed", "run_name", "cost_gamma", "star_risk_threshold", "completed", "final_step",
        "raw_eval_episodes", "raw_return", "raw_cost", "raw_evr", "train_avg_last3_reward", "train_avg_last3_cost",
        "train_cost_rate", "audit_gap_mean", "audit_gap_positive_rate", "rho_active_rate", "hidden_unsafe_rate", "pSVR",
        "checkpoint", "run_dir",
    ]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "risk_scale_runs.csv", rows, fields)
    selected, candidates = select(rows)
    candidate_fields = [
        "config", "cost_gamma", "star_risk_threshold", "tasks", "mean_return", "mean_cost", "mean_rho_active_rate",
        "mean_audit_gap", "score", "decision", "reasons",
    ]
    write_csv(args.report_dir / "risk_scale_candidates.csv", candidates, candidate_fields)
    write_md(args.report_dir, candidates, selected)
    write_selected(REPORT, selected)
    print(f"wrote {args.report_dir / 'risk_scale_runs.csv'}")
    print(f"wrote {args.report_dir / 'risk_scale_candidates.csv'}")
    print(f"wrote {args.report_dir / 'risk_scale_selection.md'}")


if __name__ == "__main__":
    main()
