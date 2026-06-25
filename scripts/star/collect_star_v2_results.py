#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TASKS = [
    "SafetyPointGoal1-v0",
    "SafetyCarGoal1-v0",
    "SafetyPointButton1-v0",
    "SafetyCarButton1-v0",
]
METHODS = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]
CORE_SEEDS = [10, 11, 12]


def fnum(value, default=0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def error_scan(log_path: Path | None) -> str:
    if not log_path or not log_path.exists():
        return "missing_log"
    patterns = ["Traceback", "ERROR", "NaN", "nan", "OOM", "out of memory", "RuntimeError", "MuJoCo", "No space left"]
    try:
        text = log_path.read_text(errors="ignore")
    except Exception as exc:
        return f"log_read_error:{exc}"
    hits = [p for p in patterns if p in text]
    return ";".join(hits)


def run_phase_from_dir(run_dir: Path, root: Path) -> str:
    try:
        return run_dir.relative_to(root).parts[0]
    except Exception:
        return ""


def summarize_run(run_dir: Path, root: Path, log_root: Path, expected_steps: int) -> dict:
    meta = read_json(run_dir / "run_metadata.json")
    train_rows = read_csv(run_dir / "train_episodes.csv")
    mech_rows = read_csv(run_dir / "mechanism.csv")
    eval_rows = read_csv(run_dir / "corrected_eval_episodes.csv")
    if not eval_rows:
        eval_rows = read_csv(run_dir / "eval_episodes.csv")
    phase = str(meta.get("ablation_group") or run_phase_from_dir(run_dir, root))
    run_name = str(meta.get("run_name") or run_dir.name)
    log_path = log_root / phase / f"{run_name}.log"
    last = train_rows[-1] if train_rows else {}
    last10 = train_rows[-10:]
    final_step = int(fnum(last.get("end_step", 0)))
    checkpoint = run_dir / "checkpoint" / "final.torch"
    err = error_scan(log_path)
    raw_eval = [r for r in eval_rows if r.get("mode") == "raw"]
    filtered_eval = [r for r in eval_rows if r.get("mode") == "star_exec"]
    key = (
        str(meta.get("task", last.get("task", ""))),
        str(meta.get("method", last.get("method", ""))),
        int(fnum(meta.get("seed", last.get("seed", 0)))),
        str(meta.get("ablation_group", phase)),
        str(meta.get("ablation_name", "")),
    )
    completed = checkpoint.exists() and final_step >= expected_steps and err == ""
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "phase": phase,
        "task": key[0],
        "method": key[1],
        "seed": key[2],
        "ablation_group": key[3],
        "ablation_name": key[4],
        "star_algorithm_version": meta.get("star_algorithm_version", ""),
        "git_sha": meta.get("git_sha", ""),
        "final_step": final_step,
        "expected_steps": expected_steps,
        "completed": completed,
        "has_final_checkpoint": checkpoint.exists(),
        "final_checkpoint": str(checkpoint) if checkpoint.exists() else "",
        "error_scan": err,
        "train_latest_reward": fnum(last.get("episode_reward", 0)),
        "train_latest_cost": fnum(last.get("episode_cost", 0)),
        "train_avg_last10_reward": mean(fnum(r.get("episode_reward", 0)) for r in last10),
        "train_avg_last10_cost": mean(fnum(r.get("episode_cost", 0)) for r in last10),
        "train_total_cost": fnum(last.get("train_total_cost", 0)),
        "train_total_cost_rate": fnum(last.get("train_total_cost_rate", 0)),
        "mechanism_rows": len(mech_rows),
        "raw_eval_episodes": len(raw_eval),
        "raw_return_mean": mean(fnum(r.get("episode_reward", 0)) for r in raw_eval),
        "raw_cost_mean": mean(fnum(r.get("episode_cost", 0)) for r in raw_eval),
        "raw_evr_mean": mean(fnum(r.get("violation_rate", 0)) for r in raw_eval),
        "raw_constraint_satisfaction_rate": mean(fnum(r.get("constraint_satisfied", 0)) for r in raw_eval),
        "filtered_eval_episodes": len(filtered_eval),
        "filtered_return_mean": mean(fnum(r.get("episode_reward", 0)) for r in filtered_eval),
        "filtered_cost_mean": mean(fnum(r.get("episode_cost", 0)) for r in filtered_eval),
        "filtered_evr_mean": mean(fnum(r.get("violation_rate", 0)) for r in filtered_eval),
        "filtered_constraint_satisfaction_rate": mean(fnum(r.get("constraint_satisfied", 0)) for r in filtered_eval),
    }


def discover_runs(root: Path, phase: str, expected_steps: int, log_root: Path) -> list[dict]:
    phase_root = root / phase
    dirs = [p.parent for p in phase_root.rglob("run_metadata.json")]
    return [summarize_run(d, root, log_root, expected_steps) for d in sorted(dirs)]


def duplicate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["task"], row["method"], row["seed"], row["ablation_name"])
        groups[key].append(row)
    out = []
    for key, vals in groups.items():
        complete = [v for v in vals if v["completed"]]
        if len(complete) > 1 or len(vals) > 1:
            for v in vals:
                item = dict(v)
                item["duplicate_key"] = "|".join(map(str, key))
                item["qualified_duplicate_count"] = len(complete)
                out.append(item)
    return out


def expected_core_rows(tasks: list[str]) -> list[dict]:
    rows = []
    for task in tasks:
        for method in METHODS:
            for seed in CORE_SEEDS:
                rows.append({"phase": "core_100k", "task": task, "method": method, "seed": seed})
    return rows


def missing_core(rows: list[dict], tasks: list[str]) -> list[dict]:
    present = {(r["phase"], r["task"], r["method"], int(r["seed"])): r for r in rows if r["completed"]}
    missing = []
    for exp in expected_core_rows(tasks):
        key = (exp["phase"], exp["task"], exp["method"], exp["seed"])
        if key not in present:
            missing.append({**exp, "reason": "missing_completed_error_free_run"})
    return missing


def summary_by_method(rows: list[dict]) -> list[dict]:
    complete = [r for r in rows if r["completed"]]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in complete:
        groups[(row["phase"], row["task"], row["method"])].append(row)
    out = []
    for (phase, task, method), vals in sorted(groups.items()):
        out.append(
            {
                "phase": phase,
                "task": task,
                "method": method,
                "seeds": len(vals),
                "train_reward_mean": mean(v["train_avg_last10_reward"] for v in vals),
                "train_reward_std": sample_std(v["train_avg_last10_reward"] for v in vals),
                "train_cost_mean": mean(v["train_avg_last10_cost"] for v in vals),
                "train_cost_std": sample_std(v["train_avg_last10_cost"] for v in vals),
                "train_total_cost_mean": mean(v["train_total_cost"] for v in vals),
                "train_total_cost_std": sample_std(v["train_total_cost"] for v in vals),
                "raw_return_mean": mean(v["raw_return_mean"] for v in vals),
                "raw_return_std": sample_std(v["raw_return_mean"] for v in vals),
                "raw_cost_mean": mean(v["raw_cost_mean"] for v in vals),
                "raw_cost_std": sample_std(v["raw_cost_mean"] for v in vals),
            }
        )
    return out


def core_gate(rows: list[dict], tasks: list[str]) -> tuple[str, str]:
    missing = missing_core(rows, tasks)
    duplicates = duplicate_rows([r for r in rows if r["phase"] == "core_100k"])
    mixed = [r for r in rows if r["phase"] == "core_100k" and r["star_algorithm_version"] != "star_v2"]
    if missing or duplicates or mixed:
        lines = [
            "# STAR-v2 Core 100k Gate",
            "",
            "Decision: PENDING",
            "",
            f"Missing completed runs: {len(missing)}",
            f"Duplicate run rows: {len(duplicates)}",
            f"Non-STAR-v2 rows: {len(mixed)}",
        ]
        return "PENDING", "\n".join(lines) + "\n"

    complete = [r for r in rows if r["phase"] == "core_100k" and r["completed"]]
    by = {(r["task"], r["method"], int(r["seed"])): r for r in complete}
    paired_wins = 0
    total_pairs = 0
    task_cost_wins = 0
    reward_retention_ok = True
    catastrophic = []
    for task in tasks:
        star_costs = []
        cur_costs = []
        star_rewards = []
        cur_rewards = []
        for seed in CORE_SEEDS:
            star = by[(task, "star_v2", seed)]
            cur = by[(task, "current_only_v2", seed)]
            # Prefer offline raw metrics when available. Until offline eval is
            # generated, fall back to training last-10 metrics and keep the gate
            # explicitly provisional in the report.
            s_cost = star["raw_cost_mean"] if star["raw_eval_episodes"] else star["train_avg_last10_cost"]
            c_cost = cur["raw_cost_mean"] if cur["raw_eval_episodes"] else cur["train_avg_last10_cost"]
            s_reward = star["raw_return_mean"] if star["raw_eval_episodes"] else star["train_avg_last10_reward"]
            c_reward = cur["raw_return_mean"] if cur["raw_eval_episodes"] else cur["train_avg_last10_reward"]
            paired_wins += int(s_cost < c_cost)
            total_pairs += 1
            star_costs.append(s_cost)
            cur_costs.append(c_cost)
            star_rewards.append(s_reward)
            cur_rewards.append(c_reward)
        if mean(star_costs) < mean(cur_costs):
            task_cost_wins += 1
        if mean(star_rewards) < 0.9 * mean(cur_rewards):
            reward_retention_ok = False
        if all(s_r < c_r and s_c > c_c for s_r, c_r, s_c, c_c in zip(star_rewards, cur_rewards, star_costs, cur_costs)):
            catastrophic.append(task)

    cost_signal = paired_wins >= 7 or task_cost_wins >= 3
    passed = cost_signal and reward_retention_ok and not catastrophic
    eval_missing = any(r["raw_eval_episodes"] == 0 for r in complete)
    decision = "PASS_PROVISIONAL" if passed and eval_missing else "PASS" if passed else "FAIL"
    lines = [
        "# STAR-v2 Core 100k Gate",
        "",
        f"Decision: {decision}",
        "",
        f"STAR-v2 lower cost paired wins vs current-only: {paired_wins}/{total_pairs}",
        f"STAR-v2 lower task-mean cost tasks: {task_cost_wins}/{len(tasks)}",
        f"Reward retention ok: {reward_retention_ok}",
        f"Catastrophic tasks: {', '.join(catastrophic) if catastrophic else 'none'}",
        f"Offline raw evaluation complete: {not eval_missing}",
    ]
    if eval_missing:
        lines.append("")
        lines.append("Gate is provisional because at least one run is missing offline raw evaluation rows.")
    return decision, "\n".join(lines) + "\n"


def load_tasks(config_path: Path) -> list[str]:
    data = read_json(config_path)
    return [str(x) for x in data.get("selected_final_tasks") or data.get("preferred_final_tasks") or TASKS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/star_v2_final")
    parser.add_argument("--log-root", default="logs/star_v2_final")
    parser.add_argument("--report-dir", default="reports/star_v2_final")
    parser.add_argument("--phase", choices=["core_100k", "resume_300k", "ablation_100k", "all"], default="all")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    log_root = Path(args.log_root)
    report = Path(args.report_dir)
    tasks = load_tasks(Path("configs/star_v2_selected_full.json"))

    rows: list[dict] = []
    if args.phase in ("core_100k", "all"):
        rows.extend(discover_runs(root, "core_100k", 100000, log_root))
    if args.phase in ("resume_300k", "all"):
        rows.extend(discover_runs(root, "resume_300k", 300000, log_root))
    if args.phase in ("ablation_100k", "all"):
        rows.extend(discover_runs(root, "ablation_100k", 100000, log_root))

    write_csv(report / "run_manifest.csv", rows)
    write_csv(report / "main_results_by_seed.csv", [r for r in rows if r["completed"]])
    write_csv(report / "main_results_summary.csv", summary_by_method(rows))
    write_csv(
        report / "training_safety_by_seed.csv",
        [
            {
                "phase": r["phase"],
                "task": r["task"],
                "method": r["method"],
                "seed": r["seed"],
                "train_total_cost": r["train_total_cost"],
                "train_total_cost_rate": r["train_total_cost_rate"],
            }
            for r in rows
            if r["completed"]
        ],
    )
    dups = duplicate_rows(rows)
    write_csv(report / "duplicate_runs.csv", dups)
    missing = missing_core(rows, tasks) if args.phase in ("core_100k", "all") else []
    missing_lines = ["# Missing Results", ""]
    if missing:
        for row in missing:
            missing_lines.append(f"- {row['phase']} {row['task']} {row['method']} seed={row['seed']}: {row['reason']}")
    else:
        missing_lines.append("No missing core_100k completed/error-free runs detected.")
    (report / "missing_results.md").write_text("\n".join(missing_lines) + "\n")

    decision, gate_text = core_gate(rows, tasks) if args.phase in ("core_100k", "all") else ("UNAVAILABLE", "")
    core_dir = report / "core_100k"
    core_dir.mkdir(parents=True, exist_ok=True)
    (core_dir / "gate.md").write_text(gate_text)
    (report / "claim_gate.md").write_text(gate_text)
    if args.strict and (missing or dups or decision in {"FAIL", "PENDING"}):
        print(f"strict collection failed: decision={decision} missing={len(missing)} duplicates={len(dups)}")
        return 2
    print(f"wrote {report / 'run_manifest.csv'} rows={len(rows)} decision={decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
