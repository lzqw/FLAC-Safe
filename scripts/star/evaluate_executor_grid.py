#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.star_agent import STARAgent
from main_star import _success, make_env, reset_env, step_env
from utilis.star_default_config import star_default_config


FIELDS = [
    "run_name",
    "task",
    "train_seed",
    "eval_seed",
    "checkpoint_path",
    "candidates",
    "margin",
    "mode",
    "episode_reward",
    "episode_cost",
    "episode_length",
    "violation_rate",
    "success",
    "return_drop_vs_raw",
    "execution_fallback_rate",
    "safe_candidate_fraction",
    "found_but_not_executed_rate",
]

SUMMARY_FIELDS = [
    "task",
    "candidates",
    "margin",
    "raw_return",
    "filtered_return",
    "return_drop_frac",
    "raw_cost",
    "filtered_cost",
    "raw_evr",
    "filtered_evr",
    "fallback_rate",
    "safe_candidate_fraction",
    "found_but_not_executed_rate",
    "decision",
]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def read_meta(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_config(meta: dict, candidates: int, margin: float):
    config = star_default_config.copy()
    config.update(meta)
    config.cuda = bool(config.cuda and torch.cuda.is_available())
    config.star_exec = True
    config.star_exec_candidates = int(candidates)
    config.star_exec_margin = float(margin)
    return config


def append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def final_checkpoint(run_dir: Path) -> Path | None:
    path = run_dir / "checkpoint" / "final.torch"
    return path if path.exists() else None


def eval_episode(agent: STARAgent, env, config, seed: int, mode: str) -> dict:
    state = reset_env(env, seed=seed)
    done = False
    reward_sum = 0.0
    cost_sum = 0.0
    steps = 0
    violations = 0
    fallbacks = 0
    safe_fraction = 0.0
    found = 0
    info = {}
    if mode == "star_exec":
        agent.method = "star"
        agent.star_exec = True
    while not done and steps < int(config.eval_numsteps):
        action = agent.select_action(state, evaluate=True, execution_mode=mode, total_numsteps=int(config.num_steps))
        next_state, reward, cost, terminated, truncated, info = step_env(env, action, config.safe_env)
        details = agent.last_action_info
        reward_sum += reward
        cost_sum += cost
        violations += int(cost > 0)
        fallbacks += int(bool(details.get("execution_fallback", False)))
        safe_fraction += float(details.get("safe_candidate_fraction", 0.0))
        found += int(bool(details.get("any_shadow_predicted_unsafe", False)) and cost <= 0)
        steps += 1
        done = terminated or truncated
        state = next_state
    return {
        "episode_reward": reward_sum,
        "episode_cost": cost_sum,
        "episode_length": steps,
        "violation_rate": violations / max(1, steps),
        "success": _success(info),
        "execution_fallback_rate": fallbacks / max(1, steps),
        "safe_candidate_fraction": safe_fraction / max(1, steps),
        "found_but_not_executed_rate": found / max(1, steps),
    }


def run_grid(
    root: Path,
    report_dir: Path,
    eval_seeds: list[int],
    candidates_list: list[int],
    margins: list[float],
    methods: set[str],
    output_name: str = "executor_grid.csv",
) -> None:
    out_path = report_dir / output_name
    if out_path.exists():
        out_path.unlink()
    for meta_path in sorted(root.rglob("run_metadata.json")):
        run_dir = meta_path.parent
        meta = read_meta(run_dir)
        if meta.get("method") not in methods:
            continue
        checkpoint = final_checkpoint(run_dir)
        if checkpoint is None:
            continue
        for candidates in candidates_list:
            for margin in margins:
                config = build_config(meta, candidates, margin)
                for seed in eval_seeds:
                    rows = []
                    raw_result = None
                    for mode in ("raw", "star_exec"):
                        env = make_env(config.task, safe_env=config.safe_env, train=False, binary_cost=config.binary_cost)
                        try:
                            agent = STARAgent(env.observation_space.shape[0], env.action_space, config)
                            agent.load_checkpoint(str(checkpoint))
                            result = eval_episode(agent, env, config, seed, mode)
                        finally:
                            env.close()
                        if mode == "raw":
                            raw_result = result
                        drop = math.nan
                        if raw_result and mode == "star_exec":
                            denom = max(1e-8, abs(float(raw_result["episode_reward"])))
                            drop = (float(raw_result["episode_reward"]) - float(result["episode_reward"])) / denom
                        row = {
                            "run_name": meta.get("run_name", run_dir.name),
                            "task": config.task,
                            "train_seed": meta.get("seed", ""),
                            "eval_seed": seed,
                            "checkpoint_path": str(checkpoint),
                            "candidates": candidates,
                            "margin": margin,
                            "mode": mode,
                            "return_drop_vs_raw": drop,
                        }
                        row.update(result)
                        rows.append(row)
                    append_csv(out_path, rows, FIELDS)


def summarize(report_dir: Path, input_name: str = "executor_grid.csv", output_name: str = "executor_grid_summary.csv") -> list[dict]:
    rows = []
    path = report_dir / input_name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["candidates"], row["margin"])].append(row)
    summary = []
    for (task, candidates, margin), items in sorted(groups.items()):
        raw = [row for row in items if row["mode"] == "raw"]
        filt = [row for row in items if row["mode"] == "star_exec"]
        mean_raw_return = mean([float(row["episode_reward"]) for row in raw]) if raw else math.nan
        mean_filtered_return = mean([float(row["episode_reward"]) for row in filt]) if filt else math.nan
        drop = (mean_raw_return - mean_filtered_return) / max(1e-8, abs(mean_raw_return)) if raw and filt else math.nan
        row = {
            "task": task,
            "candidates": candidates,
            "margin": margin,
            "raw_return": mean_raw_return,
            "filtered_return": mean_filtered_return,
            "return_drop_frac": drop,
            "raw_cost": mean([float(r["episode_cost"]) for r in raw]) if raw else math.nan,
            "filtered_cost": mean([float(r["episode_cost"]) for r in filt]) if filt else math.nan,
            "raw_evr": mean([float(r["violation_rate"]) for r in raw]) if raw else math.nan,
            "filtered_evr": mean([float(r["violation_rate"]) for r in filt]) if filt else math.nan,
            "fallback_rate": mean([float(r["execution_fallback_rate"]) for r in filt]) if filt else math.nan,
            "safe_candidate_fraction": mean([float(r["safe_candidate_fraction"]) for r in filt]) if filt else math.nan,
            "found_but_not_executed_rate": mean([float(r["found_but_not_executed_rate"]) for r in filt]) if filt else math.nan,
        }
        ok = row["filtered_evr"] <= row["raw_evr"] and row["filtered_cost"] <= row["raw_cost"] and drop <= 0.20 and row["fallback_rate"] < 0.8
        row["decision"] = "candidate" if ok else "filtered"
        summary.append(row)
    write_csv(report_dir / output_name, summary, SUMMARY_FIELDS)
    return summary


def select_global_executor(summary: list[dict]) -> dict | None:
    if not summary:
        return None
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary:
        grouped[(str(row["candidates"]), str(row["margin"]))].append(row)
    candidates = []
    for (cand, margin), rows in grouped.items():
        filtered_evr = mean([float(r["filtered_evr"]) for r in rows])
        filtered_cost = mean([float(r["filtered_cost"]) for r in rows])
        return_drop = mean([float(r["return_drop_frac"]) for r in rows])
        fallback = mean([float(r["fallback_rate"]) for r in rows])
        raw_cost = mean([float(r["raw_cost"]) for r in rows])
        raw_evr = mean([float(r["raw_evr"]) for r in rows])
        candidates.append(
            {
                "candidates": int(float(cand)),
                "margin": float(margin),
                "filtered_evr": filtered_evr,
                "filtered_cost": filtered_cost,
                "return_drop_frac": return_drop,
                "fallback_rate": fallback,
                "raw_cost": raw_cost,
                "raw_evr": raw_evr,
                # Lexicographic score: safety first, then cost, then reward
                # retention and fallback rate.
                "score": (filtered_evr, filtered_cost, max(0.0, return_drop), fallback),
            }
        )
    return sorted(candidates, key=lambda r: r["score"])[0]


def write_selection(report_dir: Path, selected: dict | None, validation_summary: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    if selected is None:
        text = "# Executor Selection\n\nNo validation rows were available; no executor config selected.\n"
        (report_dir / "selection.md").write_text(text, encoding="utf-8")
        return
    config_path = Path("configs/star_v2_selected_executor.json")
    config = {
        "schema": "star_v2_selected_executor",
        "selection_status": "selected",
        "selection_source": str(report_dir / "executor_grid_validation.csv"),
        "candidate_grid": sorted({int(float(row["candidates"])) for row in validation_summary}),
        "margin_grid": sorted({float(row["margin"]) for row in validation_summary}),
        "selected": {
            "star_exec_candidates": int(selected["candidates"]),
            "star_exec_margin": float(selected["margin"]),
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Executor Selection",
        "",
        "Selected one global STAR+Exec configuration using validation seeds only.",
        "",
        f"- candidates: `{selected['candidates']}`",
        f"- margin: `{selected['margin']}`",
        f"- validation_filtered_evr: `{selected['filtered_evr']:.6g}`",
        f"- validation_filtered_cost: `{selected['filtered_cost']:.6g}`",
        f"- validation_return_drop_frac: `{selected['return_drop_frac']:.6g}`",
        f"- validation_fallback_rate: `{selected['fallback_rate']:.6g}`",
        f"- config: `{config_path}`",
        "",
    ]
    (report_dir / "selection.md").write_text("\n".join(lines), encoding="utf-8")


def write_confirmation(report_dir: Path, summary: list[dict], selected: dict | None) -> None:
    if selected is None:
        text = "# Executor Confirmation\n\nNo selected executor config; confirmation was not run.\n"
    else:
        rows = [
            r
            for r in summary
            if int(float(r["candidates"])) == int(selected["candidates"])
            and abs(float(r["margin"]) - float(selected["margin"])) < 1e-12
        ]
        if not rows:
            text = "# Executor Confirmation\n\nNo confirmation rows were available.\n"
        else:
            text = "\n".join(
                [
                    "# Executor Confirmation",
                    "",
                    f"- candidates: `{selected['candidates']}`",
                    f"- margin: `{selected['margin']}`",
                    f"- tasks: `{len(rows)}`",
                    f"- mean_raw_cost: `{mean([float(r['raw_cost']) for r in rows]):.6g}`",
                    f"- mean_filtered_cost: `{mean([float(r['filtered_cost']) for r in rows]):.6g}`",
                    f"- mean_raw_evr: `{mean([float(r['raw_evr']) for r in rows]):.6g}`",
                    f"- mean_filtered_evr: `{mean([float(r['filtered_evr']) for r in rows]):.6g}`",
                    f"- mean_return_drop_frac: `{mean([float(r['return_drop_frac']) for r in rows]):.6g}`",
                    "",
                ]
            )
    (report_dir / "confirmation.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-seeds", default="200000,200001,200002,200003,200004,200005,200006,200007,200008,200009,200010,200011,200012,200013,200014,200015,200016,200017,200018,200019")
    parser.add_argument("--confirmation-seeds", default="800000,800001,800002,800003,800004,800005,800006,800007,800008,800009,800010,800011,800012,800013,800014,800015,800016,800017,800018,800019")
    parser.add_argument("--candidates", default="8,16")
    parser.add_argument("--margins", default="0.00,0.02,0.05")
    parser.add_argument("--methods", default="star_v2,star_collect_v2")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/star_goal"))
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"warning: ignoring compatibility arguments: {' '.join(unknown)}")
    if args.dry_run:
        print("dry_run=true")
        return
    run_grid(
        args.root,
        args.report_dir,
        parse_seed_list(args.eval_seeds),
        parse_int_list(args.candidates),
        parse_float_list(args.margins),
        parse_str_list(args.methods),
        output_name="executor_grid_validation.csv",
    )
    validation_summary = summarize(
        args.report_dir,
        input_name="executor_grid_validation.csv",
        output_name="executor_grid_validation_summary.csv",
    )
    selected = select_global_executor(validation_summary)
    write_selection(args.report_dir, selected, validation_summary)
    if selected is not None:
        run_grid(
            args.root,
            args.report_dir,
            parse_seed_list(args.confirmation_seeds),
            [int(selected["candidates"])],
            [float(selected["margin"])],
            parse_str_list(args.methods),
            output_name="executor_confirmation.csv",
        )
    confirmation_summary = summarize(
        args.report_dir,
        input_name="executor_confirmation.csv",
        output_name="executor_confirmation_summary.csv",
    )
    write_confirmation(args.report_dir, confirmation_summary, selected)
    # Backward-compatible filenames used by older artifact code.
    validation_path = args.report_dir / "executor_grid_validation.csv"
    if validation_path.exists():
        (args.report_dir / "executor_grid.csv").write_text(validation_path.read_text(), encoding="utf-8")
    validation_summary_path = args.report_dir / "executor_grid_validation_summary.csv"
    if validation_summary_path.exists():
        (args.report_dir / "executor_grid_summary.csv").write_text(validation_summary_path.read_text(), encoding="utf-8")
    print(f"wrote {args.report_dir / 'executor_grid_validation.csv'} and {args.report_dir / 'executor_confirmation.csv'}")
    print(f"wrote {args.report_dir / 'executor_grid_summary.csv'}")


if __name__ == "__main__":
    main()
