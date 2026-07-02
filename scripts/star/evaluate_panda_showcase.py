#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.star.find_panda_mechanism_episode import close_runs, evaluate_episode, load_run, load_run_dir

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
SHOWCASE_ROOT = REPORT_ROOT / "showcase"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "episode_return",
        "success",
        "episode_cost",
        "violation_rate",
        "min_clearance",
        "path_length",
        "episode_length",
    ]
    grouped = rows.groupby("method_label", sort=False)
    parts = []
    for metric in metrics:
        stat = grouped[metric].agg(["mean", "std", "count"]).reset_index()
        stat["sem"] = stat["std"] / stat["count"].pow(0.5)
        stat = stat.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "sem": f"{metric}_sem",
                "count": f"{metric}_n",
            }
        )
        keep = ["method_label", f"{metric}_mean", f"{metric}_std", f"{metric}_sem", f"{metric}_n"]
        parts.append(stat[keep])
    summary = parts[0]
    for part in parts[1:]:
        summary = summary.merge(part, on="method_label", how="outer")
    return summary


def write_showcase_definition(out_dir: Path, args: argparse.Namespace, rows: pd.DataFrame) -> None:
    star_rows = rows[rows["method_label"].isin(["STAR", "STAR+Exec"])]
    run_name = star_rows["run_name"].iloc[0] if not star_rows.empty else Path(args.star_run_dir).name
    checkpoint = star_rows["checkpoint_path"].iloc[0] if not star_rows.empty else ""
    starts = rows[["start_pos", "goal_pos", "obstacle_pos", "obstacle_radius", "safe_margin", "action_scale"]].head(1).to_dict("records")
    lines = [
        "# Panda Ref-Corridor Showcase Evaluation",
        "",
        "This is a controlled checkpoint/configuration showcase for mechanism visualization. It does not replace the main Panda quantitative benchmark.",
        "",
        f"- task: `SafetyPandaReachObstacle-v0`",
        f"- STAR run: `{run_name}`",
        f"- STAR checkpoint: `{checkpoint}`",
        f"- eval seeds: `{args.eval_start}` through `{args.eval_start + args.eval_count - 1}`",
        "- environment code: unchanged base Panda obstacle-reaching environment; the showcase uses a targeted ref-corridor STAR checkpoint and fixed evaluation seeds.",
        "",
        "Representative geometry from the first evaluation row:",
        "",
        "```json",
        json.dumps(starts[0] if starts else {}, indent=2, sort_keys=True),
        "```",
    ]
    (out_dir / "showcase_env_definition.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--star-run-dir", required=True)
    parser.add_argument("--compare-seed", type=int, default=11)
    parser.add_argument("--eval-start", type=int, default=910000)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--output-dir", default=str(SHOWCASE_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        load_run("sac_lag", args.compare_seed, mode="raw"),
        load_run("current_only_v2", args.compare_seed, mode="raw"),
        load_run_dir(args.star_run_dir, label="STAR", mode="raw"),
        load_run_dir(args.star_run_dir, label="STAR", mode="star_exec"),
    ]
    episode_rows = []
    trajectory_rows = []
    audit_rows = []
    try:
        for eval_seed in range(args.eval_start, args.eval_start + args.eval_count):
            for run in runs:
                capture = run.method == "star_v2" and run.mode == "raw"
                episode, trajectory, states = evaluate_episode(run, eval_seed, capture_audit=capture)
                episode_rows.append(episode)
                trajectory_rows.extend(trajectory)
                for state in states:
                    state = dict(state)
                    state.pop("audit_items_json", None)
                    audit_rows.append(state)
    finally:
        close_runs(runs)

    episodes = pd.DataFrame(episode_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    audits = pd.DataFrame(audit_rows)

    episodes.to_csv(out_dir / "showcase_results_by_seed.csv", index=False)
    trajectories.to_csv(out_dir / "showcase_trajectories.csv", index=False)
    audits.to_csv(out_dir / "showcase_audit_snapshots.csv", index=False)
    summarize(episodes).to_csv(out_dir / "showcase_summary.csv", index=False)
    write_showcase_definition(out_dir, args, episodes)

    print(json.dumps({"episodes": len(episodes), "trajectories": len(trajectories), "audits": len(audits), "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
