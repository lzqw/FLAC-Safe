#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / "results" / "star_1m_curves"
REPORT_ROOT = REPO / "reports" / "star_1m_curves"
FINAL_ROOT = REPORT_ROOT / "final"
TABLE1_REFERENCE = REPO / "reports" / "star_v2_final" / "main_results_summary.csv"

TASK_LABELS = {
    "SafetyPointGoal1-v0": "PointGoal1",
    "SafetyCarGoal1-v0": "CarGoal1",
    "SafetyPointPush1-v0": "PointPush1",
}
METHOD_LABELS = {
    "star_v2": "STAR",
    "sac_lag": "SAC-Lag",
    "safe_flow_q": "Safe Flow Q",
    "ppo_lag": "PPO-Lag",
    "cpo": "CPO",
    "cspo": "CSPO",
}


def run_eval(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/star/reevaluate_checkpoints.py",
        "--root",
        str(RESULT_ROOT),
        "--checkpoint-selector",
        "final",
        "--modes",
        "raw",
        "--eval-seeds",
        args.eval_seeds,
    ]
    if args.overwrite_derived:
        cmd.append("--overwrite-derived")
    subprocess.run(cmd, cwd=REPO, check=True)


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def collect_eval_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULT_ROOT.rglob("corrected_eval_episodes.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        meta = load_metadata(path.parent)
        df["run_dir"] = str(path.parent)
        df["stage"] = str(meta.get("ablation_group", ""))
        df["train_task"] = str(meta.get("task", df.get("task", "")))
        df["train_method"] = str(meta.get("method", df.get("method", "")))
        df["train_seed"] = meta.get("seed", df.get("seed", ""))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "mode" in out.columns:
        out = out[out["mode"].astype(str) == "raw"]
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        seed_columns = [
            "stage",
            "task",
            "task_label",
            "method",
            "method_label",
            "seed",
            "eval_episodes",
            "return_mean",
            "return_std",
            "cost_mean",
            "cost_std",
            "constraint_satisfaction_rate",
            "checkpoint_path",
            "run_dir",
        ]
        summary_columns = [
            "task",
            "task_label",
            "method",
            "method_label",
            "seeds",
            "return_mean",
            "return_std",
            "cost_mean",
            "cost_std",
            "constraint_satisfaction_rate",
        ]
        return pd.DataFrame(columns=seed_columns), pd.DataFrame(columns=summary_columns)

    group_cols = ["stage", "train_task", "train_method", "train_seed", "run_dir"]
    if "checkpoint_path" in df.columns:
        group_cols.append("checkpoint_path")
    seed_rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        data = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        seed_rows.append(
            {
                "stage": data.get("stage", ""),
                "task": data.get("train_task", ""),
                "task_label": TASK_LABELS.get(str(data.get("train_task", "")), str(data.get("train_task", ""))),
                "method": data.get("train_method", ""),
                "method_label": METHOD_LABELS.get(str(data.get("train_method", "")), str(data.get("train_method", ""))),
                "seed": data.get("train_seed", ""),
                "eval_episodes": len(sub),
                "return_mean": sub["episode_reward"].mean(),
                "return_std": sub["episode_reward"].std(ddof=1),
                "cost_mean": sub["episode_cost"].mean(),
                "cost_std": sub["episode_cost"].std(ddof=1),
                "constraint_satisfaction_rate": sub["constraint_satisfied"].mean()
                if "constraint_satisfied" in sub.columns
                else "",
                "checkpoint_path": data.get("checkpoint_path", ""),
                "run_dir": data.get("run_dir", ""),
            }
        )
    by_seed = pd.DataFrame(seed_rows).sort_values(["task", "method", "seed"])

    summary_rows = []
    for (task, method), sub in by_seed.groupby(["task", "method"], dropna=False):
        summary_rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS.get(str(task), str(task)),
                "method": method,
                "method_label": METHOD_LABELS.get(str(method), str(method)),
                "seeds": len(sub),
                "return_mean": sub["return_mean"].mean(),
                "return_std": sub["return_mean"].std(ddof=1),
                "cost_mean": sub["cost_mean"].mean(),
                "cost_std": sub["cost_mean"].std(ddof=1),
                "constraint_satisfaction_rate": sub["constraint_satisfaction_rate"].mean()
                if "constraint_satisfaction_rate" in sub.columns
                else "",
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["task", "method"])
    return by_seed, summary


def write_discrepancy(summary: pd.DataFrame) -> None:
    path = FINAL_ROOT / "table1_discrepancy.md"
    lines = ["# Table 1 Consistency Check", ""]
    if summary.empty:
        lines.append("No completed final raw evaluation rows are available yet.")
        path.write_text("\n".join(lines) + "\n")
        return
    if not TABLE1_REFERENCE.exists():
        lines.append(f"Reference file missing: `{TABLE1_REFERENCE.relative_to(REPO)}`.")
        path.write_text("\n".join(lines) + "\n")
        return
    reference = pd.read_csv(TABLE1_REFERENCE)
    reference = reference[reference["phase"].astype(str) == "resume_300k"].copy()
    lines.extend(
        [
            f"Reference file: `{TABLE1_REFERENCE.relative_to(REPO)}` filtered to `phase=resume_300k`.",
            "",
            "| task | method | 1M return | ref return | delta return | 1M cost | ref cost | delta cost | note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary.iterrows():
        ref = reference[
            (reference["task"].astype(str) == str(row["task"]))
            & (reference["method"].astype(str) == str(row["method"]))
        ]
        if ref.empty:
            lines.append(
                f"| {row['task']} | {row['method']} | {row['return_mean']:.4g} |  |  | {row['cost_mean']:.4g} |  |  | no reference row |"
            )
            continue
        ref_row = ref.iloc[0]
        d_return = float(row["return_mean"]) - float(ref_row["raw_return_mean"])
        d_cost = float(row["cost_mean"]) - float(ref_row["raw_cost_mean"])
        note = "review" if abs(d_return) > 5 or abs(d_cost) > 25 else "close"
        lines.append(
            "| {task} | {method} | {ret:.4g} | {ref_ret:.4g} | {dret:.4g} | {cost:.4g} | {ref_cost:.4g} | {dcost:.4g} | {note} |".format(
                task=row["task"],
                method=row["method"],
                ret=float(row["return_mean"]),
                ref_ret=float(ref_row["raw_return_mean"]),
                dret=d_return,
                cost=float(row["cost_mean"]),
                ref_cost=float(ref_row["raw_cost_mean"]),
                dcost=d_cost,
                note=note,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-seeds", default="100000,100001,100002,100003,100004,100005,100006,100007,100008,100009")
    parser.add_argument("--overwrite-derived", action="store_true")
    args = parser.parse_args()

    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    if args.run_eval:
        run_eval(args)
    raw = collect_eval_rows()
    by_seed, summary = summarize(raw)
    raw.to_csv(FINAL_ROOT / "final_eval_episodes.csv", index=False)
    by_seed.to_csv(FINAL_ROOT / "final_eval_by_seed.csv", index=False)
    summary.to_csv(FINAL_ROOT / "final_eval_summary.csv", index=False)
    write_discrepancy(summary)
    print(f"wrote final eval summaries: episodes={len(raw)} seeds={len(by_seed)} groups={len(summary)}")


if __name__ == "__main__":
    main()
