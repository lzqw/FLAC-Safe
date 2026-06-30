#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


STEP_COLUMNS = [
    "step",
    "global_step",
    "env_step",
    "env_steps",
    "total_steps",
    "timestep",
    "Timesteps",
    "Epoch",
    "Step",
    "end_step",
]
REWARD_COLUMNS = [
    "train_episode_reward",
    "train_return",
    "EpRet",
    "episode_reward",
    "eval_return",
    "raw_return",
    "reward",
    "return",
]
COST_COLUMNS = [
    "train_episode_cost",
    "train_cost",
    "EpCost",
    "episode_cost",
    "eval_cost",
    "raw_cost",
    "cost",
    "total_cost",
    "cumulative_cost",
    "train_total_cost",
    "train_total_cost_rate",
]
SKIP_SOURCE_NAMES = {
    "summary_by_seed.csv",
    "main_results_summary.csv",
    "main_results_by_seed.csv",
    "run_manifest.csv",
    "training_safety_by_seed.csv",
    "ablation_summary.csv",
}
METHODS = ["pointwise_v2", "current_only_v2", "sac_lag", "star_v2"]


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", errors="ignore") as handle:
        sample = handle.read(8192)
    if not sample.strip():
        return []
    first = sample.splitlines()[0]
    return next(csv.reader([first]))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return {}


def infer_from_path(path: Path) -> dict:
    text = path.as_posix()
    parts = path.parts
    out = {"run_dir": "", "phase": "", "task": "", "method": "", "seed": ""}
    for phase in ("resume_300k", "core_100k", "ablation_100k"):
        if phase in parts:
            out["phase"] = phase
            idx = parts.index(phase)
            if len(parts) > idx + 3:
                out["task"] = parts[idx + 1]
                out["method"] = parts[idx + 2]
                out["run_dir"] = str(Path(*parts[: idx + 4]))
            break
        if f"/{phase}/" in text:
            out["phase"] = phase
    if not out["run_dir"] and path.name in {"train_episodes.csv", "eval_episodes.csv", "corrected_eval_episodes.csv"}:
        out["run_dir"] = str(path.parent)
    meta = read_json(Path(out["run_dir"]) / "run_metadata.json") if out["run_dir"] else {}
    if meta:
        out["task"] = str(meta.get("task") or out["task"])
        out["method"] = str(meta.get("method") or out["method"])
        out["seed"] = str(meta.get("seed") or out["seed"])
        out["phase"] = str(meta.get("phase") or meta.get("ablation_group") or out["phase"])
    if not out["method"]:
        for method in METHODS:
            if method in text:
                out["method"] = method
                break
    if not out["seed"]:
        match = re.search(r"(?:^|[_/-])s(?:eed)?_?(\d+)(?:\D|$)", text)
        if not match:
            match = re.search(r"(?:^|[_/-])seed[=_-]?(\d+)(?:\D|$)", text)
        if match:
            out["seed"] = match.group(1)
    if not out["task"]:
        match = re.search(r"Safety(?:Point|Car)(?:Goal|Button)1[-_]v0", text)
        if match:
            out["task"] = match.group(0).replace("_", "-")
    return out


def inspect_csv(path: Path) -> tuple[str, list[str], list[str], str]:
    try:
        header = read_csv_header(path)
    except Exception as exc:
        return "", [], [], f"csv_read_error:{exc}"
    step = next((col for col in STEP_COLUMNS if col in header), "")
    rewards = [col for col in REWARD_COLUMNS if col in header]
    costs = [col for col in COST_COLUMNS if col in header]
    if not step:
        return "", rewards, costs, "missing_step_column"
    if not rewards and not costs:
        return step, rewards, costs, "missing_reward_or_cost_column"
    return step, rewards, costs, ""


def inspect_text_log(path: Path) -> tuple[str, list[str], list[str], str]:
    try:
        text = path.read_text(errors="ignore")
    except Exception as exc:
        return "", [], [], f"log_read_error:{exc}"
    head = text[:2_000_000]
    has_step = bool(re.search(r"\b(?:step|global_step|env_step|end_step)=", head))
    rewards = ["episode_reward"] if re.search(r"\b(?:episode_reward|reward|return)=", head) else []
    costs = ["episode_cost"] if re.search(r"\b(?:episode_cost|cost|train_total_cost)=", head) else []
    if not has_step:
        return "", rewards, costs, "missing_step_token"
    if not rewards and not costs:
        return "step", rewards, costs, "missing_reward_or_cost_token"
    return "step", rewards, costs, ""


def inspect_tensorboard(path: Path) -> tuple[str, list[str], list[str], str]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        return "", [], [], f"tensorboard_unavailable:{exc}"
    try:
        ea = event_accumulator.EventAccumulator(str(path), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
    except Exception as exc:
        return "", [], [], f"tensorboard_read_error:{exc}"
    rewards = [tag for tag in tags if any(key.lower() in tag.lower() for key in ("reward", "return", "epr"))]
    costs = [tag for tag in tags if "cost" in tag.lower()]
    if not rewards and not costs:
        return "", [], [], "missing_reward_or_cost_scalar"
    return "event_step", rewards, costs, ""


def inspect_file(path: Path) -> dict:
    inferred = infer_from_path(path)
    source_type = path.suffix.lstrip(".") or path.name
    if path.name in SKIP_SOURCE_NAMES or "reports/star_v2_final" in path.as_posix() and path.name in SKIP_SOURCE_NAMES:
        step, rewards, costs, reason = "", [], [], "summary_file_excluded"
    elif path.name.startswith("events.out.tfevents"):
        source_type = "tensorboard"
        step, rewards, costs, reason = inspect_tensorboard(path)
    elif path.suffix == ".csv":
        source_type = "csv"
        step, rewards, costs, reason = inspect_csv(path)
    elif path.suffix in {".log", ".jsonl", ".txt"}:
        source_type = path.suffix.lstrip(".")
        step, rewards, costs, reason = inspect_text_log(path)
    else:
        step, rewards, costs, reason = "", [], [], "unsupported_extension"
    usable = bool(step and (rewards or costs) and not reason)
    return {
        "file_path": str(path),
        "run_dir": inferred["run_dir"],
        "phase": inferred["phase"],
        "task": inferred["task"],
        "method": inferred["method"],
        "seed": inferred["seed"],
        "source_type": source_type,
        "step_column": step,
        "reward_columns": ";".join(rewards),
        "cost_columns": ";".join(costs),
        "usable": str(usable),
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-file", default="reports/star_v2_final/curve_audit/candidate_curve_files.txt")
    parser.add_argument("--output", default="reports/star_v2_final/curve_audit/curve_source_inventory.csv")
    args = parser.parse_args()
    candidates = []
    candidate_path = Path(args.candidate_file)
    if candidate_path.exists():
        candidates = [Path(line.strip()) for line in candidate_path.read_text().splitlines() if line.strip()]
    rows = [inspect_file(path) for path in candidates if path.exists()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file_path",
        "run_dir",
        "phase",
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
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    usable = sum(row["usable"] == "True" for row in rows)
    print(f"wrote {out} rows={len(rows)} usable={usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
