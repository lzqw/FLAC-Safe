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
CONFIG_OUT = REPORT / "selected_actor_config.json"


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
    return [v for v in values if not math.isnan(v)]


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


def config_params(config_name: str) -> dict:
    params = {
        "cost_gamma": 0.99,
        "star_risk_threshold": 0.10,
        "star_lambda": 1.0,
        "shadow_k": 16,
        "shadow_temperature": 0.05,
        "shadow_aggregation": "log_mean_exp",
        "shadow_reference_mode": "corridor",
        "star_ref_update_interval": 20,
        "star_kl_coef": 1.0,
        "star_kl_target": 0.01,
        "cost_critic_reduce": "max",
    }
    if config_name == "threshold_005":
        params["star_risk_threshold"] = 0.05
    elif config_name == "threshold_020":
        params["star_risk_threshold"] = 0.20
    elif config_name == "lambda_050":
        params["star_lambda"] = 0.5
    elif config_name == "lambda_200":
        params["star_lambda"] = 2.0
    elif config_name == "k8":
        params["shadow_k"] = 8
    elif config_name == "k32":
        params["shadow_k"] = 32
    elif config_name == "temperature_003":
        params["shadow_temperature"] = 0.03
    elif config_name == "temperature_010":
        params["shadow_temperature"] = 0.10
    elif config_name == "ref_interval10":
        params["star_ref_update_interval"] = 10
    elif config_name == "ref_interval50":
        params["star_ref_update_interval"] = 50
    elif config_name == "kl_off":
        params["star_kl_coef"] = 0.0
        params["star_kl_target"] = 0.0
    return params


KNOWN_CONFIGS = [
    "temperature_003",
    "temperature_010",
    "ref_interval10",
    "ref_interval50",
    "threshold_005",
    "threshold_020",
    "lambda_050",
    "lambda_200",
    "default",
    "k32",
    "k8",
    "kl_off",
]


def config_from_run_name(run_name: str) -> str:
    for config in KNOWN_CONFIGS:
        if f"actor_stage1_{config}_" in run_name:
            return config
    return ""


def merge_diagnostics(eval_rows: list[dict], risk_rows: list[dict]) -> list[dict]:
    risk_by_run = {row["run_name"]: row for row in risk_rows}
    rows = []
    for row in eval_rows:
        merged = dict(row)
        diag = risk_by_run.get(row.get("run_name", ""), {})
        for key, value in diag.items():
            if key not in merged:
                merged[key] = value
        rows.append(merged)
    return rows


def completed_clean(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if to_int(row.get("completed")) == 1
        and to_int(row.get("has_error")) == 0
        and row.get("decision") != "reject_error"
    ]


def raw_reward(row: dict) -> float:
    value = to_float(row.get("raw_return"))
    return value if not math.isnan(value) else to_float(row.get("train_avg_last3_reward"))


def raw_cost(row: dict) -> float:
    value = to_float(row.get("raw_cost"))
    return value if not math.isnan(value) else to_float(row.get("train_avg_last3_cost"))


def pareto_score(row: dict) -> float:
    reward = raw_reward(row)
    cost = raw_cost(row)
    audit_gap = to_float(row.get("audit_gap_mean"))
    audit_pos = to_float(row.get("audit_gap_positive_rate"))
    hidden = to_float(row.get("hidden_unsafe_rate"))
    if math.isnan(hidden):
        hidden = to_float(row.get("hidden_unsafe_rate_x", 0.0))
    score = reward - 0.05 * cost
    if not math.isnan(audit_gap):
        score += min(max(audit_gap, -1.0), 1.0)
    if not math.isnan(audit_pos):
        score += audit_pos
    if not math.isnan(hidden):
        score += hidden
    return score


def task_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("config", "")].append(row)
    return groups


def failure_reasons(items: list[dict]) -> list[str]:
    reasons = []
    if len(items) < 2:
        reasons.append("missing_task")
    for row in items:
        if raw_reward(row) < 5:
            reasons.append(f"{row.get('task')}:return_collapse")
        if raw_cost(row) > 150:
            reasons.append(f"{row.get('task')}:cost_exploded")
        if math.isnan(to_float(row.get("candidate_spread"))):
            reasons.append(f"{row.get('task')}:spread_nan")
        if abs(to_float(row.get("kl_mean"), 0.0)) > 10:
            reasons.append(f"{row.get('task')}:kl_extreme")
        if abs(to_float(row.get("audit_gap_mean"), 0.0)) < 1e-8:
            reasons.append(f"{row.get('task')}:audit_gap_zero")
    return reasons


def candidate_rows(rows: list[dict]) -> list[dict]:
    out = []
    for config, items in task_groups(rows).items():
        if len({item.get("task", "") for item in items}) < 2:
            continue
        reasons = failure_reasons(items)
        params = config_params(config)
        row = {
            "config": config,
            "tasks": ",".join(sorted(item.get("task", "") for item in items)),
            "mean_raw_return": avg([raw_reward(item) for item in items]),
            "mean_raw_cost": avg([raw_cost(item) for item in items]),
            "mean_train_cost_rate": avg([to_float(item.get("train_cost_rate")) for item in items]),
            "mean_audit_gap": avg([to_float(item.get("audit_gap_mean")) for item in items]),
            "mean_audit_gap_positive_rate": avg([to_float(item.get("audit_gap_positive_rate")) for item in items]),
            "mean_hidden_unsafe_rate": avg([to_float(item.get("hidden_unsafe_rate")) for item in items]),
            "mean_pSVR": avg([to_float(item.get("pSVR")) for item in items]),
            "mean_rho_active_rate": math.nan,
            "shadow_k": params["shadow_k"],
            "score": avg([pareto_score(item) for item in items]),
            "reasons": ";".join(sorted(set(reasons))),
            "decision": "candidate" if not reasons else "filtered",
        }
        out.append(row)
    return sorted(out, key=lambda row: (row["decision"] != "candidate", -to_float(row["score"], -1e9), to_int(row["shadow_k"], 999)))


def saturation_status(grid_rows: list[dict]) -> tuple[bool, str, float, float]:
    anchor_configs = {"default", "threshold_005", "threshold_020"}
    anchor_threshold = {"default": 0.10, "threshold_005": 0.05, "threshold_020": 0.20}
    rows = [row for row in grid_rows if config_from_run_name(row.get("run_name", "")) in anchor_configs]
    selected = []
    for row in rows:
        name = row.get("run_name", "")
        cfg = config_from_run_name(name)
        if cfg and abs(to_float(row.get("threshold")) - anchor_threshold[cfg]) < 1e-9:
            selected.append(row)
    rates = [to_float(row.get("rho_active_rate")) for row in selected]
    saturated = bool(rates) and all(rate > 0.98 for rate in rates if not math.isnan(rate))
    pooled = defaultdict(list)
    for row in grid_rows:
        th = round(to_float(row.get("threshold")), 2)
        rate = to_float(row.get("rho_active_rate"))
        if not math.isnan(rate):
            pooled[th].append(rate)
    pooled_rates = {th: avg(vals) for th, vals in pooled.items()}
    t_mid = min(pooled_rates, key=lambda th: abs(pooled_rates[th] - 0.50)) if pooled_rates else math.nan
    t_high = min(pooled_rates, key=lambda th: abs(pooled_rates[th] - 0.25)) if pooled_rates else math.nan
    note = f"anchor_rho_active_rates={','.join(fmt(r) for r in rates)}"
    return saturated, note, t_mid, t_high


def write_markdown(report_dir: Path, candidates: list[dict], saturated: bool, saturation_note: str, t_mid: float, t_high: float) -> None:
    lines = [
        "# Stage1 Decision",
        "",
        "Stage1 is a screening stage only. It may filter failed configurations and identify risk scale, but it does not establish final best parameters.",
        "",
        "## Risk Scale",
        "",
        f"- pSVR/rho-active saturated on anchor configs: `{saturated}`",
        f"- Details: {saturation_note}",
        f"- T_mid candidate: `{fmt(t_mid, 2)}`",
        f"- T_high candidate: `{fmt(t_high, 2)}`",
        "",
        "## Candidate Screening",
        "",
        "| Config | Decision | Return | Cost | Cost Rate | Audit Gap | Gap+ | Hidden Unsafe | pSVR | K | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidates:
        lines.append(
            f"| {row['config']} | {row['decision']} | {fmt(row['mean_raw_return'])} | {fmt(row['mean_raw_cost'])} | "
            f"{fmt(row['mean_train_cost_rate'], 4)} | {fmt(row['mean_audit_gap'])} | "
            f"{fmt(row['mean_audit_gap_positive_rate'])} | {fmt(row['mean_hidden_unsafe_rate'])} | "
            f"{fmt(row['mean_pSVR'])} | {row['shadow_k']} | {row['reasons']} |"
        )
    if saturated:
        lines.extend([
            "",
            "## Decision",
            "",
            "Risk-scale calibration is required before selecting the global actor configuration.",
        ])
    else:
        kept = [row for row in candidates if row["decision"] == "candidate"][:3]
        lines.extend(["", "## Decision", "", "Retain up to three candidates for the next stage:"])
        for row in kept:
            lines.append(f"- `{row['config']}`")
    (report_dir / "stage1_decision.md").write_text("\n".join(lines) + "\n")


def maybe_write_selected(report_dir: Path, candidates: list[dict], saturated: bool) -> None:
    if saturated:
        return
    kept = [row for row in candidates if row["decision"] == "candidate"]
    if not kept:
        return
    chosen = kept[0]["config"]
    data = config_params(chosen)
    data["selected_from_stage1_config"] = chosen
    data["selection_stage"] = "actor_stage1"
    (report_dir / "selected_actor_config.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def maybe_write_thresholds(report_dir: Path, saturated: bool, t_mid: float, t_high: float) -> None:
    if math.isnan(t_mid) or math.isnan(t_high):
        return
    data = {
        "pSVR_rho_active_saturated": bool(saturated),
        "T_mid": t_mid,
        "T_high": t_high,
        "threshold_grid": str(report_dir / "shadow_threshold_grid.csv"),
    }
    (report_dir / "stage1_decision_thresholds.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    args = parser.parse_args()
    eval_rows = read_csv(args.report_dir / "actor_stage1_eval_summary.csv")
    risk_rows = read_csv(args.report_dir / "shadow_risk_summary.csv")
    grid_rows = read_csv(args.report_dir / "shadow_threshold_grid.csv")
    if not eval_rows:
        raise SystemExit("missing actor_stage1_eval_summary.csv")
    rows = merge_diagnostics(completed_clean(eval_rows), risk_rows)
    if not risk_rows or not grid_rows:
        print("Stage1 decision pending: missing shadow risk diagnostics")
    candidates = candidate_rows(rows)
    saturated, note, t_mid, t_high = saturation_status(grid_rows)
    fields = [
        "config", "tasks", "mean_raw_return", "mean_raw_cost", "mean_train_cost_rate",
        "mean_audit_gap", "mean_audit_gap_positive_rate", "mean_hidden_unsafe_rate",
        "mean_pSVR", "mean_rho_active_rate", "shadow_k", "score", "reasons", "decision",
    ]
    write_csv(args.report_dir / "stage1_candidates.csv", candidates, fields)
    write_markdown(args.report_dir, candidates, saturated, note, t_mid, t_high)
    maybe_write_thresholds(args.report_dir, saturated, t_mid, t_high)
    maybe_write_selected(args.report_dir, candidates, saturated)
    print(f"wrote {args.report_dir / 'stage1_candidates.csv'}")
    print(f"wrote {args.report_dir / 'stage1_decision.md'}")


if __name__ == "__main__":
    main()
