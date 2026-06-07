#!/usr/bin/env python3
"""Collect fixed G4 transfer experiment metrics and decisions."""

from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "transfer_g4_fixed"
POINTGOAL_REF_LOG_DIR = ROOT / "logs" / "pointgoal_goal_tuning"
REPORT_DIR = ROOT / "reports" / "transfer_g4_fixed"
SUMMARY = REPORT_DIR / "summary.md"
DECISION_LOG = REPORT_DIR / "decision_log.md"
MONITOR = LOG_DIR / "gpu_monitor.csv"

POINTGOAL_COST_THRESHOLD = 45.64
POINTGOAL_REWARD_THRESHOLD = 19.85
POINTGOAL_NEAR_COST = 48.0
POINTGOAL_NEAR_REWARD = 18.0

REFERENCE_FALLBACK = {
    "avg_last3_reward": 24.16,
    "avg_last3_reward_std": 1.04,
    "avg_last3_cost": 44.29,
    "avg_last3_cost_std": 1.67,
}

ENVS = {
    "T0_PointGoal1_ref": {
        "task": "SafetyPointGoal1-v0",
        "display": "PointGoal1_ref",
        "seeds": [0, 1, 2],
        "source": "pointgoal_goal_tuning",
        "log_prefix": "G4_cdf_mean_thr005_safe07",
    },
    "T1_PointGoal2": {
        "task": "SafetyPointGoal2-v0",
        "display": "PointGoal2",
        "seeds": [0, 1, 2],
        "source": "transfer",
    },
    "T2_CarGoal1": {
        "task": "SafetyCarGoal1-v0",
        "display": "CarGoal1",
        "seeds": [0, 1, 2],
        "source": "transfer",
    },
    "T3_CarGoal2": {
        "task": "SafetyCarGoal2-v0",
        "display": "CarGoal2",
        "seeds": [0, 1, 2],
        "source": "transfer",
    },
    "T4_SwimmerVelocity": {
        "task": "SafetySwimmerVelocity-v1/v0",
        "display": "SwimmerVelocity",
        "seeds": [0, 1, 2],
        "source": "transfer",
    },
}

EVAL_RE = re.compile(
    r"Avg\. Reward:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Cost:\s*([-+]?\d+(?:\.\d+)?),\s*"
    r"Avg\. Success:\s*([-+]?\d+(?:\.\d+)?)"
)
TASK_RE = re.compile(r"task=([A-Za-z0-9_\-]+-v\d+)")
ERR_RE = re.compile(r"Traceback|RuntimeError|NaN|nan|OOM|out of memory")
NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
METRICS = [
    "loss/jvp_weighted",
    "safety/qc_target_mean",
    "safety/qc_pi_risk_over_threshold",
]


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def sci(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3e}"


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), stdev(clean)


def fmt_mean_std(pair: tuple[float | None, float | None]) -> str:
    return f"{fmt(pair[0])} / {fmt(pair[1])}"


def parse_metric(text: str, key: str) -> float | None:
    values = re.findall(r"wandb:\s+" + re.escape(key) + r"\s+(" + NUM_RE + r")\b", text, flags=re.I)
    return float(values[-1]) if values else None


def log_path(env_key: str, seed: int) -> Path:
    cfg = ENVS[env_key]
    if cfg["source"] == "pointgoal_goal_tuning":
        return POINTGOAL_REF_LOG_DIR / f"{cfg['log_prefix']}_seed{seed}.log"
    return LOG_DIR / f"{env_key}_G4_fixed_main_seed{seed}.log"


def parse_log(env_key: str, seed: int) -> dict[str, object]:
    path = log_path(env_key, seed)
    row: dict[str, object] = {
        "env": env_key,
        "seed": seed,
        "path": str(path),
        "task": ENVS[env_key]["task"],
    }
    if not path.exists():
        row.update({"status": "missing", "evals": []})
        return row
    text = path.read_text(errors="replace")
    evals = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in EVAL_RE.finditer(text)]
    task_match = TASK_RE.search(text)
    if task_match:
        row["task"] = task_match.group(1)
    lowered = text.lower()
    if "oom" in lowered or "out of memory" in lowered:
        status = "failed_oom"
    elif "nan" in text or "NaN" in text:
        status = "failed_nan"
    elif ERR_RE.search(text):
        status = "failed_error"
    elif " END " in text and evals:
        status = "completed"
    elif evals:
        status = "partial"
    else:
        status = "no evals"
    row.update(
        {
            "status": status,
            "evals": evals,
            "final_reward": evals[-1][0] if evals else None,
            "final_cost": evals[-1][1] if evals else None,
            "avg_last3_reward": mean(v[0] for v in evals[-3:]) if evals else None,
            "avg_last3_cost": mean(v[1] for v in evals[-3:]) if evals else None,
            "first3_reward": mean(v[0] for v in evals[:3]) if evals else None,
            "first3_cost": mean(v[1] for v in evals[:3]) if evals else None,
        }
    )
    for key in METRICS:
        row[key] = parse_metric(text, key)
    return row


def metric_mean(rows: list[dict[str, object]], key: str) -> float | None:
    return mean_std([row.get(key) for row in rows])[0]  # type: ignore[list-item]


def completed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["status"] == "completed"]


def pilot_decision(complete: list[dict[str, object]]) -> str:
    if not complete:
        return "pending"
    rewards = [row.get("avg_last3_reward") for row in complete]
    costs = [row.get("avg_last3_cost") for row in complete]
    first_rewards = [row.get("first3_reward") for row in complete]
    first_costs = [row.get("first3_cost") for row in complete]
    if not all(isinstance(v, float) and math.isfinite(v) and abs(v) < 1e6 for v in rewards + costs):
        return "pilot_bad"
    reward_mean = mean(rewards)  # type: ignore[arg-type]
    cost_mean = mean(costs)  # type: ignore[arg-type]
    first_reward_mean = mean([v for v in first_rewards if isinstance(v, float)]) if any(isinstance(v, float) for v in first_rewards) else reward_mean
    first_cost_mean = mean([v for v in first_costs if isinstance(v, float)]) if any(isinstance(v, float) for v in first_costs) else cost_mean
    reward_non_collapse = reward_mean >= first_reward_mean - max(10.0, abs(first_reward_mean) * 0.5)
    cost_not_exploding = cost_mean <= max(first_cost_mean * 3.0 + 50.0, 500.0)
    reward_improved = reward_mean >= first_reward_mean
    cost_improved = cost_mean <= first_cost_mean
    return "pilot_ok" if reward_non_collapse and cost_not_exploding and (reward_improved or cost_improved) else "pilot_bad"


def group_stats(env_key: str, rows: list[dict[str, object]]) -> dict[str, object]:
    env_rows = [row for row in rows if row["env"] == env_key]
    complete = completed_rows(env_rows)
    failed = [row for row in env_rows if str(row["status"]).startswith("failed")]
    fr = mean_std([row.get("final_reward") for row in complete])  # type: ignore[list-item]
    fc = mean_std([row.get("final_cost") for row in complete])  # type: ignore[list-item]
    ar = mean_std([row.get("avg_last3_reward") for row in complete])  # type: ignore[list-item]
    ac = mean_std([row.get("avg_last3_cost") for row in complete])  # type: ignore[list-item]
    expected = len(ENVS[env_key]["seeds"])
    status = "failed" if failed else ("completed" if len(complete) >= expected else "pending")
    if env_key == "T0_PointGoal1_ref" and not complete:
        ar = (REFERENCE_FALLBACK["avg_last3_reward"], REFERENCE_FALLBACK["avg_last3_reward_std"])
        ac = (REFERENCE_FALLBACK["avg_last3_cost"], REFERENCE_FALLBACK["avg_last3_cost_std"])
        status = "reference_fallback"
    if failed:
        decision = "failed"
    elif len(complete) < expected and env_key != "T0_PointGoal1_ref":
        decision = "pending"
    elif env_key in ("T0_PointGoal1_ref", "T1_PointGoal2"):
        if ac[0] is not None and ar[0] is not None and ac[0] <= POINTGOAL_COST_THRESHOLD and ar[0] >= POINTGOAL_REWARD_THRESHOLD:
            decision = "passed"
        elif ac[0] is not None and ar[0] is not None and ac[0] <= POINTGOAL_NEAR_COST and ar[0] >= POINTGOAL_NEAR_REWARD:
            decision = "near_pass"
        else:
            decision = "not_passed"
    else:
        decision = pilot_decision(complete)
    tasks = sorted({str(row.get("task")) for row in complete if row.get("task")})
    task = tasks[0] if tasks else str(ENVS[env_key]["task"])
    return {
        "rows": env_rows,
        "complete": complete,
        "failed": failed,
        "completed_count": len(complete),
        "expected": expected,
        "task": task,
        "final_reward": fr,
        "final_cost": fc,
        "avg_last3_reward": ar,
        "avg_last3_cost": ac,
        "weighted_jvp": metric_mean(complete, "loss/jvp_weighted"),
        "qc_target_mean": metric_mean(complete, "safety/qc_target_mean"),
        "qc_risk_over_threshold": metric_mean(complete, "safety/qc_pi_risk_over_threshold"),
        "status": status,
        "decision": decision,
    }


def parse_monitor() -> dict[str, str]:
    if not MONITOR.exists():
        return {}
    used = []
    util = []
    total = None
    with MONITOR.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                used.append(float(row["memory.used [MiB]"]))
                util.append(float(row["utilization.gpu [%]"]))
                total = float(row["memory.total [MiB]"])
            except (KeyError, TypeError, ValueError):
                continue
    if not used:
        return {}
    return {
        "gpu_total_memory": fmt(total / 1024 if total else None),
        "peak_memory_used": fmt(max(used) / 1024),
        "avg_memory_used": fmt(sum(used) / len(used) / 1024),
        "avg_gpu_utilization": fmt(sum(util) / len(util)) if util else "n/a",
    }


def compare_harder(a: dict[str, object], b: dict[str, object]) -> str:
    if a["decision"] == "pending" or b["decision"] == "pending":
        return "pending"
    a_reward = a["avg_last3_reward"][0]  # type: ignore[index]
    b_reward = b["avg_last3_reward"][0]  # type: ignore[index]
    a_cost = a["avg_last3_cost"][0]  # type: ignore[index]
    b_cost = b["avg_last3_cost"][0]  # type: ignore[index]
    if None in (a_reward, b_reward, a_cost, b_cost):
        return "not enough data"
    harder = b_cost > a_cost * 1.2 or b_reward < a_reward - max(5.0, abs(a_reward) * 0.2)  # type: ignore[operator]
    return "yes" if harder else "not clearly"


def scale_diff(stats: dict[str, dict[str, object]]) -> str:
    swimmer = stats["T4_SwimmerVelocity"]["avg_last3_cost"][0]  # type: ignore[index]
    goal_costs = [
        stats[key]["avg_last3_cost"][0]  # type: ignore[index]
        for key in ("T1_PointGoal2", "T2_CarGoal1", "T3_CarGoal2")
        if stats[key]["avg_last3_cost"][0] is not None  # type: ignore[index]
    ]
    if swimmer is None or not goal_costs:
        return "pending"
    goal_mean = mean(goal_costs)
    if goal_mean <= 0:
        return "not enough data"
    ratio = swimmer / goal_mean  # type: ignore[operator]
    if ratio >= 2.0 or ratio <= 0.5:
        return f"yes, cost ratio vs transfer-goal mean is {ratio:.2f}"
    return f"not clearly, cost ratio vs transfer-goal mean is {ratio:.2f}"


def write_reports(stats: dict[str, dict[str, object]], rows: list[dict[str, object]], monitor: dict[str, str]) -> None:
    failed_rows = [row for row in rows if str(row["status"]).startswith("failed")]
    pg2 = stats["T1_PointGoal2"]
    car1 = stats["T2_CarGoal1"]
    car2 = stats["T3_CarGoal2"]
    swimmer = stats["T4_SwimmerVelocity"]

    lines = [
        "# Transfer G4 Fixed Summary",
        "",
        "Config: G4_fixed_main, no Stage B, no soft feasibility gate, no risk-side gate, no directional noise rerun.",
        "",
        "| Env | Task | Seeds | Final Reward mean/std | Final Cost mean/std | Avg Last 3 Reward mean/std | Avg Last 3 Cost mean/std | weighted JVP | qc target mean | qc risk over threshold | Status | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for env_key, cfg in ENVS.items():
        stat = stats[env_key]
        lines.append(
            f"| {cfg['display']} | {stat['task']} | {stat['completed_count']}/{stat['expected']} | "
            f"{fmt_mean_std(stat['final_reward'])} | {fmt_mean_std(stat['final_cost'])} | "
            f"{fmt_mean_std(stat['avg_last3_reward'])} | {fmt_mean_std(stat['avg_last3_cost'])} | "
            f"{sci(stat['weighted_jvp'])} | {sci(stat['qc_target_mean'])} | "
            f"{sci(stat['qc_risk_over_threshold'])} | {stat['status']} | {stat['decision']} |"
        )

    lines += [
        "",
        "## Per-Run Results",
        "",
        "| Env | Seed | Task | Final Reward | Final Cost | Avg Last 3 Reward | Avg Last 3 Cost | weighted JVP | qc target mean | qc risk over threshold | Status | Log Path |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {ENVS[str(row['env'])]['display']} | {row['seed']} | {row.get('task', 'n/a')} | "
            f"{fmt(row.get('final_reward'))} | {fmt(row.get('final_cost'))} | "
            f"{fmt(row.get('avg_last3_reward'))} | {fmt(row.get('avg_last3_cost'))} | "
            f"{sci(row.get('loss/jvp_weighted'))} | {sci(row.get('safety/qc_target_mean'))} | "
            f"{sci(row.get('safety/qc_pi_risk_over_threshold'))} | {row['status']} | {row['path']} |"
        )

    pointgoal2_answer = {
        "passed": "yes",
        "near_pass": "near, but not a clean pass",
        "not_passed": "no",
        "pending": "pending",
        "failed": "failed",
    }.get(str(pg2["decision"]), str(pg2["decision"]))
    car_tune = "yes" if car1["decision"] in ("pilot_bad", "failed") else "not immediately"
    swimmer_tune = "yes" if swimmer["decision"] in ("pilot_bad", "failed") or scale_diff(stats).startswith("yes") else "not immediately"

    lines += [
        "",
        "## Transfer interpretation",
        "",
        f"1. G4_fixed direct transfer to PointGoal2: {pointgoal2_answer}.",
        f"2. CarGoal1 separate tuning needed: {car_tune}.",
        f"3. CarGoal2 harder than CarGoal1: {compare_harder(car1, car2)}.",
        f"4. SwimmerVelocity cost scale differs from Goal tasks: {scale_diff(stats)}.",
        f"5. Separate safe_threshold/lambda_safe for Car/Swimmer: Car={car_tune}, Swimmer={swimmer_tune}.",
        "",
        "## GPU",
        "",
        f"- GPU total memory: {monitor.get('gpu_total_memory', 'n/a')} GiB",
        f"- Peak memory.used: {monitor.get('peak_memory_used', 'n/a')} GiB",
        f"- Average memory.used: {monitor.get('avg_memory_used', 'n/a')} GiB",
        f"- Average GPU utilization: {monitor.get('avg_gpu_utilization', 'n/a')}%",
        "",
        "## Error Scan",
    ]
    if failed_rows:
        for row in failed_rows:
            lines.append(f"- {row['path']}: {row['status']}")
    else:
        lines.append("- No error patterns found in parsed logs.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n")

    decision_lines = [
        "# Transfer G4 Fixed Decision Log",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- PointGoal2 transfer: {pointgoal2_answer}",
        f"- CarGoal1 pilot: {car1['decision']}",
        f"- CarGoal2 pilot: {car2['decision']}",
        f"- SwimmerVelocity pilot: {swimmer['decision']}",
        f"- Best next env: {'PointGoal2/PointGoal1 paper set' if pg2['decision'] == 'passed' else 'CarGoal1 tuning' if car1['decision'] == 'pilot_bad' else 'wait for completion'}",
        f"- Car separate tuning: {car_tune}",
        f"- Swimmer separate tuning: {swimmer_tune}",
        "",
        "## Decisions By Env",
    ]
    for env_key, cfg in ENVS.items():
        stat = stats[env_key]
        decision_lines.append(
            f"- {cfg['display']}: seeds={stat['completed_count']}/{stat['expected']}, "
            f"avg_last3_reward={fmt_mean_std(stat['avg_last3_reward'])}, "
            f"avg_last3_cost={fmt_mean_std(stat['avg_last3_cost'])}, decision={stat['decision']}"
        )
    DECISION_LOG.write_text("\n".join(decision_lines) + "\n")
    print(SUMMARY)
    print("\n".join(lines))


def main() -> None:
    rows = [parse_log(env_key, seed) for env_key, cfg in ENVS.items() for seed in cfg["seeds"]]
    stats = {env_key: group_stats(env_key, rows) for env_key in ENVS}
    monitor = parse_monitor()
    write_reports(stats, rows, monitor)


if __name__ == "__main__":
    main()
