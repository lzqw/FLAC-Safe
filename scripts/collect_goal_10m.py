#!/usr/bin/env python3
import csv
import math
import re
import statistics
import subprocess
from pathlib import Path


LOG_ROOT = Path("logs/goal_10m")
REPORT_ROOT = Path("reports/goal_10m")
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "JSC_CG1_C2_safe045_bw025_10M": {
        "env": "SafetyCarGoal1-v0",
        "seeds": [0, 1, 2],
    },
    "JSC_CG2_strict_safe_schedule_bw025_10M": {
        "env": "SafetyCarGoal2-v0",
        "seeds": [0, 1, 2],
    },
    "JSC_PG1_safe070_bw050_10M": {
        "env": "SafetyPointGoal1-v0",
        "seeds": [0, 1, 2],
    },
    "JSC_PG2_strict_safe_schedule_bw050_10M": {
        "env": "SafetyPointGoal2-v0",
        "seeds": [0, 1, 2],
    },
}

BASELINE_NAMES = ("G4_fixed", "C2_safe05", "PPO")
ERROR_RE = re.compile(
    r"No space left|Traceback|RuntimeError|NaN|nan|OOM|out of memory|CUDA error|unrecognized arguments",
    re.IGNORECASE,
)
END_RE = re.compile(
    r"===== .* END (?P<config>\S+) seed=(?P<seed>\d+) exit_code=(?P<code>\d+) ====="
)
EVAL_RE = re.compile(
    r"Env: (?P<env>[^,]+), Test Episodes: .*?Avg\. Reward: "
    r"(?P<reward>[-+0-9.eE]+), Avg\. Cost: (?P<cost>[-+0-9.eE]+)"
)
SAFETY_Q_RE = re.compile(r"SAFETY_Q step=(?P<step>\d+)\s+(?P<body>.*)")
TRAIN_COST_RE = re.compile(
    r"TRAIN_COST episode=(?P<episode>\d+) step=(?P<step>\d+) "
    r"episode_cost=(?P<episode_cost>[-+0-9.eE]+) "
    r"episode_cost_rate=(?P<episode_cost_rate>[-+0-9.eE]+) "
    r"train_total_cost=(?P<train_total_cost>[-+0-9.eE]+) "
    r"train_total_env_steps=(?P<train_total_env_steps>\d+) "
    r"train_cost_rate=(?P<train_cost_rate>[-+0-9.eE]+)"
)
EPISODE_RE = re.compile(r"Episode: .*?total numsteps: (?P<step>\d+)")


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def fmt4(value):
    return fmt(value, 4)


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def parse_metrics(body):
    out = {}
    for key, value in re.findall(r"([A-Za-z0-9_./]+)=([-+0-9.eE]+)", body):
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def tmux_sessions():
    try:
        proc = subprocess.run(
            ["tmux", "ls"], check=False, capture_output=True, text=True
        )
    except FileNotFoundError:
        return set()
    sessions = set()
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if ":" in line:
                sessions.add(line.split(":", 1)[0])
    return sessions


def parse_run(config, env, seed, running_sessions):
    log_path = LOG_ROOT / config / f"seed{seed}.log"
    session = f"goal10m_{config}_s{seed}"
    if not log_path.exists():
        return {
            "config": config,
            "env": env,
            "seed": seed,
            "log_path": str(log_path),
            "status": "missing",
            "step": None,
            "latest_reward": None,
            "latest_cost": None,
            "avg_last3_reward": None,
            "avg_last3_cost": None,
            "constraint_satisfied_eval": False,
            "constraint_satisfied_training": False,
            "evals": [],
            "diagnostics": {},
            "train": {},
            "errors": [],
        }

    text = log_path.read_text(errors="replace")
    errors = [
        (idx + 1, line)
        for idx, line in enumerate(text.splitlines())
        if ERROR_RE.search(line)
    ]
    evals = []
    pending_eval = None
    diagnostics = {}
    train = {}
    latest_step = None
    end_marker = False
    exit_code = None

    for line in text.splitlines():
        m = END_RE.search(line)
        if m and m.group("config") == config and int(m.group("seed")) == seed:
            end_marker = True
            exit_code = int(m.group("code"))
        m = EVAL_RE.search(line)
        if m:
            pending_eval = {
                "reward": float(m.group("reward")),
                "cost": float(m.group("cost")),
                "step": None,
            }
        m = SAFETY_Q_RE.search(line)
        if m:
            step = int(m.group("step"))
            latest_step = step
            diagnostics = parse_metrics(m.group("body"))
            diagnostics["step"] = step
            if pending_eval is not None:
                pending_eval["step"] = step
                evals.append(pending_eval)
                pending_eval = None
        m = TRAIN_COST_RE.search(line)
        if m:
            train = {
                "episode": int(m.group("episode")),
                "step": int(m.group("step")),
                "episode_cost": float(m.group("episode_cost")),
                "episode_cost_rate": float(m.group("episode_cost_rate")),
                "train_total_cost": float(m.group("train_total_cost")),
                "train_total_env_steps": int(m.group("train_total_env_steps")),
                "train_cost_rate": float(m.group("train_cost_rate")),
            }
            latest_step = train["step"]
        m = EPISODE_RE.search(line)
        if m:
            latest_step = max(latest_step or 0, int(m.group("step")))

    if pending_eval is not None:
        pending_eval["step"] = latest_step
        evals.append(pending_eval)

    if errors or (exit_code is not None and exit_code != 0):
        status = "error_stop"
    elif session in running_sessions:
        status = "running"
    elif end_marker or (latest_step is not None and latest_step >= 10_000_000):
        status = "completed_10m"
    else:
        status = "partial"

    last3 = evals[-3:]
    avg_last3_reward = (
        statistics.mean([item["reward"] for item in last3]) if last3 else None
    )
    avg_last3_cost = (
        statistics.mean([item["cost"] for item in last3]) if last3 else None
    )
    latest_eval = evals[-1] if evals else {}
    step_candidates = [
        latest_eval.get("step"),
        diagnostics.get("step"),
        train.get("step"),
        latest_step,
    ]
    step = max([s for s in step_candidates if s is not None], default=None)

    return {
        "config": config,
        "env": env,
        "seed": seed,
        "log_path": str(log_path),
        "status": status,
        "exit_code": exit_code,
        "step": step,
        "latest_reward": latest_eval.get("reward"),
        "latest_cost": latest_eval.get("cost"),
        "avg_last3_reward": avg_last3_reward,
        "avg_last3_cost": avg_last3_cost,
        "constraint_satisfied_eval": (
            latest_eval.get("cost") is not None and latest_eval.get("cost") <= 25.0
        ),
        "constraint_satisfied_training": (
            train.get("train_cost_rate") is not None
            and train.get("train_cost_rate") <= 0.025
        ),
        "evals": evals,
        "diagnostics": diagnostics,
        "train": train,
        "errors": errors,
    }


def decision_for_run(row):
    if row["status"] == "error_stop":
        return "error_stop"
    step = row.get("step") or 0
    reward = row.get("avg_last3_reward")
    cost = row.get("avg_last3_cost")
    zero_grad = row["diagnostics"].get("safety_q/geom_zero_grad_frac")
    if step >= 5_000_000 and reward is not None and cost is not None:
        if reward < 5 and cost > 150:
            return "likely_failed"
    if step >= 1_000_000 and reward is not None and cost is not None:
        if reward < 5 and cost > 100:
            return "weak_warning"
    if zero_grad is not None and zero_grad > 0.3:
        return "geometry_warning"
    if row["status"] == "completed_10m":
        if row["constraint_satisfied_eval"] and row["constraint_satisfied_training"]:
            return "constraint_satisfied"
        return "completed_tradeoff"
    return row["status"]


def baseline_presence():
    missing = []
    present = []
    for env in sorted({cfg["env"] for cfg in CONFIGS.values()}):
        for baseline in BASELINE_NAMES:
            matches = list(LOG_ROOT.glob(f"*{baseline}*{env}*/*.log"))
            matches += list(LOG_ROOT.glob(f"*{baseline}*/*.log"))
            if matches:
                present.append((env, baseline))
            else:
                missing.append((env, baseline))
    return present, missing


def write_reports(rows):
    summary_md = REPORT_ROOT / "summary.md"
    decision_md = REPORT_ROOT / "decision_log.md"
    summary_csv = REPORT_ROOT / "summary.csv"

    for row in rows:
        row["decision"] = decision_for_run(row)

    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "Config",
            "Env",
            "Seed",
            "Step",
            "Latest Reward",
            "Latest Cost",
            "Avg Last 3 Reward",
            "Avg Last 3 Cost",
            "Train Total Cost",
            "Train Cost Rate",
            "Constraint Satisfied Eval",
            "Constraint Satisfied Training",
            "Status",
            "Decision",
            "lambda_safe_eff",
            "lambda_jvp_eff",
            "zero_grad",
            "mono_plus",
            "mono_minus",
            "fd_slope",
            "jvp_mean",
            "Log Path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            diag = row["diagnostics"]
            train = row["train"]
            writer.writerow(
                {
                    "Config": row["config"],
                    "Env": row["env"],
                    "Seed": row["seed"],
                    "Step": row.get("step"),
                    "Latest Reward": row.get("latest_reward"),
                    "Latest Cost": row.get("latest_cost"),
                    "Avg Last 3 Reward": row.get("avg_last3_reward"),
                    "Avg Last 3 Cost": row.get("avg_last3_cost"),
                    "Train Total Cost": train.get("train_total_cost"),
                    "Train Cost Rate": train.get("train_cost_rate"),
                    "Constraint Satisfied Eval": row["constraint_satisfied_eval"],
                    "Constraint Satisfied Training": row["constraint_satisfied_training"],
                    "Status": row["status"],
                    "Decision": row["decision"],
                    "lambda_safe_eff": diag.get("safety/lambda_safe_eff"),
                    "lambda_jvp_eff": diag.get("safety/lambda_jvp_eff"),
                    "zero_grad": diag.get("safety_q/geom_zero_grad_frac"),
                    "mono_plus": diag.get("safety_q/geom_mono_plus_frac"),
                    "mono_minus": diag.get("safety_q/geom_mono_minus_frac"),
                    "fd_slope": diag.get("safety_q/geom_fd_slope_mean"),
                    "jvp_mean": diag.get("safety_q/jvp_mean"),
                    "Log Path": row["log_path"],
                }
            )

    lines = [
        "# Goal 10M Summary",
        "",
        "Constraint reference: eval cost <= 25, training cost rate <= 0.025.",
        "",
        "## Runs",
        "",
        "| Config | Env | Seed | Step | Latest Reward | Latest Cost | Avg Last 3 Reward | Avg Last 3 Cost | Train Cost Rate | Status |",
        "| ------ | --- | ---: | ---: | ------------: | ----------: | ----------------: | --------------: | --------------: | ------ |",
    ]
    for row in rows:
        train_cost_rate = row["train"].get("train_cost_rate")
        lines.append(
            f"| {row['config']} | {row['env']} | {row['seed']} | {fmt(row.get('step'), 0)} | "
            f"{fmt(row.get('latest_reward'))} | {fmt(row.get('latest_cost'))} | "
            f"{fmt(row.get('avg_last3_reward'))} | {fmt(row.get('avg_last3_cost'))} | "
            f"{fmt4(train_cost_rate)} | {row['status']} |"
        )

    lines += [
        "",
        "## Diagnostics",
        "",
        "| Config | Seed | lambda_safe_eff | lambda_jvp_eff | zero_grad | mono+ | mono- | fd_slope | jvp_mean |",
        "| ------ | ---: | --------------: | -------------: | --------: | ----: | ----: | -------: | -------: |",
    ]
    for row in rows:
        diag = row["diagnostics"]
        lines.append(
            f"| {row['config']} | {row['seed']} | "
            f"{fmt4(diag.get('safety/lambda_safe_eff'))} | "
            f"{fmt4(diag.get('safety/lambda_jvp_eff'))} | "
            f"{fmt4(diag.get('safety_q/geom_zero_grad_frac'))} | "
            f"{fmt4(diag.get('safety_q/geom_mono_plus_frac'))} | "
            f"{fmt4(diag.get('safety_q/geom_mono_minus_frac'))} | "
            f"{fmt4(diag.get('safety_q/geom_fd_slope_mean'))} | "
            f"{fmt4(diag.get('safety_q/jvp_mean'))} |"
        )

    lines += [
        "",
        "## Groups",
        "",
        "| Config | Env | Seeds | Reward Mean+/-Std | Cost Mean+/-Std | Train Cost Rate Mean+/-Std | Decision |",
        "| ------ | --- | ----: | --------------: | ------------: | -----------------------: | -------- |",
    ]
    for config, cfg in CONFIGS.items():
        group_rows = [
            row
            for row in rows
            if row["config"] == config and row.get("avg_last3_reward") is not None
        ]
        rewards = [row.get("avg_last3_reward") for row in group_rows]
        costs = [row.get("avg_last3_cost") for row in group_rows]
        train_rates = [row["train"].get("train_cost_rate") for row in group_rows]
        reward_mean, reward_std = mean_std(rewards)
        cost_mean, cost_std = mean_std(costs)
        rate_mean, rate_std = mean_std(train_rates)
        decisions = sorted({row["decision"] for row in group_rows}) or ["pending"]
        lines.append(
            f"| {config} | {cfg['env']} | {len(group_rows)}/3 | "
            f"{fmt(reward_mean)}+/-{fmt(reward_std)} | "
            f"{fmt(cost_mean)}+/-{fmt(cost_std)} | "
            f"{fmt4(rate_mean)}+/-{fmt4(rate_std)} | {', '.join(decisions)} |"
        )

    present, missing = baseline_presence()
    lines += [
        "",
        "## Baselines",
        "",
        f"- Present 10M baseline markers: {len(present)}",
        f"- Missing 10M baseline markers: {len(missing)}",
    ]
    for env, baseline in missing:
        lines.append(f"- Missing: {env} {baseline} 10M")

    summary_md.write_text("\n".join(lines) + "\n")

    dlines = [
        "# Goal 10M Decision Log",
        "",
        "- Official runs use independent tmux sessions per seed; no vectorized env is used.",
        "- Stop only on numeric instability, memory errors, runtime failures, disk full, repeated CUDA failures, or explicit instruction.",
        "- Do not claim same-budget superiority until 10M baselines are present.",
        "",
        "## Run Decisions",
    ]
    for row in rows:
        dlines.append(
            f"- {row['config']} seed{row['seed']}: step={fmt(row.get('step'), 0)}, "
            f"reward={fmt(row.get('avg_last3_reward'))}, cost={fmt(row.get('avg_last3_cost'))}, "
            f"train_cost_rate={fmt4(row['train'].get('train_cost_rate'))}, "
            f"status={row['status']}, decision={row['decision']}"
        )
    dlines += [
        "",
        "## Missing Baselines",
    ]
    for env, baseline in missing:
        dlines.append(f"- {env}: {baseline} 10M")
    decision_md.write_text("\n".join(dlines) + "\n")

    print(summary_md)
    print(summary_csv)
    print(decision_md)


def main():
    sessions = tmux_sessions()
    rows = []
    for config, cfg in CONFIGS.items():
        for seed in cfg["seeds"]:
            rows.append(parse_run(config, cfg["env"], seed, sessions))
    write_reports(rows)


if __name__ == "__main__":
    main()
