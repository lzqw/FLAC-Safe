#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.85}"
GPU_MEM_RESERVE_GB="${GPU_MEM_RESERVE_GB:-3}"
PER_RUN_MEM_GB="${PER_RUN_MEM_GB:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-auto}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

LOG_DIR="logs/pointgoal_seed_sweep"
MONITOR="$LOG_DIR/gpu_monitor.csv"
mkdir -p "$LOG_DIR" reports/pointgoal_seed_sweep

RUNS=(
  "S0_penalty_only:0" "S0_penalty_only:1" "S0_penalty_only:2" "S0_penalty_only:3" "S0_penalty_only:4"
  "S1_main_R2D:0" "S1_main_R2D:1" "S1_main_R2D:2" "S1_main_R2D:3" "S1_main_R2D:4"
  "S2_bw010_R2E:0" "S2_bw010_R2E:1" "S2_bw010_R2E:2" "S2_bw010_R2E:3" "S2_bw010_R2E:4"
)

gpu_total_gb() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1 {printf "%.3f", $1/1024}'
}

gpu_used_gb() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {printf "%.3f", $1/1024}'
}

estimate_per_run_mem_gb() {
  if [[ "$PER_RUN_MEM_GB" != "auto" ]]; then
    echo "$PER_RUN_MEM_GB"
    return
  fi
  python - <<'PY'
from pathlib import Path
import re

vals = []
for path in Path("logs/4090_round2_probe").glob("*.log"):
    text = path.read_text(errors="replace")
    peaks = [int(x) for x in re.findall(r"\[GPU_MEM_PEAK_EST\] used_mb=(\d+)", text)]
    if peaks:
        vals.append(max(peaks))
if vals:
    print(f"{max(vals) / 1024:.3f}")
else:
    print("4.000")
PY
}

compute_plan() {
  local total used per free usable n_mem n_cpu final
  total="$(gpu_total_gb)"
  used="$(gpu_used_gb)"
  per="$(estimate_per_run_mem_gb)"
  read -r free usable n_mem n_cpu final < <(
    TOTAL="$total" USED="$used" PER="$per" GPU_MEM_FRACTION="$GPU_MEM_FRACTION" \
    GPU_MEM_RESERVE_GB="$GPU_MEM_RESERVE_GB" HARD_MAX_PARALLEL="$HARD_MAX_PARALLEL" \
    MIN_PARALLEL="$MIN_PARALLEL" MAX_PARALLEL="$MAX_PARALLEL" python - <<'PY'
import math
import os

total = float(os.environ["TOTAL"])
used = float(os.environ["USED"])
per = max(float(os.environ["PER"]), 0.1)
usable = total * float(os.environ["GPU_MEM_FRACTION"]) - float(os.environ["GPU_MEM_RESERVE_GB"])
n_mem = max(1, math.floor(usable / per))
n_cpu = max(1, math.floor((os.cpu_count() or 1) / 8))
hard = int(os.environ["HARD_MAX_PARALLEL"])
minp = int(os.environ["MIN_PARALLEL"])
maxp = os.environ["MAX_PARALLEL"]
if maxp == "auto":
    n = min(n_mem, n_cpu, hard)
else:
    n = min(int(maxp), n_mem, hard)
n = max(minp, n)
print(f"{max(total - used, 0):.3f} {usable:.3f} {n_mem} {n_cpu} {n}")
PY
  )
  echo "GPU total memory: ${total} GB"
  echo "GPU used memory: ${used} GB"
  echo "GPU free memory: ${free} GB"
  echo "per-run estimated memory: ${per} GB"
  echo "usable memory: ${usable} GB"
  echo "N_mem: ${n_mem}"
  echo "N_cpu: ${n_cpu}"
  echo "HARD_MAX_PARALLEL: ${HARD_MAX_PARALLEL}"
  echo "final N: ${final}"
}

final_n() {
  compute_plan | awk -F': ' '/final N/ {print $2}'
}

tag_for_spec() {
  local spec="$1"
  local group="${spec%%:*}"
  local seed="${spec##*:}"
  echo "${group}_seed${seed}"
}

short_group() {
  local group="$1"
  case "$group" in
    S0_penalty_only) echo "S0" ;;
    S1_main_R2D) echo "S1" ;;
    S2_bw010_R2E) echo "S2" ;;
    *) echo "$group" ;;
  esac
}

session_name() {
  local spec="$1"
  local group="${spec%%:*}"
  local seed="${spec##*:}"
  echo "pg_seed_$(short_group "$group")_${seed}"
}

is_running() {
  tmux has-session -t "$(session_name "$1")" 2>/dev/null
}

completed_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  [[ -f "$log" ]] && grep -q " END ${spec%%:*} seed=${spec##*:} " "$log"
}

start_run() {
  local spec="$1"
  local sess
  sess="$(session_name "$spec")"
  if completed_log "$spec"; then
    echo "skip completed $spec"
    return
  fi
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "$sess already exists"
    return
  fi
  tmux new -d -s "$sess" "cd /root/FLAC-Safe && \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate flac && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=\${WANDB_MODE:-offline} && \
BATCH_SIZE=${BATCH_SIZE} UPDATES_PER_STEP=${UPDATES_PER_STEP} HIDDEN_SIZE=${HIDDEN_SIZE} \
bash scripts/run_pointgoal_seed_sweep.sh ${spec}"
  echo "started $sess"
}

monitor_once() {
  if [[ ! -f "$MONITOR" ]]; then
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv > "$MONITOR" || true
  else
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader >> "$MONITOR" || true
  fi
}

run_status_check() {
  echo "===== tmux ====="
  tmux ls 2>/dev/null || true
  echo "===== gpu ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv || true
  echo "===== recent evals ====="
  grep -E "Env: SafetyPointGoal1-v0|Episode:|total numsteps" "$LOG_DIR"/*.log 2>/dev/null | tail -n 80 || true
  echo "===== errors ====="
  grep -E "Traceback|RuntimeError|NaN|nan|OOM|out of memory" "$LOG_DIR"/*.log 2>/dev/null || true
}

run_batch() {
  local input_queue=("$@")
  local queue=()
  local n idx active any_running
  for spec in "${input_queue[@]}"; do
    if completed_log "$spec"; then
      echo "skip completed $spec"
    else
      queue+=("$spec")
    fi
  done
  if (( ${#queue[@]} == 0 )); then
    echo "no pending runs"
    monitor_once
    run_status_check
    return
  fi
  n="$(final_n)"
  echo "dynamic final_N=$n"
  idx=0
  while (( idx < ${#queue[@]} )); do
    active=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && active=$((active + 1))
    done
    while (( active < n && idx < ${#queue[@]} )); do
      start_run "${queue[$idx]}"
      idx=$((idx + 1))
      active=$((active + 1))
    done
    monitor_once
    run_status_check
    any_running=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^pg_seed_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan)
    compute_plan
    ;;
  s0)
    run_batch "S0_penalty_only:0" "S0_penalty_only:1" "S0_penalty_only:2" "S0_penalty_only:3" "S0_penalty_only:4"
    ;;
  s1)
    run_batch "S1_main_R2D:0" "S1_main_R2D:1" "S1_main_R2D:2" "S1_main_R2D:3" "S1_main_R2D:4"
    ;;
  s2)
    run_batch "S2_bw010_R2E:0" "S2_bw010_R2E:1" "S2_bw010_R2E:2" "S2_bw010_R2E:3" "S2_bw010_R2E:4"
    ;;
  all)
    run_batch "${RUNS[@]}"
    ;;
  status)
    run_status_check
    ;;
  stop)
    stop_sessions
    ;;
  *)
    echo "Usage: bash scripts/launch_pointgoal_seed_sweep_dynamic.sh plan|s0|s1|s2|all|status|stop" >&2
    exit 2
    ;;
esac
