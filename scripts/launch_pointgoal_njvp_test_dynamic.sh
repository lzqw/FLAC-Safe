#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.85}"
GPU_MEM_RESERVE_GB="${GPU_MEM_RESERVE_GB:-3}"
PER_RUN_MEM_GB="${PER_RUN_MEM_GB:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

LOG_DIR="logs/pointgoal_njvp_test"
MONITOR="$LOG_DIR/gpu_monitor.csv"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" reports/pointgoal_njvp_test

group_name() {
  case "$1" in
    N0|N0_raw_R2D) echo "N0_raw_R2D" ;;
    N1|N1_njvp_0005) echo "N1_njvp_0005" ;;
    N2|N2_njvp_001) echo "N2_njvp_001" ;;
    N3|N3_njvp_003) echo "N3_njvp_003" ;;
    N4|N4_njvp_005) echo "N4_njvp_005" ;;
    *) echo "$1" ;;
  esac
}

tag_for_spec() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "$(group_name "$group")_seed${seed}"
}

session_name() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "pg_njvp_${group}_${seed}"
}

gpu_total_gb() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1 {printf "%.3f", $1/1024}'
}

gpu_used_gb() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {printf "%.3f", $1/1024}'
}

compute_plan() {
  local total used free usable n_mem n_cpu final
  total="$(gpu_total_gb)"
  used="$(gpu_used_gb)"
  read -r free usable n_mem n_cpu final < <(
    TOTAL="$total" USED="$used" PER_RUN_MEM_GB="$PER_RUN_MEM_GB" GPU_MEM_FRACTION="$GPU_MEM_FRACTION" \
    GPU_MEM_RESERVE_GB="$GPU_MEM_RESERVE_GB" HARD_MAX_PARALLEL="$HARD_MAX_PARALLEL" \
    MIN_PARALLEL="$MIN_PARALLEL" MAX_PARALLEL="$MAX_PARALLEL" python - <<'PY'
import math
import os

total = float(os.environ["TOTAL"])
used = float(os.environ["USED"])
per = max(float(os.environ["PER_RUN_MEM_GB"]), 0.1)
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
  echo "per-run estimated memory: ${PER_RUN_MEM_GB} GB"
  echo "usable memory: ${usable} GB"
  echo "N_mem: ${n_mem}"
  echo "N_cpu: ${n_cpu}"
  echo "HARD_MAX_PARALLEL: ${HARD_MAX_PARALLEL}"
  echo "final N: ${final}"
}

final_n() {
  compute_plan | awk -F': ' '/final N/ {print $2}'
}

is_running() {
  tmux has-session -t "$(session_name "$1")" 2>/dev/null
}

completed_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  [[ -f "$log" ]] || return 1
  grep -q " END $(group_name "${spec%%:*}") seed=${spec##*:} " "$log" || return 1
  ! grep -E "$ERR_RE" "$log" >/dev/null 2>&1
}

failed_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  [[ -f "$log" ]] && grep -E "$ERR_RE" "$log" >/dev/null 2>&1
}

start_run() {
  local spec="$1"
  local group="${spec%%:*}"
  local seed="${spec##*:}"
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
bash scripts/run_pointgoal_njvp_test.sh ${group} ${seed}"
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
  grep -E "Env: SafetyPointGoal1-v0|Episode:|total numsteps| END " "$LOG_DIR"/*.log 2>/dev/null | tail -n 100 || true
  echo "===== errors ====="
  grep -E "$ERR_RE" "$LOG_DIR"/*.log 2>/dev/null || true
}

run_queue() {
  local queue=()
  local n idx active any_running spec
  for spec in "$@"; do
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
    for spec in "${queue[@]}"; do
      if failed_log "$spec"; then
        echo "error detected in $spec; stop launching more aggressive stages"
        break
      fi
    done
    any_running=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^pg_njvp_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan)
    compute_plan
    ;;
  n0)
    run_queue "N0:0" "N0:1" "N0:2"
    ;;
  n1n2)
    run_queue "N1:0" "N1:1" "N1:2" "N2:0" "N2:1" "N2:2"
    ;;
  n3)
    run_queue "N3:0" "N3:1" "N3:2"
    ;;
  n4)
    run_queue "N4:0" "N4:1" "N4:2"
    ;;
  status)
    run_status_check
    ;;
  stop)
    stop_sessions
    ;;
  *)
    echo "Usage: bash scripts/launch_pointgoal_njvp_test_dynamic.sh plan|n0|n1n2|n3|n4|status|stop" >&2
    exit 2
    ;;
esac
