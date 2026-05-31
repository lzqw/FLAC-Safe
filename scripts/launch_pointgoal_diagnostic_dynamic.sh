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
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

LOG_DIR="logs/pointgoal_diagnostic"
MONITOR="$LOG_DIR/gpu_monitor.csv"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" reports/pointgoal_diagnostic

RUNS=(
  "PGD0:5" "PGD0:6" "PGD0:7"
  "PGD1:5" "PGD1:6" "PGD1:7"
  "PGD2:0" "PGD2:1" "PGD2:2"
  "PGD3:0" "PGD3:1" "PGD3:2"
)

group_name() {
  case "$1" in
    PGD0|PGD0_S0_extend) echo "PGD0_S0_extend" ;;
    PGD1|PGD1_S1_R2D_extend) echo "PGD1_S1_R2D_extend" ;;
    PGD2|PGD2_R2D_update1) echo "PGD2_R2D_update1" ;;
    PGD3|PGD3_mid_jvp) echo "PGD3_mid_jvp" ;;
    PGD4|PGD4_stronger_safe) echo "PGD4_stronger_safe" ;;
    PGD5|PGD5_stronger_jvp) echo "PGD5_stronger_jvp" ;;
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
  echo "pg_diag_${group}_${seed}"
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

failed_oom_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  [[ -f "$log" ]] && grep -Ei "OOM|out of memory" "$log" >/dev/null 2>&1
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
BATCH_SIZE=${BATCH_SIZE} HIDDEN_SIZE=${HIDDEN_SIZE} \
bash scripts/run_pointgoal_diagnostic.sh ${group} ${seed}"
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
  local input_queue=("$@")
  local queue=()
  local n idx active any_running spec
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
    for spec in "${queue[@]}"; do
      if failed_oom_log "$spec" && (( n > MIN_PARALLEL )); then
        n=$((n - 1))
        echo "OOM detected; reducing dynamic final_N to $n"
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
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^pg_diag_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan)
    compute_plan
    ;;
  all)
    run_queue "${RUNS[@]}"
    ;;
  pgd4)
    run_queue "PGD4:0" "PGD4:1" "PGD4:2"
    ;;
  pgd5)
    run_queue "PGD5:0" "PGD5:1" "PGD5:2"
    ;;
  status)
    run_status_check
    ;;
  stop)
    stop_sessions
    ;;
  *)
    echo "Usage: bash scripts/launch_pointgoal_diagnostic_dynamic.sh plan|all|pgd4|pgd5|status|stop" >&2
    exit 2
    ;;
esac
