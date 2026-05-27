#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.85}"
GPU_MEM_RESERVE_GB="${GPU_MEM_RESERVE_GB:-3}"
PER_RUN_MEM_GB="${PER_RUN_MEM_GB:-auto}"
MAX_PARALLEL="${MAX_PARALLEL:-auto}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-6}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

LOG_DIR="logs/pointgoal_round2"
MONITOR="$LOG_DIR/gpu_monitor.csv"
mkdir -p "$LOG_DIR" reports/pointgoal_round2

RUNS=("R2_A" "R2_B" "R2_C" "R2_D" "R2_E" "R2_F")
declare -A TAGS=(
  ["R2_A"]="r2_A_safe05_jvp00005_bw005"
  ["R2_B"]="r2_B_safe05_jvp0001_bw005"
  ["R2_C"]="r2_C_safe05_jvp0002_bw005"
  ["R2_D"]="r2_D_safe05_jvp0005_bw005"
  ["R2_E"]="r2_E_safe05_jvp0001_bw010"
  ["R2_F"]="r2_F_safe03_jvp0002_bw005"
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
        continue
    before = [int(x) for x in re.findall(r"\[GPU_MEM_BEFORE\] used_mb=(\d+)", text)]
    after = [int(x) for x in re.findall(r"\[GPU_MEM_AFTER\] used_mb=(\d+)", text)]
    if before and after:
        vals.append(max(max(after), max(before)))
if vals:
    print(f"{max(vals) / 1024:.3f}")
else:
    print("4.000")
PY
}

compute_plan() {
  local total used free per usable n_mem n_cpu n final
  total="$(gpu_total_gb)"
  used="$(gpu_used_gb)"
  per="$(estimate_per_run_mem_gb)"
  read -r free usable n_mem n_cpu n final < <(
    TOTAL="$total" USED="$used" PER="$per" GPU_MEM_FRACTION="$GPU_MEM_FRACTION" \
    GPU_MEM_RESERVE_GB="$GPU_MEM_RESERVE_GB" HARD_MAX_PARALLEL="$HARD_MAX_PARALLEL" \
    MIN_PARALLEL="$MIN_PARALLEL" MAX_PARALLEL="$MAX_PARALLEL" python - <<'PY'
import math, os
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
print(f"{max(total-used, 0):.3f} {usable:.3f} {n_mem} {n_cpu} {n} {n}")
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

session_name() {
  local run="$1"
  echo "pg_${run/R2_/r2_}"
}

is_running() {
  local run="$1"
  tmux has-session -t "$(session_name "$run")" 2>/dev/null
}

start_run() {
  local run="$1"
  local sess
  sess="$(session_name "$run")"
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
bash scripts/run_pointgoal_round2_4090.sh ${run}"
  echo "started $sess"
}

run_status_check() {
  echo "===== tmux ====="
  tmux ls 2>/dev/null || true
  echo "===== gpu ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv || true
  echo "===== logs ====="
  for run in "${RUNS[@]}"; do
    local log="$LOG_DIR/${TAGS[$run]}.log"
    [[ -f "$log" ]] || continue
    echo "----- $log -----"
    tail -n 30 "$log"
  done
  echo "===== errors ====="
  grep -E "Traceback|RuntimeError|NaN|nan|OOM|out of memory" "$LOG_DIR"/*.log 2>/dev/null || true
}

monitor_once() {
  if [[ ! -f "$MONITOR" ]]; then
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv > "$MONITOR" || true
  else
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader >> "$MONITOR" || true
  fi
}

completed_or_failed() {
  local run="$1"
  local log="$LOG_DIR/${TAGS[$run]}.log"
  [[ -f "$log" ]] || return 1
  grep -q " END ${run} " "$log" && return 0
  grep -E "Traceback|RuntimeError|NaN|nan|OOM|out of memory" "$log" >/dev/null 2>&1 && return 0
  return 1
}

run_batch() {
  local queue=("$@")
  local n
  n="$(final_n)"
  echo "dynamic final_N=$n"
  local idx=0
  while (( idx < ${#queue[@]} )); do
    local active=0
    for run in "${queue[@]}"; do
      is_running "$run" && active=$((active + 1))
    done
    while (( active < n && idx < ${#queue[@]} )); do
      start_run "${queue[$idx]}"
      idx=$((idx + 1))
      active=$((active + 1))
    done
    monitor_once
    run_status_check
    local any_running=0
    for run in "${queue[@]}"; do
      is_running "$run" && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^pg_r2_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan)
    compute_plan
    ;;
  batch1)
    run_batch R2_A R2_B
    ;;
  batch2)
    run_batch R2_C R2_D
    ;;
  batch3)
    run_batch R2_E R2_F
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
    echo "Usage: bash scripts/launch_pointgoal_round2_dynamic.sh plan|batch1|batch2|batch3|all|status|stop" >&2
    exit 2
    ;;
esac
