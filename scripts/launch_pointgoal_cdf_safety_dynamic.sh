#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-3}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

LOG_DIR="logs/pointgoal_cdf_safety"
MONITOR="$LOG_DIR/gpu_monitor.csv"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" reports/pointgoal_cdf_safety

group_name() {
  case "$1" in
    C0|C0_cumulative) echo "C0_cumulative" ;;
    C1|C1_cdf_max) echo "C1_cdf_max" ;;
    C2|C2_cdf_mean) echo "C2_cdf_mean" ;;
    *) echo "$1" ;;
  esac
}

queue_specs() {
  echo C0:0 C0:1 C0:2
  echo C1:0 C1:1 C1:2
  echo C2:0 C2:1 C2:2
}

tag_for_spec() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "$(group_name "$group")_seed${seed}"
}

session_name() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "cdf_pg_${group}_${seed}"
}

parallel_n() {
  local n="$MAX_PARALLEL"
  (( n > HARD_MAX_PARALLEL )) && n="$HARD_MAX_PARALLEL"
  (( n < MIN_PARALLEL )) && n="$MIN_PARALLEL"
  echo "$n"
}

completed_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  [[ -f "$log" ]] || return 1
  grep -q " END $(group_name "${spec%%:*}") seed=${spec##*:} " "$log" || return 1
  ! grep -E "$ERR_RE" "$log" >/dev/null 2>&1
}

start_run() {
  local spec="$1"
  local group="${spec%%:*}"
  local seed="${spec##*:}"
  local sess
  sess="$(session_name "$spec")"
  completed_log "$spec" && { echo "skip completed $spec"; return; }
  tmux has-session -t "$sess" 2>/dev/null && { echo "$sess already exists"; return; }
  tmux new -d -s "$sess" "cd /root/FLAC-Safe && \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate flac && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=\${WANDB_MODE:-offline} && \
bash scripts/run_pointgoal_cdf_safety.sh ${group} ${seed}"
  echo "started $sess"
}

monitor_once() {
  if [[ ! -f "$MONITOR" ]]; then
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv > "$MONITOR" || true
  else
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader >> "$MONITOR" || true
  fi
}

status() {
  echo "===== tmux ====="
  tmux ls 2>/dev/null || true
  echo "===== gpu ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv || true
  echo "===== progress ====="
  grep -E "Env: SafetyPointGoal1-v0|Episode:| END " "$LOG_DIR"/*.log 2>/dev/null | tail -n 120 || true
  echo "===== errors ====="
  grep -E "$ERR_RE" "$LOG_DIR"/*.log 2>/dev/null || true
}

run_all() {
  local queue=() idx=0 active=0 any_running=0 spec n
  mapfile -t queue < <(queue_specs | tr ' ' '\n' | sed '/^$/d')
  n="$(parallel_n)"
  echo "effective parallelism=$n"
  while (( idx < ${#queue[@]} )); do
    active=0
    for spec in "${queue[@]}"; do
      tmux has-session -t "$(session_name "$spec")" 2>/dev/null && active=$((active + 1))
    done
    while (( active < n && idx < ${#queue[@]} )); do
      start_run "${queue[$idx]}"
      idx=$((idx + 1))
      active=$((active + 1))
    done
    monitor_once
    any_running=0
    for spec in "${queue[@]}"; do
      tmux has-session -t "$(session_name "$spec")" 2>/dev/null && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
  monitor_once
  python scripts/collect_pointgoal_cdf_safety.py || true
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^cdf_pg_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan)
    queue_specs | tr ' ' '\n' | sed '/^$/d'
    echo "effective parallelism=$(parallel_n)"
    ;;
  all) run_all ;;
  status) status ;;
  stop) stop_sessions ;;
  *)
    echo "Usage: bash scripts/launch_pointgoal_cdf_safety_dynamic.sh plan|all|status|stop" >&2
    exit 2
    ;;
esac
