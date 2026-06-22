#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-3}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-5}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

LOG_DIR="logs/pointgoal_njvp_overnight"
REPORT_DIR="reports/pointgoal_njvp_overnight"
MONITOR="$LOG_DIR/gpu_monitor.csv"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

group_name() {
  case "$1" in
    ON0|ON0_N3_extend) echo "ON0_N3_extend" ;;
    ON1|ON1_jvp0035) echo "ON1_jvp0035" ;;
    ON2|ON2_jvp0040) echo "ON2_jvp0040" ;;
    ON3|ON3_bw0075) echo "ON3_bw0075" ;;
    ON4|ON4_safe07) echo "ON4_safe07" ;;
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
  echo "njvp_on_${group}_${seed}"
}

queue_specs() {
  echo ON0:3 ON0:4 ON0:5 ON0:6 ON0:7 ON0:8 ON0:9
  echo ON1:0 ON1:1 ON1:2 ON1:3 ON1:4
  echo ON2:0 ON2:1 ON2:2 ON2:3 ON2:4
  echo ON3:0 ON3:1 ON3:2 ON3:3 ON3:4
  echo ON4:0 ON4:1 ON4:2 ON4:3 ON4:4
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

is_running() {
  tmux has-session -t "$(session_name "$1")" 2>/dev/null
}

monitor_once() {
  if [[ ! -f "$MONITOR" ]]; then
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv > "$MONITOR" || true
  else
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader >> "$MONITOR" || true
  fi
}

current_parallel() {
  local n="$MAX_PARALLEL"
  (( n > HARD_MAX_PARALLEL )) && n="$HARD_MAX_PARALLEL"
  (( n < MIN_PARALLEL )) && n="$MIN_PARALLEL"
  echo "$n"
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
bash scripts/run_pointgoal_njvp_overnight.sh ${group} ${seed}"
  echo "started $sess"
}

status() {
  echo "===== tmux ====="
  tmux ls 2>/dev/null || true
  echo "===== gpu ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv || true
  echo "===== progress ====="
  grep -E "Env: SafetyPointGoal1-v0|Episode:|total numsteps| END " "$LOG_DIR"/*.log 2>/dev/null | tail -n 120 || true
  echo "===== errors ====="
  grep -E "$ERR_RE" "$LOG_DIR"/*.log 2>/dev/null || true
  echo "===== monitor ====="
  tail -n 8 "$MONITOR" 2>/dev/null || true
}

run_all() {
  local queue=()
  local spec idx active any_running n failed_seen
  mapfile -t queue < <(queue_specs | tr ' ' '\n' | sed '/^$/d')
  for spec in "${queue[@]}"; do
    completed_log "$spec" && echo "skip completed $spec"
  done
  n="$(current_parallel)"
  echo "MAX_PARALLEL requested=$MAX_PARALLEL effective=$n"
  idx=0
  while (( idx < ${#queue[@]} )); do
    active=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && active=$((active + 1))
    done
    while (( active < n && idx < ${#queue[@]} )); do
      spec="${queue[$idx]}"
      idx=$((idx + 1))
      completed_log "$spec" && continue
      failed_log "$spec" && continue
      start_run "$spec"
      active=$((active + 1))
    done
    monitor_once
    failed_seen=0
    for spec in "${queue[@]}"; do
      if failed_log "$spec"; then
        failed_seen=1
      fi
    done
    if (( failed_seen == 1 && n > MIN_PARALLEL )); then
      n=$((n - 1))
      echo "error pattern detected; reducing parallelism to $n"
    fi
    any_running=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
  monitor_once
  python scripts/collect_pointgoal_njvp_overnight.py || true
}

plan() {
  echo "Queue:"
  queue_specs | tr ' ' '\n' | sed '/^$/d'
  echo "MAX_PARALLEL requested: $MAX_PARALLEL"
  echo "Effective parallelism: $(current_parallel)"
  nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || true
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^njvp_on_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan) plan ;;
  all) run_all ;;
  status) status ;;
  stop) stop_sessions ;;
  *)
    echo "Usage: bash scripts/launch_pointgoal_njvp_overnight_dynamic.sh plan|all|status|stop" >&2
    exit 2
    ;;
esac
