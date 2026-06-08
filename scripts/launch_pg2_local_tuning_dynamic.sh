#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

LOG_DIR="logs/pg2_local_tuning"
REPORT_DIR="reports/pg2_local_tuning"
MONITOR="${MONITOR:-$LOG_DIR/gpu_monitor.csv}"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

group_name() {
  case "$1" in
    PG2_A1|PG2_A1_less_conservative) echo "PG2_A1_less_conservative" ;;
    PG2_A2|PG2_A2_stronger_safe) echo "PG2_A2_stronger_safe" ;;
    PG2_A3|PG2_A3_stronger_jvp) echo "PG2_A3_stronger_jvp" ;;
    PG2_A4|PG2_A4_wider_band) echo "PG2_A4_wider_band" ;;
    PG2_A5|PG2_A5_longer_G4) echo "PG2_A5_longer_G4" ;;
    *) echo "$1" ;;
  esac
}

queue_specs() {
  printf '%s\n' \
    PG2_A1:0 PG2_A1:1 \
    PG2_A2:0 PG2_A2:1 \
    PG2_A3:0 PG2_A3:1 \
    PG2_A4:0 PG2_A4:1 \
    PG2_A5:0 PG2_A5:1
}

tag_for_spec() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "$(group_name "$group")_seed${seed}"
}

session_name() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "pg2_tune_${group}_${seed}"
}

parallel_n() {
  local n="$MAX_PARALLEL"
  (( n > HARD_MAX_PARALLEL )) && n="$HARD_MAX_PARALLEL"
  (( n < MIN_PARALLEL )) && n="$MIN_PARALLEL"
  echo "$n"
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
    return 1
  fi
  if failed_log "$spec"; then
    echo "found failed log for $spec; not rerunning automatically"
    return 1
  fi
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "$sess already exists"
    return 1
  fi
  tmux new -d -s "$sess" "cd /root/FLAC-Safe && \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate flac && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=\${WANDB_MODE:-offline} && \
bash scripts/run_pg2_local_tuning.sh ${group} ${seed}"
  echo "started $sess"
  return 0
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
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv || true
  echo "===== progress ====="
  grep -E "Env: SafetyPointGoal2-v0|Episode:|total numsteps| START | END " "$LOG_DIR"/*.log 2>/dev/null | tail -n 180 || true
  echo "===== failed logs ====="
  grep -E "$ERR_RE" "$LOG_DIR"/*.log 2>/dev/null || true
}

run_all() {
  local all_specs=()
  local queue=()
  local failed_specs=()
  local spec n idx active any_running
  mapfile -t all_specs < <(queue_specs)
  for spec in "${all_specs[@]}"; do
    if failed_log "$spec"; then
      failed_specs+=("$spec")
    elif completed_log "$spec"; then
      echo "skip completed $spec"
    else
      queue+=("$spec")
    fi
  done
  if (( ${#failed_specs[@]} > 0 )); then
    echo "failed logs present; not rerunning automatically: ${failed_specs[*]}"
    run_status_check
    return 1
  fi
  if (( ${#queue[@]} == 0 )); then
    echo "no pending PG2 local tuning runs"
    monitor_once
    python scripts/collect_pg2_local_tuning.py || true
    run_status_check
    return 0
  fi
  n="$(parallel_n)"
  echo "effective parallelism=$n"
  idx=0
  while (( idx < ${#queue[@]} )); do
    active=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && active=$((active + 1))
    done
    while (( active < n && idx < ${#queue[@]} )); do
      if start_run "${queue[$idx]}"; then
        active=$((active + 1))
      fi
      idx=$((idx + 1))
    done
    monitor_once
    run_status_check
    for spec in "${queue[@]}"; do
      if failed_log "$spec"; then
        echo "error detected in $spec; not launching additional PG2 local tuning runs"
        python scripts/collect_pg2_local_tuning.py || true
        return 1
      fi
    done
    any_running=0
    for spec in "${queue[@]}"; do
      is_running "$spec" && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
  monitor_once
  python scripts/collect_pg2_local_tuning.py || true
  run_status_check
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^pg2_tune_/ {print $1}'); do
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done
}

case "$MODE" in
  plan)
    queue_specs
    echo "effective parallelism=$(parallel_n)"
    ;;
  all)
    run_all
    ;;
  status)
    run_status_check
    ;;
  stop)
    stop_sessions
    ;;
  *)
    echo "Usage: bash scripts/launch_pg2_local_tuning_dynamic.sh plan|all|status|stop" >&2
    exit 2
    ;;
esac
