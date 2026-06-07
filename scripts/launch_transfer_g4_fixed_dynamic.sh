#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

LOG_DIR="logs/transfer_g4_fixed"
REPORT_DIR="reports/transfer_g4_fixed"
MONITOR="${MONITOR:-$LOG_DIR/gpu_monitor.csv}"
RESOLUTION_LOG="$LOG_DIR/env_resolution.log"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

env_exists() {
  local task_id="$1"
  TASK_ID="$task_id" python - <<'PY' >/dev/null 2>&1
import os
import gymnasium as gym
import safety_gymnasium  # noqa: F401

raise SystemExit(0 if os.environ["TASK_ID"] in gym.registry else 1)
PY
}

record_resolution() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RESOLUTION_LOG" >&2
}

resolve_swimmer_task() {
  if env_exists "SafetySwimmerVelocity-v1"; then
    echo "SafetySwimmerVelocity-v1"
    return 0
  fi
  if env_exists "SafetySwimmerVelocity-v0"; then
    echo "SafetySwimmerVelocity-v0"
    return 0
  fi
  return 1
}

env_name() {
  case "$1" in
    T0|T0_PointGoal1_ref) echo "T0_PointGoal1_ref" ;;
    T1|T1_PointGoal2) echo "T1_PointGoal2" ;;
    T2|T2_CarGoal1) echo "T2_CarGoal1" ;;
    T3|T3_CarGoal2) echo "T3_CarGoal2" ;;
    T4|T4_SwimmerVelocity) echo "T4_SwimmerVelocity" ;;
    *) echo "$1" ;;
  esac
}

task_for_env() {
  case "$1" in
    T0_PointGoal1_ref) echo "SafetyPointGoal1-v0" ;;
    T1_PointGoal2) echo "SafetyPointGoal2-v0" ;;
    T2_CarGoal1) echo "SafetyCarGoal1-v0" ;;
    T3_CarGoal2) echo "SafetyCarGoal2-v0" ;;
    T4_SwimmerVelocity) resolve_swimmer_task ;;
    *) return 1 ;;
  esac
}

tag_for_spec() {
  local env_key="${1%%:*}"
  local seed="${1##*:}"
  echo "${env_key}_G4_fixed_main_seed${seed}"
}

session_name() {
  local env_key="${1%%:*}"
  local seed="${1##*:}"
  echo "transfer_g4_${env_key}_${seed}"
}

parallel_n() {
  local n="$MAX_PARALLEL"
  (( n > HARD_MAX_PARALLEL )) && n="$HARD_MAX_PARALLEL"
  (( n < MIN_PARALLEL )) && n="$MIN_PARALLEL"
  echo "$n"
}

queue_specs() {
  : > "$RESOLUTION_LOG"
  if env_exists "SafetyPointGoal2-v0"; then
    printf '%s\n' T1_PointGoal2:0 T1_PointGoal2:1 T1_PointGoal2:2
  else
    record_resolution "missing required env SafetyPointGoal2-v0"
  fi
  if env_exists "SafetyCarGoal1-v0"; then
    printf '%s\n' T2_CarGoal1:0 T2_CarGoal1:1 T2_CarGoal1:2
  else
    record_resolution "missing required env SafetyCarGoal1-v0"
  fi
  if env_exists "SafetyCarGoal2-v0"; then
    printf '%s\n' T3_CarGoal2:0 T3_CarGoal2:1 T3_CarGoal2:2
  else
    record_resolution "missing optional env SafetyCarGoal2-v0; skipped"
  fi
  local swimmer_task
  if swimmer_task="$(resolve_swimmer_task)"; then
    record_resolution "using swimmer env ${swimmer_task}"
    printf '%s\n' T4_SwimmerVelocity:0 T4_SwimmerVelocity:1 T4_SwimmerVelocity:2
  else
    record_resolution "missing optional env SafetySwimmerVelocity-v1/v0; skipped"
  fi
  record_resolution "T0_PointGoal1_ref is reference-only; existing G4 logs will be collected if available"
}

is_running() {
  tmux has-session -t "$(session_name "$1")" 2>/dev/null
}

completed_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  local env_key="${spec%%:*}"
  local seed="${spec##*:}"
  [[ -f "$log" ]] || return 1
  grep -q " END ${env_key} seed=${seed} " "$log" || return 1
  ! grep -E "$ERR_RE" "$log" >/dev/null 2>&1
}

failed_log() {
  local spec="$1"
  local log="$LOG_DIR/$(tag_for_spec "$spec").log"
  [[ -f "$log" ]] && grep -E "$ERR_RE" "$log" >/dev/null 2>&1
}

start_run() {
  local spec="$1"
  local env_key="${spec%%:*}"
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
BATCH_SIZE=${BATCH_SIZE} UPDATES_PER_STEP=${UPDATES_PER_STEP} HIDDEN_SIZE=${HIDDEN_SIZE} \
bash scripts/run_transfer_g4_fixed.sh ${env_key} ${seed}"
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
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv || true
  echo "===== recent transfer progress ====="
  grep -E "Env: Safety|Episode:|total numsteps| START | END " "$LOG_DIR"/T*_G4_fixed_main_seed*.log "$LOG_DIR"/smoke_*.log 2>/dev/null | tail -n 160 || true
  echo "===== failed transfer logs ====="
  grep -E "$ERR_RE" "$LOG_DIR"/T*_G4_fixed_main_seed*.log "$LOG_DIR"/smoke_*.log 2>/dev/null || true
  echo "===== env resolution ====="
  tail -n 40 "$RESOLUTION_LOG" 2>/dev/null || true
}

run_all() {
  local all_specs=()
  local queue=()
  local failed_specs=()
  local spec n idx active any_running
  mapfile -t all_specs < <(queue_specs | sed '/^$/d')
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
    echo "no pending transfer runs"
    monitor_once
    python scripts/collect_transfer_g4_fixed.py || true
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
        echo "error detected in $spec; not launching additional transfer runs"
        python scripts/collect_transfer_g4_fixed.py || true
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
  python scripts/collect_transfer_g4_fixed.py || true
  run_status_check
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^transfer_g4_/ {print $1}'); do
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
    echo "Usage: bash scripts/launch_transfer_g4_fixed_dynamic.sh plan|all|status|stop" >&2
    exit 2
    ;;
esac
