#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"

GPU_ID="${GPU_ID:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
HARD_MAX_PARALLEL="${HARD_MAX_PARALLEL:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-1}"

LOG_DIR="logs/pointgoal_goal_tuning"
MONITOR="$LOG_DIR/gpu_monitor.csv"
ERR_RE="Traceback|RuntimeError|NaN|nan|OOM|out of memory"
mkdir -p "$LOG_DIR" reports/pointgoal_goal_tuning

group_name() {
  case "$1" in
    G0|G0_legacy_best) echo "G0_legacy_best" ;;
    G1|G1_cdf_mean) echo "G1_cdf_mean" ;;
    G2|G2_cdf_mean_thr005) echo "G2_cdf_mean_thr005" ;;
    G3|G3_cdf_mean_safe07) echo "G3_cdf_mean_safe07" ;;
    G4|G4_cdf_mean_thr005_safe07) echo "G4_cdf_mean_thr005_safe07" ;;
    G5|G5_cdf_mean_jvp0035) echo "G5_cdf_mean_jvp0035" ;;
    *) echo "$1" ;;
  esac
}

queue_specs() {
  echo G0:0 G0:1 G0:2
  echo G1:0 G1:1 G1:2
  echo G2:0 G2:1 G2:2
  echo G3:0 G3:1 G3:2
  echo G4:0 G4:1 G4:2
  echo G5:0 G5:1 G5:2
}

tag_for_spec() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "$(group_name "$group")_seed${seed}"
}

session_name() {
  local group="${1%%:*}"
  local seed="${1##*:}"
  echo "goal_pg_${group}_${seed}"
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
  completed_log "$spec" && { echo "skip completed $spec"; return 1; }
  tmux has-session -t "$sess" 2>/dev/null && { echo "$sess already exists"; return 1; }
  tmux new -d -s "$sess" "cd /root/FLAC-Safe && \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate flac && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=\${WANDB_MODE:-offline} && \
bash scripts/run_pointgoal_goal_tuning.sh ${group} ${seed}"
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

status() {
  echo "===== tmux ====="
  tmux ls 2>/dev/null || true
  echo "===== gpu ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv || true
  echo "===== progress ====="
  grep -E "Env: SafetyPointGoal1-v0|Episode:| END " "$LOG_DIR"/*.log 2>/dev/null | tail -n 140 || true
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
      if start_run "${queue[$idx]}"; then
        active=$((active + 1))
      fi
      idx=$((idx + 1))
    done
    monitor_once
    for spec in "${queue[@]}"; do
      if failed_log "$spec"; then
        echo "error detected in $spec; not launching new runs"
        python scripts/collect_pointgoal_goal_tuning.py || true
        return 1
      fi
    done
    any_running=0
    for spec in "${queue[@]}"; do
      tmux has-session -t "$(session_name "$spec")" 2>/dev/null && any_running=1
    done
    (( idx >= ${#queue[@]} && any_running == 0 )) && break
    sleep 60
  done
  monitor_once
  python scripts/collect_pointgoal_goal_tuning.py || true
}

stop_sessions() {
  for sess in $(tmux ls 2>/dev/null | awk -F: '/^goal_pg_/ {print $1}'); do
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
    echo "Usage: bash scripts/launch_pointgoal_goal_tuning_dynamic.sh plan|all|status|stop" >&2
    exit 2
    ;;
esac
