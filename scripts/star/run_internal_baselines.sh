#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TASKS="${TASKS:-SafetyPointGoal1-v0,SafetyPointGoal2-v0,SafetyCarGoal1-v0,SafetyCarGoal2-v0}"
SEEDS="${SEEDS:-0,1,2,3,4}"
METHODS="${METHODS:-sac,pointwise,sac_lag,star_actor,star_exec,star}"
STEPS="${STEPS:-1000000}"
START_STEPS="${START_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_TIMES="${EVAL_TIMES:-10}"
EVAL_NUMSTEPS="${EVAL_NUMSTEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/star_aaai}"
LOG_ROOT="${LOG_ROOT:-logs/star_aaai/internal}"
ACTION="${1:-plan}"

mkdir -p "$LOG_ROOT"

IFS=',' read -r -a TASK_ARR <<< "$TASKS"
IFS=',' read -r -a METHOD_ARR <<< "$METHODS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

active_count() {
  tmux ls 2>/dev/null | cut -d: -f1 | grep -c '^star_aaai_' || true
}

session_name() {
  local task="$1" method="$2" seed="$3"
  local name="star_aaai_internal_${task}_${method}_s${seed}"
  echo "${name//[^A-Za-z0-9_]/_}"
}

run_name() {
  local task="$1" method="$2" seed="$3"
  local name="internal_${task}_${method}_seed${seed}"
  echo "${name//[^A-Za-z0-9_.-]/_}"
}

command_for() {
  local task="$1" method="$2" seed="$3" name="$4"
  local extra=""
  if [ "$method" = "star_actor" ]; then
    extra="--star_exec False"
  fi
  python_args=(
    main_star.py
    --task "$task"
    --safe_env True
    --method "$method"
    --seed "$seed"
    --num_steps "$STEPS"
    --start_steps "$START_STEPS"
    --batch_size "$BATCH_SIZE"
    --hidden_size "$HIDDEN_SIZE"
    --recent_fraction 0.0
    --binary_cost True
    --eval True
    --eval_interval_steps "$EVAL_INTERVAL"
    --eval_times "$EVAL_TIMES"
    --eval_numsteps "$EVAL_NUMSTEPS"
    --save True
    --save_interval_steps "$SAVE_INTERVAL"
    --output_root "$OUTPUT_ROOT"
    --tag aaai_internal
    --run_name "$name"
    --ablation_group internal
    --ablation_name "$method"
  )
  printf "python"
  printf " %q" "${python_args[@]}"
  if [ -n "$extra" ]; then
    printf " %s" "$extra"
  fi
}

plan() {
  printf "| Task | Method | Seed | Steps | Tmux | Log |\n"
  printf "| --- | --- | ---: | ---: | --- | --- |\n"
  for task in "${TASK_ARR[@]}"; do
    for method in "${METHOD_ARR[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        s="$(session_name "$task" "$method" "$seed")"
        log="$LOG_ROOT/$(run_name "$task" "$method" "$seed").log"
        printf "| %s | %s | %s | %s | %s | %s |\n" "$task" "$method" "$seed" "$STEPS" "$s" "$log"
      done
    done
  done
}

launch_all() {
  for task in "${TASK_ARR[@]}"; do
    for method in "${METHOD_ARR[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        s="$(session_name "$task" "$method" "$seed")"
        name="$(run_name "$task" "$method" "$seed")"
        log="$LOG_ROOT/${name}.log"
        if tmux has-session -t "$s" 2>/dev/null; then
          echo "Skip running session $s"
          continue
        fi
        if [ -f "$log" ]; then
          echo "Skip existing log $log"
          continue
        fi
        while [ "$(active_count)" -ge "$MAX_PARALLEL" ]; do
          sleep 60
        done
        cmd="$(command_for "$task" "$method" "$seed" "$name")"
        echo "Launching $s"
        tmux new -d -s "$s" "cd $(pwd) && export WANDB_MODE=offline && $cmd 2>&1 | tee $log"
      done
    done
  done
}

case "$ACTION" in
  plan) plan ;;
  all|launch) launch_all ;;
  status) tmux ls 2>/dev/null | grep '^star_aaai_' || true ;;
  stop)
    for s in $(tmux ls 2>/dev/null | cut -d: -f1 | grep '^star_aaai_internal_'); do
      echo "Stopping $s"
      tmux kill-session -t "$s" || true
    done
    ;;
  *)
    echo "Usage: $0 {plan|all|status|stop}" >&2
    exit 2
    ;;
esac
