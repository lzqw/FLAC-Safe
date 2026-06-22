#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TASK="${TASK:-SafetyCarGoal1-v0}"
SEEDS="${SEEDS:-0,1,2}"
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
LOG_ROOT="${LOG_ROOT:-logs/star_aaai/ablations}"
ACTION="${1:-plan}"

mkdir -p "$LOG_ROOT"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

declare -a RUNS=(
  "aggregation agg_mean --shadow_aggregation mean"
  "aggregation agg_log_mean_exp --shadow_aggregation log_mean_exp"
  "aggregation agg_max --shadow_aggregation max"
  "shadow_count k1 --shadow_k 1"
  "shadow_count k4 --shadow_k 4"
  "shadow_count k8 --shadow_k 8"
  "shadow_count k16 --shadow_k 16"
  "shadow_count k32 --shadow_k 32"
  "reference current_only --shadow_reference_mode current_only"
  "reference corridor --shadow_reference_mode corridor"
  "reference ref_interval5 --star_ref_update_interval 5"
  "reference ref_interval20 --star_ref_update_interval 20"
  "reference ref_interval100 --star_ref_update_interval 100"
  "reference kl_off --star_kl_coef 0.0"
  "reference kl_on --star_kl_coef 1.0"
  "components actor_audit_only --method star_actor --star_exec False"
  "components candidate_execution_only --method star_exec"
  "components full_star --method star"
  "critic_reduce mean_twin_cost --cost_critic_reduce mean"
  "critic_reduce max_twin_cost --cost_critic_reduce max"
  "temperature temp001 --shadow_temperature 0.01"
  "temperature temp003 --shadow_temperature 0.03"
  "temperature temp005 --shadow_temperature 0.05"
  "temperature temp010 --shadow_temperature 0.10"
  "execution_margin margin000 --star_exec_margin 0.0"
  "execution_margin margin001 --star_exec_margin 0.01"
  "execution_margin margin002 --star_exec_margin 0.02"
  "execution_margin margin005 --star_exec_margin 0.05"
)

active_count() {
  tmux ls 2>/dev/null | cut -d: -f1 | grep -c '^star_aaai_' || true
}

session_name() {
  local group="$1" name="$2" seed="$3"
  local s="star_aaai_ablation_${TASK}_${group}_${name}_s${seed}"
  echo "${s//[^A-Za-z0-9_]/_}"
}

run_name() {
  local group="$1" name="$2" seed="$3"
  local r="ablation_${TASK}_${group}_${name}_seed${seed}"
  echo "${r//[^A-Za-z0-9_.-]/_}"
}

command_for() {
  local group="$1" name="$2" seed="$3" run="$4" extra="$5"
  python_args=(
    main_star.py
    --task "$TASK"
    --safe_env True
    --method star
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
    --tag aaai_ablation
    --run_name "$run"
    --ablation_group "$group"
    --ablation_name "$name"
  )
  printf "python"
  printf " %q" "${python_args[@]}"
  printf " %s" "$extra"
}

plan() {
  printf "| Task | Group | Name | Seed | Steps | Tmux | Log |\n"
  printf "| --- | --- | --- | ---: | ---: | --- | --- |\n"
  for spec in "${RUNS[@]}"; do
    group="$(awk '{print $1}' <<< "$spec")"
    name="$(awk '{print $2}' <<< "$spec")"
    for seed in "${SEED_ARR[@]}"; do
      s="$(session_name "$group" "$name" "$seed")"
      log="$LOG_ROOT/$(run_name "$group" "$name" "$seed").log"
      printf "| %s | %s | %s | %s | %s | %s | %s |\n" "$TASK" "$group" "$name" "$seed" "$STEPS" "$s" "$log"
    done
  done
}

launch_all() {
  for spec in "${RUNS[@]}"; do
    group="$(awk '{print $1}' <<< "$spec")"
    name="$(awk '{print $2}' <<< "$spec")"
    extra="$(cut -d' ' -f3- <<< "$spec")"
    for seed in "${SEED_ARR[@]}"; do
      s="$(session_name "$group" "$name" "$seed")"
      run="$(run_name "$group" "$name" "$seed")"
      log="$LOG_ROOT/${run}.log"
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
      cmd="$(command_for "$group" "$name" "$seed" "$run" "$extra")"
      echo "Launching $s"
      tmux new -d -s "$s" "cd $(pwd) && export WANDB_MODE=offline && $cmd 2>&1 | tee $log"
    done
  done
}

case "$ACTION" in
  plan) plan ;;
  all|launch) launch_all ;;
  status) tmux ls 2>/dev/null | grep '^star_aaai_' || true ;;
  stop)
    for s in $(tmux ls 2>/dev/null | cut -d: -f1 | grep '^star_aaai_ablation_'); do
      echo "Stopping $s"
      tmux kill-session -t "$s" || true
    done
    ;;
  *)
    echo "Usage: $0 {plan|all|status|stop}" >&2
    exit 2
    ;;
esac
