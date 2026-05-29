#!/usr/bin/env bash
set -euo pipefail

CASE="${1:-all}"
SEED_ARG="${2:-}"
LOG_DIR="logs/pointgoal_seed_sweep"
REPORT_DIR="reports/pointgoal_seed_sweep"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

COMMON_ARGS="\
  --task SafetyPointGoal1-v0 \
  --safe_env True \
  --distributional_critic False \
  --compile_model False \
  --soft_normal_masking False \
  --epsilon 0.0 \
  --batch_size ${BATCH_SIZE} \
  --updates_per_step ${UPDATES_PER_STEP} \
  --hidden_size ${HIDDEN_SIZE} \
  --num_steps 120000 \
  --start_steps 5000 \
  --eval True \
  --eval_numsteps 5000 \
  --eval_times 5 \
  --save False \
  --steps 1 \
  --jvp_mode grad \
  --normalize_jvp False \
  --algo MF_SCTD_PointGoal_SeedSweep"

case_args() {
  local group="$1"
  case "$group" in
    S0|S0_penalty_only)
      echo "--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0 --safe_bandwidth 0.05"
      ;;
    S1|S1_main_R2D)
      echo "--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.005 --safe_bandwidth 0.05"
      ;;
    S2|S2_bw010_R2E)
      echo "--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.001 --safe_bandwidth 0.10"
      ;;
    *)
      echo "Unknown group: $group" >&2
      echo "Usage: bash scripts/run_pointgoal_seed_sweep.sh GROUP:SEED|all" >&2
      return 2
      ;;
  esac
}

canonical_group() {
  local group="$1"
  case "$group" in
    S0|S0_penalty_only)
      echo "S0_penalty_only"
      ;;
    S1|S1_main_R2D)
      echo "S1_main_R2D"
      ;;
    S2|S2_bw010_R2E)
      echo "S2_bw010_R2E"
      ;;
    *)
      echo "Unknown group: $group" >&2
      return 2
      ;;
  esac
}

run_case() {
  local spec="$1"
  local group="${spec%%:*}"
  local seed="${spec##*:}"
  if [[ "$group" == "$seed" || -z "$group" || -z "$seed" ]]; then
    echo "Case must be GROUP:SEED, got: $spec" >&2
    return 2
  fi
  local args tag log_file
  group="$(canonical_group "$group")"
  args="$(case_args "$group")"
  tag="${group}_seed${seed}"
  log_file="$LOG_DIR/${tag}.log"

  {
    echo "===== $(date '+%F %T') START $group seed=$seed tag=$tag ====="
    echo "Command: python main.py $COMMON_ARGS --seed $seed $args --tag $tag"
    echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  } | tee "$log_file"
  python main.py $COMMON_ARGS --seed "$seed" $args --tag "$tag" 2>&1 | tee -a "$log_file"
  echo "===== $(date '+%F %T') END $group seed=$seed tag=$tag =====" | tee -a "$log_file"
}

if [[ "$CASE" == "all" ]]; then
  for group in S0_penalty_only S1_main_R2D S2_bw010_R2E; do
    for seed in 0 1 2 3 4; do
      run_case "${group}:${seed}"
    done
  done
elif [[ -n "$SEED_ARG" ]]; then
  run_case "${CASE}:${SEED_ARG}"
else
  run_case "$CASE"
fi
