#!/usr/bin/env bash
set -euo pipefail

CASE="${1:-all}"
LOG_DIR="logs/pointgoal_round2"
REPORT_DIR="reports/pointgoal_round2"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
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
  --seed 0 \
  --jvp_mode grad \
  --normalize_jvp False \
  --algo MF_SCTD_PointGoal_R2_4090"

run_case() {
  local case_name="$1"
  local tag case_args

  case "$case_name" in
    R2_A)
      tag="r2_A_safe05_jvp00005_bw005"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.0005 --safe_bandwidth 0.05 --tag $tag"
      ;;
    R2_B)
      tag="r2_B_safe05_jvp0001_bw005"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.001 --safe_bandwidth 0.05 --tag $tag"
      ;;
    R2_C)
      tag="r2_C_safe05_jvp0002_bw005"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.002 --safe_bandwidth 0.05 --tag $tag"
      ;;
    R2_D)
      tag="r2_D_safe05_jvp0005_bw005"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.005 --safe_bandwidth 0.05 --tag $tag"
      ;;
    R2_E)
      tag="r2_E_safe05_jvp0001_bw010"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.001 --safe_bandwidth 0.10 --tag $tag"
      ;;
    R2_F)
      tag="r2_F_safe03_jvp0002_bw005"
      case_args="--safe_policy_loss True --lambda_safe 0.3 --lambda_jvp 0.002 --safe_bandwidth 0.05 --tag $tag"
      ;;
    *)
      echo "Unknown case: $case_name" >&2
      echo "Usage: bash scripts/run_pointgoal_round2_4090.sh R2_A|R2_B|R2_C|R2_D|R2_E|R2_F|all" >&2
      return 2
      ;;
  esac

  local log_file="$LOG_DIR/${tag}.log"
  {
    echo "===== $(date '+%F %T') START $case_name $tag ====="
    echo "Command: python main.py $COMMON_ARGS $case_args"
    echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  } | tee "$log_file"
  python main.py $COMMON_ARGS $case_args 2>&1 | tee -a "$log_file"
  echo "===== $(date '+%F %T') END $case_name $tag =====" | tee -a "$log_file"
}

if [[ "$CASE" == "all" ]]; then
  for case_name in R2_A R2_B R2_C R2_D R2_E R2_F; do
    run_case "$case_name"
  done
else
  run_case "$CASE"
fi
