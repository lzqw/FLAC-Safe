#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_pointgoal_njvp_overnight.sh ON0|ON1|ON2|ON3|ON4 seed" >&2
  exit 2
fi

LOG_DIR="logs/pointgoal_njvp_overnight"
REPORT_DIR="reports/pointgoal_njvp_overnight"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

case "$GROUP" in
  ON0|ON0_N3_extend)
    GROUP_NAME="ON0_N3_extend"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    SAFE_BANDWIDTH="0.05"
    ;;
  ON1|ON1_jvp0035)
    GROUP_NAME="ON1_jvp0035"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.0035"
    SAFE_BANDWIDTH="0.05"
    ;;
  ON2|ON2_jvp0040)
    GROUP_NAME="ON2_jvp0040"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.004"
    SAFE_BANDWIDTH="0.05"
    ;;
  ON3|ON3_bw0075)
    GROUP_NAME="ON3_bw0075"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    SAFE_BANDWIDTH="0.075"
    ;;
  ON4|ON4_safe07)
    GROUP_NAME="ON4_safe07"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.003"
    SAFE_BANDWIDTH="0.05"
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

TAG="${GROUP_NAME}_seed${SEED}"
LOG_FILE="$LOG_DIR/${TAG}.log"

COMMON_ARGS="\
  --task SafetyPointGoal1-v0 \
  --safe_env True \
  --safe_policy_loss True \
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
  --seed ${SEED} \
  --jvp_mode grad \
  --normalize_jvp True \
  --jvp_norm_mode exact \
  --lambda_safe ${LAMBDA_SAFE} \
  --lambda_jvp ${LAMBDA_JVP} \
  --safe_bandwidth ${SAFE_BANDWIDTH} \
  --algo MF_SCTD_PointGoal_NJVPOvernight \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START $GROUP_NAME seed=$SEED tag=$TAG ====="
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "NORMALIZE_JVP=True JVP_NORM_MODE=exact LAMBDA_SAFE=$LAMBDA_SAFE LAMBDA_JVP=$LAMBDA_JVP SAFE_BANDWIDTH=$SAFE_BANDWIDTH"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END $GROUP_NAME seed=$SEED tag=$TAG =====" | tee -a "$LOG_FILE"
