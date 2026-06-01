#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_pointgoal_njvp_test.sh N0|N1|N2|N3|N4 seed" >&2
  exit 2
fi

LOG_DIR="logs/pointgoal_njvp_test"
REPORT_DIR="reports/pointgoal_njvp_test"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

case "$GROUP" in
  N0|N0_raw_R2D)
    GROUP_NAME="N0_raw_R2D"
    NORMALIZE_JVP="False"
    JVP_NORM_MODE="exact"
    LAMBDA_JVP="0.005"
    ;;
  N1|N1_njvp_0005)
    GROUP_NAME="N1_njvp_0005"
    NORMALIZE_JVP="True"
    JVP_NORM_MODE="exact"
    LAMBDA_JVP="0.0005"
    ;;
  N2|N2_njvp_001)
    GROUP_NAME="N2_njvp_001"
    NORMALIZE_JVP="True"
    JVP_NORM_MODE="exact"
    LAMBDA_JVP="0.001"
    ;;
  N3|N3_njvp_003)
    GROUP_NAME="N3_njvp_003"
    NORMALIZE_JVP="True"
    JVP_NORM_MODE="exact"
    LAMBDA_JVP="0.003"
    ;;
  N4|N4_njvp_005)
    GROUP_NAME="N4_njvp_005"
    NORMALIZE_JVP="True"
    JVP_NORM_MODE="exact"
    LAMBDA_JVP="0.005"
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
  --normalize_jvp ${NORMALIZE_JVP} \
  --jvp_norm_mode ${JVP_NORM_MODE} \
  --lambda_safe 0.5 \
  --lambda_jvp ${LAMBDA_JVP} \
  --safe_bandwidth 0.05 \
  --algo MF_SCTD_PointGoal_NJVPTest \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START $GROUP_NAME seed=$SEED tag=$TAG ====="
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "NORMALIZE_JVP=$NORMALIZE_JVP JVP_NORM_MODE=$JVP_NORM_MODE LAMBDA_JVP=$LAMBDA_JVP"
} | tee "$LOG_FILE"
python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END $GROUP_NAME seed=$SEED tag=$TAG =====" | tee -a "$LOG_FILE"
