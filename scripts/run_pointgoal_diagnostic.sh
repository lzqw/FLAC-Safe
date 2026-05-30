#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_pointgoal_diagnostic.sh PGD0|PGD1|PGD2|PGD3|PGD4 seed" >&2
  exit 2
fi

LOG_DIR="logs/pointgoal_diagnostic"
REPORT_DIR="reports/pointgoal_diagnostic"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

case "$GROUP" in
  PGD0|PGD0_S0_extend)
    GROUP_NAME="PGD0_S0_extend"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0"
    SAFE_BANDWIDTH="0.05"
    UPDATES_PER_STEP="2"
    ;;
  PGD1|PGD1_S1_R2D_extend)
    GROUP_NAME="PGD1_S1_R2D_extend"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.005"
    SAFE_BANDWIDTH="0.05"
    UPDATES_PER_STEP="2"
    ;;
  PGD2|PGD2_R2D_update1)
    GROUP_NAME="PGD2_R2D_update1"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.005"
    SAFE_BANDWIDTH="0.05"
    UPDATES_PER_STEP="1"
    ;;
  PGD3|PGD3_mid_jvp)
    GROUP_NAME="PGD3_mid_jvp"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    SAFE_BANDWIDTH="0.05"
    UPDATES_PER_STEP="2"
    ;;
  PGD4|PGD4_stronger_safe)
    GROUP_NAME="PGD4_stronger_safe"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.005"
    SAFE_BANDWIDTH="0.05"
    UPDATES_PER_STEP="2"
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
  --normalize_jvp False \
  --algo MF_SCTD_PointGoal_Diagnostic"

CASE_ARGS="\
  --safe_policy_loss True \
  --lambda_safe ${LAMBDA_SAFE} \
  --lambda_jvp ${LAMBDA_JVP} \
  --safe_bandwidth ${SAFE_BANDWIDTH} \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START $GROUP_NAME seed=$SEED tag=$TAG ====="
  echo "Command: python main.py $COMMON_ARGS $CASE_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
} | tee "$LOG_FILE"
python main.py $COMMON_ARGS $CASE_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END $GROUP_NAME seed=$SEED tag=$TAG =====" | tee -a "$LOG_FILE"
