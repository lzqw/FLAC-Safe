#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_pointgoal_cdf_safety.sh C0|C1|C2 seed" >&2
  exit 2
fi

LOG_DIR="logs/pointgoal_cdf_safety"
REPORT_DIR="reports/pointgoal_cdf_safety"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

case "$GROUP" in
  C0|C0_cumulative)
    GROUP_NAME="C0_cumulative"
    SAFETY_CRITIC_MODE="cumulative"
    QC_GEOM_MODE="max"
    SAFE_THRESHOLD="${SAFE_THRESHOLD:-0.1}"
    CDF_BINARIZE_COST="True"
    CDF_TARGET_CLIP="True"
    ;;
  C1|C1_cdf_max)
    GROUP_NAME="C1_cdf_max"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="max"
    SAFE_THRESHOLD="0.1"
    CDF_BINARIZE_COST="True"
    CDF_TARGET_CLIP="True"
    ;;
  C2|C2_cdf_mean)
    GROUP_NAME="C2_cdf_mean"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="mean"
    SAFE_THRESHOLD="0.1"
    CDF_BINARIZE_COST="True"
    CDF_TARGET_CLIP="True"
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
  --directional_ref_noise False \
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
  --lambda_safe 0.5 \
  --lambda_jvp 0.003 \
  --safe_bandwidth 0.05 \
  --safe_threshold ${SAFE_THRESHOLD} \
  --safety_critic_mode ${SAFETY_CRITIC_MODE} \
  --qc_geom_mode ${QC_GEOM_MODE} \
  --cdf_binarize_cost ${CDF_BINARIZE_COST} \
  --cdf_target_clip ${CDF_TARGET_CLIP} \
  --algo MF_SCTD_PointGoal_CDFSafety \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START $GROUP_NAME seed=$SEED tag=$TAG ====="
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "SAFETY_CRITIC_MODE=$SAFETY_CRITIC_MODE QC_GEOM_MODE=$QC_GEOM_MODE SAFE_THRESHOLD=$SAFE_THRESHOLD"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END $GROUP_NAME seed=$SEED tag=$TAG =====" | tee -a "$LOG_FILE"
