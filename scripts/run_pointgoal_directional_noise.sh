#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_pointgoal_directional_noise.sh DN0|DN1|DN2|DN3 seed" >&2
  exit 2
fi

LOG_DIR="logs/pointgoal_directional_noise"
REPORT_DIR="reports/pointgoal_directional_noise"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

case "$GROUP" in
  DN0|DN0_none)
    GROUP_NAME="DN0_none"
    DIRECTIONAL_REF_NOISE="False"
    DIRECTIONAL_NOISE_MODE="none"
    TANGENT_NOISE_SCALE="0.05"
    REF_NOISE_SCALE="0.0"
    NORMAL_NOISE_SCALE="0.0"
    BETA_MAX="0.5"
    WARMUP_STEPS="10000"
    ;;
  DN1|DN1_tangent)
    GROUP_NAME="DN1_tangent"
    DIRECTIONAL_REF_NOISE="True"
    DIRECTIONAL_NOISE_MODE="tangent"
    TANGENT_NOISE_SCALE="0.05"
    REF_NOISE_SCALE="0.0"
    NORMAL_NOISE_SCALE="0.0"
    BETA_MAX="0.5"
    WARMUP_STEPS="10000"
    ;;
  DN2|DN2_reward_ref)
    GROUP_NAME="DN2_reward_ref"
    DIRECTIONAL_REF_NOISE="True"
    DIRECTIONAL_NOISE_MODE="reward_ref"
    TANGENT_NOISE_SCALE="0.05"
    REF_NOISE_SCALE="0.02"
    NORMAL_NOISE_SCALE="0.0"
    BETA_MAX="0.5"
    WARMUP_STEPS="10000"
    ;;
  DN3|DN3_ref_normal)
    GROUP_NAME="DN3_ref_normal"
    DIRECTIONAL_REF_NOISE="True"
    DIRECTIONAL_NOISE_MODE="ref_normal"
    TANGENT_NOISE_SCALE="0.05"
    REF_NOISE_SCALE="0.02"
    NORMAL_NOISE_SCALE="0.01"
    BETA_MAX="0.5"
    WARMUP_STEPS="10000"
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
  --lambda_safe 0.5 \
  --lambda_jvp 0.003 \
  --safe_bandwidth 0.05 \
  --directional_ref_noise ${DIRECTIONAL_REF_NOISE} \
  --directional_noise_mode ${DIRECTIONAL_NOISE_MODE} \
  --tangent_noise_scale ${TANGENT_NOISE_SCALE} \
  --ref_noise_scale ${REF_NOISE_SCALE} \
  --normal_noise_scale ${NORMAL_NOISE_SCALE} \
  --directional_noise_beta_max ${BETA_MAX} \
  --directional_noise_warmup_steps ${WARMUP_STEPS} \
  --algo MF_SCTD_PointGoal_DirectionalNoise \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START $GROUP_NAME seed=$SEED tag=$TAG ====="
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "DIRECTIONAL_REF_NOISE=$DIRECTIONAL_REF_NOISE DIRECTIONAL_NOISE_MODE=$DIRECTIONAL_NOISE_MODE"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END $GROUP_NAME seed=$SEED tag=$TAG =====" | tee -a "$LOG_FILE"
