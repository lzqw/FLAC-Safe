#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_pointgoal_goal_tuning.sh G0|G1|G2|G3|G4|G5 seed" >&2
  exit 2
fi

LOG_DIR="logs/pointgoal_goal_tuning"
REPORT_DIR="reports/pointgoal_goal_tuning"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"

SOFT_GATE="False"
FEAS_TAU="0.05"
FEAS_FLOOR="0.0"
RISK_GATE="False"
RISK_RHO="0.3"
RISK_TAU="0.05"

case "$GROUP" in
  G0|G0_legacy_best)
    GROUP_NAME="G0_legacy_best"
    STAGE="A"
    SAFETY_CRITIC_MODE="cumulative"
    QC_GEOM_MODE="max"
    SAFE_THRESHOLD="0.1"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    ;;
  G1|G1_cdf_mean)
    GROUP_NAME="G1_cdf_mean"
    STAGE="A"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="mean"
    SAFE_THRESHOLD="0.1"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    ;;
  G2|G2_cdf_mean_thr005)
    GROUP_NAME="G2_cdf_mean_thr005"
    STAGE="A"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="mean"
    SAFE_THRESHOLD="0.05"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    ;;
  G3|G3_cdf_mean_safe07)
    GROUP_NAME="G3_cdf_mean_safe07"
    STAGE="A"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="mean"
    SAFE_THRESHOLD="0.1"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.003"
    ;;
  G4|G4_cdf_mean_thr005_safe07)
    GROUP_NAME="G4_cdf_mean_thr005_safe07"
    STAGE="A"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="mean"
    SAFE_THRESHOLD="0.05"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.003"
    ;;
  G5|G5_cdf_mean_jvp0035)
    GROUP_NAME="G5_cdf_mean_jvp0035"
    STAGE="A"
    SAFETY_CRITIC_MODE="cdf"
    QC_GEOM_MODE="mean"
    SAFE_THRESHOLD="0.1"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.0035"
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
  --safety_critic_mode ${SAFETY_CRITIC_MODE} \
  --qc_geom_mode ${QC_GEOM_MODE} \
  --cdf_binarize_cost True \
  --cdf_target_clip True \
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
  --lambda_safe ${LAMBDA_SAFE} \
  --lambda_jvp ${LAMBDA_JVP} \
  --safe_bandwidth 0.05 \
  --safe_threshold ${SAFE_THRESHOLD} \
  --algo MF_SCTD_PointGoal_GoalTuning \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START $GROUP_NAME seed=$SEED tag=$TAG ====="
  echo "GROUP=$GROUP_NAME STAGE=$STAGE"
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "SAFETY_CRITIC_MODE=$SAFETY_CRITIC_MODE QC_GEOM_MODE=$QC_GEOM_MODE SAFE_THRESHOLD=$SAFE_THRESHOLD"
  echo "LAMBDA_SAFE=$LAMBDA_SAFE LAMBDA_JVP=$LAMBDA_JVP"
  echo "SOFT_GATE=$SOFT_GATE FEAS_TAU=$FEAS_TAU FEAS_FLOOR=$FEAS_FLOOR RISK_GATE=$RISK_GATE RISK_RHO=$RISK_RHO RISK_TAU=$RISK_TAU"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END $GROUP_NAME seed=$SEED tag=$TAG =====" | tee -a "$LOG_FILE"
