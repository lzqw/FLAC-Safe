#!/usr/bin/env bash
set -euo pipefail

ENV_KEY="${1:-}"
SEED="${2:-}"

if [[ -z "$ENV_KEY" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_transfer_g4_fixed.sh T0|T1|T2|T3|T4 seed" >&2
  exit 2
fi

LOG_DIR="logs/transfer_g4_fixed"
REPORT_DIR="reports/transfer_g4_fixed"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_STEPS="${NUM_STEPS:-120000}"
START_STEPS="${START_STEPS:-5000}"
EVAL_NUMSTEPS="${EVAL_NUMSTEPS:-5000}"
EVAL_TIMES="${EVAL_TIMES:-5}"

env_exists() {
  local task_id="$1"
  TASK_ID="$task_id" python - <<'PY' >/dev/null 2>&1
import os
import gymnasium as gym
import safety_gymnasium  # noqa: F401

raise SystemExit(0 if os.environ["TASK_ID"] in gym.registry else 1)
PY
}

resolve_swimmer_task() {
  if [[ -n "${SWIMMER_TASK:-}" ]]; then
    echo "$SWIMMER_TASK"
    return 0
  fi
  if env_exists "SafetySwimmerVelocity-v1"; then
    echo "SafetySwimmerVelocity-v1"
    return 0
  fi
  if env_exists "SafetySwimmerVelocity-v0"; then
    echo "SafetySwimmerVelocity-v0"
    return 0
  fi
  return 1
}

case "$ENV_KEY" in
  T0|T0_PointGoal1_ref)
    ENV_NAME="T0_PointGoal1_ref"
    TASK="SafetyPointGoal1-v0"
    ;;
  T1|T1_PointGoal2)
    ENV_NAME="T1_PointGoal2"
    TASK="SafetyPointGoal2-v0"
    ;;
  T2|T2_CarGoal1)
    ENV_NAME="T2_CarGoal1"
    TASK="SafetyCarGoal1-v0"
    ;;
  T3|T3_CarGoal2)
    ENV_NAME="T3_CarGoal2"
    TASK="SafetyCarGoal2-v0"
    ;;
  T4|T4_SwimmerVelocity)
    ENV_NAME="T4_SwimmerVelocity"
    TASK="$(resolve_swimmer_task)" || {
      echo "No SafetySwimmerVelocity-v1 or SafetySwimmerVelocity-v0 registered; skipping." >&2
      exit 3
    }
    ;;
  *)
    echo "Unknown env key: $ENV_KEY" >&2
    exit 2
    ;;
esac

if ! env_exists "$TASK"; then
  echo "Task is not registered: $TASK" >&2
  exit 3
fi

TAG="${ENV_NAME}_G4_fixed_main_seed${SEED}"
LOG_FILE="$LOG_DIR/${TAG}.log"

COMMON_ARGS="\
  --task ${TASK} \
  --safe_env True \
  --safe_policy_loss True \
  --safety_critic_mode cdf \
  --qc_geom_mode mean \
  --safe_threshold 0.05 \
  --lambda_safe 0.7 \
  --lambda_jvp 0.003 \
  --safe_bandwidth 0.05 \
  --normalize_jvp True \
  --jvp_norm_mode exact \
  --jvp_mode grad \
  --cdf_binarize_cost True \
  --cdf_target_clip True \
  --batch_size ${BATCH_SIZE} \
  --updates_per_step ${UPDATES_PER_STEP} \
  --hidden_size ${HIDDEN_SIZE} \
  --num_steps ${NUM_STEPS} \
  --start_steps ${START_STEPS} \
  --eval True \
  --eval_numsteps ${EVAL_NUMSTEPS} \
  --eval_times ${EVAL_TIMES} \
  --distributional_critic False \
  --compile_model False \
  --soft_normal_masking False \
  --directional_ref_noise False \
  --epsilon 0.0 \
  --save False \
  --steps 1 \
  --seed ${SEED} \
  --algo MF_SCTD_G4_Fixed_Transfer \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START ${ENV_NAME} seed=${SEED} task=${TASK} tag=${TAG} ====="
  echo "CONFIG=G4_fixed_main"
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "NUM_STEPS=$NUM_STEPS START_STEPS=$START_STEPS EVAL_NUMSTEPS=$EVAL_NUMSTEPS EVAL_TIMES=$EVAL_TIMES"
  echo "safety_critic_mode=cdf qc_geom_mode=mean safe_threshold=0.05 lambda_safe=0.7 lambda_jvp=0.003"
  echo "soft_normal_masking=False directional_ref_noise=False soft_feasibility_gate=False risk_side_gate=False"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END ${ENV_NAME} seed=${SEED} task=${TASK} tag=${TAG} =====" | tee -a "$LOG_FILE"
