#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_cargoal1_soft_gate.sh SG_CG1_1|SG_CG1_2|SG_CG1_3|SG_CG1_4 seed" >&2
  exit 2
fi

LOG_DIR="logs/cargoal1_soft_gate"
REPORT_DIR="reports/cargoal1_soft_gate"
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

SAFETY_CRITIC_MODE="cdf"
QC_GEOM_MODE="mean"
CDF_BINARIZE_COST="True"
CDF_TARGET_CLIP="True"
NORMALIZE_JVP="True"
JVP_NORM_MODE="exact"
JVP_MODE="grad"
SOFT_FEASIBILITY_GATE="True"
FEAS_GATE_DETACH="True"
SAFE_BANDWIDTH="0.05"

case "$GROUP" in
  SG_CG1_1|SG_CG1_1_C2_tau010_floor02)
    GROUP_NAME="SG_CG1_1_C2_tau010_floor02"
    SAFE_THRESHOLD="0.05"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    FEAS_GATE_TAU="0.10"
    FEAS_GATE_REWARD_FLOOR="0.2"
    ;;
  SG_CG1_2|SG_CG1_2_C2_tau010_floor03)
    GROUP_NAME="SG_CG1_2_C2_tau010_floor03"
    SAFE_THRESHOLD="0.05"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    FEAS_GATE_TAU="0.10"
    FEAS_GATE_REWARD_FLOOR="0.3"
    ;;
  SG_CG1_3|SG_CG1_3_G4_tau010_floor02)
    GROUP_NAME="SG_CG1_3_G4_tau010_floor02"
    SAFE_THRESHOLD="0.05"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.003"
    FEAS_GATE_TAU="0.10"
    FEAS_GATE_REWARD_FLOOR="0.2"
    ;;
  SG_CG1_4|SG_CG1_4_C2_tau005_floor02)
    GROUP_NAME="SG_CG1_4_C2_tau005_floor02"
    SAFE_THRESHOLD="0.05"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    FEAS_GATE_TAU="0.05"
    FEAS_GATE_REWARD_FLOOR="0.2"
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

TAG="${GROUP_NAME}_seed${SEED}"
LOG_FILE="$LOG_DIR/${TAG}.log"

COMMON_ARGS="\
  --task SafetyCarGoal1-v0 \
  --safe_env True \
  --safe_policy_loss True \
  --safety_critic_mode ${SAFETY_CRITIC_MODE} \
  --qc_geom_mode ${QC_GEOM_MODE} \
  --safe_threshold ${SAFE_THRESHOLD} \
  --lambda_safe ${LAMBDA_SAFE} \
  --lambda_jvp ${LAMBDA_JVP} \
  --safe_bandwidth ${SAFE_BANDWIDTH} \
  --normalize_jvp ${NORMALIZE_JVP} \
  --jvp_norm_mode ${JVP_NORM_MODE} \
  --jvp_mode ${JVP_MODE} \
  --cdf_binarize_cost ${CDF_BINARIZE_COST} \
  --cdf_target_clip ${CDF_TARGET_CLIP} \
  --soft_feasibility_gate ${SOFT_FEASIBILITY_GATE} \
  --feas_gate_tau ${FEAS_GATE_TAU} \
  --feas_gate_detach ${FEAS_GATE_DETACH} \
  --feas_gate_reward_floor ${FEAS_GATE_REWARD_FLOOR} \
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
  --algo MF_SCTD_CarGoal1_Soft_Gate \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START ${GROUP_NAME} seed=${SEED} task=SafetyCarGoal1-v0 tag=${TAG} ====="
  echo "TASK=SafetyCarGoal1-v0 GROUP=${GROUP_NAME}"
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "NUM_STEPS=$NUM_STEPS START_STEPS=$START_STEPS EVAL_NUMSTEPS=$EVAL_NUMSTEPS EVAL_TIMES=$EVAL_TIMES"
  echo "safety_critic_mode=$SAFETY_CRITIC_MODE qc_geom_mode=$QC_GEOM_MODE safe_threshold=$SAFE_THRESHOLD lambda_safe=$LAMBDA_SAFE lambda_jvp=$LAMBDA_JVP safe_bandwidth=$SAFE_BANDWIDTH"
  echo "soft_feasibility_gate=$SOFT_FEASIBILITY_GATE feas_gate_tau=$FEAS_GATE_TAU feas_gate_detach=$FEAS_GATE_DETACH feas_gate_reward_floor=$FEAS_GATE_REWARD_FLOOR"
  echo "soft_normal_masking=False directional_ref_noise=False risk_side_gate=False"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END ${GROUP_NAME} seed=${SEED} task=SafetyCarGoal1-v0 tag=${TAG} =====" | tee -a "$LOG_FILE"
