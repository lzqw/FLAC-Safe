#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
SEED="${2:-}"

if [[ -z "$GROUP" || -z "$SEED" ]]; then
  echo "Usage: bash scripts/run_high_fidelity_q_cargoal1.sh HQC_CG1_1|HQC_CG1_2|HQC_CG1_3|HQC_CG1_4 seed" >&2
  exit 2
fi

LOG_DIR="logs/high_fidelity_q_cargoal1"
REPORT_DIR="reports/high_fidelity_q_cargoal1"
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
SAFE_THRESHOLD="0.05"
SAFE_BANDWIDTH="0.05"
CDF_BINARIZE_COST="True"
CDF_TARGET_CLIP="True"
NORMALIZE_JVP="True"
JVP_NORM_MODE="exact"
JVP_MODE="grad"
HIGH_FIDELITY_SAFETY_Q="True"
SAFETY_Q_PRIORITY="True"
SAFETY_Q_COST_WEIGHT="2.0"
SAFETY_Q_BOUNDARY_WEIGHT="3.0"
SAFETY_Q_TD_WEIGHT="0.0"
SAFETY_Q_MAX_WEIGHT="5.0"
SAFETY_Q_BOUNDARY_WIDTH="0.05"
SAFETY_Q_FD_EPS="0.01"
DIAGNOSE_SAFETY_Q_GEOMETRY="True"

case "$GROUP" in
  HQC_CG1_1|HQC_CG1_1_G4_priority)
    GROUP_NAME="HQC_CG1_1_G4_priority"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.003"
    SAFETY_Q_EXTRA_UPDATES="0"
    ;;
  HQC_CG1_2|HQC_CG1_2_C2_priority)
    GROUP_NAME="HQC_CG1_2_C2_priority"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.003"
    SAFETY_Q_EXTRA_UPDATES="0"
    ;;
  HQC_CG1_3|HQC_CG1_3_G4_priority_extra1)
    GROUP_NAME="HQC_CG1_3_G4_priority_extra1"
    LAMBDA_SAFE="0.7"
    LAMBDA_JVP="0.003"
    SAFETY_Q_EXTRA_UPDATES="1"
    ;;
  HQC_CG1_4|HQC_CG1_4_C2_priority_weak_jvp)
    GROUP_NAME="HQC_CG1_4_C2_priority_weak_jvp"
    LAMBDA_SAFE="0.5"
    LAMBDA_JVP="0.001"
    SAFETY_Q_EXTRA_UPDATES="1"
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
  --high_fidelity_safety_q ${HIGH_FIDELITY_SAFETY_Q} \
  --safety_q_priority ${SAFETY_Q_PRIORITY} \
  --safety_q_cost_weight ${SAFETY_Q_COST_WEIGHT} \
  --safety_q_boundary_weight ${SAFETY_Q_BOUNDARY_WEIGHT} \
  --safety_q_td_weight ${SAFETY_Q_TD_WEIGHT} \
  --safety_q_max_weight ${SAFETY_Q_MAX_WEIGHT} \
  --safety_q_extra_updates ${SAFETY_Q_EXTRA_UPDATES} \
  --safety_q_boundary_width ${SAFETY_Q_BOUNDARY_WIDTH} \
  --diagnose_safety_q_geometry ${DIAGNOSE_SAFETY_Q_GEOMETRY} \
  --safety_q_fd_eps ${SAFETY_Q_FD_EPS} \
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
  --soft_feasibility_gate False \
  --soft_normal_masking False \
  --directional_ref_noise False \
  --epsilon 0.0 \
  --save False \
  --steps 1 \
  --seed ${SEED} \
  --algo MF_SCTD_High_Fidelity_Q_CarGoal1 \
  --tag ${TAG}"

{
  echo "===== $(date '+%F %T') START ${GROUP_NAME} seed=${SEED} task=SafetyCarGoal1-v0 tag=${TAG} ====="
  echo "TASK=SafetyCarGoal1-v0 GROUP=${GROUP_NAME}"
  echo "Command: python main.py $COMMON_ARGS"
  echo "BATCH_SIZE=$BATCH_SIZE UPDATES_PER_STEP=$UPDATES_PER_STEP HIDDEN_SIZE=$HIDDEN_SIZE"
  echo "NUM_STEPS=$NUM_STEPS START_STEPS=$START_STEPS EVAL_NUMSTEPS=$EVAL_NUMSTEPS EVAL_TIMES=$EVAL_TIMES"
  echo "lambda_safe=$LAMBDA_SAFE lambda_jvp=$LAMBDA_JVP safety_q_extra_updates=$SAFETY_Q_EXTRA_UPDATES"
  echo "high_fidelity_safety_q=$HIGH_FIDELITY_SAFETY_Q safety_q_priority=$SAFETY_Q_PRIORITY safety_q_boundary_width=$SAFETY_Q_BOUNDARY_WIDTH"
  echo "soft_feasibility_gate=False soft_normal_masking=False directional_ref_noise=False"
} | tee "$LOG_FILE"

python main.py $COMMON_ARGS 2>&1 | tee -a "$LOG_FILE"
echo "===== $(date '+%F %T') END ${GROUP_NAME} seed=${SEED} task=SafetyCarGoal1-v0 tag=${TAG} =====" | tee -a "$LOG_FILE"
