#!/usr/bin/env bash
set -euo pipefail

CASE="${1:-all}"
LOG_DIR="logs/convergence"
mkdir -p "$LOG_DIR"

COMMON_ARGS="\
  --task SafetyPointGoal1-v0 \
  --safe_env True \
  --distributional_critic False \
  --num_steps 80000 \
  --start_steps 5000 \
  --batch_size 512 \
  --updates_per_step 1 \
  --hidden_size 512 \
  --eval True \
  --eval_numsteps 5000 \
  --eval_times 3 \
  --save False \
  --steps 1 \
  --epsilon 0.0 \
  --compile_model False \
  --seed 0 \
  --algo MF_SCTD_Convergence"

run_case() {
  local case_name="$1"
  local case_args tag

  case "$case_name" in
    C0)
      tag="C0_base"
      case_args="--safe_policy_loss False --lambda_safe 0 --lambda_jvp 0 --tag $tag"
      ;;
    C1)
      tag="C1_weak_safety"
      case_args="--safe_policy_loss True --lambda_safe 0.1 --lambda_jvp 0 --tag $tag"
      ;;
    C2)
      tag="C2_safety"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0 --tag $tag"
      ;;
    C3)
      tag="C3_weak_jvp"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.001 --jvp_warmup_steps 20000 --jvp_mode grad --normalize_jvp False --tag $tag"
      ;;
    C4)
      tag="C4_slightly_stronger_jvp"
      case_args="--safe_policy_loss True --lambda_safe 0.5 --lambda_jvp 0.005 --jvp_warmup_steps 20000 --jvp_mode grad --normalize_jvp False --tag $tag"
      ;;
    *)
      echo "Unknown convergence case: $case_name" >&2
      echo "Usage: bash scripts/run_convergence_pointgoal.sh C0|C1|C2|C3|C4|all" >&2
      return 2
      ;;
  esac

  local log_file="$LOG_DIR/${tag}.log"
  echo "===== $(date '+%F %T') START $case_name $tag =====" | tee "$log_file"
  echo "Command: python main.py $COMMON_ARGS $case_args" | tee -a "$log_file"
  python main.py $COMMON_ARGS $case_args 2>&1 | tee -a "$log_file"
  echo "===== $(date '+%F %T') END $case_name $tag =====" | tee -a "$log_file"
}

if [[ "$CASE" == "all" ]]; then
  for case_name in C0 C1 C2 C3 C4; do
    run_case "$case_name"
  done
else
  run_case "$CASE"
fi
