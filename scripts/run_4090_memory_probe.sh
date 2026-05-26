#!/usr/bin/env bash
set -euo pipefail

LEVEL="${1:-L0}"
LOG_DIR="logs/4090_probe"
mkdir -p "$LOG_DIR"

COMMON_ARGS="\
  --task SafetyPointGoal1-v0 \
  --safe_env True \
  --safe_policy_loss True \
  --distributional_critic False \
  --num_steps 10000 \
  --start_steps 2000 \
  --eval False \
  --save False \
  --steps 1 \
  --epsilon 0.0 \
  --lambda_safe 0.5 \
  --lambda_jvp 0.001 \
  --safe_threshold 0.1 \
  --safe_bandwidth 0.05 \
  --jvp_warmup_steps 5000 \
  --jvp_mode grad \
  --normalize_jvp False \
  --soft_normal_masking False \
  --compile_model False \
  --seed 0 \
  --algo MF_SCTD_4090Probe"

run_level() {
  local level="$1"
  local level_args tag

  case "$level" in
    L0)
      tag="probe_L0_b256_u1_h512"
      level_args="--batch_size 256 --updates_per_step 1 --hidden_size 512 --tag $tag"
      ;;
    L1)
      tag="probe_L1_b512_u1_h512"
      level_args="--batch_size 512 --updates_per_step 1 --hidden_size 512 --tag $tag"
      ;;
    L2)
      tag="probe_L2_b1024_u1_h512"
      level_args="--batch_size 1024 --updates_per_step 1 --hidden_size 512 --tag $tag"
      ;;
    L3)
      tag="probe_L3_b1024_u2_h512"
      level_args="--batch_size 1024 --updates_per_step 2 --hidden_size 512 --tag $tag"
      ;;
    L4)
      tag="probe_L4_b1024_u2_h1024"
      level_args="--batch_size 1024 --updates_per_step 2 --hidden_size 1024 --tag $tag"
      ;;
    *)
      echo "Unknown level: $level" >&2
      echo "Usage: bash scripts/run_4090_memory_probe.sh L0|L1|L2|L3|L4|all" >&2
      return 2
      ;;
  esac

  local log_file="$LOG_DIR/${tag}.log"
  {
    echo "===== $(date '+%F %T') START $level $tag ====="
    echo "Command: python main.py $COMMON_ARGS $level_args"
    echo
    echo "===== nvidia-smi before ====="
    nvidia-smi || true
    echo
  } | tee "$log_file"

  python main.py $COMMON_ARGS $level_args 2>&1 | tee -a "$log_file"

  {
    echo
    echo "===== nvidia-smi after ====="
    nvidia-smi || true
    echo "===== $(date '+%F %T') END $level $tag ====="
  } | tee -a "$log_file"
}

if [[ "$LEVEL" == "all" ]]; then
  for level in L0 L1 L2 L3 L4; do
    run_level "$level"
  done
else
  run_level "$LEVEL"
fi
