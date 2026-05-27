#!/usr/bin/env bash
set -euo pipefail

LEVEL="${1:-G0}"
LOG_DIR="logs/4090_round2_probe"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

COMMON_ARGS="\
  --task SafetyPointGoal1-v0 \
  --safe_env True \
  --safe_policy_loss True \
  --distributional_critic False \
  --num_steps 15000 \
  --start_steps 3000 \
  --eval False \
  --save False \
  --steps 1 \
  --epsilon 0.0 \
  --lambda_safe 0.5 \
  --lambda_jvp 0.002 \
  --safe_threshold 0.1 \
  --safe_bandwidth 0.05 \
  --jvp_warmup_steps 8000 \
  --jvp_mode grad \
  --normalize_jvp False \
  --soft_normal_masking False \
  --compile_model False \
  --seed 0 \
  --algo MF_SCTD_4090R2Probe"

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {print $1}'
}

gpu_query() {
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv
}

run_level() {
  local level="$1"
  local tag level_args

  case "$level" in
    G0)
      tag="r2probe_G0_b1024_u1_h512"
      level_args="--batch_size 1024 --updates_per_step 1 --hidden_size 512 --tag $tag"
      ;;
    G1)
      tag="r2probe_G1_b2048_u1_h512"
      level_args="--batch_size 2048 --updates_per_step 1 --hidden_size 512 --tag $tag"
      ;;
    G2)
      tag="r2probe_G2_b2048_u2_h512"
      level_args="--batch_size 2048 --updates_per_step 2 --hidden_size 512 --tag $tag"
      ;;
    G3)
      tag="r2probe_G3_b4096_u1_h512"
      level_args="--batch_size 4096 --updates_per_step 1 --hidden_size 512 --tag $tag"
      ;;
    G4)
      tag="r2probe_G4_b4096_u2_h512"
      level_args="--batch_size 4096 --updates_per_step 2 --hidden_size 512 --tag $tag"
      ;;
    G5)
      tag="r2probe_G5_b2048_u2_h1024"
      level_args="--batch_size 2048 --updates_per_step 2 --hidden_size 1024 --tag $tag"
      ;;
    *)
      echo "Unknown level: $level" >&2
      echo "Usage: bash scripts/run_4090_round2_probe.sh G0|G1|G2|G3|G4|G5|all" >&2
      return 2
      ;;
  esac

  local log_file="$LOG_DIR/${tag}.log"
  local before after peak peak_file monitor_pid status
  before="$(gpu_used_mb || echo 0)"
  peak_file="$(mktemp)"
  echo "$before" > "$peak_file"
  {
    echo "===== $(date '+%F %T') START $level $tag ====="
    echo "Command: python main.py $COMMON_ARGS $level_args"
    echo "[GPU_MEM_BEFORE] used_mb=$before"
    echo "===== nvidia-smi before ====="
    gpu_query || true
    echo
  } | tee "$log_file"

  (
    while true; do
      local_used="$(gpu_used_mb || echo 0)"
      current_peak="$(cat "$peak_file" 2>/dev/null || echo 0)"
      if [[ "$local_used" =~ ^[0-9]+$ && "$current_peak" =~ ^[0-9]+$ && "$local_used" -gt "$current_peak" ]]; then
        echo "$local_used" > "$peak_file"
      fi
      sleep 5
    done
  ) &
  monitor_pid="$!"

  set +e
  python main.py $COMMON_ARGS $level_args 2>&1 | tee -a "$log_file"
  status="${PIPESTATUS[0]}"
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  after="$(gpu_used_mb || echo 0)"
  peak="$(cat "$peak_file" 2>/dev/null || echo "$after")"
  rm -f "$peak_file"
  {
    echo
    echo "[GPU_MEM_AFTER] used_mb=$after"
    echo "[GPU_MEM_PEAK_EST] used_mb=$peak"
    echo "===== nvidia-smi after ====="
    gpu_query || true
    echo "===== $(date '+%F %T') END $level $tag ====="
  } | tee -a "$log_file"
  return "$status"
}

if [[ "$LEVEL" == "all" ]]; then
  for level in G0 G1 G2 G3 G4; do
    run_level "$level"
    grep -E "Traceback|RuntimeError|NaN|nan|OOM|out of memory" "$LOG_DIR"/*.log || true
  done
else
  run_level "$LEVEL"
  grep -E "Traceback|RuntimeError|NaN|nan|OOM|out of memory" "$LOG_DIR"/*.log || true
fi
