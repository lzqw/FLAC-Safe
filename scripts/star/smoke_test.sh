#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export WANDB_MODE="${WANDB_MODE:-offline}"

python main_star.py \
  --task SafetyPointGoal1-v0 \
  --safe_env True \
  --method star \
  --num_steps 2000 \
  --start_steps 200 \
  --batch_size 64 \
  --hidden_size 128 \
  --shadow_k 8 \
  --star_exec_candidates 8 \
  --eval False \
  --save False \
  --cuda False
