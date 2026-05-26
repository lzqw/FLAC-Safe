#!/usr/bin/env bash
set -euo pipefail

cd /root/FLAC-Safe

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate flac

git fetch origin
git checkout main
git pull origin main

mkdir -p logs/convergence reports/convergence logs/tuning_round1 reports/tuning_round1

python -m compileall .

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("memory GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

nvidia-smi
