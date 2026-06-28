#!/usr/bin/env bash
set -euo pipefail

cd /root/FLAC-Safe-star-v2

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flac

export STAR_STORAGE_ROOT=/root/autodl-tmp/star_v2_storage
export TMPDIR=/root/autodl-tmp/star_v2_storage/tmp
export TORCH_HOME=/root/autodl-tmp/star_v2_storage/cache/torch
export MPLCONFIGDIR=/root/autodl-tmp/star_v2_storage/cache/matplotlib
export XDG_CACHE_HOME=/root/autodl-tmp/star_v2_storage/cache/xdg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_DISABLED=true
export MUJOCO_GL=egl

mkdir -p "$STAR_STORAGE_ROOT"/{results,logs,tmp,cache} reports/star_v2_final/recovery

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

count_checkpoints() {
  local spec_func="$1"
  python - "$spec_func" <<'PY'
import importlib.util
import sys
from pathlib import Path

spec_func = sys.argv[1]
path = Path("scripts/star/goal_star_v2_final.py").resolve()
module_spec = importlib.util.spec_from_file_location("goal_star_v2_final", path)
mod = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = mod
assert module_spec.loader is not None
module_spec.loader.exec_module(mod)
specs = getattr(mod, spec_func)()
done = sum(1 for spec in specs if spec.final_checkpoint.exists())
print(f"{done}/{len(specs)}")
PY
}

count_eval_csv() {
  local root="$1"
  find "$root" -type f -name corrected_eval_episodes.csv 2>/dev/null | wc -l
}

wait_for_phase() {
  local label="$1"
  local command="$2"
  local spec_func="$3"
  local sleep_seconds="${4:-600}"

  while true; do
    log "$label checkpoint progress $(count_checkpoints "$spec_func")"
    df -h /root /root/autodl-tmp || true
    nvidia-smi || true
    python scripts/star/goal_star_v2_final.py status --storage-policy unblocked --ignore-storage-gate || true
    python scripts/star/goal_star_v2_final.py "$command" --resume --max-parallel 6 --storage-policy unblocked --ignore-storage-gate

    local progress
    progress="$(count_checkpoints "$spec_func")"
    log "$label after launch $progress"
    if [[ "${progress%/*}" == "${progress#*/}" ]]; then
      log "$label training complete"
      break
    fi
    sleep "$sleep_seconds"
  done
}

log "STAR-v2 full pipeline supervisor started"
git branch --show-current
git rev-parse HEAD

wait_for_phase "core-100k" "core-100k" "core_100k_specs" 600

log "core-100k eval/collect/gate"
if [[ "$(count_eval_csv results/star_v2_final/core_100k)" -ge 48 && -f reports/star_v2_final/core_100k/gate.md ]]; then
  log "core-100k eval/gate already complete; skipping"
else
  python scripts/star/goal_star_v2_final.py eval-core-100k --storage-policy unblocked --ignore-storage-gate
  python scripts/star/goal_star_v2_final.py collect --phase core_100k --strict --storage-policy unblocked --ignore-storage-gate || true
  python scripts/star/collect_star_v2_results.py --strict || true
  python scripts/star/goal_star_v2_final.py gate-core-100k --storage-policy unblocked --ignore-storage-gate || true
fi

wait_for_phase "final-300k" "final-300k" "resume_300k_specs" 600

log "final-300k eval"
if [[ "$(count_eval_csv results/star_v2_final/resume_300k)" -ge 80 ]]; then
  log "final-300k eval already complete; skipping"
else
  python scripts/star/goal_star_v2_final.py eval-final-300k --storage-policy unblocked --ignore-storage-gate
fi

log "final-300k collect"
python scripts/star/goal_star_v2_final.py collect --phase resume_300k --strict --storage-policy unblocked --ignore-storage-gate || true
python scripts/star/collect_star_v2_results.py --strict --phase final300k || true

log "mechanism diagnostics"
python scripts/star/goal_star_v2_final.py mechanism --storage-policy unblocked --ignore-storage-gate || true

log "oracle diagnostics"
python scripts/star/inspect_safety_gym_oracle.py || true
python scripts/star/goal_star_v2_final.py oracle --storage-policy unblocked --ignore-storage-gate || true

wait_for_phase "ablation" "ablation" "ablation_specs" 600

log "ablation collect"
python scripts/star/goal_star_v2_final.py collect --phase ablation_100k --strict --storage-policy unblocked --ignore-storage-gate || true

log "executor"
python scripts/star/goal_star_v2_final.py executor --storage-policy unblocked --ignore-storage-gate || true

log "paper artifacts"
python scripts/star/goal_star_v2_final.py paper --storage-policy unblocked --ignore-storage-gate || true
python scripts/star/build_star_v2_paper_artifacts.py --strict --storage-policy unblocked --ignore-storage-gate || true

log "STAR-v2 full pipeline supervisor finished"
