#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
shift || true

GPU_ID="${GPU_ID:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOG_ROOT="logs/goal_10m"
REPORT_ROOT="reports/goal_10m"
ERR_RE="No space left|Traceback|RuntimeError|NaN|nan|OOM|out of memory|CUDA error"

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

configs() {
  printf '%s\n' \
    JSC_CG1_C2_safe045_bw025_10M \
    JSC_CG2_strict_safe_schedule_bw025_10M \
    JSC_PG1_safe070_bw050_10M \
    JSC_PG2_strict_safe_schedule_bw050_10M
}

config_env() {
  case "$1" in
    JSC_CG1_C2_safe045_bw025_10M) echo "SafetyCarGoal1-v0" ;;
    JSC_CG2_strict_safe_schedule_bw025_10M) echo "SafetyCarGoal2-v0" ;;
    JSC_PG1_safe070_bw050_10M) echo "SafetyPointGoal1-v0" ;;
    JSC_PG2_strict_safe_schedule_bw050_10M) echo "SafetyPointGoal2-v0" ;;
    *) return 1 ;;
  esac
}

common_args() {
  local task="$1"
  cat <<EOF
--task ${task}
--safe_env True
--safe_policy_loss True
--safety_critic_mode cdf
--qc_geom_mode mean
--cdf_binarize_cost True
--cdf_target_clip True
--high_fidelity_safety_q True
--safety_q_priority True
--safety_q_cost_weight 2.0
--safety_q_boundary_weight 3.0
--safety_q_td_weight 0.0
--safety_q_max_weight 5.0
--safety_q_extra_updates 0
--diagnose_safety_q_geometry True
--safety_q_fd_eps 0.01
--safety_q_boundary_width 0.05
--lambda_jvp 0.003
--lambda_jvp_schedule True
--lambda_jvp_start 0.001
--lambda_jvp_end 0.003
--lambda_jvp_warmup_steps 30000
--lambda_jvp_ramp_steps 70000
--jvp_warmup_steps 20000
--normalize_jvp True
--jvp_norm_mode exact
--jvp_mode grad
--soft_feasibility_gate False
--directional_ref_noise False
--batch_size 4096
--updates_per_step 2
--hidden_size 512
--num_steps 10000000
--start_steps 5000
--eval True
--eval_interval_steps 100000
--eval_numsteps 5000
--eval_times 1
--diagnose_interval_steps 100000
--distributional_critic False
--compile_model False
--save False
--steps 1
EOF
}

config_args() {
  local config="$1"
  local task
  task="$(config_env "$config")"
  common_args "$task"
  case "$config" in
    JSC_CG1_C2_safe045_bw025_10M)
      cat <<EOF
--lambda_safe 0.45
--lambda_safe_schedule False
--safe_bandwidth 0.025
EOF
      ;;
    JSC_CG2_strict_safe_schedule_bw025_10M)
      cat <<EOF
--lambda_safe 0.70
--lambda_safe_schedule True
--lambda_safe_start 0.30
--lambda_safe_end 0.70
--lambda_safe_warmup_steps 30000
--lambda_safe_ramp_steps 70000
--safe_bandwidth 0.025
EOF
      ;;
    JSC_PG1_safe070_bw050_10M)
      cat <<EOF
--lambda_safe 0.70
--lambda_safe_schedule False
--safe_bandwidth 0.05
EOF
      ;;
    JSC_PG2_strict_safe_schedule_bw050_10M)
      cat <<EOF
--lambda_safe 0.85
--lambda_safe_schedule True
--lambda_safe_start 0.40
--lambda_safe_end 0.85
--lambda_safe_warmup_steps 30000
--lambda_safe_ramp_steps 70000
--safe_bandwidth 0.05
EOF
      ;;
    *)
      echo "Unknown config: $config" >&2
      return 1
      ;;
  esac
}

session_name() {
  local config="$1"
  local seed="$2"
  echo "goal10m_${config}_s${seed}"
}

log_path() {
  local config="$1"
  local seed="$2"
  echo "${LOG_ROOT}/${config}/seed${seed}.log"
}

spec_config() {
  echo "${1%%:*}"
}

spec_seed() {
  echo "${1##*:}"
}

all_specs() {
  local seed config
  for seed in 0 1 2; do
    while read -r config; do
      printf '%s:%s\n' "$config" "$seed"
    done < <(configs)
  done
}

parse_seeds() {
  local seeds="0,1,2"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seeds)
        seeds="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        return 2
        ;;
    esac
  done
  echo "$seeds"
}

config_specs() {
  local config="$1"
  shift || true
  local seeds seed
  seeds="$(parse_seeds "$@")"
  IFS=',' read -ra seed_list <<< "$seeds"
  for seed in "${seed_list[@]}"; do
    [[ -n "$seed" ]] || continue
    printf '%s:%s\n' "$config" "$seed"
  done
}

env_specs() {
  local env="$1"
  local config seed
  for seed in 0 1 2; do
    while read -r config; do
      if [[ "$(config_env "$config")" == "$env" ]]; then
        printf '%s:%s\n' "$config" "$seed"
      fi
    done < <(configs)
  done
}

running_count() {
  local sessions
  sessions="$(tmux ls 2>/dev/null || true)"
  awk -F: '/^goal10m_/ {n++} END {print n+0}' <<< "$sessions"
}

is_running() {
  tmux has-session -t "$(session_name "$1" "$2")" 2>/dev/null
}

has_error() {
  local log="$1"
  [[ -f "$log" ]] && grep -E "$ERR_RE" "$log" >/dev/null 2>&1
}

is_completed() {
  local log="$1"
  [[ -f "$log" ]] && grep -q "===== .* END " "$log" && ! has_error "$log"
}

start_spec() {
  local spec="$1"
  local config seed task log dir sess args
  config="$(spec_config "$spec")"
  seed="$(spec_seed "$spec")"
  task="$(config_env "$config")"
  log="$(log_path "$config" "$seed")"
  dir="$(dirname "$log")"
  sess="$(session_name "$config" "$seed")"
  if is_running "$config" "$seed"; then
    echo "already running $sess"
    return 1
  fi
  if [[ -f "$log" ]]; then
    if is_completed "$log"; then
      echo "skip completed $config seed=$seed"
    elif has_error "$log"; then
      echo "found error log for $config seed=$seed; not rerunning automatically"
    else
      echo "existing non-completed log for $config seed=$seed; not overwriting"
    fi
    return 1
  fi
  mkdir -p "$dir"
  args="$(config_args "$config" | tr '\n' ' ')"
  tmux new -d -s "$sess" "cd /root/FLAC-Safe && \
{ source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh; } && \
conda activate flac && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=\${WANDB_MODE:-offline} && \
{ \
echo '===== '\"\$(date '+%Y-%m-%d %H:%M:%S')\"' START ${config} seed=${seed} task=${task} ====='; \
echo 'CONFIG=${config} TASK=${task} SEED=${seed} NUM_STEPS=10000000 EVAL_INTERVAL=100000 EVAL_TIMES=1 DIAGNOSE_INTERVAL=100000'; \
python main.py ${args} --seed ${seed} --algo MF_Goal10M --tag ${config}_seed${seed}; \
code=\$?; \
echo '===== '\"\$(date '+%Y-%m-%d %H:%M:%S')\"' END ${config} seed=${seed} exit_code='\"\$code\"' ====='; \
exit \$code; \
} 2>&1 | tee '${log}'"
  echo "started $sess"
  return 0
}

launch_specs() {
  local specs=("$@")
  local active slots spec
  active="$(running_count)"
  slots=$((MAX_PARALLEL - active))
  if (( slots <= 0 )); then
    echo "no launch slots available: active=$active MAX_PARALLEL=$MAX_PARALLEL"
    return 0
  fi
  echo "active=$active MAX_PARALLEL=$MAX_PARALLEL available_slots=$slots"
  for spec in "${specs[@]}"; do
    (( slots <= 0 )) && break
    if start_spec "$spec"; then
      slots=$((slots - 1))
    fi
  done
}

print_plan() {
  local spec config seed task sess
  echo "| Config | Env | Seed | Steps | eval_interval | eval_times | diagnose_interval | Tmux |"
  echo "| ------ | --- | ---: | ----: | ------------: | ---------: | ----------------: | ---- |"
  while read -r spec; do
    config="$(spec_config "$spec")"
    seed="$(spec_seed "$spec")"
    task="$(config_env "$config")"
    sess="$(session_name "$config" "$seed")"
    echo "| $config | $task | $seed | 10000000 | 100000 | 1 | 100000 | $sess |"
  done < <(all_specs)
}

status() {
  echo "===== tmux ====="
  tmux ls 2>/dev/null || true
  echo "===== gpu ====="
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv || true
  echo "===== progress tail ====="
  grep -RInE "START |END |Env: Safety|TRAIN_COST|SAFETY_Q step=|Episode:" "$LOG_ROOT" 2>/dev/null | tail -n 160 || true
  echo "===== errors ====="
  grep -RInE "$ERR_RE" "$LOG_ROOT" 2>/dev/null || true
}

stop_config() {
  local config="${1:-}"
  if [[ -z "$config" ]]; then
    echo "Usage: $0 stop_config CONFIG_NAME" >&2
    return 2
  fi
  local sess
  while read -r sess; do
    [[ -n "$sess" ]] || continue
    tmux kill-session -t "$sess"
    echo "stopped $sess"
  done < <((tmux ls 2>/dev/null || true) | awk -F: -v prefix="goal10m_${config}_s" '$1 ~ "^"prefix {print $1}')
}

case "$MODE" in
  plan)
    print_plan
    ;;
  status)
    status
    ;;
  launch_ours)
    mapfile -t specs < <(all_specs)
    launch_specs "${specs[@]}"
    ;;
  launch_env)
    env_name="${1:-}"
    if [[ -z "$env_name" ]]; then
      echo "Usage: $0 launch_env SafetyCarGoal1-v0" >&2
      exit 2
    fi
    mapfile -t specs < <(env_specs "$env_name")
    launch_specs "${specs[@]}"
    ;;
  launch_config)
    config="${1:-}"
    shift || true
    if [[ -z "$config" ]]; then
      echo "Usage: $0 launch_config CONFIG_NAME --seeds 0,1,2" >&2
      exit 2
    fi
    mapfile -t specs < <(config_specs "$config" "$@")
    launch_specs "${specs[@]}"
    ;;
  stop_config)
    stop_config "${1:-}"
    ;;
  collect)
    python scripts/collect_goal_10m.py
    ;;
  *)
    echo "Usage: $0 plan|status|launch_ours|launch_env ENV|launch_config CONFIG [--seeds 0,1,2]|stop_config CONFIG|collect" >&2
    exit 2
    ;;
esac
