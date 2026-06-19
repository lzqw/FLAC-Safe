# Fast Geometry Smoke Commands

These commands are prepared only; they were not launched by this implementation step.

## PG1 one-sided fast-geometry smoke

```bash
cd /root/FLAC-Safe
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate flac
mkdir -p logs/fast_geometry_smoke
WANDB_MODE=offline python main.py \
  --task SafetyPointGoal1-v0 \
  --tag PG1_onesided_fastgeo_smoke_s0 \
  --safe_env True \
  --safety_critic_mode cdf \
  --qc_geom_mode mean \
  --cdf_binarize_cost True \
  --cdf_target_clip True \
  --high_fidelity_safety_q True \
  --safety_q_priority True \
  --safety_q_cost_weight 2.0 \
  --safety_q_boundary_weight 3.0 \
  --safety_q_td_weight 0.0 \
  --safety_q_max_weight 5.0 \
  --safety_q_extra_updates 0 \
  --diagnose_safety_q_geometry False \
  --lambda_safe 0.70 \
  --safe_threshold 0.05 \
  --safe_bandwidth 0.05 \
  --lambda_jvp 0.003 \
  --lambda_jvp_schedule True \
  --lambda_jvp_start 0.001 \
  --lambda_jvp_end 0.003 \
  --lambda_jvp_warmup_steps 30000 \
  --lambda_jvp_ramp_steps 70000 \
  --jvp_warmup_steps 20000 \
  --jvp_batch_size 1024 \
  --jvp_sample_mode topk_gate \
  --jvp_update_interval 2 \
  --jvp_one_sided True \
  --jvp_gate_mode boundary_or_unsafe \
  --normalize_jvp True \
  --jvp_norm_mode exact \
  --jvp_mode grad \
  --batch_size 4096 \
  --updates_per_step 2 \
  --hidden_size 512 \
  --num_steps 50000 \
  --start_steps 5000 \
  --eval True \
  --eval_interval_steps 25000 \
  --eval_numsteps 5000 \
  --eval_times 1 \
  --save True \
  --save_interval_steps 50000 \
  --seed 0 2>&1 | tee logs/fast_geometry_smoke/PG1_onesided_fastgeo_smoke_s0.log
```

## CG1 one-sided fast-geometry smoke

```bash
cd /root/FLAC-Safe
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate flac
mkdir -p logs/fast_geometry_smoke
WANDB_MODE=offline python main.py \
  --task SafetyCarGoal1-v0 \
  --tag CG1_onesided_fastgeo_smoke_s0 \
  --safe_env True \
  --safety_critic_mode cdf \
  --qc_geom_mode mean \
  --cdf_binarize_cost True \
  --cdf_target_clip True \
  --high_fidelity_safety_q True \
  --safety_q_priority True \
  --safety_q_cost_weight 2.0 \
  --safety_q_boundary_weight 3.0 \
  --safety_q_td_weight 0.0 \
  --safety_q_max_weight 5.0 \
  --safety_q_extra_updates 0 \
  --diagnose_safety_q_geometry False \
  --lambda_safe 0.60 \
  --safe_threshold 0.05 \
  --safe_bandwidth 0.05 \
  --lambda_jvp 0.003 \
  --lambda_jvp_schedule True \
  --lambda_jvp_start 0.001 \
  --lambda_jvp_end 0.003 \
  --lambda_jvp_warmup_steps 30000 \
  --lambda_jvp_ramp_steps 70000 \
  --jvp_warmup_steps 20000 \
  --jvp_batch_size 1024 \
  --jvp_sample_mode topk_gate \
  --jvp_update_interval 2 \
  --jvp_one_sided True \
  --jvp_gate_mode boundary_or_unsafe \
  --normalize_jvp True \
  --jvp_norm_mode exact \
  --jvp_mode grad \
  --batch_size 4096 \
  --updates_per_step 2 \
  --hidden_size 512 \
  --num_steps 50000 \
  --start_steps 5000 \
  --eval True \
  --eval_interval_steps 25000 \
  --eval_numsteps 5000 \
  --eval_times 1 \
  --save True \
  --save_interval_steps 50000 \
  --seed 0 2>&1 | tee logs/fast_geometry_smoke/CG1_onesided_fastgeo_smoke_s0.log
```
