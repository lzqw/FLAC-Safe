# New Server State After Server Stop

- Server endpoint: `ssh -p 13335 root@connect.cqa1.seetacloud.com`
- SSH alias: `star-v2-new`
- Hostname: `autodl-container-5f5c489a58-eca64b30`
- Repo path: `/root/FLAC-Safe-star-v2`
- Branch: `codex/star-v2-aaai-final`
- Commit: `41dc8d5e6b5c863bf21a01f315198cea6472dd4e`
- Environment: `/root/miniconda3/envs/flac` (`numpy==1.23.5`, `pandas==1.5.3`, `matplotlib==3.7.5`)
- GPU state: `0, NVIDIA vGPU-48GB, 2147 MiB, 24564 MiB, 92 %
1, NVIDIA vGPU-48GB, 2325 MiB, 24564 MiB, 94 %`
- Tmux state: `starv2_after_stop_scheduler: 1 windows (created Sun Jun 28 09:03:13 2026)
starv2_resume_300k_SafetyCarGoal1_v0_selected_pointwise_v2_pointwise_v2_s10: 1 windows (created Sun Jun 28 09:03:15 2026)
starv2_resume_300k_SafetyCarGoal1_v0_selected_pointwise_v2_pointwise_v2_s11: 1 windows (created Sun Jun 28 09:03:15 2026)
starv2_resume_300k_SafetyCarGoal1_v0_selected_pointwise_v2_pointwise_v2_s12: 1 windows (created Sun Jun 28 09:03:15 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s10: 1 windows (created Sun Jun 28 09:03:15 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s11: 1 windows (created Sun Jun 28 09:03:15 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s12: 1 windows (created Sun Jun 28 09:03:15 2026)`
- Results symlink: `/root/autodl-tmp/star_v2_storage/results`
- Logs symlink: `/root/autodl-tmp/star_v2_storage/logs`
- Core-100k checkpoints: `48` / 48
- Core-100k eval CSVs: `48` / 48
- Final-300k checkpoints: `9` / 80
- Final-300k eval CSVs: `0` / 80
- Ablation checkpoints: `0` / 24
- Error-like lines after filtering: `493`

## Phase Counts From Plan

```text
{'core_100k': {'COMPLETED_TRAINING_AND_EVAL': 48}, 'core_gate': {'COMPLETED_TRAINING': 1}, 'resume_300k': {'COMPLETED_TRAINING_NEEDS_EVAL': 9, 'RUNNING': 6, 'PENDING': 65}, 'final_collect': {'COMPLETED_TRAINING': 1}, 'mechanism': {'COMPLETED_TRAINING': 3}, 'oracle': {'PENDING': 1}, 'ablation_100k': {'PENDING': 24}, 'paper_artifact': {'PENDING': 20}}
```

## Earliest Incomplete Phase

`final-300k` is incomplete (`9/80` checkpoints). Relaunch final-300k with resume flags; completed runs must be skipped by the orchestrator.

## Disk

```text
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G  6.5G   24G  22% /
/dev/md127       50G   33G   18G  66% /root/autodl-tmp
```
