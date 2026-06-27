# New Server State After Clone

- New SSH endpoint: `ssh -p 13335 root@connect.cqa1.seetacloud.com`
- SSH alias: `star-v2-new`
- Hostname: `autodl-container-5f5c489a58-eca64b30`
- Repo path: `/root/FLAC-Safe-star-v2`
- Branch: `codex/star-v2-aaai-final`
- Commit: `57543f8eb582dda6e4a1c5e101d0dfbb4faf2f43`
- Environment: `/root/miniconda3/envs/flac` (CUDA visible; pandas pinned with NumPy 1.23.5)
- GPU state: `NVIDIA vGPU-48GB, 24564 MiB
NVIDIA vGPU-48GB, 24564 MiB`
- Tmux state: no STAR-v2 tmux sessions at clone audit time
- Results symlink: `/root/autodl-tmp/star_v2_storage/results`
- Logs symlink: `/root/autodl-tmp/star_v2_storage/logs`
- Core-100k plan rows: 48
- Core-100k counts: {'PENDING': 48}
- Git status at summary time:

```text
M reports/star_v2_final/claim_gate.md
 M reports/star_v2_final/core_100k/gate.md
 M reports/star_v2_final/main_results_by_seed.csv
 M reports/star_v2_final/main_results_summary.csv
 M reports/star_v2_final/missing_results.md
 M reports/star_v2_final/recovery/post_final_continuation_status.md
 M reports/star_v2_final/run_manifest.csv
 M reports/star_v2_final/scheduler_status.csv
 M reports/star_v2_final/training_safety_by_seed.csv
 M scripts/star/goal_star_v2_final.py
?? logs
?? results
```

## Next Action

Launch `core-100k --resume --storage-policy unblocked --ignore-storage-gate` in tmux. The core-100k gate is a reporting checkpoint only and must not stop final-300k scheduling unless a real fatal technical failure occurs.
