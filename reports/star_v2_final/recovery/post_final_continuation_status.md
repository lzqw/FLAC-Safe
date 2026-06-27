# STAR-v2 Post-Final Continuation Status

- Updated: 2026-06-28T00:56:37+08:00
- Stage: final_300k_training_active
- SSH endpoint: `ssh -p 13335 root@connect.cqa1.seetacloud.com`
- SSH alias: `star-v2-new`
- Host: autodl-container-5f5c489a58-eca64b30
- Branch: codex/star-v2-aaai-final
- HEAD: 57543f8eb582dda6e4a1c5e101d0dfbb4faf2f43
- Core finals: 48 / 48
- Core eval: 48 / 48
- Final-300k finals: 0 / 80
- Final-300k active: 6
- Ablation finals: 0 / 24
- Supervisor log: `reports/star_v2_final/recovery/starv2_full_pipeline_supervisor.log`

## Core Technical Gate

```text
# STAR-v2 Core-100k Gate

- label: `PASS`
- expected_runs: `48`
- completed_training: `48`
- completed_eval: `48`
- running: `0`
- failed_or_error_logs: `0`

This gate is a reporting checkpoint only under the storage override.
The pipeline should continue unless there is a fatal technical failure.
```

## Core Claim Gate

```text
# STAR-v2 Core 100k Gate

Decision: FAIL

STAR-v2 lower cost paired wins vs current-only: 6/12
STAR-v2 lower task-mean cost tasks: 3/4
Reward retention ok: False
Catastrophic tasks: none
Offline raw evaluation complete: True
```

## Disk

```text
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   27G  3.2G  90% /
/dev/md127       50G   15G   36G  30% /root/autodl-tmp
```
