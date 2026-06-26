# New server state unblocked

Captured: 2026-06-27T06:32:39+08:00

## Identity
- hostname: autodl-container-a05746851a-389c8750
- repo: /root/FLAC-Safe-star-v2
- branch: codex/star-v2-aaai-final
- head: 35aeef0611ebc5a465a1f53683e6856b03c2df61

## Storage
- override: reports/star_v2_final/recovery/storage_override_authorization.md
- STAR_STORAGE_ROOT: /root/autodl-tmp/star_v2_storage
- results: /root/autodl-tmp/star_v2_storage/results
- logs: /root/autodl-tmp/star_v2_storage/logs

## Disk
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   27G  3.8G  88% /
/dev/md0         50G   14G   37G  28% /root/autodl-tmp

## Tmux
starv2_post_final_continuation: 1 windows (created Sat Jun 27 05:25:52 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_sac_lag_sac_lag_s10: 1 windows (created Sat Jun 27 04:22:17 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_sac_lag_sac_lag_s11: 1 windows (created Sat Jun 27 04:22:17 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_sac_lag_sac_lag_s12: 1 windows (created Sat Jun 27 04:22:17 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s10: 1 windows (created Sat Jun 27 04:44:25 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s11: 1 windows (created Sat Jun 27 04:57:06 2026)
starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s12: 1 windows (created Sat Jun 27 05:07:06 2026)
starv2_safe_final300k_scheduler: 1 windows (created Sat Jun 27 05:16:21 2026)

## Status
tmux_sessions=8
active=starv2_post_final_continuation
active=starv2_resume_300k_SafetyPointGoal1_v0_selected_sac_lag_sac_lag_s10
active=starv2_resume_300k_SafetyPointGoal1_v0_selected_sac_lag_sac_lag_s11
active=starv2_resume_300k_SafetyPointGoal1_v0_selected_sac_lag_sac_lag_s12
active=starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s10
active=starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s11
active=starv2_resume_300k_SafetyPointGoal1_v0_selected_star_v2_star_v2_s12
active=starv2_safe_final300k_scheduler
status_csv=/root/FLAC-Safe-star-v2/reports/star_v2_final/scheduler_status.csv

## Core gate
# STAR-v2 Core 100k Gate

Decision: FAIL

STAR-v2 lower cost paired wins vs current-only: 3/12
STAR-v2 lower task-mean cost tasks: 2/4
Reward retention ok: False
Catastrophic tasks: none
Offline raw evaluation complete: True

## Strict collect
strict collection failed: decision=FAIL missing=0 duplicates=0
