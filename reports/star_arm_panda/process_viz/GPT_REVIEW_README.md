# STAR-v2 Panda Gripper Process Audit Bundle

This compact bundle is for independent GPT/code review of the Panda gripper-process visualization.

## What this is

The figure and CSV files describe one selected qualitative mechanism episode for `SafetyPandaReachObstacle-v0`.

- Evaluation seed: `900078`
- Episode: `0`
- Selected audit step: `16`
- Main visual: `reports/star_arm_panda/figures/fig_panda_gripper_process.png`
- Data are exported from real PyBullet rollout states and real STAR critic/audit queries.
- Do not treat this qualitative episode as a standalone Panda benchmark claim.

## Key facts to verify

- `actor_mean_Q = 0.025621959939599`
- `d_aud = 0.05`
- `max_corridor_Q = 0.762424111366272`
- `corridor_lift = 0.21535199880599976`
- `rho_cur = 0.4085810780525207`
- `rho_cor = 0.6239330768585205`
- `risky_shadows_executed = False`
- `STAR+Exec success = 1.0`
- `STAR+Exec cost = 0.0`
- `STAR+Exec min_clearance in audit summary = 0.0396441173553466`
- `STAR+Exec min_clearance in exported per-step gripper trajectory = 0.0363994099105205`

The two min-clearance values come from related but not identical exported sources. Use the CSV columns when redrawing this exact figure.

## Main files

- `reports/star_arm_panda/figures/fig_panda_gripper_process.{png,pdf,svg}`: final multi-panel figure.
- `reports/star_arm_panda/process_viz/gripper_trajectory_rows.csv`: per-step PyBullet gripper, finger, EE, obstacle, goal, action, cost, success, and clearance rows.
- `reports/star_arm_panda/process_viz/audit_step_selection.csv`: selected process/audit steps and summary values.
- `reports/star_arm_panda/process_viz/audit_sequence_rows.csv`: per-candidate audit fan rows with endpoints and predicted `Q_C^+`.
- `reports/star_arm_panda/process_viz/audit_sequence_summary.csv`: compact per-selected-step audit summary.
- `reports/star_arm_panda/process_viz/frames/`: six top-down process frames.
- `reports/star_arm_panda/process_viz/panda_gripper_process_caption.tex`: paper caption draft.
- `reports/star_arm_panda/process_viz/panda_gripper_process_paragraph.tex`: short paper paragraph draft.
- `reports/star_arm_panda/qualitative/audit_summary.json`: original selected audit summary.
- `reports/star_arm_panda/qualitative/selection.md`: selected-case description.
- `reports/star_arm_panda/qualitative/DATA_DICTIONARY.md`: qualitative artifact data dictionary.
- `reports/star_arm_panda/qualitative/REPRODUCE_PANDA_FIGURE.md`: reproduction notes.

## Important caveats

- The checkpoint configuration exposes 16 current-only and 16 corridor shadow samples at replayed audit states, not 32 each.
- Step 16 uses the logged selected audit snapshot to preserve the known strong-audit values exactly.
- The other selected process steps use STAR+Exec replay critic queries.
- No final-summary rows were used to fabricate trajectories or curves.
- Raw checkpoints, event logs, and large result directories are intentionally excluded.
