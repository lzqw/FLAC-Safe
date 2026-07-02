# Panda Mechanism Final Audit

1. Is the figure generated from real logs? Yes: `/root/FLAC-Safe-star-v2/reports/star_arm_panda/qualitative/trajectory_rows.csv` and `/root/FLAC-Safe-star-v2/reports/star_arm_panda/qualitative/audit_snapshot.csv`.
2. Checkpoint/seed/episode/state: `/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30/checkpoint/final.torch`, eval seed `900078`, step `16`.
3. Actor mean predicted safe? `True`; q=0.025622, threshold=0.050000.
4. High-risk corridor shadows found? `True`; max q=0.762424.
5. Were risky shadows executed? `False`.
6. Corridor lift positive? `True`; lift=0.215352.
7. Did STAR reach target? `False`.
8. Did STAR avoid keep-out zone? `False`; episode_cost=12.000, min_clearance=-0.077.
9. Did Current-only or SAC-Lag provide contrast? Current-only cost=10.000; SAC-Lag cost=10.000.
9b. Did STAR+Exec reach target and avoid keep-out zone? `True`; success=1.000, episode_cost=0.000, min_clearance=0.040.
10. Is the result paper-worthy? `True` for a qualitative controlled ref-corridor mechanism figure; not as a standalone benchmark-win claim.
11. Caveats: this is a targeted showcase checkpoint, not the main Panda benchmark table. Existing original final STAR checkpoints had reference=current at save time, so they could not support a reference-to-current corridor-lift claim. Raw STAR on the selected seed does not reach the target and enters the keep-out zone; STAR+Exec on the same checkpoint provides the clean safe-success trajectory for the selected episode. Final 910000-910049 showcase evaluation is not a broad success result: STAR+Exec success_mean=0.000, episode_cost_mean=0.080, min_clearance_mean=0.082.
