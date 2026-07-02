# Reproduce Panda Mechanism Figure

Selected case:

- checkpoint: `/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30/checkpoint/final.torch`
- evaluation seed: `900078`
- episode: `0`
- step: `16`
- actor mean Q: `0.025622`
- audit threshold: `0.050000`
- max corridor Q: `0.762424`
- corridor lift: `0.215352`

Regenerate from existing checkpoint and fixed seed:

```bash
cd /root/FLAC-Safe-star-v2
export PYTHONPATH=.
/root/miniconda3/envs/flac/bin/python scripts/star/find_panda_mechanism_episode.py \
  --star-run-dir /root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30 \
  --compare-seed 11 --eval-start 900000 --eval-count 100
/root/miniconda3/envs/flac/bin/python scripts/star/package_panda_mechanism_data.py
/root/miniconda3/envs/flac/bin/python scripts/star/plot_panda_mechanism_figure.py
```

The packaged CSVs are normalized for local paper plotting:

- `trajectory_rows.csv`
- `audit_snapshot.csv`
- `audit_summary.json`

Caveat: this is a qualitative controlled ref-corridor mechanism case. Raw STAR is mixed on this seed; STAR+Exec gives the clean selected-episode trajectory. Broader Panda evaluation is not claimed as a benchmark win.

## Rendered Panda Keyframes

Regenerate PyBullet rendered keyframes and the rendered combined figure:

```bash
cd /root/FLAC-Safe-star-v2
export PYTHONPATH=.
/root/miniconda3/envs/flac/bin/python scripts/star/render_panda_selected_episode.py
```

The script writes `reports/star_arm_panda/rendered_keyframes/render_metadata.json`. Start, near-obstacle, and final frames are true STAR+Exec replay frames. The audit-state frame is rendered from the logged real audit state in `audit_snapshot.csv` so the robot pose matches the saved critic-query candidates.
