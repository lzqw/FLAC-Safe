# STAR 1M Storage Strategy

Updated: 2026-07-02 23:48 Asia/Shanghai

`/root/autodl-tmp` has about 50G total and only about 3G free because previous STAR-v2 final artifacts occupy the same writable volume. `/autodl-pub` and `/autodl-pub/data` are mounted read-only in this container, so old artifacts cannot be migrated there.

To keep the 1M training curves running without overwriting previous final results:

- Existing `results/star_v2_final/` and previous artifacts are left in place.
- The early partial `results/star_1m_curves/` run at ~30k steps was preserved as `star_1m_curves_pre_shm_restart_*`.
- New `results/star_1m_curves/` is a symlink to `/dev/shm/star_1m_curves_results`.
- New runs use `save_training_state=False` and `save_interval_steps=0`, so no replay-memory intermediate checkpoints are written.
- Final checkpoints are still enabled with `final_checkpoint=True`.

Caveat: `/dev/shm` is volatile. Curve CSVs and reports should be collected and copied back frequently while runs are active.
