# STAR 1M Autosnapshot

An autosnapshot tmux window periodically runs `collect` and `status`, then writes `/root/star_1m_curves_latest_snapshot.tar.gz` and overwrites `/root/autodl-tmp/star_v2_storage/backups/star_1m_curves_latest_snapshot.tar.gz`.

Purpose: active 1M results now write to persistent `results/star_1m_curves/` after the single-GPU restart. The latest normalized curve/report snapshot is still persisted frequently while training is running, without accumulating timestamped backup archives on the nearly full 50G volume.

Interval: 300 seconds.

Snapshot includes `reports/star_1m_curves/`, `scripts/star/goal_1m_curves.py`, `scripts/star/collect_1m_curves.py`, `scripts/star/plot_1m_curves.py`, and `scripts/star/finalize_1m_curves.py`. It intentionally excludes checkpoints and raw result directories.
