# STAR 1M Autosnapshot

An autosnapshot tmux window periodically runs `collect` and `status`, then writes `/root/star_1m_curves_latest_snapshot.tar.gz` and a timestamped copy under `/root/autodl-tmp/star_v2_storage/backups/`.

Purpose: active 1M results are stored under `/dev/shm` to avoid filling the 50G persistent volume, so normalized curve/report snapshots are persisted frequently while training is running.

Interval: 300 seconds.
