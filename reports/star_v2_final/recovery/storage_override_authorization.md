# Storage override authorization

The user explicitly overrides the previous storage-blocking rule.

The pipeline must not stop merely because no persistent writable path has >=100GB free space.
The pipeline must continue through core-100k, gate, 300k, mechanism, oracle if supported, ablation, executor, and paper artifact generation unless a fatal technical failure occurs.

Disk space must still be monitored and logged. Heavy outputs should be redirected to /root/autodl-tmp. If disk becomes tight, prune nonessential files and continue. Do not delete final checkpoints, evaluation CSVs, configs, manifests, or final summaries.
