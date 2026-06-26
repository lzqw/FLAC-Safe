# STAR-v2 New Server Recovery State

Repo: /root/FLAC-Safe-star-v2
Branch: codex/star-v2-aaai-final
HEAD: af6a8a735b5e972671f47eecc04c05674e674b96

## Resource status
- GPUs: 2 x RTX 4090 D visible
- / root free: about 3.8G
- /root/autodl-tmp free: about 50G writable
- /autodl-pub has large free space but is read-only for this container
- Launch blocked by objective requirement: free disk must be >=100GB before starting new runs.

## Existing STAR-v2 results
- Cloned core result checkpoints found: 0
- Active STAR-v2 tmux sessions: 0
- Strict collector: PENDING, missing 48/48 core runs

## Recovery decision
All expected core-100k runs are marked PENDING, but action is blocked_until_disk_ready until a writable data path with >=100GB free is available or the disk requirement is explicitly relaxed.

Resume plan: reports/star_v2_final/recovery/core100k_resume_plan.csv
