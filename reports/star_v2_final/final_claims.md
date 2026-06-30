# STAR-v2 Paper Claim Matrix

| Claim | Status | Evidence |
| --- | --- | --- |
| Core STAR-v2 rows are available | SUPPORTED | tasks_with_core_rows=4 |
| STAR-v2 raw actor improves safety over Pointwise/SAC-Lag-local | MIXED | pointwise_pairs=12; saclag_pairs=12 |
| Candidate execution improves same-checkpoint safety | NOT_SUPPORTED | raw_vs_filtered_pairs=0 |
| Simulator oracle supports predicted shadow risk | UNAVAILABLE | oracle rows are reported separately; no claim threshold is imputed here |
| Corridor/log-mean-exp design choices are empirically supported | NOT_SUPPORTED | corridor_pairs=0; logmean_pairs=0 |

Statuses use only selected completed/error-free CSV rows. Missing comparisons are not imputed.
