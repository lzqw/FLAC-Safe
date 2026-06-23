# Selected Baselines

Selected from the 100k baseline screen using final-checkpoint raw reevaluation.

| Method | Config | Parameter | Return | Cost | EVR | Train Cost Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Pointwise | `pointwise_lambda050` | `star_lambda=0.5` | 27.073 | 48.400 | 0.048 | 0.0579 |
| SAC-Lag-local | `sac_lag_lr0003` | `lagrange_lr=0.0003` | 28.160 | 38.000 | 0.038 | 0.0738 |
