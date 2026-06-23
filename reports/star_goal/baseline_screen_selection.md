# Baseline Screen Selection

This screen is used only to choose one global Pointwise and one global SAC-Lag-local setting before the decisive 300k runs.

## Candidates

| Method | Config | Decision | Return | Cost | EVR | Train Cost Rate | Reasons |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| pointwise | pointwise_lambda050 | candidate | 27.073 | 48.400 | 0.048 | 0.0579 |  |
| pointwise | pointwise_lambda100 | filtered | 21.432 | 103.700 | 0.104 | 0.0675 | SafetyPointGoal1-v0:cost_exploded |
| pointwise | pointwise_lambda200 | filtered | 10.649 | 49.950 | 0.050 | 0.0596 | SafetyPointGoal1-v0:return_collapse |
| sac_lag | sac_lag_lr0003 | candidate | 28.160 | 38.000 | 0.038 | 0.0738 |  |
| sac_lag | sac_lag_lr0001 | candidate | 28.494 | 45.300 | 0.045 | 0.0626 |  |
| sac_lag | sac_lag_lr0010 | candidate | 27.138 | 43.100 | 0.043 | 0.0546 |  |

## Selected

- Pointwise: `pointwise_lambda050`
- SAC-Lag-local: `sac_lag_lr0003`
