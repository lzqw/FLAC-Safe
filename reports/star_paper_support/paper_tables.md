# Paper Tables

## Main Results

| Task | Method | Raw Return | Raw Cost | Filtered Return | Filtered Cost | pSVR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SafetyCarGoal1-v0 | pointwise | 33.23 ± 0.37 | 52.37 ± 9.31 | 33.23 ± 0.37 | 52.37 ± 9.31 | 0.00 ± 0.00 |
| SafetyCarGoal1-v0 | sac_lag | 33.29 ± 0.39 | 51.83 ± 4.49 | 33.29 ± 0.39 | 51.83 ± 4.49 | 0.00 ± 0.00 |
| SafetyCarGoal1-v0 | star_actor | 34.21 ± 0.51 | 48.93 ± 5.14 | 34.21 ± 0.51 | 48.93 ± 5.14 | 0.16 ± 0.03 |
| SafetyCarGoal1-v0 | star | 34.95 ± 0.41 | 47.52 ± 8.19 | 34.45 ± 0.12 | 44.73 ± 9.34 | 0.15 ± 0.01 |
| SafetyPointGoal1-v0 | pointwise | 26.61 ± 0.35 | 45.73 ± 1.68 | 26.61 ± 0.35 | 45.73 ± 1.68 | 0.00 ± 0.00 |
| SafetyPointGoal1-v0 | sac_lag | 27.09 ± 0.20 | 41.28 ± 7.42 | 27.09 ± 0.20 | 41.28 ± 7.42 | 0.00 ± 0.00 |
| SafetyPointGoal1-v0 | star_actor | 26.47 ± 0.38 | 46.72 ± 9.03 | 26.47 ± 0.38 | 46.72 ± 9.03 | 0.12 ± 0.01 |
| SafetyPointGoal1-v0 | star | 26.68 ± 0.63 | 42.05 ± 2.65 | 26.46 ± 0.25 | 44.17 ± 6.21 | 0.18 ± 0.01 |

## Audit Summary

| Task | Method | pSVR | Hidden unsafe rate | Any unsafe shadow rate |
| --- | --- | ---: | ---: | ---: |
| SafetyCarGoal1-v0 | star_actor | 0.165 ± 0.029 | 0.005 ± 0.006 | 0.169 ± 0.029 |
| SafetyCarGoal1-v0 | star | 0.146 ± 0.009 | 0.000 ± 0.000 | 0.146 ± 0.009 |
| SafetyPointGoal1-v0 | star_actor | 0.116 ± 0.013 | 0.010 ± 0.010 | 0.125 ± 0.020 |
| SafetyPointGoal1-v0 | star | 0.181 ± 0.007 | 0.010 ± 0.002 | 0.190 ± 0.008 |
