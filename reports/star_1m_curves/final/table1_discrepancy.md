# Table 1 Consistency Check

Reference file: `reports/star_v2_final/main_results_summary.csv` filtered to `phase=resume_300k`.

| task | method | 1M return | ref return | delta return | 1M cost | ref cost | delta cost | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SafetyCarGoal1-v0 | sac_lag | 35.4 | 33.76 | 1.638 | 57.6 | 64.8 | -7.2 | close |
| SafetyPointPush1-v0 | sac_lag | 0.7746 |  |  | 29.57 |  |  | no reference row |
