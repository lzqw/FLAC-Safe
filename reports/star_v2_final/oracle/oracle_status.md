# STAR-v2 Oracle Status

- summary: `/root/FLAC-Safe-star-v2/reports/star_v2_final/oracle/oracle_summary.csv`
- root: `/root/FLAC-Safe-star-v2/results/star_v2_final/resume_300k`
- methods: `star_v2`
- eval_seeds: `900000,900001,900002,900003,900004`
- horizons: `1,5`
- exit_code: `0`

## Availability

Oracle actual-risk validation is unavailable for these runs. The attempted backend returned unsupported rows with reason `NoneType object has no attribute model`; no h1/h5 actual-risk samples are reported or imputed. Figure 2 and final claims therefore do not use oracle evidence.

- oracle_rows: `reports/star_v2_final/oracle/oracle_rows.csv`
- supported_rows: `0`
