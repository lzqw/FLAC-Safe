# Panda Arm Final Claim

The Panda arm add-on is a mixed/diagnostic result, not a clean STAR-win showcase.

- current_only_v2: train_total_cost=240593.7, eval_return=-1.774, eval_cost=10.333, eval_success=0.950, eval_violation_rate=0.843
- sac_lag: train_total_cost=18808.0, eval_return=-5.979, eval_cost=9.433, eval_success=0.483, eval_violation_rate=0.463
- star_v2: train_total_cost=229432.3, eval_return=-8.060, eval_cost=8.283, eval_success=0.567, eval_violation_rate=0.528

STAR reduces evaluation cost relative to Current-only in the final summary, but it does not dominate SAC-Lag and has worse return/success tradeoffs. Present as an exploratory robot-arm add-on only.

STAR+Exec summary is available in `reports/star_arm_panda/executor/executor_summary.md`.
