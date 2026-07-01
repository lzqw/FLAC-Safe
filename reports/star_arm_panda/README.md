# STAR Panda Arm Safety Showcase

Task: `SafetyPandaReachObstacle-v0`.

Backend: direct PyBullet Panda arm in Gymnasium API. `panda-gym` and `pybullet`
are installed and importable; direct PyBullet is used so the obstacle, flat
observation, and explicit safety cost are under experiment control.

Reward:
`r_t = -||p_ee - p_goal||_2 + 1.0 * I(success) - 0.01 * ||a||^2`.

Cost:
`cost_t = max(I(||p_ee - p_obs|| <= safe_margin), collision_t)`, with
`safe_margin=0.13m` and obstacle radius `0.07m`.

Action: 3D Cartesian end-effector delta, clipped to `[-1, 1]^3` and scaled by
`0.03m` per step.

Observation: flat vector containing end-effector position/velocity, target,
obstacle, target/obstacle vectors, distances, and Panda joint positions and
velocities.

Methods: SAC-Lag, Current-only-N (`current_only_v2`), STAR (`star_v2`), and
STAR+Exec for same-checkpoint evaluation once executor collection is run.

Seeds and steps:
- smoke: seed 0, 5k steps.
- calibration: seeds 0-1, 100k steps.
- final: seeds 10-12, 300k steps.

Selected STAR actor config:
```json
{
  "config_name": "star_cfg2",
  "overrides": [
    [
      "shadow_num_strata",
      16
    ],
    [
      "star_risk_threshold",
      0.03
    ],
    [
      "star_lambda",
      2.0
    ],
    [
      "star_ref_update_interval",
      100
    ]
  ],
  "selection_note": "Selected from completed calibration row panda_calibration_star_cfg2_star_v2_s1."
}
```

Selected STAR+Exec config:
```json
{
  "star_exec_candidates": 16,
  "star_exec_margin": 0.02,
  "selection_source": "/root/FLAC-Safe-star-v2/reports/star_arm_panda/executor/executor_validation.csv",
  "selection_rule": "minimize held-in validation violation/cost with success_drop <= 0.05 when feasible"
}
```

Final claim status:
# Panda Arm Final Claim

The Panda arm add-on is a mixed/diagnostic result, not a clean STAR-win showcase.

- current_only_v2: train_total_cost=240593.7, eval_return=-1.774, eval_cost=10.333, eval_success=0.950, eval_violation_rate=0.843
- sac_lag: train_total_cost=18808.0, eval_return=-5.979, eval_cost=9.433, eval_success=0.483, eval_violation_rate=0.463
- star_v2: train_total_cost=229432.3, eval_return=-8.060, eval_cost=8.283, eval_success=0.567, eval_violation_rate=0.528

STAR reduces evaluation cost relative to Current-only in the final summary, but it does not dominate SAC-Lag and has worse return/success tradeoffs. Present as an exploratory robot-arm add-on only.

STAR+Exec summary is available in `reports/star_arm_panda/executor/executor_summary.md`.

Heavy outputs are under `/root/autodl-tmp/star_v2_storage/results/star_arm_panda`. Small reports are under this directory.
