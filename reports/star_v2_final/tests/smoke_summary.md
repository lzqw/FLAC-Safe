STAR-v2 smoke summary
=====================

Smoke phases run
----------------

1. `smoke5k`: environment/checkpoint smoke with default `start_steps=5000`.
   This completed cleanly for 8 runs and saved 8 final checkpoints, but it did not exercise actor updates because the smoke length equaled `start_steps`. It is retained only as an environment/checkpoint sanity check.

2. `smoke5k_train`: actor-update smoke with `start_steps=200`, `batch_size=64`, `hidden_size=128`, `mechanism_log_interval_steps=500`, and `audit_diagnostic_interval=10`.
   This is the valid Phase-A implementation smoke.

Smoke task/method matrix
------------------------

Tasks:

- SafetyPointGoal1-v0
- SafetyCarGoal1-v0

Methods:

- pointwise_v2
- current_only_v2
- star_v2
- sac_lag

Result
------

- Completed actor-update smoke runs: 8/8.
- Final checkpoints: 8/8.
- Runtime error scan: clean for Traceback, ERROR, NaN/nan, OOM, out of memory, RuntimeError, MuJoCo.
- Mechanism CSV files: 8/8.
- STAR-v2/current-only shadow penalty nonzero on both tasks.
- Actor gradient norms finite on all runs.
- Reference refresh occurred on all runs (`reference_update_count` reached 225 in mechanism logs).
- Canonical STAR-v2 logs record `star_algorithm_version=star_v2`, `shadow_beta_mode=positive_linspace`, `shadow_penalty_mode=squared`, and `training_execution_mode=raw`.

Notes
-----

The `reference_age_post_update` sampled rows are 1 rather than 0 because mechanism logging samples after subsequent actor updates; `reference_update_count` proves refresh occurred. A later collector should use both fields rather than requiring a sampled row exactly at age zero.

Remaining gates
---------------

This smoke validates implementation execution only. Calibration, 100k core gate, 300k final runs, oracle diagnostics, ablations, executor validation, final aggregation, paper artifacts, and claim gate remain incomplete.
