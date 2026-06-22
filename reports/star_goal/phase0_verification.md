# Phase 0 Verification

## pytest
.............                                                            [100%]
=============================== warnings summary ===============================
tests/test_shadow_audit.py::test_shadow_action_shape_and_bounds
tests/test_shadow_audit.py::test_beta_grid_contains_reference_and_current_endpoints
tests/test_shadow_audit.py::test_shadow_spread_k1_is_zero_not_nan
tests/test_star_agent.py::test_shadow_loss_gives_actor_gradient_but_not_cost_critic_gradient
tests/test_star_agent.py::test_raw_select_action_records_predicted_risk_without_changing_execution
tests/test_star_agent.py::test_sac_lag_local_logs_residual_and_mean_qc
tests/test_star_agent.py::test_checkpoint_save_load_restores_training_state
tests/test_star_agent.py::test_lambda_zero_kl_zero_exec_false_degenerates_to_sac_actor_update
  /root/miniconda3/envs/flac/lib/python3.11/site-packages/gymnasium/spaces/box.py:130: UserWarning: [33mWARN: Box bound precision lowered by casting to float32[0m
    gym.logger.warn(f"Box bound precision lowered by casting to {self.dtype}")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
13 passed, 8 warnings in 3.99s

## GPU tensor/forward/backward/optimizer smoke
gpu=0 ok loss=0.047806 mem_alloc=19671552
gpu=1 ok loss=0.066938 mem_alloc=19671552

## Safety-Gymnasium env reset/step smoke
env smoke FAIL

## Safety-Gymnasium env reset/step smoke retry
task=SafetyPointGoal1-v0 step_return_len=6
task=SafetyPointGoal1-v0 obs_shape=(60,) action_shape=(2,) reward=-0.0006241171269489865 cost=0.0 done=False
task=SafetyCarGoal1-v0 step_return_len=6
task=SafetyCarGoal1-v0 obs_shape=(72,) action_shape=(2,) reward=-0.0009706502766635428 cost=0.0 done=False
