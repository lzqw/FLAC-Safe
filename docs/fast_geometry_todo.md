# Fast geometry implementation TODO

Goal: reduce the cost of the JVP geometry term in `model/algo.py` without changing the main algorithm.

## Key idea

Keep the current detached action-gradient behavior in `_grad_dot_directional`. Do not build a second-order graph through the safety critic. The geometry term should remain a stop-gradient local metric for the actor velocity.

## Add args in `main.py`

```python
arg.add_arg("jvp_batch_size", 0, "Max samples for JVP geometry; 0 means full batch")
arg.add_arg("jvp_sample_mode", "full", "JVP sample mode: full, random, boundary, topk_gate")
arg.add_arg("jvp_update_interval", 1, "Compute JVP geometry every N actor updates")
arg.add_arg("jvp_min_samples", 32, "Minimum selected JVP samples")
```

Read the same values in `flowAC.__init__`. Defaults must reproduce current behavior.

## Add helper in `flowAC`

Implement `_select_jvp_indices(g_mid)`:

- `full`: return `None`
- `random`: random subset up to `jvp_batch_size`
- `boundary`: select samples with `g_mid > 1e-3`, cap to `jvp_batch_size`, fallback to top-k if too few
- `topk_gate`: select top `jvp_batch_size` samples by detached `g_mid`

Return `(idx, mode_id)` where mode IDs are: full=0, random=1, boundary=2, topk_gate=3, fallback=-1.

## Modify `update_policy`

Current code calls `compute_jvp_scd(state_batch, action, velocity_action, g_mid)` on the full batch whenever JVP is enabled. Replace this with:

1. `jvp_active_this_update = jvp_enabled and (current_step_or_updates % jvp_update_interval == 0)`
2. if active, select indices using `_select_jvp_indices(g_mid)`
3. compute JVP only on the selected subset
4. keep reward critic loss and scalar safety penalty on the full actor batch
5. add `lambda_jvp_eff * jvp_loss` only when JVP is active

## Add logs

Add these keys to the dict returned by `update_policy`:

```text
safety/jvp_active_this_update
safety/jvp_update_interval
safety/jvp_batch_size
safety/jvp_selected_count
safety/jvp_sample_frac
safety/jvp_sample_mode_id
```

Keep all existing JVP logs.

## Recommended test config

```bash
--jvp_batch_size 1024 \
--jvp_sample_mode topk_gate \
--jvp_update_interval 2 \
--diagnose_safety_q_geometry False
```

Do not change batch size, updates per step, lambda schedule, replay buffer, environment, or evaluation protocol in the first ablation.

## Smoke test

Run a short PointGoal1 job with `num_steps=20000`, `eval=False`, `save=False`. Confirm that:

- default args still reproduce current behavior
- logs contain the new JVP keys
- `jvp_selected_count` is about 1024 for batch 4096 in top-k mode
- `jvp_active_this_update` alternates when interval is 2
- no second-order graph through the safety critic is introduced

## Implementation notes

- Fast geometry defaults preserve full-batch JVP: `jvp_batch_size=0`, `jvp_sample_mode=full`, `jvp_update_interval=1`.
- Optional one-sided JVP uses `relu(grad_QC dot velocity)^2`, leaving the detached safety normal path unchanged.
- `jvp_gate_mode=unsafe_side` and `boundary_or_unsafe` extend the boundary gate without changing scalar safety penalty.
- Periodic checkpoints are opt-in with `save=True --save_interval_steps N`; default `0` keeps old checkpoint behavior.
- Optional vectorized env collection was not mixed into this implementation because safe batched cost/final-observation handling should be validated separately from the JVP change.
