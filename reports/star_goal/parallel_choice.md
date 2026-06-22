# Parallel Choice

| Runs/GPU | Total runs | Aggregate transitions/s | Min per-run transitions/s | Max GPU mem MB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 47.13 | 23.45 | 319.9 |
| 2 | 4 | 90.50 | 21.94 | 319.9 |
| 3 | 6 | 132.14 | 21.36 | 319.9 |

## Decision

- Selected `RUNS_PER_GPU=3`.
- rpg3 improves aggregate throughput over rpg2 by 46.0%, above the 8% threshold.
- Early rpg3 monitoring showed GPU utilization around 95%, so rpg4 is not required before moving to tuning; the GPUs are already effectively saturated.
- Per-run rpg3 speed remains above 65% of the one-run/GPU baseline.
- Peak model memory is far below 85% of 24GB, so the limiting resource is compute scheduling, not memory.

## Selected parallel config

```json
{
  "RUNS_PER_GPU": 3,
  "MAX_TOTAL_RUNS": 6,
  "CPU_CORES_PER_RUN": "unbound_initial; BLAS threads fixed to 1",
  "compile_model": false,
  "num_envs": 1,
  "wandb_log_interval_steps": 1000,
  "mechanism_log_interval_steps": 1000,
  "online_eval_mode": "none"
}
```
