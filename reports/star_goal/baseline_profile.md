# STAR Phase 1 Baseline Profile

Configuration: method=star, 30k real transitions, seed=910, eval=False, save=False, batch_size=256, hidden_size=256, updates_per_step=1, shadow_k=16, star_exec=True.

| Task | GPU | Step | Wall sec | Transitions/s | Updates/s | Env time | Update time | GPU util mean/p95 | GPU mem peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SafetyPointGoal1-v0 | 0 | 30000 | 1391.3 | 21.56 | 17.97 | 46.92 | 1143.84 | 12.3/13.0 | 777 |
| SafetyCarGoal1-v0 | 1 | 30000 | 1342.1 | 22.35 | 18.63 | 63.02 | 1087.37 | 12.5/15.0 | 777 |

Current code did not separately instrument action selection, replay sample/push, observation normalization, logging, checkpoint time, CPU utilization, or RSS. These are marked `not_instrumented`/blank in the CSV and are the target of the next profiling/optimization pass.

Observation: with one run per GPU, GPU utilization is low and per-run speed settles near 22 transitions/s after updates start. This supports benchmarking more than one independent run per GPU after logging/diagnostic throttling is verified.
