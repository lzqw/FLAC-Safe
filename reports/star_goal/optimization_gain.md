# Phase 2 Optimization Gain

| Task | Baseline t/s | Optimized t/s | Gain | Wall-time reduction |
| --- | ---: | ---: | ---: | ---: |
| SafetyPointGoal1-v0 | 21.56 | 23.68 | 9.8% | 8.9% |
| SafetyCarGoal1-v0 | 22.35 | 23.45 | 4.9% | 4.7% |

Applied optimizations: W&B disabled by default, W&B logging throttled, mechanism logging interval separated, online eval default disabled, raw action diagnostics throttled, non-diagnostic STAR execution avoids candidate-stat synchronization, torch threads set to 1 per process.
