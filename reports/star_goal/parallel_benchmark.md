# Parallel Benchmark Update

## rpg2

- completed runs: 4/4
- aggregate transitions/sec: 90.50
- min per-run transitions/sec: 21.94
- max per-run GPU memory MB: 319.9

| Run | Task | Step | t/s | updates/s | cost rate |
| --- | --- | ---: | ---: | ---: | ---: |
| parallel_rpg2_cg1_g0_slot1_s9401 | SafetyCarGoal1-v0 | 30000 | 22.83 | 19.02 | 0.0531 |
| parallel_rpg2_cg1_g1_slot1_s9403 | SafetyCarGoal1-v0 | 30000 | 22.72 | 18.93 | 0.0884 |
| parallel_rpg2_pg1_g0_slot0_s9400 | SafetyPointGoal1-v0 | 30000 | 21.94 | 18.28 | 0.0792 |
| parallel_rpg2_pg1_g1_slot0_s9402 | SafetyPointGoal1-v0 | 30000 | 23.01 | 19.18 | 0.0526 |
