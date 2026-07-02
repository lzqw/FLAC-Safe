# Panda Ref-Corridor Showcase Evaluation

This is a controlled checkpoint/configuration showcase for mechanism visualization. It does not replace the main Panda quantitative benchmark.

- task: `SafetyPandaReachObstacle-v0`
- STAR run: `panda_showcase_refcorridor_star_v2_s30`
- STAR checkpoint: `/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30/checkpoint/final.torch`
- eval seeds: `910000` through `910049`
- environment code: unchanged base Panda obstacle-reaching environment; the showcase uses a targeted ref-corridor STAR checkpoint and fixed evaluation seeds.

Representative geometry from the first evaluation row:

```json
{
  "action_scale": 0.03,
  "goal_pos": [
    0.5657281875610352,
    0.15428900718688965,
    0.23395906388759613
  ],
  "obstacle_pos": [
    0.4945596754550934,
    0.008381843566894531,
    0.27797478437423706
  ],
  "obstacle_radius": 0.07,
  "safe_margin": 0.13,
  "start_pos": [
    0.4591338038444519,
    -0.16681917011737823,
    0.2489236295223236
  ]
}
```
