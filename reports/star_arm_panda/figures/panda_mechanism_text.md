# Panda Mechanism Figure Text

The figure is generated from real policy rollouts and critic evaluations, not hand-drawn or imputed data.

Selected STAR checkpoint: `/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30/checkpoint/final.torch`
Selected eval seed / step: `900078` / `16`
Actor mean predicted cost: `0.025622` with threshold `0.050000`; actor mean safe = `True`.
Max corridor shadow predicted cost: `0.762424`; high-risk corridor shadow found = `True`.
Corridor lift over matched current-only samples: `0.215352`.
Risky corridor shadows executed: `False`.

Episode comparison on the selected seed:
- SAC-Lag: success=1.000, cost=10.000, min_clearance=-0.084
- Current-only-N: success=1.000, cost=10.000, min_clearance=-0.112
- STAR: success=0.000, cost=12.000, min_clearance=-0.077
- STAR+Exec: success=1.000, cost=0.000, min_clearance=0.040

Interpretation: the selected ref-corridor showcase checkpoint exposes high-risk nearby shadow actions; STAR+Exec on the same checkpoint reaches the target while avoiding keep-out cost. Raw STAR on this seed is not a clean safe-success trajectory and should be reported as a caveat.
This is a controlled showcase result for mechanism visualization, not a replacement for the main quantitative benchmark tables.
