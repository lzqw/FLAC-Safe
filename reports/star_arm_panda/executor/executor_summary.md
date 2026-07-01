# STAR+Exec Panda Evaluation

Same STAR checkpoints are used; candidate execution is evaluation-only.

- selected candidates: `16`
- selected margin: `0.02`
- validation raw cost -> STAR+Exec cost: `8.000 -> 7.433`
- validation raw violation -> STAR+Exec violation: `0.476 -> 0.396`
- validation success drop: `0.033`

Held-out confirmation at selected config:
- raw cost -> STAR+Exec cost: `8.550 -> 8.267`
- raw violation -> STAR+Exec violation: `0.628 -> 0.571`
- raw success -> STAR+Exec success: `0.667 -> 0.633`
- raw return -> STAR+Exec return: `-5.743 -> -6.377`
- fallback rate: `0.996`
- found-but-not-executed rate: `0.001`
- mean action-selection latency ms: `2.791`
