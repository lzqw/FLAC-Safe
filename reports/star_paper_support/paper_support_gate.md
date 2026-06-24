# STAR Paper Support Gate

Claim A: supported
Evidence: predicted pSVR is positive for STAR methods. Oracle simulator snapshot diagnostics are unsupported on this wrapper and are not used as evidence.

Claim B: weak
Evidence: STAR-Actor cost lower than pointwise on 4/6 paired task-seeds; lower than SAC-Lag-local on 3/6 paired task-seeds.

Claim C: supported
Evidence: Default Full STAR filtered cost lower than raw on 3/6 paired task-seeds. Selected executor grid improved filtered EVR on 2/2 task-level comparisons with candidates=8 margin=0.0.

Claim D: supported
Evidence: reports/star_goal/corridor_vs_current_only.csv.

Do not hide weak or unsupported results; PointGoal1 remains weaker than CarGoal1 for raw actor improvements.
