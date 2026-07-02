# STAR 1M Training-Curve Plan

Stage A launches STAR (`star_v2`) for 3 tasks x 5 seeds = 15 runs.
Stage B launches SAC-Lag for 3 tasks x 3 seeds = 9 runs.
Safe Flow Q, PPO-Lag, CPO, and CSPO are documented as unavailable in this checkout and are not faked.

Use `python scripts/star/goal_1m_curves.py launch-star` first.
Use `python scripts/star/goal_1m_curves.py launch-baselines1` after or alongside Stage A if resources allow.
