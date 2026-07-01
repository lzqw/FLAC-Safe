# STAR Panda Arm Safety Showcase

Task: `SafetyPandaReachObstacle-v0`.

Backend: direct PyBullet Panda arm in Gymnasium API. `panda-gym` and `pybullet`
are installed and importable; direct PyBullet is used so the obstacle, flat
observation, and explicit safety cost are under experiment control.

Reward:
`r_t = -||p_ee - p_goal||_2 + 1.0 * I(success) - 0.01 * ||a||^2`.

Cost:
`cost_t = max(I(||p_ee - p_obs|| <= safe_margin), collision_t)`, with
`safe_margin=0.13m` and obstacle radius `0.07m`.

Action: 3D Cartesian end-effector delta, clipped to `[-1, 1]^3` and scaled by
`0.03m` per step.

Observation: flat vector containing end-effector position/velocity, target,
obstacle, target/obstacle vectors, distances, and Panda joint positions and
velocities.

Methods: SAC-Lag, Current-only-N (`current_only_v2`), STAR (`star_v2`), and
STAR+Exec for same-checkpoint evaluation once executor collection is run.

Heavy outputs are under `/root/autodl-tmp/star_v2_storage/results/star_arm_panda`. Small reports are under this directory.
