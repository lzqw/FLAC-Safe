# Panda Mechanism Data Dictionary

`trajectory_rows.csv` contains one row per real rollout step for Current-only-N, SAC-Lag, STAR raw, and STAR+Exec.

- `method`: plotted method label.
- `seed`, `episode`, `step`: selected evaluation identifier.
- `p_ee_*`, `p_goal_*`, `p_obs_*`: end-effector, goal, and obstacle positions in meters.
- `safe_margin`: keep-out radius used for clearance.
- `clearance`: `||p_ee - p_obs|| - safe_margin`; negative means inside keep-out.
- `cost`, `violation`, `collision`, `success`: values logged from environment step/info.
- `action_*`, `selected_action_*`: executed action components from the policy/executor.

`audit_snapshot.csv` contains the selected STAR boundary-state critic queries.

- `candidate_type`: `actor_mean`, `current_only`, `corridor_shadow`, or `selected_exec`.
- `action_d*`: queried action vector.
- `endpoint_*`: end-effector endpoint implied by the queried action.
- `predicted_QC`: critic-predicted safety cost.
- `predicted_safe`: `predicted_QC <= d_aud`.
- `executed`: whether the row is the action actually executed.
- `is_high_risk`: `predicted_QC > d_aud`.
- `rho_cor`, `rho_cur`, `corridor_lift`: matched audit risks for the selected state.
- `selection_score`: deterministic score used to prefer safe actor mean, high-risk non-executed corridor shadows, positive lift, and STAR+Exec success/safety.
