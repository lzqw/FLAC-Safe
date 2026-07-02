# Method Availability

Available in the current repo through `main_star.py` / `agents.star_agent.STARAgent`:
- `STAR` (`star_v2`)
- `SAC-Lag` (`sac_lag`)

Not launched because no matching implementation/training entry point was found in this checkout:
- `Safe Flow Q` (`safe_flow_q`)
- `PPO-Lag` (`ppo_lag`)
- `CPO` (`cpo`)
- `CSPO` (`cspo`)

Evidence: `STARAgent.valid_methods` accepts `sac`, `pointwise`, `sac_lag`, `star_actor`, `star_exec`, `star`, `pointwise_v2`, `current_only_v2`, `star_v2`, and `star_collect_v2`; grep over configs/scripts/agents did not find runnable Safe Flow Q, PPO-Lag, CPO, or CSPO trainers.
