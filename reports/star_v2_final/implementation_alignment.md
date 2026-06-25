STAR-v2 implementation alignment
================================

| Paper object | Paper formula / behavior | Configuration key | Code function | Tensor shape | Gradient behavior | Test coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Algorithm version | STAR-v1 and STAR-v2 are explicit, non-mixed modes | `star_algorithm_version` | `STARAgent.__init__`, `save_checkpoint`, `load_checkpoint` | scalar metadata | loading STAR-v1 as STAR-v2 requires explicit override | `test_star_v2_checkpoint_preserves_algorithm_metadata`, `test_loading_star_v1_checkpoint_as_v2_requires_override` |
| Positive beta grid | `beta_k = k / K_beta`, `k=1..K_beta` | `shadow_beta_mode=positive_linspace` | `ShadowAuditModule.beta_grid` | `[B,K,1]` | beta=0 excluded from training candidates | `test_positive_linspace_excludes_reference_and_includes_current_endpoint`, `test_positive_linspace_k1_is_current_endpoint` |
| Legacy endpoint ablation | `linspace(0,1,K)` | `shadow_beta_mode=legacy_endpoints` | `ShadowAuditModule.beta_grid` | `[B,K,1]` | retained for ablation only | `test_legacy_endpoints_reproduce_old_grid` |
| Shadow strata and samples | `N = K_beta * L` | `shadow_num_strata`, `shadow_samples_per_stratum` | `ShadowAuditModule.generate_shadow_actions` | actions preflatten `[B,K,L,A]`, flattened `[B,N,A]` | vectorized; no Python loops over batch/strata/samples | `test_shadow_action_shape_and_bounds_with_samples_per_stratum` |
| Reference endpoint diagnostic | Detached beta=0 reference endpoint is diagnostic-only | implicit via audit output | `ShadowAuditModule.generate_shadow_actions` | `[B,A]` optional field | detached; not concatenated into `actions`, not in rho | `test_reference_endpoint_is_diagnostic_only_not_in_shadow_actions` |
| Matched current-only audit | corridor/current-only share `eps, xi` | `shadow_reference_mode`; diagnostic API | `ShadowAuditModule.generate_matched_audits` | both `[B,N,A]` | diagnostic can run under `no_grad`; training current-only uses same N | `test_matched_audits_share_eps_and_xi`, `test_current_only_v2_and_star_v2_have_same_candidate_count` |
| Shadow risk | `tau * (logsumexp(q/tau)-log(N))` | `shadow_aggregation=log_mean_exp`, `shadow_temperature` | `log_mean_exp_risk`, `ShadowAuditModule.shadow_risk` | risks `[B,N]`, rho `[B]` | differentiable through shadow actions to actor | `test_log_mean_exp_matches_manual_result`, `test_log_mean_exp_bounds` |
| Squared exceedance | `0.5 * mean(relu(rho-d)^2)` | `star_shadow_penalty_mode=squared` | `exceedance_penalty`, `STARAgent._shadow_actor_terms`, `_pointwise_penalty` | risk `[B]` or `[B,1]` | gradient magnitude increases with exceedance | `test_squared_exceedance_penalty_value_and_gradients`, `test_threshold_changes_squared_hinge_gradient_magnitude` |
| Linear STAR-v1 compatibility | `mean(relu(rho-d))` | `star_shadow_penalty_mode=linear` | `exceedance_penalty` | risk `[B]` | reproduces prior hinge | `test_linear_exceedance_reproduces_v1_behavior` |
| Cost critic isolation | actor loss must not update cost critic parameters | shared config | `STARAgent.update_actor`, `_shadow_actor_terms` | n/a | cost critic params frozen during actor loss; action gradients preserved | `test_shadow_loss_gives_actor_gradient_but_not_cost_critic_gradient`, `test_cost_critic_action_input_gradient_nonzero` |
| Reference refresh timing | loss uses current reference, actor updates, then reference age increments/refreshes | `star_ref_update_interval` | `STARAgent.update_actor`, `_reference_update_if_needed` | scalar counters | after refresh, policy/reference can produce matching paired audits | `test_after_reference_refresh_corridor_and_current_only_match_with_same_noise` |
| Canonical methods | `pointwise_v2`, `current_only_v2`, `star_v2`, `star_collect_v2` | `method` | `STARAgent.__init__`, `_method_uses_shadow_loss`, `_method_uses_kl` | n/a | v2 methods use proximal KL when `star_use_kl=True` | `test_star_v2_all_positive_strata_have_actor_gradient_path` |

Current scope note
------------------

This alignment report covers the implemented STAR-v2 core and unit-testable
mechanics. The later experiment scheduler, oracle repair, decisive 100k/300k
experiments, paper figures, and final claim gates are not yet completed.
