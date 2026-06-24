# Corridor vs Current-Only Minimal Ablation

Seeds paired: [10, 11, 12]

| Metric | Mean delta current-corridor | Sample std | Direction |
| --- | ---: | ---: | --- |
| delta_return_current_minus_corridor | -1.80986 | 0.974926 | positive favors current-only return; negative favors corridor return |
| delta_cost_current_minus_corridor | 3.69321 | 16.6513 | negative favors current-only cost; positive favors corridor cost |
| delta_pSVR_current_minus_corridor | -0.0118001 | 0.00272229 | interpret as mechanism delta, not executed violations |
| delta_hidden_current_minus_corridor | -0.00651042 | 0.0090211 | interpret as mechanism delta, not executed violations |

Per-seed rows are in `reports/star_goal/corridor_vs_current_only.csv`.

This comparison uses completed, error-free final-checkpoint raw evaluations only.
