STAR-v2 implementation pre-edit audit
=====================================

Base commit: f85f711b7558109b101c25c93f5e93de6d729a73

Observed STAR-v1 behavior before this edit:

- `ShadowAuditModule` generated `[B,K,A]` actions with one sample per beta.
- The legacy beta grid included beta=0 and beta=1 through `linspace(0, 1, K)`.
- `shadow_k` was the only shadow count parameter.
- Shadow risk used normalized log-mean-exp, mean, or max aggregation.
- Actor shadow penalty was a linear hinge: `mean(relu(rho - threshold))`.
- Canonical STAR-v2 methods (`pointwise_v2`, `current_only_v2`, `star_v2`,
  `star_collect_v2`) were not recognized.
- Checkpoints did not record STAR algorithm version, beta mode, penalty mode,
  or separated stratum/sample counts.
- Reference endpoint risk was not separated as a diagnostic-only quantity.

This report intentionally does not include STAR-v1 experiment results and should
not be used as a STAR-v2 result source.
