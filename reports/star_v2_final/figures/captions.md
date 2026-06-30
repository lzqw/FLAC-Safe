# Figure Captions

- `fig_mechanism_validation`: Mechanism and diagnostic validation panel generated from available STAR-v2 summary rows.


## Figure 3 Training Curves

Figure 3 reports training-log episode return and cumulative training cost over environment steps. Seed means are shown with 95% normal-approximation standard-error bands. No final-summary rows are used to draw the curves. Some resumed seeds only have per-step resume logs after the 100k checkpoint; the mean and uncertainty bands use available seeds at each environment step. Table 1 uses all completed final evaluations.

## Figure 2 Mechanism Validation

Figure 2 uses training-time logged paired-audit diagnostics from final STAR-v2 runs. The primary comparison is corridor shadow risk versus equal-budget current-only samples under paired base noise. Final-run reference age is collapsed to the `age=1-5` bin, so age-dynamic claims are not made. Short-horizon simulator oracle validation is unavailable for these runs, so Figure 2 does not claim oracle-confirmed actual risk.
