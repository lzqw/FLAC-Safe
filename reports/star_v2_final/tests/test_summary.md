STAR-v2 test summary
====================

Commands run in conda env `flac` on `/root/FLAC-Safe-star-v2`:

```bash
python -m py_compile agents/shadow_audit.py agents/star_agent.py main_star.py utilis/star_default_config.py
pytest -q tests/test_shadow_audit.py tests/test_star_agent.py
pytest -q
```

Results:

- Target STAR tests: 26 passed, 15 warnings in 3.06s.
- Full pytest suite: 26 passed, 15 warnings in 4.08s.

Coverage notes:

- Positive beta grid excludes beta=0 and includes beta=1.
- Legacy endpoint grid reproduces STAR-v1 beta behavior.
- Reference endpoint is diagnostic-only and not part of shadow rho.
- K-by-L shadow shape, matched corridor/current-only noises, and K=1 spread are tested.
- Squared and linear exceedance penalties are tested.
- Actor gradients flow through shadow actions while cost critic parameters stay gradient-free during actor loss.
- STAR-v2 checkpoint metadata and STAR-v1-to-v2 checkpoint override protection are tested.

Remaining required pipeline work:

- Oracle diagnostics repair, staged scheduler, calibration, 100k/300k experiments, ablations, executor validation, final collectors, paper artifacts, and final claim gates remain incomplete.
