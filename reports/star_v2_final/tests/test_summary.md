# STAR-v2 Recovery Test Summary

Date: 2026-06-26
Server: autodl-container-a05746851a-389c8750
Branch: codex/star-v2-aaai-final
Commit: af6a8a735b5e972671f47eecc04c05674e674b96

## Environment

- Conda env: flac
- Python: 3.11.15
- Torch: 2.12.0+cu130
- CUDA available: yes
- CUDA devices: 2
- safety_gymnasium: 0.4.1
- gymnasium: 0.28.1
- mujoco: 2.3.3

## Task availability

- SafetyPointGoal1-v0: OK
- SafetyCarGoal1-v0: OK
- SafetyPointButton1-v0: OK
- SafetyCarButton1-v0: OK

## Tests

Targeted:

```text
pytest -q tests/test_shadow_audit.py tests/test_star_agent.py
26 passed, 15 warnings in 4.35s
```

Full:

```text
pytest -q
26 passed, 15 warnings in 5.40s
```

## Launch status

No training was launched during recovery because writable disk space is below the objective threshold of 100GB. `/root/autodl-tmp` is writable but has about 50GB free; `/autodl-pub` has large free space but is read-only from this container.
