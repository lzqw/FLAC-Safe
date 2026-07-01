# Panda Arm Environment Setup

- branch: `codex/star-v2-aaai-final`
- head: `a5dc6710a139fe474839bddb34fc5f2a7864b3b5`
- python: `/root/miniconda3/envs/flac/bin/python`
- backend decision: direct PyBullet Panda environment; panda-gym and pybullet imports are checked.
- dependency isolation: installed into existing `flac` conda env after restoring `numpy==1.23.5` for Safety-Gymnasium compatibility.

## Import Check

```
AdroitHandRelocateDense-v1, AdroitHandHammerDense-v1, AdroitHandDoorDense-v1 environment's reward functions were updated in v1.2.1 without an environment version update. Therefore, use gymnasium-robotics==1.2.0 for v1 reproducibility or use v2 in gymnasium-robotics>=1.4.3. See https://github.com/Farama-Foundation/Gymnasium-Robotics/pull/220 for more details
AdroitHandRelocateDense-v1, AdroitHandHammerDense-v1, AdroitHandDoorDense-v1 environment's reward functions were updated in v1.2.1 without an environment version update. Therefore, use gymnasium-robotics==1.2.0 for v1 reproducibility or use v2 in gymnasium-robotics>=1.4.3. See https://github.com/Farama-Foundation/Gymnasium-Robotics/pull/220 for more details
pybullet build time: Jan 29 2025 23:17:20
torch 2.12.0+cu130 cuda True 2
gymnasium 0.28.1
panda_gym 3.0.7
numpy 1.23.5
```

## Headless Reset/Step Check

```
AdroitHandRelocateDense-v1, AdroitHandHammerDense-v1, AdroitHandDoorDense-v1 environment's reward functions were updated in v1.2.1 without an environment version update. Therefore, use gymnasium-robotics==1.2.0 for v1 reproducibility or use v2 in gymnasium-robotics>=1.4.3. See https://github.com/Farama-Foundation/Gymnasium-Robotics/pull/220 for more details
AdroitHandRelocateDense-v1, AdroitHandHammerDense-v1, AdroitHandDoorDense-v1 environment's reward functions were updated in v1.2.1 without an environment version update. Therefore, use gymnasium-robotics==1.2.0 for v1 reproducibility or use v2 in gymnasium-robotics>=1.4.3. See https://github.com/Farama-Foundation/Gymnasium-Robotics/pull/220 for more details
pybullet build time: Jan 29 2025 23:17:20
obs_shape (34,) action_shape (3,) cost 0.0
```

Required checks: imports succeed; reset/step succeeds; rendering disabled for training; PyBullet DIRECT headless mode is used.
