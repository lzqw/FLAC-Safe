# Seeta RTX 4090 Remote Workflow

This project can run on a single RTX 4090 SeetaCloud server. Do not put the SSH password in any script, README, config file, or commit. Type the password only in the terminal prompt when SSH asks for it.

## SSH Config

Recommended local `~/.ssh/config` entry:

```sshconfig
Host seeta-flacsafe
    HostName connect.cqa1.seetacloud.com
    Port 41939
    User root
    ServerAliveInterval 60
    ServerAliveCountMax 10
    StrictHostKeyChecking accept-new
```

First connection:

```bash
ssh seeta-flacsafe
```

## Project And Conda

On the server:

```bash
cd /root/FLAC-Safe
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate flac
```

Sync code:

```bash
git fetch origin
git checkout main
git pull origin main
git log --oneline -5
```

Basic check:

```bash
python -m compileall .
python - <<'PY'
import torch
import gymnasium
import safety_gymnasium
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
print("gymnasium ok")
print("safety_gymnasium ok")
PY
```

## tmux

```bash
tmux new -s flac_conv
tmux attach -t flac_conv
```

Press `Ctrl+b`, then `d`, to detach without stopping the task.

## GPU And Logs

Watch GPU:

```bash
watch -n 1 nvidia-smi
```

Follow a log:

```bash
tail -f logs/convergence/xxx.log
```

For a compact GPU view:

```bash
bash scripts/watch_gpu.sh
```

In `nvidia-smi`, watch:

- `memory.used / memory.total`
- `utilization.gpu`
- `power.draw`
- `temperature.gpu`

Interpretation:

1. If `memory.used < 3GB` and GPU utilization is very low, the batch is too small or environment interaction is the bottleneck.
2. If `memory.used` is 6-12GB and GPU utilization moves clearly, the RTX 4090 is already being used enough for this phase.
3. If `memory.used` approaches 22GB, OOM risk is high and the configuration is not recommended.
4. If GPU utilization is low but CPU is busy, use vectorized environments or higher `updates_per_step` later instead of only increasing batch size.
5. Phase 1 does not use vectorized environments because that adds engineering complexity.
6. For one RTX 4090, start with `batch_size=512` or `batch_size=1024`.

## 4090 24GB Configuration Principles

The current program is off-policy RL. Environment interaction is often CPU/env-bound, and small network training may not fill GPU memory. Do not blindly increase model size just to occupy 24GB.

Use this order:

1. First make convergence stable.
2. Then increase `batch_size`.
3. Then increase `updates_per_step`.
4. Then consider `hidden_size`.
5. Last, consider `torch.compile` or more parallel environments.

Recommended levels:

| Level | Purpose | batch_size | updates_per_step | hidden_size | num_steps |
| --- | --- | ---: | ---: | ---: | ---: |
| L0 | Stable test | 256 | 1 | 512 | 50000 |
| L1 | Light GPU use | 512 | 1 | 512 | project default |
| L2 | Higher training intensity | 1024 | 1 | 512 | project default |
| L3 | More gradient updates | 1024 | 2 | 512 | project default |
| L4 | Only after earlier levels are stable | 1024 | 2 | 1024 | project default |

Do not start with:

- `batch_size >= 2048`
- `updates_per_step >= 4`
- `hidden_size >= 1024`
- `compile_model=True`

These settings can mix convergence problems with engineering problems.

## Recommended Remote Flow

1. Connect:

```bash
ssh seeta-flacsafe
```

2. Initialize:

```bash
cd /root/FLAC-Safe
bash scripts/remote_setup_seeta.sh
```

3. Open tmux:

```bash
tmux new -s flac_probe
```

4. Run memory probes:

```bash
bash scripts/run_4090_memory_probe.sh L0
bash scripts/run_4090_memory_probe.sh L1
bash scripts/run_4090_memory_probe.sh L2
```

5. In another SSH window, watch the GPU:

```bash
ssh seeta-flacsafe
cd /root/FLAC-Safe
bash scripts/watch_gpu.sh
```

6. If L2 is stable, try:

```bash
bash scripts/run_4090_memory_probe.sh L3
```

Do not run L4 first.

7. After probes pass, run convergence tests:

```bash
tmux new -s flac_conv
cd /root/FLAC-Safe
conda activate flac
bash scripts/run_convergence_pointgoal.sh C0
bash scripts/run_convergence_pointgoal.sh C1
bash scripts/run_convergence_pointgoal.sh C2
bash scripts/run_convergence_pointgoal.sh C3
bash scripts/run_convergence_pointgoal.sh C4
```

The convergence script deliberately avoids forward/hutchinson JVP and soft masking in phase 1. Test those only after C0-C4 are stable.
