# STAR 1M Curves Status

Updated: `2026-07-03 00:07:33`

## stage_a_star
`pending`=9 `running`=6

| task | method | seed | status | max_step | errors |
|---|---:|---:|---|---:|---|
| PointGoal1 | STAR | 0 | running | 27000 |  |
| PointGoal1 | STAR | 1 | running | 26000 |  |
| PointGoal1 | STAR | 2 | running | 26000 |  |
| PointGoal1 | STAR | 3 | running | 26000 |  |
| PointGoal1 | STAR | 4 | running | 26000 |  |
| CarGoal1 | STAR | 0 | running | 26000 |  |
| CarGoal1 | STAR | 1 | pending | 0 |  |
| CarGoal1 | STAR | 2 | pending | 0 |  |
| CarGoal1 | STAR | 3 | pending | 0 |  |
| CarGoal1 | STAR | 4 | pending | 0 |  |
| PointPush1 | STAR | 0 | pending | 0 |  |
| PointPush1 | STAR | 1 | pending | 0 |  |
| PointPush1 | STAR | 2 | pending | 0 |  |
| PointPush1 | STAR | 3 | pending | 0 |  |
| PointPush1 | STAR | 4 | pending | 0 |  |

## stage_b_baselines1
`pending`=9

| task | method | seed | status | max_step | errors |
|---|---:|---:|---|---:|---|
| PointGoal1 | SAC-Lag | 0 | pending | 0 |  |
| PointGoal1 | SAC-Lag | 1 | pending | 0 |  |
| PointGoal1 | SAC-Lag | 2 | pending | 0 |  |
| CarGoal1 | SAC-Lag | 0 | pending | 0 |  |
| CarGoal1 | SAC-Lag | 1 | pending | 0 |  |
| CarGoal1 | SAC-Lag | 2 | pending | 0 |  |
| PointPush1 | SAC-Lag | 0 | pending | 0 |  |
| PointPush1 | SAC-Lag | 1 | pending | 0 |  |
| PointPush1 | SAC-Lag | 2 | pending | 0 |  |

## Resources

```
bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
Fri Jul  3 00:07:33 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA vGPU-48GB               On  |   00000000:18:00.0 Off |                  Off |
| 62%   62C    P2            179W /  450W |    2503MiB /  24564MiB |     97%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA vGPU-48GB               On  |   00000000:39:00.0 Off |                  Off |
| 53%   61C    P2            177W /  450W |    2503MiB /  24564MiB |     98%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           80170      C   ...niconda3/envs/flac/bin/python        828MiB |
|    0   N/A  N/A           80223      C   ...niconda3/envs/flac/bin/python        828MiB |
|    0   N/A  N/A           80243      C   ...niconda3/envs/flac/bin/python        828MiB |
|    1   N/A  N/A           80218      C   ...niconda3/envs/flac/bin/python        828MiB |
|    1   N/A  N/A           80232      C   ...niconda3/envs/flac/bin/python        828MiB |
|    1   N/A  N/A           80256      C   ...niconda3/envs/flac/bin/python        828MiB |
+-----------------------------------------------------------------------------------------+
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   28G  2.3G  93% /
/dev/md127      7.0T  6.8T  220G  97% /autodl-pub
AutoFS:fs1      4.0T  2.6T  1.5T  65% /autodl-pub/data
tmpfs            64M     0   64M   0% /dev
shm              90G  168K   90G   1% /dev/shm
/dev/nvme0n1p2  1.8T   16G  1.7T   1% /usr/bin/nvidia-smi
tmpfs           504G   12K  504G   1% /proc/driver/nvidia
tmpfs           504G  4.0K  504G   1% /etc/nvidia/nvidia-application-profiles-rc.d
tmpfs           504G     0  504G   0% /proc/asound
tmpfs           504G     0  504G   0% /proc/acpi
tmpfs           504G     0  504G   0% /proc/scsi
tmpfs           504G     0  504G   0% /sys/firmware
tmpfs           504G     0  504G   0% /sys/devices/virtual/powercap
```
