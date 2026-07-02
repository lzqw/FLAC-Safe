# STAR 1M Curves Status

Updated: `2026-07-02 23:35:06`

## stage_a_star
`pending`=9 `running`=6

| task | method | seed | status | max_step | errors |
|---|---:|---:|---|---:|---|
| PointGoal1 | STAR | 0 | running | 20000 |  |
| PointGoal1 | STAR | 1 | running | 19000 |  |
| PointGoal1 | STAR | 2 | running | 19000 |  |
| PointGoal1 | STAR | 3 | running | 19000 |  |
| PointGoal1 | STAR | 4 | running | 20000 |  |
| CarGoal1 | STAR | 0 | running | 19000 |  |
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
Thu Jul  2 23:35:07 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA vGPU-48GB               On  |   00000000:18:00.0 Off |                  Off |
| 62%   61C    P2            167W /  450W |    2503MiB /  24564MiB |     77%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA vGPU-48GB               On  |   00000000:39:00.0 Off |                  Off |
| 54%   60C    P2            169W /  450W |    2503MiB /  24564MiB |     97%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           74003      C   ...niconda3/envs/flac/bin/python        828MiB |
|    0   N/A  N/A           74034      C   ...niconda3/envs/flac/bin/python        828MiB |
|    0   N/A  N/A           74056      C   ...niconda3/envs/flac/bin/python        828MiB |
|    1   N/A  N/A           74029      C   ...niconda3/envs/flac/bin/python        828MiB |
|    1   N/A  N/A           74043      C   ...niconda3/envs/flac/bin/python        828MiB |
|    1   N/A  N/A           74065      C   ...niconda3/envs/flac/bin/python        828MiB |
+-----------------------------------------------------------------------------------------+
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   28G  2.3G  93% /
/dev/md127      7.0T  6.8T  220G  97% /autodl-pub
AutoFS:fs1      4.0T  2.6T  1.5T  65% /autodl-pub/data
tmpfs            64M     0   64M   0% /dev
shm              90G     0   90G   0% /dev/shm
/dev/nvme0n1p2  1.8T   16G  1.7T   1% /usr/bin/nvidia-smi
tmpfs           504G   12K  504G   1% /proc/driver/nvidia
tmpfs           504G  4.0K  504G   1% /etc/nvidia/nvidia-application-profiles-rc.d
tmpfs           504G     0  504G   0% /proc/asound
tmpfs           504G     0  504G   0% /proc/acpi
tmpfs           504G     0  504G   0% /proc/scsi
tmpfs           504G     0  504G   0% /sys/firmware
tmpfs           504G     0  504G   0% /sys/devices/virtual/powercap
```
