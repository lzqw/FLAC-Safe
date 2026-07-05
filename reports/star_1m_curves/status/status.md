# STAR 1M Curves Status

Updated: `2026-07-05 18:25:17`

## Host Load

- CPU loadavg: `21.45 24.20 24.37 16/12126 129157`
- nproc: `20`

## stage_a_star
`completed`=15

| task | method | seed | status | max_step | recent steps/s | ETA h | errors |
|---|---:|---:|---|---:|---:|---:|---|
| PointGoal1 | STAR | 0 | completed | 1000000 |  |  |  |
| PointGoal1 | STAR | 1 | completed | 1000000 |  |  |  |
| PointGoal1 | STAR | 2 | completed | 1000000 |  |  |  |
| PointGoal1 | STAR | 3 | completed | 1000000 |  |  |  |
| PointGoal1 | STAR | 4 | completed | 1000000 |  |  |  |
| CarGoal1 | STAR | 0 | completed | 1000000 |  |  |  |
| CarGoal1 | STAR | 1 | completed | 1000000 |  |  |  |
| CarGoal1 | STAR | 2 | completed | 1000000 |  |  |  |
| CarGoal1 | STAR | 3 | completed | 1000000 |  |  |  |
| CarGoal1 | STAR | 4 | completed | 1000000 |  |  |  |
| PointPush1 | STAR | 0 | completed | 1000000 |  |  |  |
| PointPush1 | STAR | 1 | completed | 1000000 |  |  |  |
| PointPush1 | STAR | 2 | completed | 1000000 |  |  |  |
| PointPush1 | STAR | 3 | completed | 1000000 |  |  |  |
| PointPush1 | STAR | 4 | completed | 1000000 |  |  |  |

## stage_b_baselines1
`completed`=9

| task | method | seed | status | max_step | recent steps/s | ETA h | errors |
|---|---:|---:|---|---:|---:|---:|---|
| PointGoal1 | SAC-Lag | 0 | completed | 1000000 |  |  |  |
| PointGoal1 | SAC-Lag | 1 | completed | 1000000 |  |  |  |
| PointGoal1 | SAC-Lag | 2 | completed | 1000000 |  |  |  |
| CarGoal1 | SAC-Lag | 0 | completed | 1000000 | 23.79 |  |  |
| CarGoal1 | SAC-Lag | 1 | completed | 1000000 | 24.78 |  |  |
| CarGoal1 | SAC-Lag | 2 | completed | 1000000 | 26.23 |  |  |
| PointPush1 | SAC-Lag | 0 | completed | 1000000 | 25.48 |  |  |
| PointPush1 | SAC-Lag | 1 | completed | 1000000 | 25.55 |  |  |
| PointPush1 | SAC-Lag | 2 | completed | 1000000 | 26.27 |  |  |

## Resources

```
Sun Jul  5 18:25:19 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA vGPU-48GB               On  |   00000000:18:00.0 Off |                  Off |
| 30%   39C    P8             16W /  450W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   28G  3.0G  91% /
/dev/md127      7.0T  6.7T  371G  95% /autodl-pub
AutoFS:fs1      4.0T  2.6T  1.5T  65% /autodl-pub/data
tmpfs            64M     0   64M   0% /dev
shm              45G     0   45G   0% /dev/shm
/dev/nvme0n1p2  1.8T   16G  1.7T   1% /usr/bin/nvidia-smi
tmpfs           504G   12K  504G   1% /proc/driver/nvidia
tmpfs           504G  4.0K  504G   1% /etc/nvidia/nvidia-application-profiles-rc.d
tmpfs           504G     0  504G   0% /proc/asound
tmpfs           504G     0  504G   0% /proc/acpi
tmpfs           504G     0  504G   0% /proc/scsi
tmpfs           504G     0  504G   0% /sys/firmware
tmpfs           504G     0  504G   0% /sys/devices/virtual/powercap
```

