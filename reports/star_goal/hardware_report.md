# Hardware Report

## git
de4ff660cd5f2ebcec347f08596b49cba0ee4059
codex/star-dual4090d-goal
?? directional_noise.patch
?? paper/
?? results

## nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 4090 D (UUID: GPU-63ff0132-2f16-81c2-cbc5-c5e1ba3427e0)
GPU 1: NVIDIA GeForce RTX 4090 D (UUID: GPU-a242f06c-63c2-f67a-496e-1aded6600932)

## nvidia-smi
Mon Jun 22 15:44:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090 D      On  |   00000000:A9:00.0 Off |                  Off |
|  0%   38C    P8             16W /  425W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090 D      On  |   00000000:B8:00.0 Off |                  Off |
| 30%   35C    P8             12W /  425W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

## nvidia-smi topo -m
	[4mGPU0	GPU1	NIC0	NIC1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NODE	NODE	NODE	56-111,168-223	1		N/A
GPU1	NODE	 X 	NODE	NODE	56-111,168-223	1		N/A
NIC0	NODE	NODE	 X 	PIX				
NIC1	NODE	NODE	PIX	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1


## lscpu
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 57 bits virtual
Byte Order:                         Little Endian
CPU(s):                             224
On-line CPU(s) list:                0-223
Vendor ID:                          GenuineIntel
Model name:                         Intel(R) Xeon(R) Platinum 8481C
CPU family:                         6
Model:                              143
Thread(s) per core:                 2
Core(s) per socket:                 56
Socket(s):                          2
Stepping:                           8
Frequency boost:                    enabled
CPU max MHz:                        2001.0000
CPU min MHz:                        800.0000
BogoMIPS:                           4000.00
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization:                     VT-x
L1d cache:                          5.3 MiB (112 instances)
L1i cache:                          3.5 MiB (112 instances)
L2 cache:                           224 MiB (112 instances)
L3 cache:                           210 MiB (2 instances)
NUMA node(s):                       2
NUMA node0 CPU(s):                  0-55,112-167
NUMA node1 CPU(s):                  56-111,168-223
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Not affected
Vulnerability Spec rstack overflow: Not affected
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected

## nproc
224

## free -h
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        69Gi       192Gi       4.1Gi       744Gi       927Gi
Swap:             0B          0B          0B

## ulimit -a
real-time non-blocking time  (microseconds, -R) unlimited
core file size              (blocks, -c) unlimited
data seg size               (kbytes, -d) unlimited
scheduling priority                 (-e) 0
file size                   (blocks, -f) unlimited
pending signals                     (-i) 4126369
max locked memory           (kbytes, -l) unlimited
max memory size             (kbytes, -m) unlimited
open files                          (-n) 1024
pipe size                (512 bytes, -p) 8
POSIX message queues         (bytes, -q) 819200
real-time priority                  (-r) 0
stack size                  (kbytes, -s) 8192
cpu time                   (seconds, -t) unlimited
max user processes                  (-u) unlimited
virtual memory              (kbytes, -v) unlimited
file locks                          (-x) unlimited
