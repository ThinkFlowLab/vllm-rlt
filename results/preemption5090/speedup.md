# Ouro-1.4B priority preemption E2E

RTX 5090, BF16, Triton, d=2/D=4, K=4. One low-priority 32-token request runs first; a high-priority 16-token request arrives after two low-priority outputs. One warmup and alternating paired trials. Both arms share one resident model and allow only one active request. Every paired output token and exit depth matched. Lower latency is better.

| Metric | Wait for active request, s | Preempt and resume, s | Speedup |
|---|---:|---:|---:|
| Two-request E2E | 2.240 | 2.311 | 0.969× |
| High-priority TTFT | 1.148 | 0.122 | 9.403× |

[Raw paired trials](raw.json)

Checkpoint: ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. Physical GPU 2 was mapped to CUDA device 0. Runs used `/home/gongji/0z5a` in a Docker container capped at 96 GiB RAM. Each preempted trial recorded one preemption and one resumption. Priority admission reduced the high-priority request's TTFT while increasing total E2E time by about 0.071 seconds.
