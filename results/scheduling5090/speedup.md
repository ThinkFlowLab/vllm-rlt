# Ouro-1.4B speculative intra-round scheduling E2E

NVIDIA GeForce RTX 5090; BF16, Triton, d=2/D=4, K=4, 32 output tokens per request. The second prompt arrives after the first request's initial output. One resident model, three warmups per arm, 5 alternating paired trials. Timings include prefill, draft, verification, coda and KV commit. Every paired output token and exit depth matched. Lower latency is better; speedup is atomic / interleaved.

| Metric | Atomic, s | Interleaved, s | Speedup |
|---|---:|---:|---:|
| Two-request E2E | 1.914 | 1.907 | 1.003× |
| Second-request TTFT | 0.285 | 0.283 | 1.005× |

[Raw paired trials](raw.json)

Checkpoint: ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. The tested card was physical GPU 2, mapped to CUDA device 0. Runs used `/home/gongji/0z5a` in a Docker container capped at 96 GiB RAM. Every interleaved trial executed exactly one prefill between draft and verification. The 0.3% E2E difference is too small to establish a material speedup in this shared-host run.
