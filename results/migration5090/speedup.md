# Ouro-1.4B cross-GPU KV migration E2E

Two RTX 5090 GPUs, BF16, Triton, d=2/D=4, K=4, 32 output tokens. One warmup and three alternating paired trials; one request migrates after its first committed speculative round. Timings include prefill, draft, verification, CPU KV snapshot and restore, coda and KV commit. Every paired output token and exit depth matched. Lower latency is better.

| Path | Median E2E, s | Relative speed |
|---|---:|---:|
| Single GPU | 1.672 | 1.000× |
| GPU 0 → GPU 1 | 1.704 | 0.981× |

[Raw paired trials](raw.json)

Checkpoint: ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. Physical GPUs 2 and 3 were mapped to CUDA devices 0 and 1. Runs used `/home/gongji/0z5a` in a Docker container capped at 96 GiB RAM. Migration added about 0.032 seconds to the median E2E time; this is a compatibility result, not a speedup claim.
