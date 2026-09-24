# Ouro-1.4B speculative CUDA Graph E2E

NVIDIA GeForce RTX 5090; BF16, Triton, d=2/D=4, K=4, 32 output tokens. One resident model, separate engine state, five warmups per arm, 3 alternating paired trials. Timings include prefill, draft, verification, coda and KV commit. Every paired output token and exit depth matched.

| Workload / requests | Eager E2E s | Graph E2E s | Speedup | Output |
|---|---:|---:|---:|---|
| repeated / 1 | 1.494 | 0.425 | 3.518× | exact match |
| repeated / 4 | 1.367 | 0.780 | 1.752× | exact match |
| prose / 1 | 0.929 | 0.401 | 2.317× | exact match |

| Graph stage | Captures | Replays | Eager fallbacks |
|---|---:|---:|---:|
| recurrent | 8 | 2368 | 0 |
| coda | 8 | 984 | 0 |

[Raw paired trials](raw.json). Checkpoint: ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. Physical GPU 2 was mapped to CUDA device 0. Runs used `/home/gongji/0z5a` in Docker capped at 96 GiB RAM. This run exercises CUDA Graph replay with the split draft/verify runner and atomic rounds. A separate tiny-model GPU test covers Graph replay with an interleaved prefill; this table does not measure that workload.
