# Ouro-1.4B speculative CUDA Graph E2E

NVIDIA GeForce RTX 5090; BF16, Triton, d=2/D=4, K=4, 64 output tokens. One resident model, separate engine state, five warmups per arm, 5 alternating paired trials. Timings include prefill, draft, verification, coda and KV commit. Every paired output token and exit depth matched.

| Workload / requests | Eager E2E s | Graph E2E s | Speedup | Output |
|---|---:|---:|---:|---|
| repeated / 1 | 3.028 | 0.763 | 3.969× | exact match |
| repeated / 4 | 2.223 | 1.209 | 1.838× | exact match |
| prose / 1 | 1.865 | 0.753 | 2.477× | exact match |

| Graph stage | Captures | Replays | Eager fallbacks |
|---|---:|---:|---:|
| recurrent | 11 | 5400 | 0 |
| coda | 11 | 2240 | 0 |
