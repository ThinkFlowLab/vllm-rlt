# Ouro-1.4B speculative CUDA Graph E2E

NVIDIA GeForce RTX 5090; BF16, Triton, d=2/D=4, K=4, 64 output tokens. One resident model, separate engine state, five warmups per arm, 5 alternating paired trials. Timings include prefill, draft, verification, coda and KV commit. Every paired output token and exit depth matched.

| Workload / requests | Eager E2E s | Graph E2E s | Speedup | Output |
|---|---:|---:|---:|---|
| repeated / 1 | 1.694 | 0.717 | 2.361× | exact match |
| repeated / 4 | 4.326 | 1.396 | 3.099× | exact match |
| prose / 1 | 4.584 | 0.852 | 5.380× | exact match |

| Graph stage | Captures | Replays | Eager fallbacks |
|---|---:|---:|---:|
| recurrent | 11 | 5400 | 0 |
| coda | 11 | 2240 | 0 |
