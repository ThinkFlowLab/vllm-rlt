# Speculative CUDA Graph E2E on RTX 5090

Ouro-1.4B, BF16, Triton, `last_exited`, d=2/D=4, K=4, 64 output tokens per request. Eager and graphed engines share one resident model but keep separate request state. Five warmups per arm precede five alternating paired trials. Timings include prefill, drafting, verification, LM head, and KV commit. Every paired output token and exit depth matched. Values are median ± median absolute deviation (seconds); speedup is eager median / graphed median.

The table below uses one CPU thread for OMP, MKL, and OpenBLAS in both arms. The host load fell from 67 to 12 during the run, so the single-request repeated prompt remains sensitive to host scheduling.

| Workload | Eager, s | CUDA Graph, s | Speedup | Raw trials |
|---|---:|---:|---:|---|
| Repeated prompt, 1 request | 3.028 ± 0.161 | 0.763 ± 0.003 | 3.969× | [JSON](rtx5090/raw.json) |
| Repeated prompt, 4 requests | 2.223 ± 0.001 | 1.209 ± <0.001 | 1.838× | [JSON](rtx5090/raw.json) |
| Prose prompt, 1 request | 1.865 ± 0.017 | 0.753 ± <0.001 | 2.477× | [JSON](rtx5090/raw.json) |

The graph arm recorded 11 recurrent captures and 5,400 replays, plus 11 LM-head captures and 2,240 replays, with zero eager fallbacks. A separate run without the single-thread settings during host load around 162/128 logical CPUs showed much larger timing drift; its [report](rtx5090-highload/speedup.md) and [raw trials](rtx5090-highload/raw.json) are retained. These measurements do not establish a stable speedup under a saturated host.

Checkpoint SHA256: `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. Source: `e1336ac` on top of #44 `d9fca50`. Runs used the existing `/home/gongji/0z5a` environment inside Docker with a verified 96 GiB RAM cap. RTX 5090 tests used Triton; packed FA4 graph capture remains unverified because the installed paged FA4 backend does not support SM12.
