# Ouro-1.4B speculative execution optimization, RTX 5090 E2E

ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, weight SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. BF16, Triton paged attention, shallow depth 2 / target depth 4, draft length 4, 32 output tokens. Physical GPU 2 maps to CUDA 0; migration also uses physical GPU 3 as CUDA 1. Runs used the existing `/home/gongji/0z5a` environment inside Docker capped at 96 GiB RAM. Timings include prefill, draft, verification, coda, and KV commit; migration also includes CPU KV snapshot and restore. All paired output tokens and exit depths matched.

## Before / after code change

Median of five real-model trials for the CUDA Graph path. Lower E2E time is better. Each arm ran in its own process with the same checkpoint and GPU. The repeated single-request workload used twelve warmups per arm because five did not reach steady state; the other workloads used five. These small differences are sensitive to shared-host load and should not be treated as a guaranteed speedup.

| Workload / requests | Before, s | After, s | Before / after |
|---|---:|---:|---:|
| repeated / 1 | 0.390 | 0.388 | 1.004× |
| repeated / 4 | 0.793 | 0.775 | 1.023× |
| prose / 1 | 0.404 | 0.400 | 1.009× |

Raw trials: [single request before](base-warm12/repeated1/raw.json), [single request after](optimized-warm12/repeated1/raw.json), [other workloads before](base-warm5/graph/raw.json), [other workloads after](optimized-warm5/graph/raw.json). The before/after outputs matched for every trial and workload.

## Final code path comparisons

Each row below compares two arms within the same real-model benchmark. `A / B` greater than one means B is faster. Five alternating paired trials were run for Graph, scheduling, and preemption; migration used three. One warmup per arm was used for preemption and migration, three for scheduling, and twelve for the repeated single-request Graph workload.

| Workload / metric | A | A, s | B | B, s | A / B |
|---|---|---:|---|---:|---:|
| repeated / 1 E2E | eager | 0.890 | Graph | 0.388 | 2.292× |
| repeated / 4 E2E | eager | 1.347 | Graph | 0.775 | 1.738× |
| prose / 1 E2E | eager | 0.931 | Graph | 0.400 | 2.327× |
| two-request scheduling E2E | atomic | 1.065 | interleaved | 1.062 | 1.002× |
| second-request TTFT | atomic | 0.158 | interleaved | 0.158 | 1.003× |
| two-request priority E2E | wait | 2.887 | preempt/resume | 2.998 | 0.963× |
| high-priority TTFT | wait | 1.483 | preempt/resume | 0.163 | 9.109× |
| migration request E2E | GPU 0 | 1.935 | GPU 0 → 1 | 1.988 | 0.973× |

Raw trials: [Graph](optimized-warm5/graph/raw.json), [scheduling](scheduling/raw.json), [preemption](preemption/raw.json), [migration](migration/raw.json). Every interleaved trial executed one prefill between draft and verification. Every preempted trial recorded one preemption and one resumption. The Graph path recorded 2,960 recurrent replays, 1,230 coda replays, and zero eager fallbacks in the full workload run.

Priority preemption improves the arriving high-priority request's TTFT while increasing total completion time. Cross-GPU migration preserves outputs but adds CPU copy cost. Intra-round scheduling has no material E2E gain in this workload.
