# Speculative scheduler and CUDA Graph verification on RTX 5090

ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`,
weight SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
BF16, Triton paged attention, speculative depths 2/4, and four draft tokens.
The code is based on upstream `3314c1b` and current #48 Graph parent `b574eb3`.
All runs used the existing `/home/gongji/0z5a` Python and Docker environments;
no package or driver changed. Physical GPU 4 was CUDA 0, with GPU 5 added for
migration. Task checkpoint files and transfer parts were removed after E2E.

## Incremental change relative to prior #52

Baseline implementation `892d82f` versus candidate `6b9ef19`. The admission
guard skips queue scans when all active slots are full and priority preemption
cannot admit waiting work. Both arms used the same Graph code and benchmark
inputs. Queue processes alternated baseline/candidate/candidate/baseline on
one GPU, with three warmups and five trials per process. Lower is better.

| Workload | Metric | Baseline median | Candidate median | Baseline / candidate |
|---|---|---:|---:|---:|
| 256 waiting, 2 active slots | `_admit`, μs/call | 56.39 | 31.37 | **1.797×** |
| 1 request | Complete E2E, s | 0.39568 | 0.39666 | 0.998× |
| 1 request | Median TTFT, s | 0.08780 | 0.08907 | 0.986× |
| 8 queued requests | Complete E2E, s | 2.38872 | 2.41746 | 0.988× |
| 8 queued requests | Median TTFT, s | 0.96335 | 0.97626 | 0.987× |

The CPU result is the median of two sessions, each with five 1,000-call batches
per arm. [Microbenchmark](../../benchmarks/spec_admission_micro.py) and its
[baseline](raw/base-micro.json), [baseline repeat](raw/base-micro-2.json),
[candidate](raw/candidate-micro.json), and
[candidate repeat](raw/candidate-micro-2.json) trials retain the inputs.
Queue E2E sessions: [one request baseline](raw/q1-final-1-base.json),
[candidate](raw/q1-final-2-candidate.json),
[candidate repeat](raw/q1-final-3-candidate.json),
[baseline repeat](raw/q1-final-4-base.json);
[eight requests baseline](raw/q8-final-1-base.json),
[candidate](raw/q8-final-2-candidate.json),
[candidate repeat](raw/q8-final-3-candidate.json),
[baseline repeat](raw/q8-final-4-base.json).
All output token IDs and exit depths matched. The eight-request baseline
process medians varied from 2.278 to 2.392 s, more than the candidate
difference. These E2E rows establish no material gain or regression.

## Complete model paths after the change

Each comparison kept one model resident and alternated arms within one process.
Speedup is the **median of per-trial A/B ratios**; arm columns are their
separate medians. All paired token IDs and exit depths matched. Time includes
prefill, drafting, verification, coda, and KV commit; migration also includes
CPU KV snapshot and restore.

| Workload / metric | A | A median, s | B | B median, s | Paired A / B |
|---|---|---:|---|---:|---:|
| Repeated prompt / 1, E2E | eager | 1.725 | Graph | 0.439 | **3.924×** |
| Repeated prompt / 4, E2E | eager | 1.347 | Graph | 0.777 | **1.734×** |
| Prose prompt / 1, E2E | eager | 1.898 | Graph | 0.452 | **4.208×** |
| Two-request scheduling, E2E | atomic | 1.173 | interleaved | 1.038 | 1.003× |
| Second-request TTFT | atomic | 0.238 | interleaved | 0.158 | 1.001× |
| Priority workload, E2E | wait | 2.530 | preempt/resume | 2.620 | 0.957× |
| High-priority TTFT | wait | 1.294 | preempt/resume | 0.155 | **8.372×** |
| Migration, E2E | one GPU | 1.953 | GPU 0 → 1 | 1.991 | 0.975× |

Graph used five warmups and five paired trials per workload, recording 2,960
recurrent and 1,230 coda replays with zero eager fallbacks. Scheduling used
three warmups and five trials; preemption used one warmup and five trials;
migration used one warmup and three trials. The scheduling E2E pair ratios
were 1.003, 0.992, 1.131, 1.001, 1.003. Its TTFT pair ratios were 1.001,
0.992, 1.513, 1.001, 1.005. The isolated large ratios came from a GPU
performance-mode change rather than a repeatable scheduling gain.

Raw paired trials: [Graph](raw/graph-final/raw.json),
[scheduling](raw/sched-final/raw.json),
[preemption](raw/preemption-final/raw.json), and
[migration](raw/migration-final/raw.json).
Direct reuse of a Graph-owned verification output was tried and removed:
the paired ratios were 0.998× for one request and 1.003× for eight requests.
The [single](raw/paired-q1.json) and [eight-request](raw/paired-q8.json)
diagnostic trials remain for review.

The supported incremental gain is lower saturated-admission CPU cost. Graph
replay and priority preemption retain their workload-specific benefits. This
shared-host run did not establish an all-metric E2E improvement.

## Validation

- CPU suite: 291 passed, 11 skipped, 119 GPU cases deselected.
- RTX 5090 Triton Graph/speculative suite: 22 passed, 4 skipped, 61 deselected.
  Paged FlashAttention is unsupported on SM120 in the pinned backend. An
  earlier run included those parameters and had two unseeded speculative
  test mismatches; both passed when isolated and in the corrected combined run.
- Ruff check/format and `git diff --check` passed.
