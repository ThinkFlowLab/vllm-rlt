# Example contributor A/B report

**ILLUSTRATIVE ONLY: every result below is synthetic. No benchmark was run.**
This is a report-format reference, not executable harness output or evidence for
current main. Replace all numbers, placeholders and artifact paths with verified
results before submitting a report. Omit inapplicable sections with a reason.

## Change and frozen experiment

Hypothesis: removing redundant decode synchronization reduces single-request
latency without changing arithmetic, generated answers or KV ownership.
A is the intended base; B adds only this optimization.

| Control | Recorded value in this example |
| --- | --- |
| Sources | A: `<full-base-SHA>`; B: `<full-head-SHA>`; both clean; diff attached |
| Model/tokenizer | ByteDance/Ouro-1.4B; exact revisions and file hashes in `contract.json` |
| Precision | BF16 parameters, ordinary activations and KV verified; FP32 reductions and accumulation flags unchanged and recorded |
| Hardware | Same `<host>`, `<GPU-model/UUID/exact-ID>`, CPU affinity and NUMA placement for A/B |
| Environment | Server/client Python paths, package versions and benchmark source hashes in `environment.json` |
| Speed client | Pinned `vllm bench serve` client against the vllm-rlt server |
| Speed workload | Concurrency 1; 100 requests/run; each 128 actual input tokens and 64 output tokens; fixed seed and identical prompt hashes |
| Generation | Greedy, four loops; speed uses ignore-EOS and 64 output tokens; accuracy retains its frozen stop rules and token limit |
| Server controls | Identical capacity, scheduling and cache policy; exact launch arguments in `commands.sh`; no cross-request prefix reuse in this example |
| Timing boundary | Client-observed streaming HTTP timings; request throughput uses total benchmark duration |

Predeclared budget: CPU preparation probe; feasibility on each configuration;
then exactly two measured runs per configuration, ordered A1/B1/B2/A2. A restarts
before A2; B1/B2 reuse one server. Warmups occur before every measured block.
Separate diagnostic profiles follow; they never contribute to speed results.
Stop on failed/incomplete requests, nonfinite values or the recorded job timeout.
Do not silently add runs if inconclusive.

Example acceptance rule, frozen before execution: at least 1% output-throughput
gain in both pairs, nonoverlapping observed throughput ranges, no more than 2%
regression in the listed latency/memory controls, and no net accuracy loss.
These thresholds belong only to this example.

## Accuracy smoke result — synthetic

Fixed GSM8K-10 source IDs:
`11, 109, 351, 392, 669, 683, 914, 956, 997, 1033`.
Subset selection precedes A/B execution and is independent of answers/timings.
Prompt hashes, three-shot demonstrations and strict scoring are frozen.

| Metric | A | B |
| --- | ---: | ---: |
| Completed questions | 10/10 | 10/10 |
| Correct answers | 7/10 | 7/10 |
| Parse failures | 0 | 0 |
| Length-limited outputs | 0 | 0 |

Paired losses: 0; gains: 0; extracted-answer disagreements: 0; accuracy delta:
0 percentage points. Verdict under the example rule: **smoke pass**. This does
not establish full GSM8K accuracy or reuse the historical 59/87 reference floor.
For this arithmetic-preserving example, targeted token/state/KV/RNG comparisons
also pass exactly; retain their actual test commands and logs separately.

## Serving speed result — synthetic, profiling disabled

Every run completes 100/100 requests, 12,800 input tokens and 6,400 output tokens.
Failed requests, incomplete streams and nonfinite outputs: zero in all runs.

| Run, in execution order | Duration (s) | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean completion (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 200 | 32.000 | 100 | 30.0 | 1990.0 |
| B1 | 180 | 35.556 | 90 | 27.0 | 1791.0 |
| B2 | 182 | 35.165 | 91 | 27.3 | 1810.9 |
| A2 | 202 | 31.683 | 101 | 30.3 | 2009.9 |

Throughput is `6400 / duration`; per-request TPOT excludes the first token.
For these fixed-length requests, mean completion is mean TTFT + 63 × mean TPOT.
Benchmark duration also includes client gaps between requests.

- Pair 1, B1 versus A1: throughput +11.11%; mean completion latency −10.00%.
- Pair 2, B2 versus A2: throughput +10.99%; mean completion latency −9.90%.
- Observed throughput ranges: A 31.683–32.000; B 35.165–35.556 tok/s.
- No listed latency control regresses. Full per-request timings, token gaps and
  percentile summaries remain in the raw client artifacts.

Verdict under the example rule: **speed pass for this workload**. Two runs per
configuration describe observed variability, not a statistical confidence
interval. Concurrent serving and longer prompts are untested by this example.

## Single-request profiling analysis — synthetic, separate captures

Tool: `<profiler and exact version>`. Reproduction commands and capture settings:
`profiles/commands.sh`. Traces: `profiles/A.trace`, `profiles/B.trace`.
Both captures cover one warmed-up 128-input/64-output request with the same
controls as the speed test. Prefill and 63 decode steps have explicit markers.

Required figure in a real report: `profiles/A-vs-B-timeline.png`, linked alongside
the original traces. Arrange A above B, align request start at 0 ms, and use the
same milliseconds-per-pixel scale and equivalent CPU/runtime/GPU tracks. Include
the full request and matching decode zooms; mark prefill/decode boundaries and
box the synchronization/idle regions that change. Keep the axis and event names
readable. Do not stretch the shorter B trace to match A's width.

Example caption (synthetic): "A above, B below; matching warmed-up requests.
Boxed decode gaps total 240 ms in A and 45 ms in B (−195 ms); arrows connect
representative corresponding gaps. Decode phase wall time is 1920 → 1720 ms.
Removed host synchronization calls align with reduced GPU idle gaps; kernel
busy time is nearly unchanged. See linked traces for the remaining decode steps."
This document supplies the figure specification and caption, not a fabricated
profiler screenshot. Contributors must attach the figure from their actual A/B
captures; the following table supplements that figure.

| Profile observation | A | B | B − A |
| --- | ---: | ---: | ---: |
| Prefill phase wall time | 95 ms | 86 ms | −9 ms |
| Decode phase wall time | 1920 ms | 1720 ms | −200 ms |
| Decode GPU idle-gap union within that phase | 240 ms | 45 ms | −195 ms |
| Decode GPU kernel busy-time union | 1680 ms | 1675 ms | −5 ms |
| Decode kernel launches | 12,600 | 12,600 | 0 |
| Redundant host synchronization calls | 63 | 0 | −63 |
| Host/device transfer volume | 0.5 MiB | 0.5 MiB | 0 MiB |

Analysis: trace events place the removed synchronization calls immediately before
the gaps that shrink in B. Decode remains dominant. Most of the observed decode
reduction corresponds to less GPU idle time; kernel busy time, launch counts
and transfers are essentially unchanged. CPU activity and GPU gaps overlap, so
their durations must not be added as independent savings. The prefill reduction
is not explained by this decode change and remains an open observation.

The traces support reduced decode waiting as the main mechanism, consistent
with the roughly 10% completion-latency reduction in unprofiled runs. Profiled
phase durations differ from client timings because capture overhead and timing
boundaries differ; do not substitute them for the speed table or claim exact
causal attribution of every millisecond.

## Memory, preparation and artifacts — synthetic

| Server session | Peak allocated | Peak reserved | Peak occupied KV blocks | Requests/KV blocks after drain | Allocated after cleanup |
| --- | ---: | ---: | ---: | --- | ---: |
| A-first | 4.20 GiB | 4.50 GiB | 12 | 0 / 0 | 0 bytes |
| B | 4.20 GiB | 4.50 GiB | 12 | 0 / 0 | 0 bytes |
| A-last | 4.20 GiB | 4.50 GiB | 12 | 0 / 0 | 0 bytes |

Allocated KV capacity: 512 blocks of 16 tokens in each session; peak occupied
blocks describe useful occupancy, not allocated pool size. Warmed first-request
finite checks and task-owned process cleanup succeed in each session.

Report excluded work separately: preparation/downloads 0 s (existing local
artifacts); process-to-health 12.0/11.9/12.1 s for A-first/B/A-last;
feasibility and warmup durations in `setup.json`; profile durations in
`profiles/metadata.json`. No startup improvement is claimed.

Attach an artifact bundle containing `contract.json`, `environment.json`,
`commands.sh` (exact scheduler/server/client/evaluator commands), `setup.json`,
raw A1/B1/B2/A2 client JSON, A/B accuracy responses and rescoring, correctness
logs, server logs, memory records, profiles and a SHA256 manifest. Paths here
are suggested bundle names, not existing files. A real report must provide
accessible links; missing artifacts leave the associated claim unqualified.

Overall illustrative conclusion: the single-request workload meets its declared
speed, smoke-accuracy and memory gates, with profiling that supports the proposed
mechanism. Broader accuracy and concurrency performance remain unqualified.
