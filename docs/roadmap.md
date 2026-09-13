# Roadmap: faster Ouro inference first

The next objective is measurable latency and throughput improvement for
**ByteDance/Ouro-1.4B on one GPU**, while preserving the chosen numerical and
last-exited KV contracts. Keep the vLLM-style separation between the engine,
scheduler, runner, model, and attention backend. Loop-level scheduling and
depth-aware cache management remain owned by vllm-lt.

This roadmap records the priority agreed on 2026-09-10. Optimization order
after the first baseline is conditional on profiling. Set timing and
performance targets using the measurements from each completed milestone.

## Starting point

The preview at `fd3e45b71a9cb53421fdbb08ed2474ece37f3f92` loads real weights,
implements full-depth chunked prefill, synchronous refill/no-refill decoding,
last-exited paged KV, and Torch/Triton attention. CPU tests and the recorded GPU
kernel checks passed. The bounded real-checkpoint FP32 comparison passed at all
four loops, with maximum absolute logit error `6.2943e-5`.

BF16 failed the predeclared logit tolerances. Matching outputs and exit depths
on three short prompts does not qualify BF16 generally. There is no measured
throughput improvement or task-quality result yet. See the
[validation record](validation.md) for the exact coverage and original failure.

## Milestones and dependencies

| Milestone | Deliverable | Completion criterion |
| --- | --- | --- |
| M0 — preview | Current synchronous engine and validation record | Implemented; draft PR remains under review, with BF16 limitations recorded. |
| M1 — baseline, next | Reproducible benchmark, bounded profiler captures, and bottleneck report | Correct timing and replay accounting, raw results, variability, and a ranked next optimization. No speedup required. |
| Q1 — numerical qualification, parallel with M1 | Expanded FP32/BF16 diagnostics and independent adaptive-history oracle | A documented numerical acceptance decision; BF16 remains experimental until it passes justified criteria. |
| M2 — reduce host overhead | Reuse batch metadata across layers; batch result transfers where profiling supports it | Preserve outputs, gates, cache isolation, and sampling state; demonstrate the declared end-to-end benefit. |
| M3 — recurrent CUDA graphs | Safe inactive rows, persistent buffers, then capture one recurrent traversal | Eager/graph equivalence, safe slot reuse, bounded graph memory, and measured benefit including routing overhead. |
| M4 — attention and KV efficiency | One measured attention or KV bottleneck per PR | Improve the chosen workload within declared latency, correctness, and memory guardrails. |
| Q2 — quality and comparison | Bounded task-quality screen and equivalent cached external baseline | State which dtype/depth policy is qualified and which speed/quality comparisons are supported. |
| M5 — asynchronous routing, later | Evaluate the paper's lookahead method only after the synchronous baseline is fast | Available or separately trained/calibrated gate, held-out quality evidence, and benefit over the optimized synchronous engine. |

M1 and Q1 can proceed independently using FP32 as the numerical baseline.
M3 uses scheduler-sized persistent buckets and captures one recurrent traversal.
It reduces launch overhead while retaining actual gate readback and host routing
after each traversal; it does not implement M5 lookahead. Observed throughput
ratios are workload measurements, not a theoretical ceiling. See
[the CUDA graph design and qualification boundaries](m3-cuda-graphs.md).

M2–M4 retain synchronous gating initially. Attention work can precede graphs if
M1 identifies it as the larger cost. Q2 depends on Q1 for BF16 promotion and is
required before publishing adaptive speed/quality claims. Serving, more models,
distributed inference, quantization, speculative decoding, and prefix sharing
are deferred until the single-model performance case is established.

## M1: first benchmark and profiler PR

Add a reusable offline benchmark command, frozen workload fixtures, a JSON
result schema, and a report script. Keep profiling optional and outside timed
runs. Reuse a loaded model per prepared configuration and clear only request
state between workloads. Preserve commands, raw results, and manifest hashes;
commit compact reports and reproducible fixtures.

Start with this bounded suite, using the pinned checkpoint/tokenizer revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`:

| Workload | Requests | Prompt / output tokens | Policy and purpose |
| --- | --- | --- | --- |
| W1 | 1 | 128 / 64 | Fixed four loops; single-request decode latency. |
| W2 | 8, simultaneous | 128 / 64 each | Fixed four loops; batched throughput. |
| W3 | 1 | 512 / 32 | Fixed four loops; prefill cost with chunk size 128. |
| W4 | 8, simultaneous | Alternating 64 / 32 and 128 / 64 | Frozen output IDs and mixed exit depths 2/3/4; refill versus no-refill. |
| W5 | Same requests as W4 | Same lengths and output IDs | Replay with depth four everywhere; uniform-depth control. |

Use frozen token IDs, greedy selection, `ignore_eos=True`, FP32, Triton
attention, full-depth prefill, and one device. Initial configuration:
`block_size=16`, `num_blocks=1024`, `max_num_seqs=8`,
`max_num_batched_tokens=128`, and `min_coda_batch_size=1`. W1–W3 use refill.
Run W4 and W5 under both refill and no-refill; within each workload, change
only scheduling mode. Confirm capacity from the actual model configuration in
the CPU probe: this pool alone uses 6 GiB
in FP32, in addition to weights, activations, and runtime allocations.

The replay path must execute the same declared model, gate, coda, cache, and
result-transfer work in both modes, then impose the frozen token/exit trace
at a documented boundary. The first output always records prefill depth four;
mixed depths apply only to subsequent outputs. Validate per-request executed
depths and token histories. Batch shapes and stage invocation counts may differ
as a consequence of scheduling. Replay measures scheduling under a controlled workload;
separate live-gate generation is needed for real adaptive behavior and quality.

Record these metrics with explicit denominators and timing boundaries:

- Wall time from request submission to last output, generated tokens/second,
  per-request time to first token (TTFT), completion latency, mean time per
  subsequent output token (TPOT), and individual token gaps. Record queue and
  prefill time separately. Engine steps return cumulative outputs; record only
  newly emitted tokens once. Eight requests provide descriptive distributions,
  not a stable serving-tail estimate.
- Prefill and decode execution separately. The first output is predicted by
  full-depth prefill; exclude it from adaptive decode depth averages. Report
  the executed loop-depth distribution and recurrent batch occupancy.
- CPU scheduling, metadata preparation, launch and gate/result-transfer costs;
  GPU stage spans, attention, LM head, and KV write/finalization work. Use
  profiler traces for attribution and GPU gaps. Avoid double-counting nested
  spans or adding per-stage synchronizations to measured runs.
- Peak allocated/reserved GPU memory, request-reserved and populated KV
  blocks, copy bytes, and zero retained request blocks after completion.

Use wall-clock timestamps and CUDA events with synchronization at experiment
boundaries. Take at most two separate profiler captures initially: W1 and W4
(refill), with a bounded window of 16 decode outputs. The first report should
identify the largest costs and choose the next PR from evidence.

### Run contract

Before execution, write the hypothesis, isolated variable, fixed controls,
primary metric, success criterion, and stop conditions into the manifest.
Freeze repository SHA and dirty-file hashes, model/tokenizer and fixture
revisions, exact GPU IDs, hardware, software, arithmetic flags, thread count,
NUMA placement, cache state, and all engine arguments.

Run a CPU-only configuration/fixture/replay probe first, then one excluded GPU
feasibility suite. Preflight memory before execution. After feasibility, allow
one bounded warmup per configuration/workload and two measured iterations per
configuration/workload. Order paired comparisons A–B, then B–A on the same
exact device IDs. Separate preparation, downloads, compilation, and warmups
from execution; preserve shared caches and reuse the prepared environment.

On the shared host, verify the scheduler and its status, select available exact
IDs, and execute all device work through `gpu run --gpu-ids ... --timeout ...
--note ... -- ...`, leaving visibility assignment to the scheduler. Start with
a two-hour reservation cap, ten minutes per timed workload, and a finite
scheduler-step guard derived from the fixtures. The seven workload/configuration
cells require fourteen measured runs and seven warmups, plus the feasibility
suite and two profiler captures. The overall cap takes precedence over per-run
caps; an unfinished suite is partial, with no automatic extension. Release only
task-owned resources. Infeasibility, timeout, nonfinite values, invalid replay, or cleanup
failure stops the affected experiment and preserves partial results. Revise
and record the contract before a new experiment; do not silently shrink work
or add trials.

The harness succeeds when it produces reproducible, correctly accounted
baseline evidence. For each later optimization, choose a primary latency or
throughput target and maximum regressions on the frozen control workloads
before its runs. Report both measured values and their range; two runs are a
bounded screen, not a strong statistical claim. Results within observed
variation remain inconclusive at the run limit.

## Q1/Q2: qualification alongside performance work

Separate implementation fidelity, adaptive KV semantics, and task quality:

1. **Arithmetic and model fidelity.** Expand the independent dense comparison
   to 16 deterministic text fixtures across prompt lengths 16/64/128/256 and
   short teacher-forced decode histories. Inspect every loop, layer boundaries,
   hidden/logit RMS and maximum error, top-1 agreement, top-two margins, gate
   logits, and cumulative probability distance from the exit threshold.
   Stream comparisons to bound memory. Validate fixed-depth behavior against
   the pinned official model as well as the local independent equations.
2. **Adaptive history.** Add an independent serial incremental oracle for
   last-exited KV. Cover adjacent tokens with different exit depths, a shallow
   exit followed by deeper execution, chunk/block boundaries, mixed batches,
   cancellation, and reuse. Full-depth dense recomputation and the official
   model's adaptive output selection do not reproduce this cache history.
3. **Quality screen.** Predeclare a pinned, deterministic 64-example GSM8K
   subset, prompt template, exact answer parser, maximum 512 prompt tokens and
   256 generated tokens, and disjoint feasibility examples. Record exclusions,
   truncation, paired answer accuracy, disagreements, realized depths, and
   uncertainty. Compare fixed-depth FP32 with fixed-depth BF16 first, then
   fixed-depth with adaptive BF16 at threshold 0.7 and minimum two loops if
   BF16 is qualified. This is an initial regression screen; broader quality
   claims need a separately budgeted evaluation. Do not tune the gate on the
   evaluation examples.

Retain the original BF16 failure at `atol=0.25, rtol=0.02`. Any revised tolerance
needs independent justification and must be declared before the new comparison;
matching a few greedy continuations is insufficient. Decide arithmetic
acceptance and task-quality acceptance separately. Predeclare a maximum
acceptable quality loss and use a paired uncertainty interval; if the small
screen cannot exclude unacceptable loss, report it as inconclusive and retain
experimental status. Deterministic quality checks use one pass/configuration,
explicitly overriding the performance repetition budget.

For a claim of faster Ouro inference than the released implementation, prepare
a pinned external baseline using cached incremental generation, equivalent
fixed-depth policy, dtype, token histories/lengths, and timing boundaries.
Verify its fixed-depth outputs first. The local dense oracle computes all
positions and depths and is not a generation speed baseline. Compare adaptive
policies separately with their actual quality and KV differences disclosed.

## M2/M3: runtime work in small PRs

These are candidates from source inspection, not measured bottlenecks yet.
Keep each behavioral change independently reviewable and benchmarkable.

| PR-sized change | Dependency and acceptance |
| --- | --- |
| Prepare attention metadata once per recurrent traversal | Reuse positions, depth-specific block tables, lengths, and write indices across the 24 physical layers; prepare depth-specific metadata for each traversal within a prefill stage. Preserve validation and per-layer KV initialization checks. Test fragmentation, partial pages, mixed depths, and cancellation/reuse. Measure metadata work scaling with traversals rather than layers. |
| Batch token-result transfers | Replace per-request sampling readbacks with one batch transfer while preserving each request's RNG state and draw order. Removing unused gate readback in explicitly fixed-depth batches is a separate change. Gates before the minimum adaptive exit depth still contribute to cumulative probability. |
| Define inactive rows and persistent buffers | Mask attention, KV writes, hidden-state updates, and output routing; test zero/one live row and bucket boundaries. Zero-length rows currently lack a defined finite attention result. Keep stable pointers and explicit slot ownership; padded rows must not affect KV or RNG. |
| Capture one recurrent traversal | After buffer and masking correctness, capture all shared physical layers plus the gate tensor for bounded batch/context-table buckets. Read the actual gate after replay and retain eager fallback. Test graph/eager equivalence, mixed depths, cancellation, slot reuse, and completion before KV release. Record capture cost, memory, hit rate, and end-to-end benefit. |

Graphs can reduce launch overhead while retaining synchronous scheduling.
They do not supply lookahead decisions or remove the host routing dependency.
Capture prefill or stochastic coda only if later profiling justifies a separate
PR. Stop expanding graph buckets when the predeclared memory/capture budget is
reached or representative workloads show no benefit.

## M4: profile-selected attention and KV work

Choose one attention target per PR. If prefill dominates, add a query-tiled
causal kernel that reuses context across prompt rows and preserves each row's
position and depth. If decode dominates, start with a bounded tile/warp search;
consider split-context reduction only with evidence from small batches and
long contexts. Ouro has 16 query and 16 KV heads, so generic GQA specialization
is not the first model-specific opportunity. Retain numerical tests and require
an end-to-end benefit within the declared regression budget.

Treat KV improvements as three separate questions:

| Work | Contract and decision gate |
| --- | --- |
| Reduce skipped-depth propagation cost | First consider batching exits into fewer copy launches. Preserve every layer's final computed KV at each skipped depth before dependent execution. Compare complete populated prefixes for mixed exits, neighboring tokens, page boundaries, and reuse. Stop if copy work is negligible. |
| Avoid copies through depth indirection | Define token-level source-depth addressing and ownership before implementation. Previously exited tokens must resolve each skipped depth to their last computed state, while shallower states remain intact. Adjacent tokens can exit at different depths, so whole-page aliasing is generally invalid. Measure mapping overhead against copy savings. |
| Improve admission/capacity | Specify incremental growth, atomic allocation, and a progress mechanism such as protected capacity or preemption/recomputation before removing full reservation. Test the case where every active request needs another page and the pool is exhausted. Require bounded progress or explicit rejection, no starvation, and correct cancellation/reuse. Compare useful completed work, recomputation, and tail latency under the same fixed pool. |

The current pool allocates its full physical tensors at construction. Avoiding
copies alone does not reduce allocated VRAM. Report allocated bytes, occupied
physical pages, reserved pages, and admitted capacity separately. Actual sharing
requires a physical-page occupancy and future-write policy, not only logical
depth indirection. Retain the conservative allocator as the correctness baseline
while qualifying a new policy.

## M5 and later scope

The [paper](https://arxiv.org/pdf/2608.09444v1) uses a separately distilled
lookahead gate for Ouro. Its performance experiments replay outputs/exits with
shared KV, which differs from this engine's live-gate, last-exited policy. The
[inspected author repository](https://github.com/kschwethelm/continuous-depth-batching/tree/260ca350cc35bb579879f40fee47c077a2c244ca)
contains no implementation or lookahead checkpoint. See
[source notes](paper-notes.md) for the pinned evidence.

Recheck gate availability when this milestone starts. If unavailable, write a
separate data, training, and evaluation proposal before scheduling training.
Evaluate routing overlap and held-out quality against the optimized synchronous
engine; delaying the stock gate is not the paper's lookahead method. Keep the
synchronous path available until the asynchronous path passes its own criteria.

After useful single-model performance is established, add an asynchronous
request frontend, backpressure, cancellation, and serving metrics. Validate the
first real inference after readiness; report preparation separately from
process-to-readiness, reuse a server per configuration, and keep health tied to
actual readiness. An HTTP compatibility layer and multi-device/model support
follow demonstrated needs rather than blocking the Ouro runtime work.

## Immediate backlog

1. **Next PR:** M1 benchmark/profiler harness and FP32 baseline report.
2. **Parallel PR:** Q1 BF16 diagnosis and serial last-exited oracle; define the
   numerical acceptance contract before promoting BF16.
3. **Following PR:** the highest-impact measured runtime change, with metadata
   reuse the initial candidate. Choose attention first if the profile warrants it.

Each completed milestone updates this roadmap with its commit, artifact links,
acceptance outcome, and the next evidence-backed decision. New workload families
such as long contexts, cache pressure, and staggered arrivals receive separate
budgets before execution. The initial short-context suite supports claims only
within its measured envelope.
