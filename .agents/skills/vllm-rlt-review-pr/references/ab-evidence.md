# Accuracy and speed A/B evidence

## Choose checks from the PR's impact

Use only the sections relevant to the changed behavior and claims. This is a
review reference, not a requirement to run the full benchmark for every PR.
Assess the diff and callers rather than relying only on the PR's category.

For a concrete contributor report, see the [illustrative A/B output](ab-example.md).
Its numbers are synthetic, not benchmark evidence or universal thresholds.

| PR impact | Appropriate validation |
| --- | --- |
| Documentation, review skills, or tests with no runtime or measurement change | Review the content and relevant tests; no accuracy/speed A/B required. |
| Interface or configuration change outside inference/timing paths | Check compatibility, defaults and affected user flows; no speedup requirement. If it changes precision, generation controls or timed work, assess that impact separately. |
| Correctness fix or execution refactor | Validate the changed invariants and outputs. Use a targeted accuracy or performance regression check where the affected path warrants it; improvement is not required. |
| Optimization or an explicit speed claim | Matched BF16 speed A/B plus the relevant correctness/accuracy and memory evidence. Apply the declared improvement and control-regression gates. |
| Arithmetic, sampling, exit policy or model-quality change | Numerical/decision checks and relevant paired accuracy evaluation. Add speed comparisons only for performance impact or claims. |
| Benchmark or report change | Validate protocol, coverage, accounting and verdict logic. Use preserved/synthetic artifacts when sufficient; new GPU runs are needed only to resolve execution or measurement questions. |

Briefly state why a selected check applies. When a dimension is unaffected,
mark it not applicable with a reason; do not call it passed or block the PR
merely because there is no speed measurement. A claim that performance is
unaffected still needs scrutiny when the diff changes a hot path or timed work.

## What to monitor when A/B is applicable

A is the frozen intended base; B is that base plus the PR change. Both use BF16.
For a stacked PR, use its immediate prerequisite stack as A.

| Dimension | Evidence to inspect |
| --- | --- |
| Accuracy | Correct counts, paired losses/gains, answer disagreements, parse failures and length-limited outputs. |
| Speed | Output tokens/s, TTFT, TPOT, token gaps and request completion latency; both measured observations and their ranges. |
| Reliability | Failed requests, timeouts, incomplete streams, missing/duplicate outputs and nonfinite values. |
| Memory | Peak allocated/reserved GPU bytes, occupied/reserved KV, request cleanup and task-owned memory after shutdown. |
| Comparison controls | Source/model/data/client pins, actual dtype and accumulation, exact GPU, CPU/NUMA affinity, prompts, output lengths, loop policy, capacity, cache state and measurement order. |
| Changed behavior | Relevant KV ownership, request isolation, exit and RNG invariants; existing-client compatibility for interface changes. |

For quick iteration, a fixed ten-question subset of GSM8K-87 can serve as the
accuracy smoke screen when accuracy validation is relevant. Freeze the subset
before either run, select by source IDs independently of answers or observed
timings, and preserve prompts, BF16 generation and scoring controls. Compare
native A/B on the same ten questions with no net lost correct answer, reporting
paired losses/gains even when they cancel. Do not reuse or prorate the 59/87 HF
floor: a subset needs its own measured reference if an absolute floor is claimed.
Label the result as a smoke screen; it does not qualify the full 87-question
case or broader quality. Use the full case when the PR's accuracy impact or
claim requires its coverage. Neither case is mandatory for unaffected PRs.

Separate primary metrics from controls before execution. The earlier proposed
GSM8K-87 screen used a 59/87 floor and at most one percentage point loss against
A (therefore no net lost answer on 87 questions). The proposed optimization
screen used at least 1% primary throughput improvement in both pairs, at most
2% control regression, and overlapping ranges as inconclusive. These are
examples for that frozen protocol, not universal PR acceptance thresholds.
Use the reviewed experiment's actual contract; do not impose these numbers
on correctness-only changes or a different dataset/workload.

## Single-request profiling analysis

For single-request (concurrency 1) speed A/B tests, require the contributor to
provide profiling analysis for both A and B. This is a repository requirement;
it does not make performance testing mandatory for otherwise unaffected PRs.

- Capture comparable warmed-up requests with the same BF16 workload and frozen
  controls as the speed test. Record profiler/tool versions, commands, capture
  boundaries and links to traces or exported reports.
- Include an annotated timeline figure with A (before) above B (after), or side
  by side. Align the same request/phase start, use the same time scale and units,
  and show equivalent CPU thread, runtime and GPU stream tracks. Keep labels and
  the time axis readable; do not resize each trace independently to fill its panel.
  Show the complete request with prefill/decode boundaries, plus matching zoomed
  regions when needed to expose the bottleneck. Use boxes/arrows to highlight
  removed waits, idle gaps, launches or other changed work, with before/after
  durations and an explanation. Link the original traces behind the figure.
  A summary table alone does not satisfy this visual comparison requirement;
  a screenshot alone does not establish causality or an unprofiled speedup.
- Compare prefill and decode separately. Identify the dominant CPU/GPU work,
  kernel time and launch counts, synchronization, transfers and GPU idle gaps
  where relevant to the change. Report absolute before/after values and deltas,
  not only percentages or a screenshot of the fastest kernel.
- Explain how the changed code affects the observed bottleneck and how that
  accounts for the end-to-end result, including regressions or unchanged speed.
  Distinguish trace-supported conclusions from hypotheses and unexplained time.
- Collect profiles in separate diagnostic runs; profiler overhead must not enter
  measured A/B timings. A kernel improvement alone does not establish a request
  speedup. If profiling is missing, mark the required performance analysis
  incomplete rather than inferring that the implementation is incorrect.

## Validate the evidence

Verify that the intended implementation actually ran on the measured workload.
Trace flags/defaults through dispatch to the executed path, using existing
traces, counters, logs or targeted tests. Accepted arguments, successful output
and capture/setup logs alone do not establish fast-path execution. Distinguish
a correct fallback from evidence for the claimed optimization. Check warmup
covers the measured shapes, batch sizes and execution modes; first-use compile,
capture or allocation costs belong outside warm timing unless explicitly part
of the declared experiment. Do not require new telemetry when existing evidence
is sufficient.

The standard accuracy and speed A/B tests both use BF16 weights, ordinary
activations and KV. Check explicit configuration and actual parameter/cache
tensor dtypes, not just a CLI default or declared metadata. Full-model FP32 is
only an explicitly requested diagnostic; do not select it as a fallback baseline
because an older harness requires it. Retain justified FP32 reductions and
intermediates within BF16 inference and record accumulation flags separately.
Historical FP32 measurements or numerical failures remain scoped to their
original contracts; neither silently qualifies nor automatically blocks a new
BF16 experiment.

When reporting current main's accuracy or speed, identify the execution SHA,
dtype, workload and timing boundary for each number. A stored regression score,
a full-dataset result from another branch, and a measurement of the latest main
are distinct evidence. Merging serving support or accuracy tests does not rerun
the benchmark. If the current BF16 baseline has not been measured, say so; do
not substitute old FP32 throughput. Consult the current fixture and frozen
contract for accuracy floors and performance gates instead of hard-coding a
historical score, merge SHA or local proposed threshold into the review.

- Metadata/data-movement-only changes preserving arithmetic require exact
  selected state, populated KV, tokens/exits and RNG behavior. Arithmetic
  changes need independently justified, predeclared numerical/decision bounds;
  do not demand blanket bitwise equality with a different arithmetic path or
  invent near-tie exceptions after seeing failures.
- For accuracy, verify model/dataset/tokenizer/package pins, exact question and
  prompt IDs, four-loop/greedy or declared adaptive policy, demonstrations,
  stopping/token limits and strict scoring. Check raw-response rescoring,
  missing/duplicate questions, paired losses/gains, parse/length failures and
  the actual frozen floor/loss budget. Equal total scores can hide different
  failures. A selected regression cohort is not a held-out quality estimate.
- Fixed-depth accuracy and synthetic token/exit replay do not qualify live
  adaptive quality. Compare adaptive history against the same last-exited
  semantics; official/full-depth recomputation can have different KV history.
- Speed A/B must isolate the PR change, use matching workloads and declared
  arithmetic, exact devices, CPU/NUMA placement, package/client code, capacity
  and cache controls. Check the actual measured order against the frozen plan;
  ABBA means A1/B1/B2/A2, paired by repetition. Disclose any restarts and warmups.
- Exclude feasibility, first-request diagnostics, warmups and profiling from
  measured timings. Require complete successful requests and correct token/work
  accounting; failed, truncated or selectively resumed runs cannot enter a
  passing comparison. Keep original failures and exhausted budgets visible.
- Report both observations and variability under the declared acceptance rule.
  Separate aggregate throughput from per-request latency, HTTP/client timings
  from engine-only timings, and allocation bytes from useful KV occupancy.
  Changed output lengths can distort accuracy-run timing; those durations alone
  do not establish a controlled speedup. Missing memory/capture evidence remains
  unqualified where that claim requires it.
- Apply thresholds frozen for this experiment; do not impose unmerged proposals
  or a speedup requirement on correctness-only PRs. A reproducibly invalid speed
  claim is actionable; missing evidence is otherwise a qualification limitation,
  not proof that the implementation is wrong.
