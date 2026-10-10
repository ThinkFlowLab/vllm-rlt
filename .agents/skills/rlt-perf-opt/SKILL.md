---
name: rlt-perf-opt
description: Analyze and optimize vllm-rlt inference performance using reproducible unprofiled benchmarks, paired ops-only/full profiles, source-level attribution, and correctness checks. Use for quantitative PR performance reviews and optimization of looped Transformer inference features.
---

# Looped Transformer Performance Analysis and Optimization

The goal is to answer: how does the feature work, which code or execution stages account for performance changes, how do those changes arise quantitatively, and which further optimizations are worth implementing? Final conclusions must be verifiable from configurations, raw results, traces, and source locations.

This skill covers performance analysis and optimization after feature implementation. Code-style refactoring belongs to a separate workflow; do not mix unrelated refactoring into performance experiments. When the user requests analysis only, deliver measurements and recommendations; implementation changes must stay within the user's authorization.

## 1. Establish Requirements and Understand the Implementation

Before measuring, use the requirements, PR diff, and actual code to explain:

- What problem the feature solves, which metric it is expected to improve, and what costs are acceptable.
- The main request path from arrival to output, including key computations, scheduling, KV state transitions, and synchronization points.
- How loop depth, exit conditions, and the lifecycle of KV and hidden states at each depth participate in that path.
- Whether the implementation meets the requirements and whether the relevant feature combinations, backends, and hardware are actually supported.

A line-by-line walkthrough is unnecessary, but do not infer behavior from a feature name alone. For unsupported combinations, record the current code version, the source location checked or minimal reproduction results, and mark the combination as untested. Do not combine gains from two independent tests and present them as a result for the combined mode.

## 2. Fix a Reproducible Test Contract

First read the project's existing benchmarks, user guides, and profiling interfaces, and prefer reusing existing tools. Put any missing diagnostic scripts in a local experiment directory; do not add a new benchmark framework to the repository by default.

Save three types of entry points: service/engine startup commands, single-request reproduction commands, and batch benchmark commands. Associate each experiment with the following information:

- Repository commit and workspace diff; model version, tokenizer, dtype, and quantization method.
- Hardware count and model, interconnects, and device mapping; driver, runtime, framework, and attention backend versions.
- KV capacity, maximum request count and token budget, chunked prefill, prefix caching, asynchronous scheduling, and CUDA Graph configuration.
- Loop and exit configuration; speculative draft/target depths, draft length, and sampling configuration.
- Data source and version, prompt template, tokenized inputs, output-length policy, and random seeds.
- Concurrency, arrival model, total request count or duration, warmup rules, number of repetitions, timeouts, and failure handling.

State whether the experiment isolates the cost of a particular feature or compares the best tuned configurations of different modes. Do not mix these two types of results in the same performance-gain claim. Explain any configuration differences that cannot be held constant.

Resource fairness: compare aggregate throughput for a PD 1P1D deployment against a conventional inference deployment using the same two GPUs. For single requests, an additional comparison against single-GPU latency is acceptable, but state the resource cost. For speculative versus target-only inference, hold the model, target exit policy, and hardware budget fixed.

### Choose Data and Workloads

Choose based on the mechanism being tested; no particular dataset is mandatory:

- Fixed-token-length benchmarks isolate the effects of input length, concurrency, and scheduling. Retain the actual inputs and token IDs; describe any truncation, repetition, or random synthesis.
- Real datasets validate natural length distributions, speculative acceptance rates, early-exit behavior, and task quality. Fix the sampling method; do not replace the test set because another sample batch yields better gains.
- Performance benchmarks are not accuracy tests. Experiments with random weights or synthetically enlarged models may only support explicitly labeled capacity, performance, and latency studies; they cannot represent real acceptance rates, exit distributions, or task quality.

Optional starting scenarios: 128 input tokens / 64 output tokens / one request; 1024 input tokens / 64 output tokens / 16 simultaneous arrivals; 4096 input tokens / 64 output tokens / fixed-rate arrivals. Adjust to the objective. Short bursts cannot establish steady-state throughput or saturation capacity. High-concurrency conclusions require a load sweep and a sufficiently long stable measurement window.

Record fixed-shape ignore-EOS experiments separately from natural generation. Also separate cold/warm cache conditions, graph compilation/capture, and steady-state execution.

## 3. Establish an Unprofiled Baseline First

Performance numbers must come from independent runs with the **profiler disabled**. Warmup must cover the actual shape, batch, and graph paths. By default, collect at least three valid repetitions and retain each run's results and variability. For noise-sensitive or small gains, check for drift using methods such as interleaved A/B runs.

Record at least:

| Metric | Definition and reporting requirements |
| --- | --- |
| Output throughput | Successfully generated output tokens / an explicitly defined measurement window; also report total output tokens, window duration, and GPU count |
| Request throughput | Successful requests / the same window |
| TTFT | Time from the client sending the request to the first valid output; distinguish server-side queuing and network timing boundaries |
| TPOT | Specify the first-to-last token time difference and denominator used; explain observation limitations for speculative output emitted in batches |
| ITL / chunk gap | Use ITL only when per-token timestamps are available; SSE chunk intervals must not be presented as speculative per-token intervals |
| E2E latency | Time from sending a single request to its completion |
| Distributions and failures | Latency p50/p95/p99, sample counts, and error/timeout/cancellation rates; tail percentiles from small samples are descriptive only |
| Resource usage | Relevant peak device memory, GPU/CPU activity, KV usage, and transfer volume; state the measurement method |

When SLOs apply, also report goodput subject to TTFT/TPOT constraints. Do not silently exclude failed requests. Means, medians of per-run percentiles, and percentiles of pooled samples are different statistics; specify which is being reported.

## 4. Capture ops-only and full Profiles Separately

Independently capture both modes using the same representative request configuration, with fresh sessions and output directories:

| Setting | ops-only | full |
| --- | --- | --- |
| CPU + device operator activity | On | On |
| record_shapes | Off | On |
| profile_memory | Off | On |
| with_stack | Off | On |
| with_flops | Off by default | Off by default; enable only when needed |

Use ops-only to identify operators, launches, copies, synchronization, and device timelines. Use full to further locate shapes, allocations, and call stacks. Full does not mean enabling only shapes or only stacks. On non-CUDA platforms, use the device profiler actually supported by that platform and disclose missing capabilities; do not describe a CPU-only capture as a complete device capture.

### Keep Captures Small and Adapt the Window

Start with the smallest window that covers the mechanism under investigation, usually one complete representative speculative round or decode iteration after warmup. An engine step is not necessarily a complete token iteration; ensure the window includes the relevant draft, verification, commit/rollback, or transfer stages. Increase the window only when the initial trace lacks necessary evidence. Benchmark repetition counts do not determine profiling step counts.

Inspect the first capture's file size, event count, export time, and parsing cost before launching a profiling matrix. If traces become too large, export or analysis takes too long, or capture/export fails, first reduce the number of active steps or narrow the capture to the relevant stage. Keep the request configuration representative, and use matching windows for the baseline and candidate and for ops-only/full. Split distinct stages into separate short captures when needed. Do not repeatedly rerun the same oversized window or broaden the matrix before validating a small capture.

Reduce the window, not the definition of full: keep record_shapes, profile_memory, and with_stack enabled. Preserve enough events to support the conclusion; an incomplete round cannot stand in for the whole round. If reducing the window does not resolve an export error, retain the error and then investigate profiler/runtime compatibility. Do not assume file size caused an error without evidence.

Inspect summaries and selected events before loading an entire large trace. Preserve existing valid captures and reuse them; do not recapture completed evidence merely to obtain smaller files. Report unprofiled A/B results as soon as they are available, with their validation status, rather than withholding all results while profiling or export issues are being resolved.

- Choose a short window containing the suspected bottleneck. Warm up the shapes used in that window first, so initialization, JIT compilation, and first graph capture do not contaminate steady-state conclusions.
- If cold startup is the subject of study, create a separate experiment that retains initialization costs.
- For PD, capture P and D separately. For speculation, cover drafting, verification, and commit/rollback. For combined modes, capture according to the actual execution roles.
- Verify request success, actual step counts, rank/session mappings, export completion, and readability of the full profile's shape, memory, and stack artifacts.
- Disclose missing trace events, incomplete windows, export errors, or missing ranks. Do not interpret missing activity as zero cost.
- Profiling changes CPU scheduling, batch composition, and execution time. Do not claim gains using latency or throughput measured with profiling enabled, and do not interpret full being slower than ops-only as an engine regression.

Retain traces, operator statistics, metadata, capture commands, and corresponding requests. Do not commit large traces directly to the code repository.

## 5. Attribute Changes to Source Code and Execution Flow

Start from the end-to-end difference, then examine: request queuing and batch composition → prefill/loops/decode → host scheduling and device idle gaps → operator hotspots, memory, and transfers. Link each finding to `file:line`, the call path, trace events, and the measurement definition.

Distinguish carefully between:

- Total operator time and device critical-path time. Times that overlap across streams or ranks cannot simply be added to obtain end-to-end duration.
- CPU inclusive and self time. Do not double-count parent and child calls.
- Host waits in `.cpu()`, `.item()`, or synchronization APIs and actual D2H transfer time. A wait may include preceding GPU computation; do not treat the entire wait as an eliminable copy.
- Gaps with no device kernels and their root causes. They may arise from scheduling, queues, synchronization, waiting for input, or missing observations; they do not establish a CPU bottleneck on their own.
- Observed request latency, worker message timestamps, and pure computation or transfer time. Cross-machine timestamps require clock synchronization.
- The union of GPU active intervals and hardware utilization. A trace-derived busy ratio is not SM utilization.

Provide an evidence chain for each major gain or regression:

`configuration change → code/flow change → change in work, event counts, or waits → unprofiled metric difference → added costs and remaining uncertainty`

Where quantification is possible, report both absolute differences and relative changes. If the current data cannot isolate a causal contribution, explicitly label it as a hypothesis and design an A/B test, ablation, or additional counter to validate it. Do not force contributions into an explanation that adds up to 100%.

### Select Additional Analysis by Feature

- **PD:** Examine P queuing/computation, D capacity reservation, KV handoff/transfer/activation, and decode queuing separately. Check whether prefill interference decreases and whether P becomes the throughput bottleneck. Smoother tail decode behavior and aggregate throughput can move in opposite directions.
- **Speculation:** Record the acceptance-rate numerator and denominator, committed tokens per round, draft/verify row counts, depths, and additional KV work. Separate drafting, verification, and commit/rollback. State whether cumulative statistics include warmup and profiled requests; do not present them as the acceptance rate of an individual case.
- **CUDA Graph:** Record capture/replay/fallback counts and reasons. Check request counts, actual token rows, padding, and graph keys. Do not equate a maximum batch parameter directly with request capacity. Also measure the memory and capture costs of expanding graph coverage.
- **Loops and early exit:** Distinguish fixed depth from adaptive exit. Examine exit-depth distributions, per-loop post-processing, prefill/decode state handling, KV/hidden-state reuse, and host-device round trips.
- **PD + speculation:** Test only when the combination is actually supported. Also examine draft/target state after handoff, temporary KV capacity, rollback on rejection, and output commit boundaries.

## 6. Drive Optimization Through Experiments

Prioritize candidates by critical-path cost, expected gain, complexity, and correctness risk. Distinguish proven bottlenecks, hypotheses awaiting validation, and measured optimizations.

Where possible, change one interpretable factor per iteration and record the code/configuration differences before and after optimization. First examine existing capabilities: backends, batch/token budgets, graph coverage, redundant synchronization/copies, repeated computation, and metadata preparation. Introduce new kernels or complex scheduling only when supported by evidence.

Once implementation is authorized, iterate: implement → check correctness → run unprofiled A/B measurements under the same definitions → reprofile when necessary → decide whether to retain, revert, or continue. Analysis-only tasks may deliver concrete change proposals and validation experiments; do not expand them into implementation without authorization.

Do not change request semantics, exit policies, or accuracy for performance without disclosure. Report gains from shorter outputs, fewer loops, or different sampling algorithms separately as quality/performance tradeoffs.

## 7. Correctness and Quality Gates

Deliver correctness results alongside performance results:

- Verify request completion and whether input/output lengths, token commits, and stopping behavior satisfy the test contract.
- For optimizations that preserve exact semantics, compare token IDs using identical inputs and sampling configurations; add logits/state checks where applicable.
- Do not require token-by-token equality under random sampling as proof of distributional correctness; choose tests appropriate to the actual semantics. Likewise, do not dismiss greedy mismatches as “floating-point error” without investigation. Check repeated runs, batch/kernel changes, and the first divergence.
- For features allowed to change outputs, such as early exit or post-training policies, evaluate real task metrics, exit distributions, and an explicit quality budget.
- Performance tests on a corpus slice establish behavior only for the covered requests; they do not constitute a complete task-accuracy evaluation.

If unexplained correctness differences remain, performance numbers may be retained as diagnostic results, but the optimization must not be declared to have passed acceptance.

## 8. Deliver a Report and Reusable Configurations

The report must let readers understand the conclusions without rerunning the experiments and reproduce them when needed. Include:

1. Verified gains/regressions, scope of applicability, unsupported combinations, and correctness status.
2. Environment, code/model versions, inputs and workload, comparison baselines, and statistical methods.
3. An unprofiled results table: absolute values, relative changes, variability across repetitions, and resource costs.
4. ops-only/full evidence and source attribution: specific locations, work reduced or added, measured values, and limits of inference.
5. A/B results for completed optimizations, unresolved issues, and prioritized next experiments.
6. The three types of reproduction commands, raw results, and trace paths.

Describe the best result within the current hardware, model, and tested cases only as “best among tested configurations” or “the most suitable tradeoff under current conditions.” State which dimensions were not searched; do not claim a global optimum.

After correctness checks pass and performance conclusions stabilize, collect the configurations, commands, metrics, versions, and applicability limits needed for a recipe. If a recipe already exists, update it using the repository's format. Otherwise, first save reusable materials locally; publication must follow the current task's authorization. New features, parameters, or usage patterns also require updates to the relevant documentation and README.

## Methodology Sources and Scope

This skill draws on the reproducible baselines, profiling-based attribution, and iterative validation approach of [vllm-omni diffusion-perf-opt](https://github.com/vllm-project/vllm-omni/tree/main/.claude/skills/diffusion-perf-opt), reorganized for looped Transformers. It does not incorporate diffusion-specific parallel-strategy search. The definitions of ops-only and full in this skill take precedence.
