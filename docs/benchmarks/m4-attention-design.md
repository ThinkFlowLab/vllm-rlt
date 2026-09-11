# M4 attention tile qualification

The candidate changes one production launch constant: Triton attention `BLOCK_T=32` on A becomes `BLOCK_T=64` on B; `num_warps=4` and all other attention arithmetic remain unchanged. The accepted [PR14](https://github.com/hsliuustc0106/vllm-lt/pull/14) runtime supplies the common baseline. Benchmark inference uses compact synchronous execution with the private persistent option disabled. The unqualified M3 capture implementation is outside this baseline.

**Status: experiment complete; candidate failed performance acceptance.** The [result](m4-attention-20260911.md) records the unchanged protocol and actual failed W1 gate. The [operative pre-run contract](m4-attention-contract-20260911.md) froze source, input, environment, device, affinity and output-path identities before execution.

## Selection and scope

The selected [M4 issue](https://github.com/hsliuustc0106/vllm-lt/issues/7) candidate is the finite decode-kernel configuration change. The constant also affects prefill and masked/persistent attention; those consequences belong to B and receive correctness coverage. There is no autotuner, additional tile, split-context reduction, dtype change, cache-layout change, graph capture or asynchronous scheduling.

[M1](m1-20260910.md) recorded 24.670 ms of W1 attention kernels over 16 subsequent outputs, about 1.542 ms/output, alongside roughly 56.38 ms measured TPOT. Different instrumentation and the older metadata path prevent an end-to-end speed prediction. This supports testing a modest predeclared 1% target; it does not predict that the target will pass. Both configurations are measured anew on the accepted runtime described in [M2](m2-20260911.md).

AC-M4-04 finalization, AC-M4-05 indirection and AC-M4-06 incremental admission are **N/A, unselected**. M1's finalization device work was small, and its broad host scope included benchmark observer scans. Indirection and incremental admission would change different addressing or progress policies. This PR establishes no copy-elimination, admission-capacity or physical-memory saving. Common correctness, memory-accounting, measurement and evidence requirements still apply.

## Implementation boundary

The production diff is one constant. The common qualification harness covers experiment planning, execution, numerical/held evidence and offline reporting; that code is separate from the inference implementation.

| Module | Contract |
| --- | --- |
| `benchmarks/m4_attention_schema.py` | Strict M4 identities, exact source difference, frozen controls, 139-row order and byte estimates. |
| `benchmarks/m4_attention.py` | Eight workers, first-request feasibility, deadlines, acknowledgments, cap checks and cleanup. Reuses the existing model loader and `run_loaded_rows`. |
| `benchmarks/m4_attention_report.py` | Offline raw-event, chronology, prerequisite, pair, memory/setup and profile checks. |
| `validation/m4_attention.py` | `build_numerical_plan`, `validate_numerical_plan`, `run_numerical_rows(..., after_case=...)`, `audit_numerical`. Reuses the existing numerical executor and typed-payload audits. |
| `validation/m4_attention_held.py` | `build_held_plan`, `validate_held_plan`, `run_held_row`, `audit_held`. Composes finite direct inputs with the genuine existing lifecycle executor. |

The parent embeds distinct `numerical` and `held` subplans. The held runner receives the full frozen nested row resolved from the outer execution ID. Outer A/B means tile/source; inner lifecycle A/B means allocating/persistent storage, with separate directories. Existing M2 and M3 contracts retain their original meanings. No module-global patching or fabricated old full-suite verdict substitutes for M4 evidence.

## Finite execution matrix

| Worker | Declared work | Rows |
| --- | --- | ---: |
| N-A | 7 feasibility, 15 direct kernels, 2 lifecycle, 16 numerical | 40 |
| N-B | 7 feasibility, 15 direct kernels, 2 lifecycle, 11 numerical | 35 |
| A1, B1, B2, A2 | Each: 7 warmups and 7 measured cells | 56 |
| P-A, P-B | Each: 2 diagnostic warmups and 2 profiles | 8 |
| Total | Eight fresh workers, eight model loads | 139 |

N-A and N-B begin inference with W1 feasibility. The four timing workers begin with W1 warmup, and profile workers with W1 diagnostic warmup. A prerequisites pass before N-B; both sides and their raw prerequisite audit pass before A1. Feasibility, warmup, load/setup, compilation and profiles are excluded from measured comparisons. No hidden additional inference, replacement row or extra sample is permitted.

The 27 numerical cases contain five independent dense oracles and 11 native cases per source: four Torch serial, four Triton serial, refill/no-refill packed cases and one live-gate case. Fixtures are `Q1-L16-F0`, `Q1-L256-F0`, `Q1-L64-F2` and `Q1-L128-F3`, with nine predictions and unchanged forced histories or declared live routing. There are 34 oracle comparisons and 17 A/B comparisons, totaling 51 streams.

The 30 direct evaluations use 15 layouts per source. They retain the 13 inactive-row layouts and add lengths 63/64/65/129 plus 128 queries sharing one fragmented causal prefix with lengths 1–128. Geometry is FP32, 16 query/KV heads, dimension 128 and 16-token pages. Frozen inputs include poisoned inactive rows, invalid inactive addresses and a strided target layer in an 80-MiB two-layer pool. Each evaluation calls physical and compact attention once: 60 wrappers, 52 nonempty device launches. All 40 pool chunks are checked before and after, totaling 2,400 guard checks.

Four genuine Triton lifecycle evaluations select allocating and persistent storage under each tile. Each retains the unchanged seed plus 11 actions, 173 typed records and 480 guard checks. They cover shrinking/reordering, recycled ownership, held coda/RNG state, positions 510/511/512, fallback, empty dispatch, stable persistent pointers and cleanup. Totals are 692 lifecycle tensor records and 1,920 guards. These visits qualify shared kernel correctness; the performance path remains compact.

## Numerical and measurement gates

All model comparisons retain Q1 FP32 final-logit `atol=0.001`, `rtol=0.0001`, exact actual top-1/token and exit/history agreement, and finite observed tensors. The four unchanged Torch A/B streams require exact tensors. The 13 Triton A/B streams use fidelity because the tile can change reduction order. Layer/hidden/gate/KV differences are diagnostic, without invented tolerances. Full layer observations and final populated KV coverage remain required. A live divergence fails at its matched prefix; subsequent incompatible states stay labeled incomparable.

Direct attention must satisfy predeclared `atol=rtol=2e-5` against independent CPU FP64 dense attention on the exact moderate-valued inputs. Same-tile compact/physical active outputs, inactive positive zeros and whole-pool bytes must be exact. Within each tile, allocating/persistent lifecycle tensors are exact; across tiles, attention uses the same held bound and other computed values remain finite diagnostics. This does not qualify BF16, arbitrary input magnitudes or additional architectures.

W1-refill B/A throughput must be at least **1.01 in both pairs**. Each of the six other cells must be at least **0.95 in each pair**. Both peak-allocated and peak-reserved B-minus-A increases must be at most **64 MiB**, and engine-setup increase at most **100 ms**. W1 also requires `min(B) > max(A)`. Raw values and ranges remain visible; averages cannot rescue a failed pair, and unresolved variation is inconclusive.

Matched W1/W4 compact-prepared profiles must preserve the declared work and 16-subsequent-output windows, with actual CUDA kernel/copy attribution and separate prefill/decode evidence. CPU and GPU annotation categories are distinguished. Observer page scans are benchmark overhead. Profile durations alone do not establish end-to-end savings, and generic kernel names alone do not identify the tile.

## Resource, completion and failure contract

Each full case has a **600-second** deadline and the controller **7,200 seconds**; the earlier limit wins. Markers cover validation, allocation, execution, export, byte checks and cleanup. Numerical markers remain active through the parent callback. Worker lifetime records additionally cover held/benchmark return and acknowledgment. Missing observations or completion proof cannot pass.

The total artifact cap is **12 GiB**, including at most 4 GiB of profiles (2 GiB each), 312 MiB of held evidence and 1 GiB of auxiliary artifacts. Numerical storage preserves the 8-GiB cumulative-write, 2-GiB group and 256-MiB dump limits. Typed indexes are capped at 1,024 bytes/record, comparisons at 2,048 bytes/record, with 64-MiB matching and explicit estimated totals. The held plan freezes 752 typed records and 4,320 guards. A memory-access checker is unavailable: zero checker executions and no checker-coverage claim.

Keep the same reserved device/UUID, prepared environment, NUMA/thread binding, arithmetic, workload and cache/admission controls across A/B. Fresh case engines preserve worker allocator state; load/setup are reported separately. Requests and KV ownership must be fully released, held cache/buffer references settled, and final worker CUDA allocated/reserved bytes zero after task-owned model/workspace cleanup. Shared caches are not dropped.

Any source/control, numerical, guard, history, ownership, cleanup, deadline or artifact-cap failure stops later work. Preserve completed and partial evidence; an independently audited required failure remains a failure when the suffix is missing, while corruption is reported separately. There is no retry, extra tile or additional measured/profile sample. Adoption requires every applicable gate; the completed experiment leaves this candidate unqualified because the required W1 performance gate failed.
