# M4 attention tile experiment — 2026-09-11

**Complete evidence; failed performance acceptance. Do not adopt tile64.** The only production difference was Triton attention `BLOCK_T=32` (A) versus `64` (B), with four warps unchanged. W1's first pair missed the predeclared 1% improvement, and observed throughput ranges overlap. Correctness, six control workloads, memory/setup and profile accounting passed. The PR remains draft and issue #7 remains open. Q2 continues from the accepted compact PR14 baseline.

## Frozen experiment

The [design](m4-attention-design.md) and [operative pre-run contract](m4-attention-contract-20260911.md) define the selection, exact 139 executions, resource limits and acceptance. A is `55f93ce51eaa4e0f490d32fc455d0bbbf6ff3851`; B and the unchanged primary auditor are `576c468d79d971d3585d7cf85d159bb763a5b543`. Both contain 135 tracked source files and differ only in the attention launch literal. They share the accepted [PR14](https://github.com/hsliuustc0106/vllm-lt/pull/14) runtime; benchmark inference is compact, synchronous FP32, with the private persistent option disabled.

The checkpoint is ByteDance/Ouro-1.4B at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` (1,434,652,673 parameters, four loops). Both sides used reserved GPU0, NVIDIA L20X UUID `GPU-006eb78c-23cb-f37d-eb7c-0ccb578b8f11`, CPUs 56–63, NUMA node 1, one intra/inter-op thread and seed 0. Python 3.12.13, Torch 2.13.0+cu130, Triton 3.7.1 and Transformers 4.55.0 were fixed; TF32 and reduced-precision reductions were disabled. The resolved contract records remaining hashes, capacity, input and launch controls.

Eight workers completed all 139 rows in **1,241.532 seconds**: N-A/N-B, A1/B1/B2/A2, then P-A/P-B. The 14 feasibility runs, 28 timing warmups, four diagnostic warmups and four profiler captures are excluded from the 28 measured records. Numerical and held prerequisites passed before measured execution. Downloads, model loading, setup and compilation are outside token-generation measurements. Every worker recorded zero final CUDA allocated and reserved bytes; the post-run scheduler/device probe verified GPU0 available with 0 MiB used. No retry, extra candidate or additional sample occurred.

## Paired throughput

Values are generated tokens/second, in execution order. Pair 1 is B1/A1; pair 2 is B2/A2. W1 required ratios ≥1.01 in both pairs and `min(B) > max(A)`; all other cells required ≥0.95 in each pair.

| Cell | A1 | B1 | B2 | A2 | B/A pair 1 / 2 | Throughput outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| W1-refill | 20.876602 | 20.850592 | 20.920867 | 20.484478 | 0.998754 / 1.021303 | failed |
| W2-refill | 144.843996 | 144.439776 | 143.660451 | 142.607377 | 0.997209 / 1.007384 | passed |
| W3-refill | 18.462674 | 18.358174 | 18.211291 | 18.092894 | 0.994340 / 1.006544 | passed |
| W4-refill | 142.376313 | 136.835390 | 142.264195 | 139.048567 | 0.961083 / 1.023126 | passed |
| W4-no_refill | 111.664596 | 109.225378 | 111.536321 | 109.943439 | 0.978156 / 1.014488 | passed |
| W5-refill | 112.853181 | 112.092699 | 112.519439 | 110.984278 | 0.993261 / 1.013832 | passed |
| W5-no_refill | 113.144101 | 112.507361 | 113.024614 | 110.320768 | 0.994372 / 1.024509 | passed |

W1 changes were **−0.1246% / +2.1303%**. A's observed range was 20.484478–20.876602 tokens/s; B's was 20.850592–20.920867. The first pair fails the target and the ranges overlap. Two observations per side provide observed variation, not statistical confidence; pooling or averaging cannot rescue the failed gate. W1 mean TPOT was A1/B1/B2/A2 = 47.742755 / 47.794380 / 47.620696 / 48.659700 ms. TTFT remains diagnostic in this experiment.

All 12 control pairs passed. Peak allocated and peak reserved increases were **zero in all 14 pairs**, against a 64-MiB ceiling. Maximum setup increase was **0.126005 ms**, below 100 ms. The physical KV pool, page accounting, admission policy, request work and copy policy remain unchanged; the raw results distinguish occupied pages, reserved pages and populated slots. This candidate establishes no physical-memory or capacity saving.

## Correctness and profile evidence

All 27 model cases and 51 comparison streams passed: 243,495 compared boundaries, zero incomparable boundaries, zero required numerical failures and zero behavior failures. Coverage includes unchanged Torch exact comparisons, Triton fidelity, independent serial last-exited history, mixed/live routing and full layer/final populated KV observations. Final FP32 logits retain `atol=0.001`, `rtol=0.0001`; actual greedy IDs, exits and work histories match. Intermediate floating gate/hidden/KV deltas are finite diagnostics, not a widened final-logit tolerance.

All 34 held evaluations passed: 30 direct attention evaluations and four genuine allocating/persistent lifecycle evaluations, with 752 typed records and 4,320 guards. Direct inputs cover partial and fragmented pages, lengths 63/64/65/129, a causal 128-query shared prefix, poisoned inactive rows, invalid inactive addresses and strided layers. Independent CPU FP64 attention uses the frozen `atol=rtol=2e-5`; active compact/physical outputs and untouched pool bytes match exactly. Lifecycle coverage includes reorder/shrink, explicit free/reallocation, positions 510/511/512, fallback, coda/RNG state and persistent pointer lifetime. BF16 remains unqualified. Compute Sanitizer was unavailable: zero checker executions.

The independent streaming profiler audit verified all four captures, source bindings, actual CPU/GPU correlations and matching work histories. CPU annotations are counted separately from GPU annotations. W1 steps 2–97 contain 64 recurrent traversals and 1,536 attention kernels per side. W4 steps 2–35 contain 640 interleaved prefill tokens and seven decode traversals: 480 prefill plus 168 decode attention kernels. Each window contains 16 subsequent outputs.

| Profile | Attention GPU duration A → B | Change | All kernel calls A/B | GPU copy events A/B |
| --- | ---: | ---: | ---: | ---: |
| W1/refill | 24.978887 → 22.739640 ms | −8.9656% | 74,672 / 74,672 | 560 / 560 |
| W4/refill | 16.605661 → 15.955637 ms | −3.9145% | 34,936 / 34,936 | 205 / 205 |

These are diagnostic sums of GPU event durations, not generation latency or device busy-time unions. W4 attention splits into prefill 13.789856→13.363573 ms and decode 2.815805→2.592064 ms. Attention accounts for 11.2143%→10.3036% of W1 summed kernel time and 5.7635%→5.5585% of W4. Most device work and all launch/copy counts remain. No hardware-counter explanation or repeated-profile variability is claimed. The measured component improvement did not satisfy the end-to-end milestone.

## Acceptance criteria / 验收标准

| Criterion | Outcome | Evidence or disposition |
| --- | --- | --- |
| AC-M4-01 — frozen scope and thresholds | Passed | One candidate, committed pre-run contract, numeric targets, source/control identities and finite budgets. |
| AC-M4-02 — inference contract | Passed within declared FP32 envelope | 27 model cases / 51 streams and 34 held evaluations; exact discrete histories and guards. BF16 and unavailable checker coverage remain excluded. |
| AC-M4-03 — attention candidate | Passed within declared shape/input envelope | Both tile configurations retained; dense causal/fragmented/partial/poisoned-row and lifecycle checks pass. No split-context variant or scratch. |
| AC-M4-04 — finalization | N/A | Unselected; earlier profiling did not justify this independent copy-policy change. |
| AC-M4-05 — indirection | N/A | Unselected; no token source-depth addressing change or copy-elimination claim. |
| AC-M4-06 — admission | N/A | Unselected; conservative full reservation retained. |
| AC-M4-07 — memory accounting | Passed | Same pool and work, zero peak increases; separate raw allocation/page/slot accounting. No savings claim. |
| AC-M4-08 — measured guardrails | **Failed** | W1 pair 1 ratio 0.998754 < 1.01; strict observed-range separation also absent. Controls and memory/setup pass. |
| AC-M4-09 — auditable disposition | Recorded rejection | Commands, all raw records, source snapshots, CPU/GPU outcomes, profiles and cleanup preserved in linked evidence; roadmap issue #2 records rejection. This is not an accepted optimization. |

The primary frozen auditor produced `evidence_status=complete`, `decision=failed`, with no errors or missing evidence. Its CLI exited 1 for the failed milestone, not an audit crash. Independent manual and streaming profile audits agree on their respective scopes. A conservative inherited limitation remains: an unexpected negative lifecycle execution is treated as an unqualified audit error unless independently reconstructed; it cannot silently pass. No such error occurred here.

## Reproduction and evidence

[Raw evidence release](https://github.com/hsliuustc0106/vllm-lt/releases/tag/m4-attention-evidence-20260911) contains the unchanged primary report, all raw tensors/events/traces and completion records, both execution source trees, the pre-run contract, exact producers/commands/guards, CPU preparation failures and final passes, and independent reviews. Large evidence stays outside source Git. Archive member SHA256 verification and GitHub asset verification are recorded separately; a relocated report execution is not claimed.

Before device execution, the CUDA-blocked full CPU suite passed **770 tests, 17 skipped** in 215.53 seconds; Ruff check/format and diff checks passed. Focused runner/schema/report/validation checks and earlier preparation failures are retained. The 12-GiB raw cap passed: the pre-report artifact audit counted 6,824,496,864 bytes, including 1,459,913,538 profile bytes, 5,316,289,825 numerical bytes and 30,271,664 held bytes. Package inventories additionally include the completed report bytes.

| Identity | SHA256 |
| --- | --- |
| Canonical plan | `5c0e9a4b3e2d3546d849cf3cd4cc941353f249de494f671718a9a148cce0825b` |
| Exact plan file | `11baf97a9d1b9a9618b189dec150d8f51e45153890bab023b3d13e19b01ccd77` |
| Operative pre-run contract | `be68ba9d55728eee8c590961e13cab1bee6d6c67d79486dc8fc9267ae0308b7f` |
| Primary report | `753984d0038d0a662b33fe209021b7f5ca2beb3ed9f6a5001980f2483fc71d2b` |
| Full CPU log | `7a405ffebdc91f21fa14ce798d474db83552b66ece78b7a8569463c1682a97f1` |
| Independent manual audit | `cd638e57553c82aced7102bd32d16e1eb97dc5fab4482d9c4ef8ee548f9450cd` |
| Independent profile audit | `c7bf1b1bcd2119ff5bf375842f087fb16b26b0e8eda22de563d94db2b5a5655c` |
