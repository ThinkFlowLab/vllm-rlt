# M3 persistent eager buffers: correctness result — 2026-09-11

**The persistent eager-storage prerequisite passed. M3 remains open.**
Allocating padded A and persistent padded B completed all 17 declared executions
on the same reserved GPU, and the complete offline audit passed. This qualifies
reusable boundary storage, publication/lease isolation and eager fallback behavior.
It does not establish CUDA graph capture, performance improvement or BF16 support.

The [resolved pre-run contract](m3-persistent-contract-20260911.md) defines the
inputs, controls, exact order and limits. [The interface guide](../m3-persistent-buffers.md)
describes ownership and failure handling. Full raw evidence is published with the
[persistent evidence release](https://github.com/hsliuustc0106/vllm-lt/releases/tag/m3-persistent-evidence-20260911);
generated tensors and JSON are outside source Git.

## Frozen comparison

| Side | Implementation SHA | Behavior |
| --- | --- | --- |
| A | `ee4e91c27e7890f3e30ae57be1e479bb98099c68` | Allocating padded eager traversal |
| B | `9a28b62edd86ddb3bf84d8d0dd640bb6dfe687b8` | Persistent padded eager traversal |

Both use eight physical rows, 32 table columns and at most four live requests
mapped to odd slots. Only the three declared production files differ; A restores
them from accepted inactive-row commit `a17abb047ef893e115b43b7bd14315525a344075`.
All 121 probed source files per side were frozen and independently hash checked.
The common harness hash is
`3154fd61b38dd8c3f77cddb82a79193b38de663524f494a2814bc801df76c9f0`.

Real weights: ByteDance/Ouro-1.4B at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, FP32.
Physical GPU 0: NVIDIA L20X, UUID `006eb78c-23cb-f37d-eb7c-0ccb578b8f11`.
Both workers recorded that UUID, CPU cores 56–63, active NUMA binding to node 1,
one intra-op/inter-op thread, seed 0 and the same arithmetic/software controls.
The prepared environment uses Torch 2.13.0+cu130 and Triton 3.7.1.

The canonical plan hash is
`d513ee88df0a83ce7a9b4386c2f83d57feadf46cf61aa993b0048580515e28f5`.
The resolved protocol was committed before device execution at `bdb102bf`.
A later documentation correction clarified the plan's existing allowance for up
to two failure-selected dump fixtures beyond its two preselected fixtures.
The executed plan, sources, budgets and matrix did not change; no extra fixtures
were selected by a failure.

## Outcomes

| Check | Result |
| --- | --- |
| Exact executions | 17/17: 11 in N-A, then six in N-B |
| Real-model cases | 13/13: 11 qualification and two excluded feasibility |
| Comparison streams | 28/28: 27 qualification and one excluded feasibility |
| Compared tensor boundaries | 129,663 qualification + 4,113 excluded feasibility; no required numerical/behavior failure |
| Actual token/depth/history matches | 243 qualification + nine excluded feasibility |
| Actual input/publication storage checks | 240 native dispatches: 120 A and 120 B |
| Held-input lifecycle checks | 4/4: A/B on Torch and Triton |
| Whole-pool guards | 1,920/1,920 chunk checks |
| Lifecycle typed records | 692: 173 per evaluation |
| Worker model loads | Exactly one per side |
| Final allocated/reserved CUDA bytes | Zero on both sides |

Full-model comparisons retain the original Q1 FP32 final-logit bounds
`atol=0.001, rtol=0.0001`, exact actual token/exit/history gates, finite selected
intermediates and full populated-KV coverage. Intermediate/KV deltas remain
diagnostic. Refill, no-refill, mixed depths, fixed/forced histories and live
routing all use their declared fixtures. Feasibility results are excluded from
qualification counts.

The independent audit found all 47,334 B/A tensor records exactly equal,
including the 4,113 excluded feasibility records. Against the dense oracle,
the largest final-logit absolute difference was 0.000118255615234375, within
the original bound. All 1,540 populated-KV chunks were present; their finite
numerical deltas retain diagnostic status.

Every B native dispatch used stable owned tensor/staging addresses and the
expected generation/completion sequence. Actual model arguments and actual
published request rows were checked independently of runner snapshots.
Fresh live publications did not alias reusable output storage.

Each lifecycle evaluation executed seven supported traversals, one empty call,
two compact fallbacks and a host-only cancellation/reallocation action. B ended
at generation 7, with seven prepares/completions and one fallback for each of
live-count and table-width overflow. Same-backend A/B raw outputs and actual
gates matched exactly. Held coda state and RNG survived subsequent reuse;
expired leases and changed allocation ownership were rejected. All pool chunks
matched the frozen expected writes/guards after seed and each of 11 actions.
The held-input model has two layers and no pretrained weights; the separate
13 real-model cases qualify Ouro behavior.

## Resources, cleanup and verification

B owns 132,392 device tensor payload bytes and 1,256 unpinned CPU staging bytes,
within the 256-KiB/16-KiB limits. Eager model intermediates still allocate;
this payload is not total process memory. The native model pool is
1,006,632,960 bytes; the separate held-input pool is 80 MiB.

Execution completed in 221.910748480 seconds within the 3,600-second protocol
limit. Setup, loading, exports and cleanup are included in this execution
lifetime; it is not a speed measurement. The finite lifecycle evidence occupies
22,741,188 bytes within its 48-MiB allowance.

All model cases released their requests/pages; all lifecycle cases confirmed
completion and released cache/buffer references. After garbage collection each
worker retained 32 MiB of cuBLAS workspace. Explicit workspace release and cache
cleanup brought both allocated and reserved CUDA bytes to zero. The reservation
ended, and GPU 0 was observed available with 0 MiB used at
`2026-09-10T20:27:51.395506+00:00`.

The CUDA-blocked full candidate CPU suite passed 667 tests, with 16 GPU tests
skipped. After the final export/deadline correction, its targeted lifecycle and
controller suite passed 57 tests. The frozen allocating control passed all
557 predecessor CPU tests, with 16 skipped. Ruff lint/format and diff checks
passed. The final offline audit required no
checkpoint loading or GPU execution.

Primary offline report SHA-256:
`54086d9f0a14abf52da3442a5369a838a8604b7fa9ffa2bccc6babf86ffe9aac`.
The archive includes the frozen sources/inputs, complete run records,
typed evidence, pointer/guard records, CPU logs, review producers and cleanup
evidence. Its manifest and checksums support byte verification after extraction.

## Acceptance scope / 验收标准

| Criterion | Decision |
| --- | --- |
| AC-M3-02 | Inherited eager inactive-row masking remains required and passed |
| AC-M3-03 | Eager stable storage, publication isolation, leases and reuse passed; captured pointers/bucket alternation remain open |
| AC-M3-04 | Eager numerical/history/bookkeeping and failure tests passed; capture-safe bookkeeping remains open |
| AC-M3-05 | Bounded eager storage and counted fallback passed; graph scratch/setup limits remain open |
| AC-M3-01/06/07 | Graph rationale/limits, end-to-end benefit and adoption remain open |

No device memory-access checker is installed, and zero checker runs were made.
Guards and CPU fault tests do not replace that coverage. No GPU fault was
injected, no BF16 qualification was attempted, and no timing/profile experiment
supports a speed claim. The next M3 PR must qualify actual capture/replay and
meet the separately frozen performance gates before M3 can close.
