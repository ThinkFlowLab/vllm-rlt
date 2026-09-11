# M3 persistent buffers: resolved correctness contract — 2026-09-11

This protocol is committed before device execution. Later documentation commits
do not change the separate clean execution checkouts.

Resolved at `2026-09-10T20:21:26.348378+00:00`.
Plan: `/data/hsliu2/tmp/vllm-lt-m3-persistent/artifacts/m3-persistent-plan-20260911/plan.json`.
Canonical plan SHA-256: `d513ee88df0a83ce7a9b4386c2f83d57feadf46cf61aa993b0048580515e28f5`.
Plan file SHA-256: `7eb40d06449f9430e3b9bef5bd1907e65eeef89a390d4a7b5efb7dc27d1705ee`.

## Hypothesis and source controls

Reusing bounded eager decode storage preserves live numerical behavior, ownership
and safe request lifetime. A allocates padded inputs/metadata/outputs per traversal;
B reuses one owned bundle and publishes fresh live output storage. Both use eight
physical rows, 32 table columns, at most four live rows in odd slots, and the same
valid-shape compact fallbacks. The independent dense oracle stays compact.

| Side | Frozen commit | Clean execution checkout |
| --- | --- | --- |
| A | `ee4e91c27e7890f3e30ae57be1e479bb98099c68` | `/data/hsliu2/tmp/vllm-lt-m3-persistent-control` |
| B | `9a28b62edd86ddb3bf84d8d0dd640bb6dfe687b8` | `/data/hsliu2/tmp/vllm-lt-m3-persistent-candidate` |

A restores only the three production files below from inactive-row commit
`a17abb047ef893e115b43b7bd14315525a344075`. Every other probed source file
matches, including all validation code, fixtures and tests.
Common harness SHA-256: `3154fd61b38dd8c3f77cddb82a79193b38de663524f494a2814bc801df76c9f0`.

| Production path | A SHA-256 | B SHA-256 |
| --- | --- | --- |
| `vllm_lt/core/kv_cache_manager.py` | `3afb020da127d24a1c579fc097ffa0313b39ab476f92fd89dcd711fbd8b65daa` | `9f8c430871feede8981047be1a1ca3079dcaa7e59183342c884fec6a86371f1e` |
| `vllm_lt/engine/llm_engine.py` | `c8402dd2398e0a312d02e75fd3cbbb422f0899d7729befcce3beed09908f1d01` | `9d482c317d8685ecaa7c840edcb67f935968abebcca6fe7611f2bb5dac6ae166` |
| `vllm_lt/worker/model_runner.py` | `6704a2884480b99d95e040a34aa89fdff300e90ac6474f709d1e1ac85b21822a` | `efc7450d82ea9a3909debee77a27071214bc270cd88720fb20c74346d761bc81` |

The default compact engine remains unchanged in use. This comparison qualifies
the private opt-in eager storage prerequisite of [M3 #6](https://github.com/hsliuustc0106/vllm-lt/issues/6).
It does not qualify graph capture, graph performance, BF16 or default adoption.

## Fixed controls

Real checkpoint: ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1,
FP32, at `/data/hsliu2/tmp/vllm-lt-models/ouro-1.4b`. Interpreter: `/home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python`.
Both sides reuse Python 3.12.13, Torch 2.13.0+cu130, Triton 3.7.1 and
Transformers 4.55.0. Exact dependency versions and model/config/tokenizer file
identities are in the plan; optional kernels is absent.

Host: dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw; account: hsliu2.
Physical GPU 0 is an NVIDIA L20X, management UUID
006eb78c-23cb-f37d-eb7c-0ccb578b8f11, available with zero used memory at selection.
The reserved workers must record that same UUID. The scheduler controls visibility.
CPU cores 56–63 and active NUMA memory policy bind to node 1 are fixed; the allowed
memory mask remains 0–1. No GPU-locality claim is made. Seed 0, one intra-op and
inter-op thread, disabled TF32/FP16/BF16 reduced-precision reductions, and
OMP_NUM_THREADS=1 are fixed. Other recorded optional environment variables are
unset. The CPU probe ran under these bindings with CUDA entry points blocked.

One fresh worker and one real-model load per side; one per-case pool at a time.
A and B run sequentially, with task CUDA cleanup before the next worker. Preserve
shared caches and each worker's allocator between cases. Record loading, setup,
compilation, export and cleanup separately; no execution time is a speed claim.

| Input | SHA-256 |
| --- | --- |
| suite | `9b1fee4778fe77c20f937417946ce4ad0b5f21510659221d6cdedffaf92da197` |
| contract | `61622b6e50670b1a46808acb913dc29be5e86f57332882f5784c35f5378057a5` |
| persistent contract | `d7ed54ae8d44647fc51b56c5ea78f0a773499332cbe53b074b75b007ad8b7058` |
| config.json | `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3` |
| merges.txt | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| model.safetensors | `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` |
| special_tokens_map.json | `55087bb8409060d9cb0f80495e34f4e0d8a84f68a799a6b0cab3132be0aae319` |
| tokenizer.json | `fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d` |
| tokenizer_config.json | `7e010da95d71b0fa1aa809552922c74cc1dd98d628f21ba39ca6c35aa18d5d91` |
| vocab.json | `7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c` |

## Exact executions and required gates

Exactly 17 executions, with no retries or replacement trials: N-A runs one excluded
feasibility case, Torch/Triton lifecycle checks, five oracle cases and three native
qualification cases; N-B runs one excluded feasibility case, Torch/Triton lifecycle
checks and three native qualification cases. This correctness prerequisite uses
one deterministic pass per case, zero performance warmups/measured repetitions
and zero profiler captures.

The 13 real-model cases contain 11 qualification cases and two excluded feasibility
cases, with 27 qualification comparison streams and one excluded B/A feasibility
stream. Required failures in either stop subsequent cases. Fixtures remain
Q1-L16-F0, Q1-L256-F0, Q1-L64-F2 and Q1-L128-F3, plus L16-F0 live routing.
The oracle is serial; native mixed groups use refill and no-refill. Each trajectory
records nine genuine predictions. Fixed/forced histories consume eight supplied
inputs; live routing consumes generated tokens with threshold 0.7 and depth 2–4.

All model comparisons retain original Q1 FP32 final-logit atol 0.001/rtol 0.0001,
exact actual top1/exit/history checks, finite selected hidden/gate boundaries and
full populated-KV coverage. Intermediate/KV deltas remain diagnostics. L64-F2 and
L256-F0 are preselected for paired dumps; the frozen plan also permits up to two
additional first-failure fixtures within the unchanged diagnostic caps. This
wording was clarified after execution began; the committed plan and its budgets
were unchanged. Per-native-dispatch evidence binds actual model
inputs and request publications to the recorded logical schedule, verifies inactive
finite zero outputs, and verifies B's stable tensor/staging pointers and generation
lifetime. All ordinary real-model dispatches must use the supported padded shape.

## Finite lifecycle matrix

Lifecycle plan SHA-256: `b8bb058aecb4f3dbd5a124e587135fb445d2d0823bdba6dba235ad1b577cf0ab`.
Each side executes Torch and Triton once. The deterministic two-layer held-input
model contains no pretrained weights; separate real-model cases qualify Ouro.
It uses real cache writes/attention with FP32 H=2048, 16 heads, D=128 and an 80-MiB
pool of 160 pages, 16 tokens/page. The CPU probe freezes hidden/QKV/metadata formulas,
allocation maps, action inputs and expected whole-pool hashes.

| Step | Action | Dispatch | Positions |
| --- | --- | --- | --- |
| 1 | four_at_510 | supported | `[510, 0, 0, 0]` |
| 2 | four_at_511_hold_r2 | supported | `[511, 1, 1, 1]` |
| 3 | shrink_one | supported | `[2]` |
| 4 | empty | empty | `[]` |
| 5 | reorder_two | supported | `[2, 3]` |
| 6 | reallocate_r1_and_reject_stale | host_only | `[]` |
| 7 | recycled_two | supported | `[0, 3]` |
| 8 | width_33_fallback | fallback | `[512]` |
| 9 | retire_long_then_four | supported | `[1, 4, 0, 0]` |
| 10 | five_live_fallback | fallback | `[2, 5, 1, 1, 0]` |
| 11 | return_one | supported | `[1]` |

Each evaluation contains seven supported traversals, two compact fallbacks and
one empty invocation; step 6 changes host ownership without executing a core.
B must finish generation 7 with 10 calls, seven prepares/completions, one empty
call and one fallback for each of live-count and table-width overflow.
Check held r2 coda state and RNG through later reuse; independently reject expired
leases and changed allocation identity after reusing r1. Empty calls execute no
core and leave fixed buffers unchanged. Required 32-to-33-column and four-to-five
live-row transitions return to supported execution without replacing the bundle.

Each evaluation retains 173 typed tensor records and checks all 40 whole-pool
chunks after seed and each of 11 actions: 480 checks/evaluation, 1,920 total.
Same-backend A/B raw outputs and actual gates must match exactly. B owns 132,392
device payload bytes and 1,256 CPU staging bytes; model intermediates still allocate
eagerly. Request publications cannot alias reusable output storage. Complete
cleanup requires confirmed completion, zero requests/pages, released cache/buffer
references and zero allocated/reserved CUDA bytes at worker termination.

## Budgets, stop rules and acceptance

The internal total cap is 3,600 seconds and each whole model/lifecycle case has
600 seconds, including validation, setup, evidence export and cleanup. The outer
scheduler timeout is 70 minutes to permit startup and owned cleanup around that
unchanged internal cap. Only task child process groups/resources may be cleaned.
The lifecycle terminal marker remains unpublished through export/cap validation;
post-write deadline checks prevent late persistence from qualifying.

One fixed 8 × 32 bucket; device payload ≤ 256 KiB and CPU staging ≤ 16 KiB. The native
real-model pool is 160 × 16 and 1,006,632,960 bytes. Live matching ≤ 64 MiB, retained
group ≤ 2 GiB, cumulative tensors ≤ 8 GiB and paired dumps ≤ 256 MiB. Lifecycle per-evaluation
typed data ≤ 8 MiB and JSON ≤ 4 MiB: 48 MiB for the finite suite within the parent 256 MiB
allowance. All artifacts ≤ 12 GiB; profiler and sanitizer executions are zero.

Model retained-tensor upper bound 3,067,709,328 bytes; largest case 939,262,464;
index upper bound 71,583 records × 1,024 bytes; comparison upper bound 133,776 × 2,048.
Pointer evidence is bounded to 2,916 records × 32,768 bytes = 95,551,488 bytes.

Stop subsequent cases on execution, numerical/discrete, pointer, publication,
lease, guard, source/input/device/control, cleanup, deadline or resource failures.
Unsupported valid shapes take only the declared counted compact fallback.
Execution/device failures never retry eagerly. Unconfirmed completion quarantines
the cache and cannot be reported as safe page reuse. Preserve partial evidence;
a stopped failure or corrupt/incomplete audit cannot pass.

compute-sanitizer and cuda-memcheck are unavailable in PATH and the installed
CUDA 13 tree. Zero checker executions are planned. CPU failure injection does not
establish real GPU-fault recovery. These gaps remain explicit.

Acceptance requires every declared model, storage, lifecycle, control and cleanup
gate plus complete offline evidence. This supports only the eager subset of
AC-M3-03 and related eager AC-M3-02/04/05 checks. Captured storage/bucket alternation,
capture-safe bookkeeping, scratch setup, graph performance and adoption remain
open. Correct buffers cannot close M3; BF16 remains separately gated by Q1.

## Reserved command

Run from the frozen B checkout:

```bash
gpu run --gpu-ids 0 --nonblock --timeout 70m --note 'vllm-lt M3 persistent A/B 9a28b62' -- \
  numactl --physcpubind=56-63 --membind=1 -- env OMP_NUM_THREADS=1 \
  /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python -m vllm_lt.validation.m3_persistent_run run \
  --plan /home/hsliu2/tmp/vllm-lt-m3-persistent/artifacts/m3-persistent-plan-20260911/plan.json \
  --output /home/hsliu2/tmp/vllm-lt-m3-persistent/artifacts/m3-persistent-run-20260911
```
