# M3 prerequisite: persistent eager decode storage

This follows the accepted inactive-row prerequisite in [PR #13](https://github.com/hsliuustc0106/vllm-lt/pull/13).
It gives the runner one bounded set of reusable inputs and output destinations,
with explicit request ownership and completion rules. It prepares the storage
boundary for [M3 #6](https://github.com/hsliuustc0106/vllm-lt/issues/6); it does not
implement capture/replay or establish a performance improvement.

## Scope and capacity

The path is private and opt-in through `ModelRunner._enable_persistent_decode()`.
Normal engine construction retains compact execution. Setup permits FP32 on the
existing Torch/Triton backends, with one fixed capacity: eight physical rows,
32 block-table columns, and at most four live decode rows at slots `2*i+1`.
The model and cache must share a device. Replacing or growing the bundle is an
error; a graph cache or public configuration interface is not introduced.

Prefill, prelude, coda, synchronous gate readback, cumulative hazards and
LAST-EXITED finalization keep their existing semantics. A request's RoPE
position remains fixed while its recurrence depth advances. Padding cannot
create a request, advance its RNG or enter coda.

Selection uses current causal table width, not lifetime reservation length.
With 16-token pages, position 511 fits; position 512 needs compact fallback.
Valid batches over four live rows also fall back compactly. Both reasons have
separate counters. Invalid metadata, stale ownership, failed execution and
changed streams are errors, never fallback or retry conditions. Empty calls
perform no traversal and leave storage unchanged.

For real Ouro FP32, the fixed device tensor payload is **132,392 bytes**:
hidden input/output `[8,2048]`, gate output `[8]`, six metadata tensors and
four int64 live indices. The cap is 256 KiB. The six unpinned CPU staging
tensors occupy 1,256 bytes, within a 16-KiB staging cap. Allocator reservation
and temporary eager tensors are separate quantities. No model intermediate
allocation is removed or hidden by this payload accounting.

## Ownership and private interfaces

| Owner | Storage or state | Lifetime |
| --- | --- | --- |
| Cache manager | Request/depth KV pages and initialized-prefix bookkeeping | Existing request reservation |
| Runner | One fixed input/metadata/output bundle and CPU staging | Enabled runner |
| Prepared descriptor | Logical rows, allocation identities and storage generation | One completed traversal |
| Request | Independently gathered live output | Until that request's next execution or completion |

`_prepare_host_batch(ids, depths, positions)` validates logical rows and unique
write destinations without creating device metadata. Both the allocating and
persistent paths use that helper. `_prepare_into(storage, host)` checks live
allocation identities and capacity, fills every staging slot, and copies into
the existing device tensors. Inactive rows get zero positions/lengths and
invalid address sentinels. Shrinking a batch cannot retain old active bits or
table tails. No compact device descriptor is allocated as an intermediate.

A fresh `_PreparedKVBatch` borrows those tensors with a generation number.
Only one generation can be in flight. Cache consumers reject a released,
superseded or failed generation, even if its old request ID still exists.
Allocation-object identity independently rejects a canceled/reallocated
request. These descriptors are private borrowed state, never request storage.

`_recurrent_persistent(hidden, ids, depths, positions)` copies current compact
hidden states into the live physical slots, calls the existing eager prepared
core, copies its final hidden/gate tensors into fixed output destinations, then
gathers live outputs into fresh storage. Model-produced outputs and
intermediates remain temporary. A later traversal may overwrite every fixed
output slot without changing a request paused in coda.

`_persistent_snapshot()` returns detached host metadata: tensor device, dtype,
shape, stride, data/storage pointers, payload sizes, generation/status,
dispatch/fallback counts and publication descriptors. It reads no device
tensor contents. The validation observer separately records actual model
arguments and published request rows; snapshots alone cannot prove use of
the fixed buffers.

## Completion and failure

Setup and execution stay on one recorded ordered stream. A different stream
is rejected; concurrent/reentrant execution is unsupported. During normal
`execute`, the generation remains in flight through the existing real gate
conversion/readback. That readback completes earlier input/output copies and
publication before releasing the lease. A direct private recurrent call has
no caller gate-readback boundary, so it explicitly synchronizes before
releasing its lease and returning. Neither path permits overlapping reuse.

Direct-call input and ownership validation precedes storage/KV mutation. Once work
has been submitted, an error invalidates the executor and attempts completion.
Confirmed completion permits the engine's affected-request abort/free path;
the executor still rejects retries. If completion fails, the cache is
quarantined: allocation, preparation, lookup and page release fail, and engine
admission/execution cannot continue. Reservation counters are preserved;
uncertain cleanup cannot be reported as safely recycled pages.

Finalization can submit work after the runner has released its generation.
The engine therefore settles errors and interruptions from `_update` too.
Secondary completion/cleanup errors cannot replace the original exception.
The CPU tests inject these failures; they do not establish recovery from an
actual GPU fault. No recovery, asynchronous execution or overlapping streams
are implemented.

Per-layer Python initialized-prefix bookkeeping remains eager, with the
existing write-before-attend checks. Moving that bookkeeping outside a
captured tensor traversal belongs to the later capture PR.

## A/B validation contract

A executes the accepted allocating padded path. B executes the persistent
path with the **same** physical shapes, odd live mapping and valid-shape
fallback decisions. This isolates storage reuse from the previous padding
change. The independent dense oracle remains compact.

The frozen protocol contains 17 executions in two fresh workers: 13 real-model
cases and four held-input lifecycle evaluations. Each worker executes its
excluded feasibility case, its two lifecycle evaluations, then its remaining
model cases. The real-model subset preserves 11 qualification/two feasibility
cases and 27 qualification/one feasibility comparison streams, with original
Q1 FP32 final-logit bounds and exact actual token/exit histories. Hidden, gate
and populated-KV deltas remain finite diagnostics. No additional timing or
performance repetitions are implied.

For every native model dispatch, validation binds actual input/metadata and
publication pointers to the independently recorded logical schedule. B must
show stable owned tensor/staging addresses, one valid generation, correct
completion counters, and no alias between fixed storage and request outputs.
Missing evidence or a pointer/isolation failure stops subsequent cases.
The same checks run offline; their default device requirement is CUDA.
CPU stand-in tests explicitly select CPU and cannot qualify device pointers.

The four lifecycle evaluations compare A/B on Torch and Triton. Their
deterministic two-layer held-input boundary exercises the actual cache
write/attention operations and buffer interfaces; it contains no pretrained
weights. The separate 13 model cases use the real Ouro checkpoint. The
finite action sequence covers reuse, shrinking, an empty call, reordered live
rows, held-coda publication, cancellation/reallocated IDs, the 32-to-33-column
boundary and the four-to-five-live-row fallback. Its exact inputs, page
ownership, actions and guard expectations are frozen by the CPU probe.

Each lifecycle evaluation permits seven supported dispatches, two compact
fallbacks and one empty call. The fixed two-layer KV pool is 80 MiB. All 40
whole-pool chunks are checked after seed initialization and each of 11 action
steps. Raw typed tensors are bounded to 8 MiB and JSON to 4 MiB per evaluation.
No device fault or memory-access-checker execution is added.

Global execution is bounded to 3,600 seconds and each whole case to 600 seconds,
including setup/export/cleanup. Model matching/reference/dump limits remain
the retained Q1/M3 bounds. The finite lifecycle plan permits 48 MiB of evidence
across its four evaluations, within the parent contract's 256-MiB allowance.
All raw artifacts are capped at 12 GiB. Plans record exact source/checkpoint/software/GPU/CPU/NUMA
identities before reservation; there are no retries or hidden extra cases.
The scheduler's outer timeout is 70 minutes, allowing process startup and owned
cleanup around the controller's unchanged 60-minute execution limit.

## Acceptance scope / 验收标准

- The selected eager subset of **AC-M3-03** requires stable owned pointers,
  independent published state, valid lease/allocation reuse and complete
  successful cleanup. Captured pointers and alternating captured buckets
  remain untested.
- The earlier **AC-M3-02** masking contract remains required. **AC-M3-04**
  retains eager numerical/history and initialized-prefix checks; capture-safe
  bookkeeping and actual device-fault recovery are not claimed.
- **AC-M3-05** gets bounded storage and explicit eager fallback evidence only.
  Capture scratch state, graph setup budgets and graph transitions remain open.
- **AC-M3-01/06/07** graph limits, end-to-end performance and adoption gates
  remain open. Correct reusable buffers alone cannot close M3 or justify a
  speed/default-adoption claim. BF16 remains separately gated by Q1.

The final PR must link the resolved pre-run contract, individual outcomes,
raw source/input/tensor evidence, pointer and guard checks, cleanup records
and unavailable checks. Generated JSON and tensors belong outside source Git.

## Commands

Run the CPU probe with the same explicit CPU/NUMA environment later used by
the reserved comparison. Substitute the same selected available GPU ID in
both commands; the examples do not reserve a device by themselves.

```bash
python -m vllm_lt.validation.m3_persistent_run probe \
  --baseline-root /absolute/path/control --candidate-root /absolute/path/candidate \
  --contract /absolute/path/candidate/benchmarks/fixtures/ouro-m3-persistent-contract.json \
  --model-path /absolute/path/ouro-1.4b --gpu-id <selected-id> \
  --output /absolute/path/persistent-plan

gpu run --gpu-ids <selected-id> --nonblock --timeout 70m \
  --note 'vllm-lt M3 persistent storage A/B' -- \
  python -m vllm_lt.validation.m3_persistent_run run \
  --plan /absolute/path/persistent-plan/plan.json --output /absolute/path/persistent-run

python -m vllm_lt.validation.m3_persistent_run report \
  --run-dir /absolute/path/persistent-run
```

The offline report requires no checkpoint or GPU. Complete evidence must pass
all numerical, storage, lifecycle, control and cleanup checks. A stopped
failure remains failed with incomplete coverage; corrupt identity or order is
invalid. Neither outcome qualifies this prerequisite.

## Review clarifications: configuration and terminal failures

The private executor supports four live rows in eight physical rows and at most
32 block-table columns. `SchedulerConfig.max_num_seqs` is an admission/batching
limit, not a promise of persistent dispatch: larger recurrent batches deliberately
use compact execution. Callers enabling this experimental path should inspect
`_persistent_snapshot()` capacity, `last_dispatch` and `fallback_counts`; an
assertion tying scheduler capacity to four would incorrectly reject supported
mixed persistent/compact workloads. A high scheduler limit may make every step
fall back, so this configuration alone proves neither storage reuse nor speed.

Every exception while the persistent executor is enabled disables it, including
prefill errors and interrupts raised during engine update. This conservative
policy is intentional: the executor is not restartable after partial request
failure, even when the particular stage did not borrow persistent buffers.
Successful gate readback and later coda/LAST-EXITED completion remain separate
lifetime boundaries. Request-owned published tensors must outlive buffer refill.

If stream completion cannot be confirmed, the cache is quarantined and ordinary
abort/free cannot safely reclaim it. There is no in-process persistent-executor
drain/recovery API in this scope; the owning worker must terminate and release
its device context. Do not bypass quarantine or reuse its pages. A recoverable
replacement protocol needs separate completion/ownership proofs; stable pointers
alone do not establish those proofs.
