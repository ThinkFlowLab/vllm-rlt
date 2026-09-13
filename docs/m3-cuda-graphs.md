# Ouro recurrent CUDA graphs

The opt-in FP32 executor captures one Ouro recurrent traversal: all shared
physical layers, normalization, actual gate logits and copies into persistent
output buffers. It implements part of [M3 #6](https://github.com/hsliuustc0106/vllm-lt/issues/6),
following the inactive-row and persistent-buffer prerequisites in #13 and #14.
Compact execution remains the default; BF16 qualification belongs to Q1.

Graphs reduce CPU/kernel launch overhead. Every traversal still reads the actual
gate on the host before routing and LAST-EXITED KV propagation. Prefill, prelude,
coda, sampling and scheduling are outside capture. This is not lookahead gating,
and an observed roughly 2× result is not a theoretical speedup ceiling.

## Module ownership

The runtime graph implementation belongs in `vllm_lt/worker/`: its lifetime and
transactions are owned by `ModelRunner`, and its captured body is the Ouro
recurrent traversal. Keep these responsibilities together:

| Module | Responsibility |
| --- | --- |
| [`model_runner.py`](../vllm_lt/worker/model_runner.py) | Execution selection, gate synchronization and executor lifetime |
| [`recurrent_graph.py`](../vllm_lt/worker/recurrent_graph.py) | Recurrent capture/replay and KV transactions |
| [`capture_resources.py`](../vllm_lt/worker/capture_resources.py) | CUDA stream leases and graph-pool ownership |
| [`decode_buffers.py`](../vllm_lt/worker/decode_buffers.py) | Buffer layout and allocation shared by persistent eager and graph execution |
| [`graph_diagnostics.py`](../vllm_lt/worker/graph_diagnostics.py) | Runtime budget limits, errors and executor snapshots |

Experiment controllers, profiling adapters, validation and report generation
belong in [`benchmarks/capture/`](../benchmarks/capture/README.md). They may inspect
the worker's runtime state; worker modules must not import experiment tooling.
In particular, `benchmarks/capture/runtime.py` is an experiment adapter, while
`worker/recurrent_graph.py` implements inference execution.

Upstream vLLM also separates reusable graph mechanisms from runner ownership.
At revision `756794a9a7f08900c00fbfaa6d8332631503f528`, its general
[`CUDAGraphWrapper`](https://github.com/vllm-project/vllm/blob/756794a9a7f08900c00fbfaa6d8332631503f528/vllm/compilation/cuda_graph.py)
lives under `compilation/`, its V1
[mode/batch dispatcher](https://github.com/vllm-project/vllm/blob/756794a9a7f08900c00fbfaa6d8332631503f528/vllm/v1/cudagraph_dispatcher.py)
lives under `v1/`, and its newer
[GPU-runner graph managers](https://github.com/vllm-project/vllm/blob/756794a9a7f08900c00fbfaa6d8332631503f528/vllm/v1/worker/gpu/cudagraph_utils.py)
live under `v1/worker/gpu/`.

For this repository, retain the runner-specific implementation under `worker/`.
Extract a general capture/replay layer when another execution path needs to
share it; the current executor does not require a separate `compilation/` package.

## Lifetime and bucket layout

```python
engine._enable_recurrent_graph(use_graphs=True)  # before request admission
# Reuse this engine/model/cache and its captured graphs across request waves.
engine.close()  # after completion or safe abort
```

`use_graphs=False` runs the same padded tensor body eagerly. Comparing it with
replay isolates launch savings. Compare compact execution separately to measure
the combined cost of padding, staging, routing and graph execution.

The engine passes `SchedulerConfig.max_num_seqs` into the shared persistent and
graph buffer layout. Both select the smallest fitting bucket:

| Live requests | Physical rows | Included for scheduler capacity |
| --- | --- | --- |
| 1–2 | 4 | All capacities |
| 3–4 | 8 | At least 3 |
| 5–8 | 16 | At least 5; includes the default capacity of 8 |
| 9–16 | 32 | At least 9 |

The ladder doubles up to the configured capacity, subject to setup budgets.
Metadata retains the established interleaved layout: live rows occupy odd
slots, so at least half the physical rows are inactive. Each bucket has 32
block-table columns. Over-capacity batches, longer tables and the Torch backend
use counted compact fallbacks. No capture occurs during request processing.

`decode_buffers.py` supplies selection, tensor specifications and allocation to
both executors. Device and staging budgets sum those same shapes and dtypes;
there is no hidden-size magic formula. The engine default produces physical
buckets 4/8/16; a standalone private runner retains its four-request default.

All bucket metadata, staging, hidden/gate destinations and gather indices remain
owned for the executor lifetime. Model weights and the KV pool must remain
stable. `_make_tensor_decode_view()` resolves tensor metadata and kernel views
before setup; it contains no request ownership or written-prefix state.

## Pool and stream ownership

All graphs within one executor share one public `torch.cuda.MemPool`. They replay
serially on the original ordered execution stream. Their inputs and persistent
output destinations live outside the graph pool. Each dispatch gathers fresh
outputs before another replay; published request state does not alias captured
storage. Independent graphs may share a pool in varying order when their live
inputs/outputs cannot be overwritten by another replay and they do not execute
concurrently. See [PyTorch's pool-sharing guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#sharing-memory-across-captures).

Live executors retain separate pools and separate capture-stream leases. Safe
close synchronizes, resets every graph, drops captured outputs, and releases the
pool owner last. Only then is the capture stream returned for reuse by a later
executor on that device. A failed synchronization or reset retains ownership and
the lease for safe settlement. No global allocator-cache drop occurs in the
executor. The idle-stream cache grows with peak concurrent executor count; it
is not a promise of constant memory under increasing concurrency.

The earlier 34 MiB/executor growth came from **inactive default-pool allocations
on fresh setup streams**, not surviving graph-private allocations. An isolated
same-source diagnostic held model/cache and per-bucket pools fixed: fresh streams
added 34 MiB per close, while a reused stream held post-close reservation flat.
CUDA snapshots showed the private pool segments disappeared in both variants.
The [pinned allocator source](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp)
restricts cached-block reuse to its allocation stream. Reusing released streams
addresses that growth; sharing pools within an executor also reduces private
pool ownership. Pools are not shared across simultaneously live engines.

Resource handling is in `capture_resources.py`; diagnostic snapshots and limits
are in `graph_diagnostics.py`. The executor retains transaction and capture logic.

## Transactions, setup and budgets

A traversal validates logical rows and preceding KV prefixes, prepares and binds
the selected bucket's lease, updates dynamic tensor contents, executes or replays,
and gathers independent outputs. After actual gate readback confirms completion,
it commits initialized KV positions, releases the lease and publishes request
state. Routing/finalization follows. A partial host commit failure quarantines
the cache; uncertain device completion retains borrowed resources. Repeated
failure settlement preserves secondary diagnostics without repeating a sync.

Setup precedes admission. It saves four scratch pages from the actual KV pool,
seeds preceding history, and runs three eager warmups, one capture recording and
one verification replay per bucket. It requires the maximum absolute warmup/replay difference for both hidden outputs and gate logits to be at most `1e-5` (FP32), plus finite physical outputs and
positive-zero inactive rows, then restores saved bytes and exact free-list order.
The default three-bucket ladder records three graphs and executes twelve setup
traversals. Capture recording itself does not execute the recorded kernels.

| Resource | Default limit |
| --- | ---: |
| Common device tensors, complete ladder | 1 MiB |
| Common CPU staging, complete ladder | 16 KiB |
| Additional retained graph allocated / reserved memory | 256 MiB each |
| Setup peak allocated / reserved increase | 512 MiB each |
| Complete setup | 60 seconds |

For Ouro-1.4B and the default capacity, common tensors total 463,372 bytes and
CPU staging 4,396 bytes. Saving four scratch pages needs 24 MiB of temporary host
memory. Setup timing and memory baselines are recorded separately from inference.
A configured budget overflow may decline capture only after successful completion,
scratch restoration and resource release. Device/allocation errors and uncertain
cleanup do not trigger silent fallback. Experiment adapters require successful
setup. A process watchdog must bound blocked device calls; the setup deadline is
checked between phases rather than interrupting a CUDA call at exactly 60 seconds.

## Evidence and qualification

The [milestone evidence record](benchmarks/m3-capture.md) contains the latest device attempt, lifecycle and live-gate checks, archive checksums, and an attempt log. Profiling time and artifact-budget failures keep the graph path unqualified. Historical results do not qualify the current source.

The original numerical and persistent validation adapters explicitly retain their
frozen 4/8-row and 8-row shapes. They do not qualify the expanded scheduler ladder.
New experiments must freeze new source, shape coverage and budgets. Do not pool
samples or relabel a historical fallback-only profile as replay evidence.

A performance plan must prove intended dispatch coverage on CPU before GPU work,
then confirm actual live-gate coverage in feasibility. Report compact/eager/graph
comparisons, padding, bucket hits and fallback reasons, setup amortization, and
arrival-to-token latency. Profiles need actual CUDA launch-to-kernel correlations;
profile time is excluded from measured inference.

Prefer one executor per resident model/cache. Replacement tests must separately
freeze a post-close allocated/reserved-memory plateau criterion without dropping
shared caches. Process-terminal zero memory alone does not prove a plateau.
Two throughput samples show observed variability, not a confidence interval.
CPU stand-ins verify transactions; only reserved-device runs verify CUDA replay.

## Benchmark tooling

Graph benchmarks and numerical validation live in
[`benchmarks/capture/`](../benchmarks/capture/README.md), outside the installed
runtime package. Run `python -m benchmarks.capture --help` from the repository
root. Generated plans, traces, results and logs belong in external experiment
storage or the ignored `artifacts/` directory.

Metadata uses one packed device buffer and one pinned host buffer per bucket on CUDA (ordinary host storage in CPU tests). Disjoint typed views retain the kernel shapes. One asynchronous copy is ordered on the execution stream before replay; the lease and gate readback prevent staging reuse while DMA is pending. Allocation generations reject stale decode tickets; preceding KV history is validated once before submission. Full tensor signatures are checked at setup; dispatch checks object identity and borrowed-view identity. Executor-private tensors, weights and cache pools must not be resized or rebound during the executor lifetime.

`engine.close()` releases partial setup resources after an OOM or compilation failure and may be retried if synchronization or graph reset failed. It is terminal for the installed graph executor and never revives a quarantined cache. Errors during host `_update` bookkeeping also permanently fail the executor: finalization may have submitted additional device work, so settlement must precede request-page release.
