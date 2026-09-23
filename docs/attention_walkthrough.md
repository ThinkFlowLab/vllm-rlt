# M8 Attention walkthrough and first interface increment

Baseline: `60af1cf7bee9e44b9d8f284c42b3b97074ba12bb` on `ThinkFlowLab/vllm-rlt` main, 2026-09-23. File ranges below refer to that commit. This review covers Attention selection, synchronous metadata preparation, backend execution, and the other metadata producers that constrain the boundary. It does not claim to review all logical KV allocation or worker scheduling code.

## Boundary and call graph

```text
LLMEngine -> KVCacheManager(backend)
ModelRunner / Ouro recurrent -> KVCacheManager._prepare_batch
  -> _validate_rows (logical allocation, depth and position)
  -> selected depth plane and physical page table
  -> Attention tensors (page table, context length, optional packed CU lengths)
Ouro layers -> _write_prepared -> _attend_prepared
  -> Torch / Triton / FlashAttention paged kernel
```

`core/kv_cache_manager.py:82-147` currently creates the backend and physical K/V tensors together. `:383-479` validates request rows, selects depth planes, computes write addresses, and constructs Attention metadata. `:499-587` writes K/V and dispatches attention. A prepared descriptor is borrowed for one traversal; `:480-485` rejects a different manager or a released/reallocated request. This identity check is essential even when an external request ID is reused.

Other producers cannot be replaced by the synchronous builder without changing their lifetime contract: `worker/buffers.py:65-102` copies metadata through reusable pinned host/device buffers; `worker/async_state.py:169-209` makes a descriptor backed by an asynchronous routing bank; `worker/prefill_metadata.py:19-108` stages packed FA4 prefill data with a completion event and Triton expansion; `worker/cuda_graph.py:43-116` copies metadata into graph-private, fixed-shape device buffers. `worker/model_runner.py:229-315,322-433` chooses these paths. `engine/llm_engine.py:65-70` also gates prefill UVA on FA4, and `worker/model_runner.py:272-274` chooses packed prefill from the backend generation. These are callers of the M8 contract, while M5 retains buffer, DMA, event, and graph lifetimes.

## State and tensor contract

| State | Writer and owner | Consumer / lifetime |
| --- | --- | --- |
| `_Allocation.block_tables`, written prefixes, refs and leases | KV manager (M4 logical state) | M8 receives selected physical tables for one prepared traversal; a released/reused allocation invalidates it. |
| `key_cache`, `value_cache` `[block, layer, token, KV head, head dim]` | KV manager today; M5 target device owner | M8 receives per-layer views `[block, token, KV head, head dim]`; no copy or gather is required. |
| `position_ids`, `write_blocks`, `write_offsets` | KV preparation or M5 staging | M5 writes K/V at `[block, layer, offset]`; `position_ids` also feeds rotary embedding. |
| `block_tables` `int32 [sequence, width]` | M8 semantic construction; M5 may stage storage | Each row selects the M4 physical plane for its depth; unused entries are `-1` in the synchronous path. Packed prefill has one row per contiguous request/depth chunk. |
| `context_lengths` `int32 [sequence]` | M8 semantic construction | Causal visible length is the last query position plus one; no future K/V may be read. |
| `cu_seqlens_q` `int32 [chunks+1]`, `max_seqlen_q` | M8 semantic construction | Only packed FA4 prefill uses them; `cu_seqlens_q` groups contiguous query chunks. |
| `attention` implementation, `attention_info` | M8 backend selection, currently stored on KV manager | Model runner, engine, graph path, PD reporting, and backend calls. Selection is explicit; no silent fallback. |

The selected logical plane is `depth` for LAST_EXITED and `0` for SHARED (`core/kv_cache_manager.py:350-351`). M4's prefix validity check at `:536-544` prevents any backend from reading unwritten history. Early exit finalization at `:589-615` copies the last computed token to skipped planes only in LAST_EXITED, then marks those positions written. M8 must not infer that every depth has its own plane under SHARED.

## Executable branch review

| Baseline code | Inputs, branches, side effects and failures |
| --- | --- |
| `core/kv_cache_manager.py:82-147` | Validates positive dimensions, backend name, dtype and layout; Triton requires CUDA and head dimension at most 256. Flash backend construction checks architecture/package separately. Allocates physical tensors and records `attention_info`. |
| `:359-397` | Validates layer/depth/position, one-dimensional integer positions and equal row counts. Missing allocation raises `KeyError`; tensor positions are copied to host before preparing a CPU-owned row list. |
| `:398-479` | Computes write addresses, rejects duplicate writes, pads page tables to the largest queried page width. Packed mode requires LAST_EXITED + FA4; adjacent positions of a request/depth must be contiguous and each sequence appears in one chunk. Emits five metadata tensors for ordinary rows, six for packed rows, once per traversal. Empty input produces shaped empty tensors. |
| `:480-498` | Rejects wrong/stale prepared descriptors and mismatched query or KV shape/dtype/device/head count before device access. |
| `:499-535` | Public write prepares a descriptor. Prepared write rejects read-only rows, writes K/V to physical addresses and records logical written positions; empty rows return without mutation. |
| `:536-588` | Attend requires a complete written prefix for every row and layer. Empty rows return an empty output. Packed rows call `prefill`; ordinary rows call the backend with selected tables and causal lengths. |
| `kernels/paged_attention.py:12-37` | Torch reference gathers only visible tokens, repeats KV heads for GQA and computes FP32 scores/softmax/output. Zero length produces zero output. The per-row `.tolist()` is a synchronization for CUDA; this backend is a reference path. |
| `kernels/paged_attention.py:39-54` | Triton adapter explicitly rejects non-CUDA or head dimension above 256 and imports the kernel lazily; no silent fallback. |
| `kernels/triton_attention.py:13-98,100-125` | One program per query head, depth-selected page table per row, bounded online softmax, masked last tile and 64-bit block offset math. Empty batch launches nothing. Inputs are assumed validated by the manager. |
| `kernels/flash_attention.py:16-32` | Auto selects FA2 on SM8, FA3 on SM9 and FA4 on SM10; explicit generations are checked. Paged FA4 on SM12 is rejected before import. |
| `kernels/flash_attention.py:34-64` | Requires NVIDIA CUDA, FP16/BF16, head dimension divisible by eight and at most 256; FA2 requires page size divisible by 256. Imports the chosen package and reports exact implementation/version; missing packages raise without fallback. |
| `kernels/flash_attention.py:66-96` | Packed FA4 prefill uses `seqused_k`, page tables and bottom-right causal alignment; returns the tensor from a possible `(output, LSE)` tuple. Other generations reject packed mode. |
| `kernels/flash_attention.py:98-129` | Single-query rows use already written K/V; FA4 fixes `num_splits=1` for batch-stable BF16 reduction order, FA2/3 use their distinct page-table keyword. Empty rows return immediately; FA4 tuple outputs are normalized. |
| `worker/buffers.py:65-102` | Static execution workspace validates rows and duplicate write addresses, zeroes padded host rows, copies position/table/length tensors to reusable GPU buffers, and returns a borrowed descriptor. Its `acquire`/`release` event guards DMA source reuse (`:49-55`); this is M5 lifetime control. |
| `worker/async_state.py:144-209` | Request slots and versioned pinned tables are prepared for the asynchronous bank. Recurrent/finalize rows resolve physical addresses and reject duplicates; the descriptor borrows bank tensors sized for padded execution. `:211-217` releases request slot ownership; bank event handling is in `RoutingBank:39-93`. |
| `worker/prefill_metadata.py:19-108` | Waits for the prior event before reusing host storage; packs contiguous request chunks, end lengths, cumulative query lengths and depth tables, stages one pinned buffer and expands device addresses per depth. `metadata(depth)` borrows those tensors; `release()` records the consumer event. This path assumes the upstream packed sequence order. |
| `worker/cuda_graph.py:43-121` | Falls back for oversized, empty or packed batches and when graph count is capped. Otherwise checks allocation/prefix liveness, copies current addresses into graph-private tensors, captures or replays, clones output before reuse, records completion, then advances written-prefix bookkeeping. |

## Findings and first migration

1. **Design concern, confirmed by code:** Backend selection and implementation capability live in the KV manager constructor (`:110-131`). Engine and ModelRunner inspect the concrete Flash object's `generation` (`engine/llm_engine.py:69`, `worker/model_runner.py:274`). This couples logical allocation to an Attention implementation detail. Extract a small factory and a `packed_prefill` capability now, without changing selection semantics.
2. **Design concern, confirmed by code:** `_prepare_batch` combines M4 allocation identity and depth selection with M8 causal metadata packing (`:398-479`). Extract a pure M8 builder that accepts already selected physical tables, positions, and opaque request/depth identity. Keep M4 row/address/prefix/liveness validation in the manager. Continue returning `_PreparedKVBatch` so current M5 producers and model callers remain compatible.
3. **Necessary complexity:** M5's static, async, UVA and graph buffers are distinct storage strategies (`worker/buffers.py`, `async_state.py`, `prefill_metadata.py`, `cuda_graph.py`). Moving those allocations into M8 would transfer stream/event ownership incorrectly. Their semantic table/length rules should converge in later increments, after M4/M5 agree on a borrowed metadata lifetime.
4. **Open question for M4/M5 review:** Current `_PreparedKVBatch` borrows allocation identities and device tensors but does not represent an explicit GPU completion lease. Synchronous liveness is checked before dispatch; async buffers use their own events. Define any shared completion/borrowing contract jointly before replacing those paths. This review does not claim a correctness defect here.

The first increment leaves public backend strings, `attention_info`, `_PreparedKVBatch` fields and kernel call signatures unchanged. It introduces `AttentionRows` (parallel selected page tables and causal positions, with opaque sequence/depth keys only for packed prefill), `AttentionMetadata` (kernel-ready tensors), `create_backend`, and `BackendCapabilities.packed_prefill`. M4 resolves the table and checks allocation validity; M8 groups/pads the rows and reports kernel capability; M5 owns any reusable storage. No extra per-layer tensor construction, host readback, or backend fallback is permitted.

## Validation and removal conditions

Run direct builder/factory tests and existing prepared-KV, attention, Flash selection, prefix, model and engine tests on CPU. On the RTX 4090 run matched-input Torch versus Triton attention for mixed request/depth, fragmented physical pages, long contexts, GQA, all supported dtypes, and shared/LAST_EXITED layouts. FA2 can be checked if its package is available; FA3/FA4 and packed FA4 cannot be claimed on SM8.9. Compare before/after latency only under identical device, PyTorch/Triton, inputs and warmup. For packed FA4, preserve existing metadata shape/error tests and leave actual GPU execution for SM9/SM10. Regressions in output, selected backend, error type, or synchronization invalidate the structural increment.

After M4 and M5 agree on a borrowed descriptor and completion lifetime, migrate the reusable buffer producers one at a time. Remove duplicate construction only after each path has equivalent tests for cancellation, stale allocations, request-ID reuse, empty/inactive rows, graph replay and prefix growth. Treat any behavioral fix as a separate change with its own failing regression.

## First-increment validation on RTX 4090

Environment: autodl-4090, NVIDIA GeForce RTX 4090 (SM 8.9, 24 GiB), driver 580.76.05, PyTorch 2.14.0+cu130, Triton 3.8.0. Baseline is the pinned commit above; the changed checkout uses the same packages and GPU. No model weights were downloaded. Benchmark scripts in `benchmarks/bench_attention_m8_metadata.py` and `benchmarks/bench_attention_m8_engine.py` run on either checkout via `PYTHONPATH=<checkout>`.

| Check | Result |
| --- | --- |
| `python -m pytest -q` on the changed checkout | 251 passed, 114 skipped (GPU-marked tests skipped). |
| `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_attention.py tests/test_async_pipeline.py tests/test_cuda_graph.py::test_graph_cache_limit_fallback_and_abort -m gpu --run-gpu -q` | 27 passed, 15 deselected. |
| `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_cuda_graph.py -k triton --run-gpu -q` | 12 passed, 14 deselected; sync/async/multi-stream, SHARED/LAST_EXITED and static/nonstatic cases. |
| `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_attention.py --run-gpu -q` on the untouched baseline | 20 passed. |

The metadata benchmark prepares eight mixed-depth rows with unequal causal lengths. Output page tables and context lengths match exactly. Across eight alternating samples of 300 preparations per checkout in one Python process, the medians were **169.7 µs per batch** for baseline and **180.1 µs** for this increment. A separate final-script run gave **164.7/175.0 µs**, respectively. Both show about **6% slower** metadata preparation. The extra Python-side interface work is a known performance cost; it has not been characterized at production model sizes.

The small-model benchmark uses a random-weight Ouro `tiny` model with BF16, Triton, eight requests, chunked prefill and six generated tokens each. Both checkouts produced the same token/depth digest (`a811f023118451da`) and released all KV blocks. Three process-level runs, each with one warmup and five measured generations, gave median total times of **129.5/124.4/132.6 ms** for baseline and **133.5/125.9/123.4 ms** for the changed checkout. Median TTFT was **12.4/12.0/12.6 ms** and **12.6/13.9/12.0 ms**, respectively. These measurements are noisy and do not establish performance equivalence; the metadata microbenchmark indicates a small regression worth revisiting before broader M5 integration.

FA2 was not installed in this environment. FA3 and FA4 execution, including packed FA4 prefill, cannot be tested on SM 8.9. Their selection/error and packed-metadata construction paths are covered by CPU tests, but kernel numerics and performance remain unverified on compatible hardware.
