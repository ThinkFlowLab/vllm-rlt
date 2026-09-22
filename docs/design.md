# vllm-rlt engine design

The engine schedules one traversal of Ouro's shared transformer layers at a time. It owns request state and depth-aware KV allocation.

## Components and ownership

```text
LLM facade: tokenization, output collection
    -> engine: request lifecycle
        -> scheduler: admission and stage selection
        -> model runner: execute a selected batch
            -> native Ouro: embedding, recurrent core, output head
            -> attention backend: PyTorch reference or Triton
        -> KV manager: reservations, block tables, release
```

The scheduler owns queue membership, request progress, and loop depth. The runner owns tensor execution and the hidden states that survive between stages. The KV manager owns physical block allocation. Model code receives the positions and cache addresses required by a batch; it does not admit requests or decide which request executes next.

Each request has at most one token undergoing autoregressive decoding. Different requests may contribute tokens at different positions and loop depths to the same recurrent batch. Their data remains separate through per-request positions, block tables, and lengths.

## Request lifecycle

Waiting requests enter prefill after obtaining a complete KV reservation. LAST-EXITED prefill embeds a prompt chunk and executes its complete recurrent depth before moving to another chunk. SHARED prefill completes one position at all depths before advancing each request, batching across requests; this preserves chunk invariance. Causal attention uses the selected layout's history. After the last chunk, the final prompt hidden state enters coda and produces the first generated token.

If generation continues, the sampled token enters prelude for embedding, then recurrent execution. After each recurrent pass, the gate either returns that token to recurrent work or routes its final hidden state to coda. Coda computes logits and samples the following token. A completed or cancelled request releases its reservation and persistent state.

Full-depth prefill is an implementation choice. The last prompt token is not reprocessed, and a sampled token enters KV only after its forward pass.

## Scheduling modes

Refill scheduling allows newly prepared decode tokens to join continuing recurrent tokens. A prelude queued by coda runs immediately to prepare the sampled token for re-entry. Otherwise ready coda work runs first, followed by eligible prompt admission/prefill and recurrent work. Thus a request beginning loop one can execute alongside a different request beginning loop four.

No-refill holds a decode cohort. Tokens leave its recurrent batch when they exit, but its coda waits until the remaining cohort finishes. The next cohort starts after that boundary. Both modes use the same model, gate, and cache semantics.

Synchronous execution remains the default. Optional random lookahead or trace replay
supports a pipeline that submits the next required loop before consuming the previous
signal, and allows boundary stages on another CUDA stream. Async mode uses pinned,
reusable inputs and event-protected resource lifetimes. CUDA graphs are deferred.
See [CDB runtime details](cdb_runtime.md) for implementation, semantics and validation.

## Gate and model contract

Ouro's embedding is the prelude. One recurrent operation runs all physical transformer layers and the shared end-of-loop normalization. The normalized hidden state persists for the next loop. Coda applies the LM head to the exited state.

Decode exits when `1 - product(1 - sigmoid(gate_i)) >= exit_threshold`, subject to the loop bounds. Every loop contributes, including those before the minimum allowed exit depth. Default adaptive bounds are two through four loops; threshold `1.0` selects fixed depth explicitly. Token position and its RoPE phase stay unchanged while loop depth advances.

Checkpoint loading defaults to BF16; explicit dtype overrides are supported. Existing model objects retain their dtype, and engine KV storage uses that same dtype. RMSNorm and RoPE use FP32 intermediates; Triton attention accumulates in FP32.

## Last-exited paged KV

Keys and values have separate physical tensors, each laid out as:

```text
[physical_block, physical_layer, token_offset, kv_head, head_dimension]
```

A request's block table maps `(loop_depth, logical_token_block)` to a physical block. There is no assumption that a recurrent batch shares one loop depth. Attention selects the depth-specific block-table row for each token and reads only its request's populated context, including the current token after its KV write.

When a token exits at depth `d`, the manager fills its slots at each skipped deeper depth with that token's computed depth-`d` keys and values. This operation copies each physical layer's KV independently. It does not copy the final hidden vector, neighboring tokens, or the whole partially populated block. Shallower computed depths remain intact. Future tokens can consequently attend at any supported depth without encountering missing entries.

The initial allocator reserves enough blocks for every input position the request can execute, at every depth. The last sampled output token needs no forward pass or KV entry. Required physical blocks are:

```text
model.total_ut_steps * ceil((prompt_tokens + max_tokens - 1) / block_size)
```

The depth factor is the model's full prefill depth, even when a request sets a lower decode loop limit. SHARED uses one physical plane and omits the depth multiplier. Admission can bypass temporarily blocked requests within configured scan and fairness limits. A protected long request waits until this full reservation fits. A request that cannot fit even in an empty cache must fail with an actionable capacity error. An admitted request can then reach completion without requesting additional blocks; this prevents admission from consuming space needed to finish existing requests. The cost is conservative memory use. Prefix sharing, eviction, swapping, and incremental reservation are outside this version.

Unused reserved positions are not valid KV. Attention lengths and block metadata must exclude them. Release occurs only after task-owned execution no longer references a request's blocks; reusing a block must not expose a previous request's contents.

For the released BF16 Ouro configuration, each token at one depth requires `2 * 24 * 16 * 128 * 2 = 196,608` bytes of KV. Four depths require 768 KiB per token before block rounding. Cache capacity is either explicitly overridden or profiled from actual CUDA memory and execution limits, not inferred solely from model parameter size.

## Correctness coverage

Core tests compare fixed-depth model execution with a dense reference and packed generation with serial engine execution. They also cover mixed depths, admission pressure, cancellation, partial blocks, skipped-depth copies, isolation, and block reuse. See [test commands and historical results](../README.md#tests).
