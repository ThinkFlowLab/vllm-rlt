# Persistent eager decode storage

Normal generation uses compact rows. The private
`engine.model_runner._enable_persistent_decode()` hook opts into reusable eager
decode storage for experiments with stable input, output and metadata pointers.
It supports matching BF16 or FP32 model and KV dtypes. This is not a public
serving option, CUDA graph implementation or speedup claim.

The runner owns one fixed capacity: four live requests mapped to odd slots of
eight physical rows, with 32 block-table columns. More live requests or a
position requiring a wider table use compact allocating execution. These
fallbacks are counted by `_persistent_snapshot()`. The capacity is independent
of `SchedulerConfig.max_num_seqs`; enabling it does not promise that every
scheduled batch uses persistent storage.

The hidden and gate buffers use the model dtype. Tensor payload is checked
against a 256 KiB cap before allocation; metadata also has 1,256 bytes of
unpinned CPU staging. For each persistent traversal, six staging tensors are
refilled and copied to the execution device. Model intermediates and the
independently published live outputs still allocate. Stable buffers do not
imply allocation-free execution.

The cache owns allocation identities; each prepared descriptor borrows one
generation of the runner's storage. A new generation rejects stale descriptors,
including descriptors for a freed and reallocated request ID. The runner uses
its original ordered CUDA stream. The real gate readback completes publication
before the lease is released; direct helper calls synchronize explicitly.
Published request state owns separate storage so a request paused at CODA can
survive subsequent buffer refills without changing its hidden state or RNG.

After an execution failure, the persistent executor cannot be retried. If stream
completion is confirmed, affected requests can be aborted. If completion cannot
be confirmed, the cache is quarantined and its allocation identities are kept;
pages cannot be recycled. Finalization failures are settled before cleanup too.
This conservative policy also disables an enabled executor after a prefill or
coda failure. Hardware fault recovery and asynchronous lookahead are outside
this contract.

Keeping compact execution is the simpler alternative and remains the default.
Allocating padded execution is the same-shape correctness control for storage
reuse. Comparing it with persistent execution isolates storage lifetime from
the numerical effects of changing the GEMM row count. A capacity ladder could
serve larger batches, but requires a separate design and qualification.

The inherited inactive-row prerequisite has an unresolved
[BF16 compact-versus-padded numerical qualification failure](https://github.com/hsliuustc0106/vllm-lt/releases/tag/pr27-gpu-qualification-20260913).
Passing same-shape storage checks does not resolve that failure or qualify
compact-to-persistent numerical equivalence, task accuracy, or performance.
Historical FP32 evidence is available in the
[experiment archive](https://github.com/hsliuustc0106/vllm-lt/releases/tag/implementation-notes-archive-20260913).
