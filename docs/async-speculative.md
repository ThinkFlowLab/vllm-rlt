# Cross-round asynchronous self-speculation

This opt-in path uses fixed-depth Ouro, `last_exited` KV, greedy sampling,
Triton attention, eager CUDA execution, and refill scheduling. It supports H20;
FlashAttention is not required. Draft depth is configurable, with `d=2, D=4`
as the initial validation configuration. The target depth must match the model.

```python
from vllm_rlt import LLM, ExecutionConfig, SamplingParams, SpeculativeConfig

llm = LLM(
    "artifacts/models/Ouro-1.4B",
    device="cuda",
    attention_backend="triton",
    execution_config=ExecutionConfig(async_scheduling=True),
    speculative_config=SpeculativeConfig(num_speculative_tokens=4),
)
outputs = llm.generate(
    ["Explain speculative decoding."],
    SamplingParams(temperature=0, max_loops=4, exit_threshold=1.0, max_tokens=64),
)
```

The CLI combines `--async-scheduling --attention-backend triton
--speculative-tokens 4 --draft-loops 2 --target-loops 4 --temperature 0`.
Random sampling, adaptive exit, CUDA Graphs, preemption and PD are rejected.
The speculative runner owns its compute and result-copy streams. Setting
`multi_stream=False` orders the copy on the compute stream as well; cross-round
CPU/GPU submission remains asynchronous, but D2H no longer overlaps computation.

## Execution and ownership

The synchronous runner reads the acceptance result before constructing the next
round. Cross-round submission needs a different ownership contract: the GPU must
advance the next token and actual KV position while the CPU reserves a safe upper
bound. `_DeviceKVBatch` describes that device-owned frontier without treating
uncommitted positions as populated CPU KV. The scheduler tracks outstanding
output reservations separately from committed tokens.

A simpler alternative is to copy each result asynchronously but wait before
submitting the next round for that request. That can overlap independent requests,
but retains a CPU dependency between consecutive rounds. The two-bank design
removes that dependency at the cost of device state, reservation slack and up to
one discarded round after EOS. The existing synchronous speculative path remains
available by disabling `async_scheduling`; ordinary decoding remains the default
when no `SpeculativeConfig` is supplied.

Prefill and the first sampled token use the native bootstrap path. Thereafter,
the GPU retains each request's next input token and actual position. A round
drafts candidates, retains shallow hidden states, verifies the deep loops in a
batch, and selects the accepted prefix plus correction/bonus on device. Each
query row has its own context length; no packed FlashAttention path is used.

Two round banks allow the CPU to submit the next round before delivering the
previous result. All request computation runs in stream order. CPU positions
include outstanding output reservations and are upper bounds used only for
capacity allocation; GPU positions determine RoPE, KV writes and attention.
Requests near output/context limits may wait for a result to recover rejected
reservation capacity. An effective K of zero is a full-depth single-token step.

For a round beginning at position `s` with `a` accepted candidates, the valid
KV frontier and next input position are `s+a+1`. Invalid suffix bytes may remain
in allocated pages, but attention lengths exclude them. The CPU never truncates
GPU state when collecting an older round. CPU `written` bookkeeping continues
to describe native prefill, not in-flight speculative decode; device descriptors
use the reserved allocation and stream-ordered writes instead. Only completed
full prompt pages are eligible for prefix caching.

Every ticket owns its result buffer and page-table snapshot until compute and
copy events complete. EOS/length stopping is applied in output order on the CPU;
one already-submitted later round can finish but its outputs are discarded.
Cancellation removes scheduling eligibility immediately. KV allocations and
device state survive until their final GPU user completes. Request object
identity isolates stale tickets when a request ID is reused.

## Validation

Inside a Slurm GPU allocation, use the repository environment:

```bash
.venv/bin/python -m pytest tests/test_async_speculative.py --run-gpu -q
.venv/bin/python -m pytest tests/test_serving.py --run-gpu -k speculative -q
```

The tests exercise two GPU rounds before CPU delivery, every rejection position,
K/length boundaries, mixed requests, prefix reuse, EOS, cancellation, ID reuse
and partially submitted failures. CPU-only test results do not qualify the GPU
path. Compare BF16 runs on the same hardware and Triton backend: synchronous
speculation versus asynchronous speculation, and fixed-depth native async versus
async speculation. Record committed tokens, variability, memory and traces;
do not infer a speedup from successful asynchronous submission alone.

The matched engine-only benchmark uses the same local checkpoint, BF16, prompt,
output budget and Triton backend for native async, synchronous speculation and
asynchronous speculation:

```bash
.venv/bin/python -m benchmarks.async_speculative \
  --model artifacts/models/Ouro-1.4B --output /path/to/results.json \
  --concurrency 1 8 --k 1 2 4 8 --repeats 5
```

Run this inside a Slurm GPU allocation. Each observation follows a warmup;
measurement starts after all first tokens and ends after completed GPU work.
JSON includes every output sequence, source/model hashes, throughput, delivery
times and peak allocated/reserved memory. Any mismatch against the matched native
output makes the command fail after retaining the observations. `--profile`
adds separate warmed single-request captures with a `measured_decode` region;
profiled timings are excluded from throughput observations. This is a controlled
engine microbenchmark, not an HTTP load test or a model-quality evaluation.

The initial H20 BF16 results, test coverage, aligned profiler figures and
qualification limits are recorded in
[async-speculative-validation.md](async-speculative-validation.md).
