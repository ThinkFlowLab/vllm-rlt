# Nanbeige4.2 recurrent CUDA Graph on RTX 5080

This recipe validates fixed two-loop Nanbeige4.2 inference with the shared
recurrent graph runner. It covers BF16 weights and Triton attention on one RTX
5080. It does not validate FlashAttention, speculative decoding, PD, or every
checkpoint and workload.

## Environment and checkpoint

- Code: `ThinkFlowLab/vllm-rlt` main at `299bf14b117f42a38d15852886d673f89e123307`.
- GPU: NVIDIA GeForce RTX 5080 (SM120, 16 GiB); PyTorch `2.14.0+cu130`, Triton
  `3.8.0`, Python `3.12.3`.
- Checkpoint: `Nanbeige/Nanbeige4.2-3B` at
  `b82e54bd609793562a75cbf9337970a93369eab5` (BF16 safetensors).
- Backend: Triton, fixed two loops, `last_exited` KV, greedy decoding, EOS ignored
  for fixed-length comparisons.

From the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Nanbeige/Nanbeige4.2-3B",
    revision="b82e54bd609793562a75cbf9337970a93369eab5",
    local_dir="artifacts/models/Nanbeige4.2-3B",
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py"],
)
PY
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  tests/test_cuda_graph.py -k nanbeige --run-gpu
```

The GPU tests compare eager and captured generation for four concurrent requests
with different prompt and generation lengths. They exercise synchronous and
multi-stream asynchronous scheduling, KV release, and request-ID reuse. Both
tests passed on the configuration above. A separate real-checkpoint smoke run
generated the same three tokens and exit depths with and without graphs:
`[13, 295, 152406]` at depths `[2, 2, 2]`. The captured run recorded one graph
capture and four replays.

## Repeated real-checkpoint measurement

The measurement used one warmup run followed by three timed runs for each mode.
Each run reused request IDs after completion. Timing synchronized CUDA before
and after generation and excluded model loading. The four-request workload used
synthetic token prompts of lengths 4, 8, 12, and 16, with 16 generated tokens
per request. The one-request control used a four-token prompt and 16 generated
tokens. Both used `CacheConfig(num_blocks=128, block_size=16)`,
`SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32,
prefill_chunk_size=8)`, and default synchronous `ExecutionConfig` apart from
`cuda_graphs`.

To repeat the comparison from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python - <<'PY'
import gc
import statistics
import time

import torch

from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import NanbeigeForCausalLM

model = NanbeigeForCausalLM.from_pretrained(
    "artifacts/models/Nanbeige4.2-3B", device="cuda", dtype=torch.bfloat16
)


def measure(prompts):
    eager_outputs = None
    for graphs in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=128, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=32, prefill_chunk_size=8
            ),
            execution_config=ExecutionConfig(cuda_graphs=graphs),
            attention_backend="triton",
        )
        times, outputs = [], []
        for trial in range(4):  # warmup, then three measured runs
            for index, prompt in enumerate(prompts):
                engine.add_request(
                    str(index), prompt,
                    SamplingParams(
                        max_tokens=16, min_loops=2, max_loops=2,
                        temperature=0, ignore_eos=True,
                    ),
                )
            torch.cuda.synchronize()
            start = time.perf_counter()
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            torch.cuda.synchronize()
            if trial:
                times.append(time.perf_counter() - start)
                outputs.append(finished)
            assert engine.cache_manager.num_used_blocks == 0
        if eager_outputs is None:
            eager_outputs = outputs
        else:
            assert outputs == eager_outputs
        stats = engine.model_runner.graphs
        print(len(prompts), graphs, times, statistics.median(times),
              None if stats is None else (stats.captures, stats.replays, stats.fallbacks))
        del engine
        gc.collect()
        torch.cuda.empty_cache()


measure([[2 + i, 3, 4, 5] * (i + 1) for i in range(4)])
measure([[2, 3, 4, 5]])
PY
```

| Workload | Eager times (s) | Graph times (s) | Median eager / graph (s) | Captures / replays / fallbacks |
| --- | --- | --- | --- | --- |
| 4 requests, 64 output tokens | 0.4452, 0.4520, 0.4447 | 0.3783, 0.3785, 0.3779 | 0.4452 / 0.3783 | 2 / 124 / 0 |
| 1 request, 16 output tokens | 0.3639, 0.3613, 0.3674 | 0.2995, 0.2998, 0.2999 | 0.3639 / 0.2998 | 1 / 120 / 0 |

All generated token IDs and exit depths matched across eager and graph runs.
These are local observations on one machine, not a general throughput guarantee.
CUDA Graph capture happened during warmup; changing shapes or graph cache limits
can change memory use and the hit rate.
