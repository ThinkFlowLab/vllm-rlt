"""Measure a fixed small Ouro generation workload and check output identity."""

import hashlib
import json
import statistics
import time
from dataclasses import replace

import torch

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM

torch.set_num_threads(1)
torch.manual_seed(73)
model = OuroForCausalLM(replace(OuroConfig.tiny(), head_dim=64)).to("cuda", torch.bfloat16).eval()
prompts = [[2 + i, 3, 4, 5] * (1 + i % 3) for i in range(8)]
params = SamplingParams(max_tokens=6, exit_threshold=1.0, ignore_eos=True)
measurements = []
final_digest = None
for trial in range(6):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=128, block_size=16, layout="last_exited"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8, max_num_batched_tokens=16, prefill_chunk_size=4
        ),
        attention_backend="triton",
    )
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, params)
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = None
    finished = {}
    steps = 0
    with torch.inference_mode():
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.token_ids and first is None:
                    first = time.perf_counter()
                if output.finished:
                    finished[output.request_id] = (output.token_ids, output.exit_depths)
            steps += 1
            assert steps < 300
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert len(finished) == len(prompts)
    assert engine.cache_manager.num_used_blocks == 0
    digest = hashlib.sha256(json.dumps(finished, sort_keys=True).encode()).hexdigest()[:16]
    if final_digest is not None:
        assert digest == final_digest
    final_digest = digest
    if trial:
        measurements.append((elapsed * 1000, (first - start) * 1000))
print("digest", final_digest)
print("total_ms_median", round(statistics.median(x[0] for x in measurements), 1))
print("ttft_ms_median", round(statistics.median(x[1] for x in measurements), 1))
print("total_ms_samples", ",".join(f"{x[0]:.1f}" for x in measurements))
print("gpu", torch.cuda.get_device_name())
