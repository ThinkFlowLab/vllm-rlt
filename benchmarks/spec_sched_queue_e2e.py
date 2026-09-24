"""Real-model speculative E2E with and without a saturated admission queue."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def run(engine, prompts, params):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, params)
    first, finished = {}, {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.token_ids and output.request_id not in first:
                first[output.request_id] = time.perf_counter() - start
            if output.finished:
                finished[output.request_id] = (output.token_ids, output.exit_depths)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return dict(
        seconds=seconds,
        median_ttft=statistics.median(first.values()),
        max_ttft=max(first.values()),
        tokens_per_second=sum(len(tokens) for tokens, _ in finished.values()) / seconds,
        outputs=finished,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, choices=(1, 8), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    model = OuroForCausalLM.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=1024),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=128),
        execution_config=ExecutionConfig(cuda_graphs=True),
        speculative_config=SpeculativeConfig(4, interleave_round=True),
        attention_backend="triton",
    )
    prompts = [([2 + index, 43, 314, 2718, 9] * 12)[:60] for index in range(args.requests)]
    params = SamplingParams(max_tokens=32, min_loops=4, max_loops=4, ignore_eos=True)
    for _ in range(3):
        run(engine, prompts, params)
    rows = [run(engine, prompts, params) for _ in range(args.repeats)]
    if any(row["outputs"] != rows[0]["outputs"] for row in rows[1:]):
        raise AssertionError("repeated output mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(requests=args.requests, rows=rows), indent=2))


if __name__ == "__main__":
    main()
