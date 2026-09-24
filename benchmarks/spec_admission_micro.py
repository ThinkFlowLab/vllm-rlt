"""Measure saturated speculative admission with a large waiting queue."""

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.request import Stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = LLMEngine(
        OuroForCausalLM(OuroConfig.tiny()),
        cache_config=CacheConfig(num_blocks=1024),
        scheduler_config=SchedulerConfig(max_num_seqs=2, policy="priority"),
        speculative_config=SpeculativeConfig(4),
        attention_backend="torch",
    )
    for index in range(256):
        engine.add_request(
            str(index), [2, 3], SamplingParams(max_tokens=32, priority=0, ignore_eos=True)
        )
    engine.scheduler._admit()
    waiting = tuple(engine.scheduler.queues[Stage.WAITING])
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(1000):
            engine.scheduler._admit()
        timings.append((time.perf_counter() - start) / 1000)
    assert tuple(engine.scheduler.queues[Stage.WAITING]) == waiting
    assert len(waiting) == 254
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(seconds_per_call=timings, median=statistics.median(timings)))
    )


if __name__ == "__main__":
    main()
