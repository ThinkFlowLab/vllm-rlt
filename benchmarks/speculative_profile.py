"""Out-of-band phase timings for one warmed synchronous speculative trial.

CUDA events bracket queued GPU work and can include host dispatch gaps. CPU wall
times include Python work and implicit synchronization. The two quantities
overlap and must not be added or read as hardware utilization.
Profiling hooks are removed after the trial and are never used in throughput
measurements.
"""

import argparse
import importlib.metadata
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.speculative import cache_blocks_for_case, make_prompts, run_decode_trial
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def profile_trial(engine, prompts, params):
    """Record call counts, CPU wall time and CUDA event time by phase."""
    runner = engine.speculative_runner
    if runner is None:
        raise ValueError("a speculative engine is required")
    model = engine.model
    device = next(model.parameters()).device
    originals = {
        "core": runner._core,
        "execute": runner.execute,
        "prelude": model.prelude,
        "coda": model.coda,
        "commit": engine._update_speculative,
    }
    records = defaultdict(list)
    active = False
    final_positions = {}

    def measure(label, function, *args, **kwargs):
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        wall_ms = (time.perf_counter() - start) * 1000
        if end_event is not None:
            end_event.record()
        records[label].append((wall_ms, start_event, end_event))
        return result

    def core(hidden, ids, positions, depth, **kwargs):
        if depth >= runner.config.draft_loops:
            label = "target_core"
        elif all(
            position == final_positions[request_id] for request_id, position in zip(ids, positions)
        ):
            label = "final_shallow_core"
        else:
            label = "draft_core"
        return measure(label, originals["core"], hidden, ids, positions, depth, **kwargs)

    def execute(batch):
        nonlocal active, final_positions
        final_positions = {
            item.request.request_id: item.token_start + item.token_count - 1 for item in batch.items
        }
        active = True
        try:
            return measure("execute_wall", originals["execute"], batch)
        finally:
            active = False

    def prelude(*args, **kwargs):
        if active:
            return measure("prelude", originals["prelude"], *args, **kwargs)
        return originals["prelude"](*args, **kwargs)

    def coda(*args, **kwargs):
        if active:
            return measure("coda", originals["coda"], *args, **kwargs)
        return originals["coda"](*args, **kwargs)

    def commit(*args, **kwargs):
        return measure("commit", originals["commit"], *args, **kwargs)

    runner._core = core
    runner.execute = execute
    model.prelude = prelude
    model.coda = coda
    engine._update_speculative = commit
    try:
        trial = run_decode_trial(engine, prompts, params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phases = {}
        for label, samples in records.items():
            phases[label] = {
                "calls": len(samples),
                "cpu_wall_ms": sum(row[0] for row in samples),
                "cuda_event_ms": sum(row[1].elapsed_time(row[2]) for row in samples)
                if device.type == "cuda"
                else None,
            }
        return {"trial": trial, "phases": phases}
    finally:
        runner._core = originals["core"]
        runner.execute = originals["execute"]
        model.prelude = originals["prelude"]
        model.coda = originals["coda"]
        engine._update_speculative = originals["commit"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args(argv)
    if args.prompt_tokens < 1 or args.output_tokens < 2 or args.concurrency < 1 or args.k < 1:
        parser.error("prompt, output, concurrency and K must be positive; output >= 2")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA with BF16 support is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = make_prompts(tokenizer, args.prompt_tokens, args.concurrency)
    model = OuroForCausalLM.from_pretrained(str(args.model), device="cuda", dtype=torch.bfloat16)
    params = SamplingParams(
        max_tokens=args.output_tokens,
        max_loops=model.config.total_ut_steps,
        exit_threshold=1.0,
        temperature=0,
        ignore_eos=True,
    )
    scheduler = SchedulerConfig(
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=max(128, args.prompt_tokens * args.concurrency),
        prefill_chunk_size=args.prompt_tokens,
    )
    blocks = cache_blocks_for_case(
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        concurrency=args.concurrency,
        depth=model.config.total_ut_steps,
        k=args.k,
    )
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(blocks, 16),
        scheduler_config=scheduler,
        attention_backend="flash_attn_4",
        speculative_config=SpeculativeConfig(args.k, target_loops=model.config.total_ut_steps),
    )
    run_decode_trial(engine, prompts, params)
    result = {
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": args.model_revision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "flash_attn_4": importlib.metadata.version("flash-attn-4"),
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "k": args.k,
        "measurement": (
            "out-of-band instrumentation; CUDA event spans can include host gaps "
            "and overlap CPU wall times"
        ),
        **profile_trial(engine, prompts, params),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["phases"]), flush=True)


if __name__ == "__main__":
    main()
