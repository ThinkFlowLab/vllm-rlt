"""Matched synchronous Ouro decode evaluation for native and speculation.

Example::

    python -m benchmarks.speculative \
      --model artifacts/models/Ouro-1.4B \
      --model-revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
      --output /tmp/speculative-decode.json
"""

import argparse
import gc
import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM

PROMPT_CONTEXT = (
    "A research notebook contains observations from several unrelated fields. "
    "In an optics experiment, sunlight passes through a clear atmosphere and short "
    "wavelengths scatter more strongly than long wavelengths. In a history seminar, "
    "students compare primary sources with later interpretations and identify where "
    "the accounts disagree. A small software service receives bursts of requests, "
    "records their arrival times, and stores recently used data in a limited cache. "
    "A gardener changes only the amount of water given to two otherwise similar "
    "plots, then records growth over several weeks. For each question below, use "
    "only the relevant details, explain your reasoning, and state any uncertainty. "
)
PROMPT_QUESTIONS = (
    "Why does a clear daytime sky usually look blue, and why can the horizon look paler?",
    "What evidence would help distinguish a reliable eyewitness account from a later retelling?",
    "How would you measure whether a cache reduces request latency during a burst of traffic?",
    "Which variables should the gardener hold constant to compare the two watering schedules?",
    "Write a short Python function that counts distinct words while ignoring case and punctuation.",
    "Explain why the sum of two odd integers is even, using both an example and algebra.",
    "Describe a practical way to check whether a headline accurately summarizes a study.",
    "Give a concise recipe for lentil soup and explain when to add acidic ingredients.",
)


def run_decode_trial(engine, prompts, params):
    """Time output tokens after every request has completed prefill and first coda."""
    if params.max_tokens < 2:
        raise ValueError("decode timing requires max_tokens >= 2")
    ids = [f"bench-{index}" for index in range(len(prompts))]
    for request_id, prompt in zip(ids, prompts):
        engine.add_request(request_id, prompt, params)

    first = {}
    steps = 0
    while len(first) < len(ids):
        if not engine.has_unfinished_requests():
            raise RuntimeError("requests finished before the decode boundary")
        for output in engine.step():
            if len(output.token_ids) > 1 and len(first) < len(ids):
                raise RuntimeError("some requests decoded before the matched prefill boundary")
            if output.token_ids:
                first[output.request_id] = output.token_ids[0]
        steps += 1
        if steps > 10000:
            raise RuntimeError("prefill did not reach the first-token boundary")

    device = next(engine.model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    stats = engine.speculative_runner.stats if engine.speculative_runner else None
    before = vars(stats).copy() if stats else None
    start = time.perf_counter()
    finished = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
        steps += 1
        if steps > 10000:
            raise RuntimeError("decode did not finish")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    tokens = [finished[request_id].token_ids for request_id in ids]
    decode_tokens = sum(len(row) - 1 for row in tokens)
    return {
        "seconds": seconds,
        "decode_tokens": decode_tokens,
        "tokens_per_second": decode_tokens / seconds,
        "token_ids": tokens,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "speculative_stats": {key: value - before[key] for key, value in vars(stats).items()}
        if stats
        else None,
    }


def make_prompts(tokenizer, prompt_length, concurrency):
    """Build deterministic, equal-length token inputs for a matched batch."""
    if prompt_length < 1 or concurrency < 1:
        raise ValueError("prompt_length and concurrency must be positive")
    prompts = []
    for index in range(concurrency):
        text = PROMPT_CONTEXT + PROMPT_QUESTIONS[index % len(PROMPT_QUESTIONS)]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < prompt_length:
            raise ValueError("prompt corpus is shorter than the requested token length")
        prompts.append(tokens[-prompt_length:])
    return prompts


def cache_blocks_for_case(*, prompt_tokens, output_tokens, concurrency, depth, k):
    """Reserve all per-depth pages plus one speculative frontier per request."""
    positions = prompt_tokens + output_tokens - 1 + k
    return max(256, concurrency * depth * math.ceil(positions / 16) + 16)


def run_case(model, prompts, *, output_tokens, k, trials, backend):
    """Warm both paths, then alternate matched native/speculative decode runs."""
    if not prompts or trials < 1:
        raise ValueError("prompts must be nonempty and trials must be positive")
    params = SamplingParams(
        max_tokens=output_tokens,
        max_loops=model.config.total_ut_steps,
        exit_threshold=1.0,
        temperature=0,
        ignore_eos=True,
    )
    scheduler = SchedulerConfig(
        max_num_seqs=len(prompts),
        max_num_batched_tokens=max(128, sum(len(row) for row in prompts)),
        prefill_chunk_size=max(len(row) for row in prompts),
    )
    common = dict(
        cache_config=CacheConfig(
            cache_blocks_for_case(
                prompt_tokens=max(len(row) for row in prompts),
                output_tokens=output_tokens,
                concurrency=len(prompts),
                depth=model.config.total_ut_steps,
                k=k,
            ),
            16,
        ),
        scheduler_config=scheduler,
        attention_backend=backend,
    )
    engines = {
        "native": LLMEngine(model, **common),
        "speculative": LLMEngine(
            model,
            speculative_config=SpeculativeConfig(k, target_loops=model.config.total_ut_steps),
            **common,
        ),
    }
    for engine in engines.values():
        run_decode_trial(engine, prompts, params)

    runs = []
    for trial in range(trials):
        order = ("native", "speculative") if trial % 2 == 0 else ("speculative", "native")
        results = {label: run_decode_trial(engines[label], prompts, params) for label in order}
        runs.append(
            {
                "trial": trial,
                **results,
                "same_tokens": results["native"]["token_ids"]
                == results["speculative"]["token_ids"],
            }
        )
    summary = {
        "native_median_tokens_per_second": statistics.median(
            row["native"]["tokens_per_second"] for row in runs
        ),
        "speculative_median_tokens_per_second": statistics.median(
            row["speculative"]["tokens_per_second"] for row in runs
        ),
        "paired_median_speedup": statistics.median(
            row["native"]["seconds"] / row["speculative"]["seconds"] for row in runs
        ),
        "exact_agreement_fraction": sum(row["same_tokens"] for row in runs) / trials,
    }
    return {
        "prompt_tokens": len(prompts[0]),
        "output_tokens": output_tokens,
        "concurrency": len(prompts),
        "k": k,
        "runs": runs,
        "summary": summary,
    }


def measure_isolated_memory(model, prompts, *, output_tokens, k, backend):
    """Measure one engine at a time with the same resident model weights."""
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
    params = SamplingParams(
        max_tokens=output_tokens,
        max_loops=model.config.total_ut_steps,
        exit_threshold=1.0,
        temperature=0,
        ignore_eos=True,
    )
    scheduler = SchedulerConfig(
        max_num_seqs=len(prompts),
        max_num_batched_tokens=max(128, sum(len(row) for row in prompts)),
        prefill_chunk_size=max(len(row) for row in prompts),
    )
    cache = CacheConfig(
        cache_blocks_for_case(
            prompt_tokens=max(len(row) for row in prompts),
            output_tokens=output_tokens,
            concurrency=len(prompts),
            depth=model.config.total_ut_steps,
            k=k,
        ),
        16,
    )
    results = {}
    for label in ("native", "speculative"):
        model_bytes = torch.cuda.memory_allocated(device) if cuda else None
        config = SpeculativeConfig(k, target_loops=model.config.total_ut_steps)
        engine = LLMEngine(
            model,
            cache_config=cache,
            scheduler_config=scheduler,
            attention_backend=backend,
            speculative_config=config if label == "speculative" else None,
        )
        run_decode_trial(engine, prompts, params)
        resident = torch.cuda.memory_allocated(device) if cuda else None
        trial = run_decode_trial(engine, prompts, params)
        results[label] = {
            "model_allocated_bytes": model_bytes,
            "engine_resident_allocated_bytes": resident,
            "peak_allocated_bytes": trial["peak_allocated_bytes"],
            "decode_tokens": trial["decode_tokens"],
        }
        del engine
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned local Ouro checkpoint")
    parser.add_argument("--model-revision", required=True, help="Checkpoint commit for the report")
    parser.add_argument("--output", type=Path, required=True, help="Raw JSON result path")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--output-lengths", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--backend", default="flash_attn_4", choices=["flash_attn_4"])
    parser.add_argument("--resume", action="store_true", help="Continue a compatible result file")
    parser.add_argument("--memory-only", action="store_true", help="Measure one engine at a time")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA with BF16 support is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = OuroForCausalLM.from_pretrained(str(args.model), device="cuda", dtype=torch.bfloat16)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    results = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": args.model_revision,
        "model_path": str(args.model.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn_4": importlib.metadata.version("flash-attn-4"),
        "trials_per_case": args.trials,
        "timing_boundary": (
            "after every request's first coda; excludes matched prefill and first token"
        ),
        "execution": "synchronous eager, fixed D=4, last_exited, BF16, greedy, ignore_eos",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.memory_only:
        if args.resume:
            parser.error("--memory-only cannot be combined with --resume")
        results["scope"] = "isolated process allocations with one engine resident at a time"
        for prompt_length in args.prompt_lengths:
            for output_tokens in args.output_lengths:
                for concurrency in args.concurrencies:
                    prompts = make_prompts(tokenizer, prompt_length, concurrency)
                    for k in args.ks:
                        memory = measure_isolated_memory(
                            model,
                            prompts,
                            output_tokens=output_tokens,
                            k=k,
                            backend=args.backend,
                        )
                        case = {
                            "prompt_tokens": prompt_length,
                            "output_tokens": output_tokens,
                            "concurrency": concurrency,
                            "k": k,
                            "memory": memory,
                        }
                        results["cases"].append(case)
                        print(json.dumps(case), flush=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        return
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for field in (
            "code_sha",
            "model_revision",
            "gpu",
            "torch",
            "cuda",
            "flash_attn_4",
            "trials_per_case",
            "timing_boundary",
            "execution",
        ):
            if previous[field] != results[field]:
                raise ValueError(f"cannot resume: {field} changed")
        results = previous
    completed = {
        (case["prompt_tokens"], case["output_tokens"], case["concurrency"], case["k"])
        for case in results["cases"]
    }
    for prompt_length in args.prompt_lengths:
        for output_tokens in args.output_lengths:
            for concurrency in args.concurrencies:
                prompts = make_prompts(tokenizer, prompt_length, concurrency)
                for k in args.ks:
                    if (prompt_length, output_tokens, concurrency, k) in completed:
                        continue
                    case = run_case(
                        model,
                        prompts,
                        output_tokens=output_tokens,
                        k=k,
                        trials=args.trials,
                        backend=args.backend,
                    )
                    results["cases"].append(case)
                    temp = args.output.with_suffix(args.output.suffix + ".tmp")
                    temp.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
                    temp.replace(args.output)
                    print(
                        json.dumps(
                            {
                                "prompt_tokens": prompt_length,
                                "output_tokens": output_tokens,
                                "concurrency": concurrency,
                                "k": k,
                                **case["summary"],
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
