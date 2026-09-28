"""Matched BF16/Triton decode trials; profiling is a separate optional run.

Run inside Slurm with a local Ouro checkpoint. JSON retains every trial and
output sequence, including mismatches; timing never includes profiler overhead.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_engine(model, mode, k, concurrency, prompt_len, output_len):
    blocks = concurrency * math.ceil((prompt_len + output_len) / 16) * model.config.total_ut_steps
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(blocks, 16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency,
            max_num_batched_tokens=concurrency * max(128, k + 1),
            prefill_chunk_size=128,
        ),
        execution_config=ExecutionConfig(async_scheduling=mode != "sync_spec"),
        exit_config=ExitConfig("ouro_delayed" if mode == "native_async" else "ouro"),
        speculative_config=None if mode == "native_async" else SpeculativeConfig(k),
    )


def trial(engine, prompts, output_len, *, profile=False):
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, SamplingParams(max_tokens=output_len, ignore_eos=True))
    outputs = {}
    # Equal-length prompts and a batch budget covering all prefill chunks ensure
    # every first output is delivered together, before any decode is scheduled.
    while len(outputs) < len(prompts):
        for output in engine.step():
            outputs[output.request_id] = output
    assert all(len(o.token_ids) == 1 for o in outputs.values())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    delivery = []
    region = torch.profiler.record_function("measured_decode") if profile else nullcontext()
    with region:
        while engine.has_unfinished_requests():
            for output in engine.step():
                previous = len(outputs[output.request_id].token_ids)
                delivery.append(
                    (
                        time.perf_counter() - started,
                        output.request_id,
                        len(output.token_ids) - previous,
                    )
                )
                outputs[output.request_id] = output
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert all(o.finished and len(o.token_ids) == output_len for o in outputs.values())
    committed = len(prompts) * (output_len - 1)
    return dict(
        seconds=elapsed,
        committed_tokens=committed,
        tokens_per_second=committed / elapsed,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        delivery=delivery,
        token_ids={rid: o.token_ids for rid, o in outputs.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run inside a Slurm GPU allocation")
    if args.output_length < 2 or args.prompt_length < 1 or args.repeats < 1:
        parser.error("positive prompt/repeats and at least two outputs required")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model).resolve()
    model = OuroForCausalLM.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    text = "Explain how a computer verifies a mathematical proof, with a concrete example. "
    base = tokenizer.encode(text, add_special_tokens=False)
    prompt = (base * math.ceil(args.prompt_length / len(base)))[: args.prompt_length]
    record = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_hashes={
            str(p.relative_to(root)): file_hash(p)
            for p in sorted((root / "vllm_rlt").rglob("*.py"))
        },
        benchmark_hash=file_hash(Path(__file__)),
        model_hashes={p.name: file_hash(p) for p in sorted(model_dir.glob("*.safetensors"))},
        config_hash=file_hash(model_dir / "config.json"),
        node=platform.node(),
        job=os.environ.get("SLURM_JOB_ID"),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        dtype="bfloat16",
        backend="triton",
        arguments=vars(args),
        prompt=prompt,
        trials=[],
    )
    torch.set_num_threads(1)
    for concurrency in args.concurrency:
        prompts = [prompt] * concurrency
        for k in args.k:
            baseline = None
            for repetition in range(args.repeats):
                modes = ["native_async", "sync_spec", "async_spec"]
                if repetition % 2:
                    modes.reverse()
                for mode in modes:
                    engine = make_engine(
                        model, mode, k, concurrency, args.prompt_length, args.output_length
                    )
                    trial(engine, prompts, args.output_length)
                    result = trial(engine, prompts, args.output_length)
                    if baseline is None and mode == "native_async":
                        baseline = result["token_ids"]
                    result.update(
                        mode=mode,
                        k=k,
                        concurrency=concurrency,
                        repetition=repetition,
                        exact_native_agreement=result["token_ids"] == baseline,
                    )
                    if engine.speculative_runner is not None:
                        result["cumulative_stats_including_warmup"] = asdict(
                            engine.speculative_runner.stats
                        )
                    record["trials"].append(result)
                    output.write_text(json.dumps(record, indent=2))
                    print(
                        json.dumps(
                            {
                                key: result[key]
                                for key in (
                                    "mode",
                                    "k",
                                    "concurrency",
                                    "repetition",
                                    "tokens_per_second",
                                    "exact_native_agreement",
                                )
                            }
                        ),
                        flush=True,
                    )
                    engine.close()
                    del engine
                    gc.collect()
                    torch.cuda.empty_cache()
            if args.profile and concurrency == 1:
                for mode in ("sync_spec", "async_spec"):
                    engine = make_engine(
                        model, mode, k, concurrency, args.prompt_length, args.output_length
                    )
                    trial(engine, prompts, args.output_length)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        trial(engine, prompts, args.output_length, profile=True)
                    profiler.export_chrome_trace(str(output.with_suffix(f".{mode}.k{k}.json")))
                    engine.close()
                    del engine
                    gc.collect()
                    torch.cuda.empty_cache()
    if not all(result["exact_native_agreement"] for result in record["trials"]):
        raise RuntimeError(f"output mismatch; retained all sequences in {output}")


if __name__ == "__main__":
    main()
