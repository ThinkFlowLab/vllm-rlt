"""Finite local-checkpoint Huginn Graph/eager generation and paired timing."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
import triton
from transformers import AutoTokenizer

from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM
from vllm_rlt.worker.cuda_graph import RecurrentGraphs

PROMPTS = (
    "The capital of France is",
    "Explain why water freezes at low temperatures in one sentence.",
    "Write a short Python function that adds two integers and returns the result.",
    "Question: A box contains 12 apples. We add 8 and remove 5. How many remain? Answer:",
)


@torch.inference_mode()
def validate_core(model):
    config = model.config
    caches = [
        KVCacheManager(
            num_layers=8,
            num_kv_heads=55,
            head_dim=96,
            num_blocks=128,
            block_size=16,
            max_loops=32,
            device="cuda",
            dtype=torch.bfloat16,
            backend="triton",
            recurrent_layers=model.recurrent_kv_layers,
        )
        for _ in range(2)
    ]
    graphs = RecurrentGraphs(model, caches[1], ExecutionConfig(cuda_graphs=True), False)
    for cache in caches:
        assert cache.allocate("state-kv-oracle", 2)
    for position, token in enumerate((101, 202)):
        states = []
        for cache in caches:
            torch.manual_seed(123 + position)
            metadata = cache._prepare_batch(["state-kv-oracle"], [0], [position])
            states.append(
                model.prelude_prepared(torch.tensor([token], device="cuda"), metadata, cache)
            )
        torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
        for depth in range(32):
            metadata = [
                cache._prepare_batch(["state-kv-oracle"], [depth], [position]) for cache in caches
            ]
            states[0], _ = model.recurrent_prepared(
                states[0], metadata[0], caches[0], compute_gate=False
            )
            states[1], _ = graphs.run(states[1], metadata[1])
            torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
            for layer in model.recurrent_kv_layers:
                for eager, replay in zip(
                    caches[0].read(layer, "state-kv-oracle", depth),
                    caches[1].read(layer, "state-kv-oracle", depth),
                ):
                    torch.testing.assert_close(eager, replay, atol=0, rtol=0)
        logits = [
            model.coda_prepared(
                hidden, cache._prepare_batch(["state-kv-oracle"], [0], [position]), cache
            )
            for hidden, cache in zip(states, caches)
        ]
        torch.testing.assert_close(logits[0], logits[1], atol=0, rtol=0)
        for cache in caches:
            cache.finalize_token("state-kv-oracle", position, 31)
    assert graphs.captures == 1 and graphs.replays == 64 and graphs.fallbacks == 0
    for cache in caches:
        cache.free("state-kv-oracle")
        assert cache.num_used_blocks == 0
    return {
        "positions": 2,
        "depths_per_position": 32,
        "state_logits_kv_bit_exact": True,
        "captures": graphs.captures,
        "replays": graphs.replays,
        "fallbacks": graphs.fallbacks,
        "kv_used_blocks_after": 0,
        "hidden_width_including_injection": config.hidden_size,
    }


def graph_stats(engine):
    graphs = engine.model_runner.graphs
    return (
        {"captures": graphs.captures, "replays": graphs.replays, "fallbacks": graphs.fallbacks}
        if graphs is not None
        else dict(captures=0, replays=0, fallbacks=0)
    )


@torch.inference_mode()
def burst(engine, prompts, sampling, seed):
    torch.manual_seed(seed)
    before = graph_stats(engine)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, sampling)
    finished = {}
    # Every request has a bounded prompt, output length and recurrent depth.
    for steps in range(20000):
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = {
                    "token_ids": output.token_ids,
                    "exit_depths": output.exit_depths,
                    "finish_reason": output.finish_reason,
                }
        if not engine.has_unfinished_requests():
            break
    else:
        raise RuntimeError("Finite generation stage limit exceeded")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    assert len(finished) == len(prompts)
    assert all(len(out["token_ids"]) == sampling.max_tokens for out in finished.values())
    assert all(out["exit_depths"] == [32] * sampling.max_tokens for out in finished.values())
    assert engine.cache_manager.num_used_blocks == 0
    after = graph_stats(engine)
    return {
        "seconds": seconds,
        "tokens_per_second": len(prompts) * sampling.max_tokens / seconds,
        "steps": steps + 1,
        "outputs": finished,
        "kv_used_blocks_after": 0,
        "graph_delta": {name: after[name] - before[name] for name in before},
    }


def measure(model, prompts, sampling, deadline):
    cache = CacheConfig(num_blocks=1024, block_size=16)
    scheduler = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=32,
        prefill_chunk_size=8,
    )
    engines = {
        mode: LLMEngine(
            model,
            cache_config=cache,
            scheduler_config=scheduler,
            execution_config=ExecutionConfig(cuda_graphs=mode == "graph"),
            attention_backend="triton",
        )
        for mode in ("eager", "graph")
    }
    warmups = []
    for trial in range(5):
        arms = {
            mode: burst(engine, prompts, sampling, 123 + trial) for mode, engine in engines.items()
        }
        assert arms["eager"]["outputs"] == arms["graph"]["outputs"]
        warmups.append(arms)
    pairs = []
    for trial in range(5):
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Wall budget reached before next timing pair; no active step interrupted"
            )
        order = ("eager", "graph") if trial % 2 == 0 else ("graph", "eager")
        arms = {mode: burst(engines[mode], prompts, sampling, 123) for mode in order}
        assert arms["eager"]["outputs"] == arms["graph"]["outputs"]
        assert arms["graph"]["graph_delta"]["captures"] == 0
        assert arms["graph"]["graph_delta"]["replays"] > 0
        assert arms["graph"]["graph_delta"]["fallbacks"] == 0
        pairs.append(
            {
                "trial": trial + 1,
                "order": order,
                **arms,
                "speedup": arms["eager"]["seconds"] / arms["graph"]["seconds"],
            }
        )
        print(
            json.dumps(
                {
                    "requests": len(prompts),
                    "pair": trial + 1,
                    "eager_s": arms["eager"]["seconds"],
                    "graph_s": arms["graph"]["seconds"],
                    "speedup": pairs[-1]["speedup"],
                }
            ),
            flush=True,
        )
    eager = [pair["eager"]["seconds"] for pair in pairs]
    replay = [pair["graph"]["seconds"] for pair in pairs]
    result = {
        "requests": len(prompts),
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "prompt_token_ids": prompts,
        "sampling": asdict(sampling),
        "cache": asdict(cache),
        "scheduler": asdict(scheduler),
        "warmups": warmups,
        "pairs": pairs,
        "graph_stats": graph_stats(engines["graph"]),
        "mean_eager_seconds": statistics.mean(eager),
        "mean_graph_seconds": statistics.mean(replay),
        "median_eager_seconds": statistics.median(eager),
        "median_graph_seconds": statistics.median(replay),
        "mean_time_speedup": statistics.mean(eager) / statistics.mean(replay),
        "median_time_speedup": statistics.median(eager) / statistics.median(replay),
        "median_pair_speedup": statistics.median(pair["speedup"] for pair in pairs),
    }
    del engines
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise ValueError("Provide a complete local checkpoint")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    deadline = time.monotonic() + 6000
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in PROMPTS]
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    assert model.config.n_embd == 5280 and model.config.n_heads == 55
    assert model.config.total_ut_steps == 32 and model.config.head_dim == 96
    assert model.lm_head.weight is model.transformer.wte.weight
    sampling = SamplingParams(max_tokens=16, min_loops=32, max_loops=32, ignore_eos=True)
    gpu = torch.cuda.get_device_properties(0)
    results = {
        "status": "RUNNING",
        "model": "tomg-group-umd/huginn-0125",
        "revision": args.revision,
        "runtime": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "transformers": transformers.__version__,
            "gpu": gpu.name,
            "capability": [gpu.major, gpu.minor],
            "total_memory": gpu.total_memory,
        },
        "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "model_load_seconds_excluded": load_seconds,
        "warmups_per_arm": 5,
        "pairs_per_case": 5,
        "graph_scope": "recurrent decode core only; prelude, coda, prefill and sampling eager",
        "timing_scope": "CUDA-synchronized admission through finished outputs and KV reclamation",
        "initial_state_rng": "torch.manual_seed reset identically before each paired burst",
        "cases": [],
    }
    results["core_validation"] = validate_core(model)
    gc.collect()
    torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for count in (1, 4):
        result = measure(model, prompts[:count], sampling, deadline)
        result["decoded_outputs"] = {
            rid: tokenizer.decode(out["token_ids"])
            for rid, out in result["pairs"][0]["eager"]["outputs"].items()
        }
        results["cases"].append(result)
        results["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    results["status"] = "PASS"
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("HUGINN_CUDA_GRAPH_E2E_PASS", flush=True)


if __name__ == "__main__":
    main()
