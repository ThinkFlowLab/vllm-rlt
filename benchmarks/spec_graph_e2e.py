"""Paired Ouro speculative E2E with CUDA graphs disabled and enabled."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.models import OuroForCausalLM


def timed(llm, prompts, params):
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    torch.cuda.synchronize()
    return [
        dict(tokens=o.token_ids, depths=o.exit_depths) for o in outputs
    ], time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.repeats < 1:
        parser.error("max-tokens must be at least 2 and repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    repeated = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 12)
    prose = tokenizer.encode(
        "Explain how a matrix multiplication works, including input and output shapes "
        "and one numerical example."
    )
    workloads = {
        "repeated / 1": [repeated],
        "repeated / 4": [repeated] * 4,
        "prose / 1": [prose],
    }
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
    engines = {
        graph: LLM(
            model,
            cache_config=CacheConfig(num_blocks=1024),
            scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128),
            attention_backend="triton",
            speculative_config=SpeculativeConfig(4),
            execution_config=ExecutionConfig(
                cuda_graphs=graph, cuda_graph_max_batch_size=32, cuda_graph_max_graphs=16
            ),
        )
        for graph in (False, True)
    }
    params = SamplingParams(max_tokens=args.max_tokens, min_loops=4, max_loops=4, ignore_eos=True)
    rows = []
    for name, prompts in workloads.items():
        for _ in range(5):
            warmup = {graph: timed(engines[graph], prompts, params)[0] for graph in (False, True)}
            if warmup[False] != warmup[True]:
                raise AssertionError(f"{name}: graph warmup output mismatch")
        for trial in range(args.repeats):
            pair = {}
            for graph in (False, True) if trial % 2 == 0 else (True, False):
                outputs, seconds = timed(engines[graph], prompts, params)
                pair[graph] = outputs
                rows.append(
                    dict(workload=name, trial=trial, graphs=graph, seconds=seconds, outputs=outputs)
                )
            if pair[False] != pair[True]:
                raise AssertionError(f"{name}: trial {trial} output mismatch")

    runner = engines[True].engine.speculative_runner
    counters = {
        "recurrent": dict(
            captures=runner.graphs.captures,
            replays=runner.graphs.replays,
            fallbacks=runner.graphs.fallbacks,
        ),
        "coda": dict(
            captures=runner.coda_graphs.captures,
            replays=runner.coda_graphs.replays,
            fallbacks=runner.coda_graphs.fallbacks,
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps(
            dict(
                model="ByteDance/Ouro-1.4B",
                device=torch.cuda.get_device_name(0),
                dtype="bfloat16",
                backend="triton",
                draft_loops=2,
                target_loops=4,
                k=4,
                output_budget=args.max_tokens,
                prompts=workloads,
                rows=rows,
                graphs=counters,
            ),
            indent=2,
        )
    )
    table = [
        "| Workload / requests | Eager E2E s | Graph E2E s | Speedup | Output |",
        "|---|---:|---:|---:|---|",
    ]
    for name in workloads:
        eager = statistics.median(
            r["seconds"] for r in rows if r["workload"] == name and not r["graphs"]
        )
        graphed = statistics.median(
            r["seconds"] for r in rows if r["workload"] == name and r["graphs"]
        )
        table.append(
            f"| {name} | {eager:.3f} | {graphed:.3f} | {eager / graphed:.3f}× | exact match |"
        )
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B speculative CUDA Graph E2E\n\n"
        f"{torch.cuda.get_device_name(0)}; BF16, Triton, d=2/D=4, K=4, "
        f"{args.max_tokens} output tokens. One resident model, separate engine state, "
        f"five warmups per arm, {args.repeats} alternating paired trials. "
        "Timings include prefill, draft, verification, coda and KV commit. "
        "Every paired output token and exit depth matched.\n\n"
        + "\n".join(table)
        + "\n\n| Graph stage | Captures | Replays | Eager fallbacks |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {stage} | {value['captures']} | {value['replays']} | {value['fallbacks']} |"
            for stage, value in counters.items()
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
