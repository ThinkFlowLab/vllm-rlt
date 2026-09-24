"""Paired Ouro E2E with a high-priority arrival after low-priority decoding begins."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def run(engine, prompts, params):
    torch.cuda.synchronize()
    start = time.perf_counter()
    before = engine.preemption.preemptions, engine.preemption.resumptions
    engine.add_request("low", prompts[0], params[0])
    while len(engine.scheduler.requests["low"].generated_token_ids) < 2:
        engine.step()
    torch.cuda.synchronize()
    arrival = time.perf_counter()
    engine.add_request("high", prompts[1], params[1])
    outputs = {}
    high_ttft = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id == "high" and high_ttft is None and output.token_ids:
                torch.cuda.synchronize()
                high_ttft = time.perf_counter() - arrival
            if output.finished:
                outputs[output.request_id] = dict(
                    tokens=output.token_ids, depths=output.exit_depths
                )
    torch.cuda.synchronize()
    if high_ttft is None:
        raise AssertionError("high-priority request produced no output")
    return dict(
        seconds=time.perf_counter() - start,
        high_ttft=high_ttft,
        preemptions=engine.preemption.preemptions - before[0],
        resumptions=engine.preemption.resumptions - before[1],
        outputs=outputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [
        tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 8),
        tokenizer.encode("Explain matrix multiplication with one numerical example."),
    ]
    params = [
        SamplingParams(max_tokens=32, min_loops=4, max_loops=4, ignore_eos=True, priority=10),
        SamplingParams(max_tokens=16, min_loops=4, max_loops=4, ignore_eos=True, priority=0),
    ]
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
    engines = {
        flag: LLMEngine(
            model,
            speculative_config=SpeculativeConfig(4, interleave_round=True),
            cache_config=CacheConfig(num_blocks=1024, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=1, policy="priority", enable_preemption=flag
            ),
            attention_backend="triton",
        )
        for flag in (False, True)
    }
    rows = []
    for trial in range(args.repeats + 1):
        pair = {}
        for flag in (False, True) if trial % 2 == 0 else (True, False):
            pair[flag] = run(engines[flag], prompts, params)
            if flag and not (pair[flag]["preemptions"] and pair[flag]["resumptions"]):
                raise AssertionError("priority request did not preempt and resume")
            if trial:
                rows.append(dict(trial=trial - 1, preemption=flag, **pair[flag]))
        if pair[False]["outputs"] != pair[True]["outputs"]:
            raise AssertionError(f"trial {trial} output mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps(
            dict(
                model="ByteDance/Ouro-1.4B",
                device=torch.cuda.get_device_name(0),
                dtype="bfloat16",
                backend="triton",
                prompts=prompts,
                rows=rows,
            ),
            indent=2,
        )
    )
    table = []
    for key, label in (("seconds", "Two-request E2E"), ("high_ttft", "High-priority TTFT")):
        serial = statistics.median(row[key] for row in rows if not row["preemption"])
        priority = statistics.median(row[key] for row in rows if row["preemption"])
        table.append(f"| {label} | {serial:.3f} | {priority:.3f} | {serial / priority:.3f}× |")
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B priority preemption E2E\n\n"
        "RTX 5090, BF16, Triton, d=2/D=4, K=4. One low-priority 32-token "
        "request runs first; a high-priority 16-token request arrives after "
        "two low-priority outputs. One warmup and alternating paired trials. "
        "Both arms share one resident model and allow only one active request. "
        "Every paired output token and exit depth matched. Lower latency is better.\n\n"
        "| Metric | Wait for active request, s | Preempt and resume, s | Speedup |\n"
        "|---|---:|---:|---:|\n" + "\n".join(table) + "\n\n[Raw paired trials](raw.json)\n"
    )


if __name__ == "__main__":
    main()
