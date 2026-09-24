"""Paired real-model E2E for a prompt arriving at a speculative round boundary."""

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
from vllm_rlt.request import Stage


def run(engine, prompts, params):
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.add_request("first", prompts[0], params)
    while not engine.scheduler.requests["first"].generated_token_ids:
        engine.step()
    torch.cuda.synchronize()
    arrival = time.perf_counter()
    engine.add_request("second", prompts[1], params)
    outputs = {}
    second_ttft = None
    interleaved_prefills = 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id == "second" and second_ttft is None and output.token_ids:
                torch.cuda.synchronize()
                second_ttft = time.perf_counter() - arrival
            if output.finished:
                outputs[output.request_id] = dict(
                    tokens=output.token_ids, depths=output.exit_depths
                )
        interleaved_prefills += int(
            engine.last_schedule is not None
            and engine.last_schedule.stage == Stage.PREFILL
            and engine._spec_drafted is not None
        )
    torch.cuda.synchronize()
    if second_ttft is None:
        raise AssertionError("second request produced no output")
    if engine.speculative_config.interleave_round and not interleaved_prefills:
        raise AssertionError("intra-round prefill did not execute")
    return dict(
        seconds=time.perf_counter() - start,
        second_ttft=second_ttft,
        interleaved_prefills=interleaved_prefills,
        outputs=outputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.repeats < 1:
        parser.error("max-tokens must be at least 2 and repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [
        tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 10),
        tokenizer.encode(
            "Explain matrix multiplication with concrete input and output shapes. " * 4
        ),
    ]
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
    engines = {
        flag: LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=1024),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=256),
            attention_backend="triton",
            speculative_config=SpeculativeConfig(4, interleave_round=flag),
        )
        for flag in (False, True)
    }
    params = SamplingParams(max_tokens=args.max_tokens, min_loops=4, max_loops=4, ignore_eos=True)
    rows = []
    for _ in range(3):
        pair = {flag: run(engines[flag], prompts, params) for flag in (False, True)}
        if pair[False]["outputs"] != pair[True]["outputs"]:
            raise AssertionError("warmup output mismatch")
    for trial in range(args.repeats):
        pair = {}
        for flag in (False, True) if trial % 2 == 0 else (True, False):
            pair[flag] = run(engines[flag], prompts, params)
            rows.append(dict(trial=trial, interleave=flag, **pair[flag]))
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
                k=4,
                output_budget=args.max_tokens,
                prompts=prompts,
                rows=rows,
            ),
            indent=2,
        )
    )
    table = []
    for key, label in (("seconds", "Two-request E2E"), ("second_ttft", "Second-request TTFT")):
        atomic = statistics.median(row[key] for row in rows if not row["interleave"])
        interleaved = statistics.median(row[key] for row in rows if row["interleave"])
        table.append(
            f"| {label} | {atomic:.3f} | {interleaved:.3f} | {atomic / interleaved:.3f}× |"
        )
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B speculative intra-round scheduling E2E\n\n"
        f"{torch.cuda.get_device_name(0)}; BF16, Triton, d=2/D=4, K=4, "
        f"{args.max_tokens} output tokens per request. The second prompt arrives after "
        "the first request's initial output. One resident model, three warmups "
        f"per arm, {args.repeats} alternating paired trials. Timings include "
        "prefill, draft, verification, coda and KV commit. Every paired output "
        "token and exit depth matched. Lower latency is better; speedup is "
        "atomic / interleaved.\n\n"
        "| Metric | Atomic, s | Interleaved, s | Speedup |\n"
        "|---|---:|---:|---:|\n" + "\n".join(table) + "\n\n[Raw paired trials](raw.json)\n"
    )


if __name__ == "__main__":
    main()
