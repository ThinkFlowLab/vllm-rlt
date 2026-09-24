"""Paired Ouro E2E for a request migrated across two GPUs after KV commit."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import CacheConfig, SamplingParams, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def drain(engine):
    result = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                result = dict(tokens=output.token_ids, depths=output.exit_depths)
    return result


def run_baseline(engine, prompt, params):
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    engine.add_request("r", prompt, params)
    output = drain(engine)
    torch.cuda.synchronize(0)
    return dict(seconds=time.perf_counter() - start, output=output)


def run_migrated(source, target, prompt, params):
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    start = time.perf_counter()
    source.add_request("r", prompt, params)
    while not source.scheduler.requests["r"].generated_token_ids:
        source.step()
    source.step()  # Provisional draft.
    source.step()  # Verification commits the round.
    packet = source.export_request("r")
    with torch.cuda.device(1):
        if not target.import_request(packet):
            raise AssertionError("target could not reserve migrated KV")
        output = drain(target)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    return dict(seconds=time.perf_counter() - start, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("benchmark requires exactly two visible GPUs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompt = tokenizer.encode(
        "Explain how matrix multiplication works, with one numerical example."
    )
    models = [
        OuroForCausalLM.from_pretrained(args.model, device=f"cuda:{device}", dtype=torch.bfloat16)
        for device in (0, 1)
    ]
    config = SpeculativeConfig(4, interleave_round=True)
    baseline = LLMEngine(
        models[0],
        speculative_config=config,
        cache_config=CacheConfig(num_blocks=1024),
        attention_backend="triton",
    )
    source = LLMEngine(
        models[0],
        speculative_config=config,
        cache_config=CacheConfig(num_blocks=1024),
        attention_backend="triton",
    )
    with torch.cuda.device(1):
        target = LLMEngine(
            models[1],
            speculative_config=config,
            cache_config=CacheConfig(num_blocks=1024),
            attention_backend="triton",
        )
    params = SamplingParams(max_tokens=32, min_loops=4, max_loops=4, ignore_eos=True)
    rows = []
    for trial in range(args.repeats + 1):
        pair = {}
        for migrated in (False, True) if trial % 2 == 0 else (True, False):
            value = (
                run_migrated(source, target, prompt, params)
                if migrated
                else run_baseline(baseline, prompt, params)
            )
            pair[migrated] = value
            if trial:
                rows.append(dict(trial=trial - 1, migrated=migrated, **value))
        if pair[False]["output"] != pair[True]["output"]:
            raise AssertionError(f"trial {trial} migration output mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps(
            dict(
                model="ByteDance/Ouro-1.4B",
                devices=[torch.cuda.get_device_name(i) for i in (0, 1)],
                dtype="bfloat16",
                backend="triton",
                prompt=prompt,
                rows=rows,
            ),
            indent=2,
        )
    )
    baseline_s = statistics.median(row["seconds"] for row in rows if not row["migrated"])
    migrated_s = statistics.median(row["seconds"] for row in rows if row["migrated"])
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B cross-GPU KV migration E2E\n\n"
        "Two RTX 5090 GPUs, BF16, Triton, d=2/D=4, K=4, 32 output tokens. "
        "One warmup and three alternating paired trials; one request migrates "
        "after its first committed speculative round. Timings include prefill, "
        "draft, verification, CPU KV snapshot and restore, coda and KV commit. "
        "Every paired output token and exit depth matched. Lower latency is better.\n\n"
        "| Path | Median E2E, s | Relative speed |\n"
        "|---|---:|---:|\n"
        f"| Single GPU | {baseline_s:.3f} | 1.000× |\n"
        f"| GPU 0 → GPU 1 | {migrated_s:.3f} | {baseline_s / migrated_s:.3f}× |\n\n"
        "[Raw paired trials](raw.json)\n"
    )


if __name__ == "__main__":
    main()
