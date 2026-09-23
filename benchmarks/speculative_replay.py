"""Replay the first greedy divergence on a fixed BF16 token chain."""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.speculative import cache_blocks_for_case, make_prompts
from benchmarks.speculative_accuracy import compare_fixed_chain
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage


def first_divergence(results, case_index=None):
    """Return the first case/trial/request/output-position that differs."""
    for index, case in enumerate(results["cases"]):
        if case_index is not None and index != case_index:
            continue
        for trial_index, trial in enumerate(case["runs"]):
            for request_index, (native, speculative) in enumerate(
                zip(trial["native"]["token_ids"], trial["speculative"]["token_ids"])
            ):
                for output_index, (native_id, speculative_id) in enumerate(
                    zip(native, speculative)
                ):
                    if native_id != speculative_id:
                        return index, trial_index, request_index, output_index
    return None


def capture_native_margins(model, prompts, case, trial):
    """Replay the native batch and record margins at each first divergence."""
    targets = {}
    for request_index, (native, speculative) in enumerate(
        zip(trial["native"]["token_ids"], trial["speculative"]["token_ids"])
    ):
        for output_index, (native_id, speculative_id) in enumerate(zip(native, speculative)):
            if native_id != speculative_id:
                targets[request_index] = (output_index, native_id)
                break
    if not targets:
        return []
    params = SamplingParams(
        max_tokens=case["output_tokens"],
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
    blocks = cache_blocks_for_case(
        prompt_tokens=case["prompt_tokens"],
        output_tokens=case["output_tokens"],
        concurrency=len(prompts),
        depth=model.config.total_ut_steps,
        k=case["k"],
    )
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(blocks, 16),
        scheduler_config=scheduler,
        attention_backend="flash_attn_4",
    )
    for request_index, prompt in enumerate(prompts):
        engine.add_request(f"bench-{request_index}", prompt, params)
    original_coda = model.coda
    captured = {}

    def coda(hidden):
        logits = original_coda(hidden)
        batch = engine.last_schedule
        if batch is not None and batch.stage == Stage.CODA:
            for row, item in enumerate(batch.items):
                request_index = int(item.request.request_id.removeprefix("bench-"))
                target = targets.get(request_index)
                if target is None or len(item.request.generated_token_ids) != target[0]:
                    continue
                top2 = logits[row].float().topk(2)
                prefix_matches = (
                    item.request.generated_token_ids
                    == trial["native"]["token_ids"][request_index][: target[0]]
                )
                captured[request_index] = {
                    "request_index": request_index,
                    "output_index_zero_based": target[0],
                    "native_output_token_id": target[1],
                    "replayed_native_top1": int(top2.indices[0]),
                    "native_top1_top2_margin": (top2.values[0] - top2.values[1]).item(),
                    "prefix_matches_recorded_native": prefix_matches,
                    "top1_matches_recorded_native": int(top2.indices[0]) == target[1],
                }
        return logits

    model.coda = coda
    try:
        while engine.has_unfinished_requests() and len(captured) < len(targets):
            engine.step()
    finally:
        model.coda = original_coda
        for request_id in list(engine.scheduler.requests):
            engine.abort_request(request_id)
    if len(captured) != len(targets):
        raise RuntimeError("native replay ended before every target margin was captured")
    return [captured[index] for index in sorted(captured)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--decode-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int, help="Select one zero-based matrix case")
    args = parser.parse_args(argv)
    results = json.loads(args.decode_result.read_text())
    location = first_divergence(results, args.case_index)
    if location is None:
        parser.error("decode result has no divergent output token")
    case_index, trial_index, request_index, output_index = location
    case = results["cases"][case_index]
    trial = case["runs"][trial_index]
    native = trial["native"]["token_ids"][request_index]
    speculative = trial["speculative"]["token_ids"][request_index]
    result = {
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": args.model_revision,
        "case_index": case_index,
        "trial_index": trial_index,
        "request_index": request_index,
        "output_index_zero_based": output_index,
        "native_token_id": native[output_index],
        "speculative_token_id": speculative[output_index],
        "case": {key: case[key] for key in ("prompt_tokens", "output_tokens", "concurrency", "k")},
    }
    if output_index > 0:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error("CUDA with BF16 support is required")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        prompts = make_prompts(tokenizer, case["prompt_tokens"], case["concurrency"])
        prefix = prompts[request_index] + native[: output_index - 1]
        current = native[output_index - 1]
        model = OuroForCausalLM.from_pretrained(
            str(args.model), device="cuda", dtype=torch.bfloat16
        )
        result["native_first_divergence_margins"] = capture_native_margins(
            model, prompts, case, trial
        )
        result["fixed_chain"] = compare_fixed_chain(
            model,
            backend="flash_attn_4",
            prefix=prefix,
            current=current,
            k=case["k"],
        )
        result["replay_input_tokens"] = len(prefix) + 1
        result["replay_caveat"] = (
            "serial fixed-chain oracle may differ numerically from batched native prefill"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "fixed_chain"}))


if __name__ == "__main__":
    main()
