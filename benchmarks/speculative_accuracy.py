"""Fixed-chain numerical comparison for Ouro self-speculation."""

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import CacheConfig, SamplingParams, SpeculativeConfig
from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, Stage


def error_statistics(actual, expected):
    """Summarize a tensor difference in FP32, regardless of source precision."""
    if actual.shape != expected.shape:
        raise ValueError("tensor shape mismatch")
    difference = actual.float() - expected.float()
    return {
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "elements": difference.numel(),
    }


@torch.inference_mode()
def compare_fixed_chain(model, *, backend, prefix, current, k):
    """Compare reused verification with serial full-depth execution on one chain."""
    if not prefix or k < 1:
        raise ValueError("prefix must be nonempty and k must be positive")
    device = next(model.parameters()).device
    depth_count = model.config.total_ut_steps
    speculative = LLMEngine(
        model,
        cache_config=CacheConfig(128, 16),
        attention_backend=backend,
        speculative_config=SpeculativeConfig(k),
    )
    reuse_cache = speculative.cache_manager
    reuse_cache.allocate("r", len(prefix) + k + 1)
    for position, token in enumerate(prefix):
        hidden = model.prelude(torch.tensor([token], device=device))
        for depth in range(depth_count):
            hidden, _ = model.recurrent(hidden, ["r"], [depth], [position], reuse_cache)

    request = Request("r", prefix, SamplingParams(ignore_eos=True), generated_token_ids=[current])
    runner = speculative.speculative_runner
    original_core = runner._core
    seen = {}

    def record(hidden, ids, positions, depth, **kwargs):
        result = original_core(hidden, ids, positions, depth, **kwargs)
        for row, position in enumerate(positions):
            seen[depth, position] = result[row].detach().clone()
        return result

    head_outputs = []
    runner._core = record
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: head_outputs.append(output.detach().clone())
    )
    try:
        runner.execute(
            SchedulerOutput(
                Stage.SPECULATIVE,
                [ScheduledItem(request, len(prefix), k + 1)],
            )
        )
    finally:
        hook.remove()
        runner._core = original_core

    verified_logits = head_outputs[-1]
    candidates = [int(model.coda(seen[1, len(prefix) + offset]).argmax()) for offset in range(k)]
    tokens = prefix + [current] + candidates
    oracle = LLMEngine(
        model, cache_config=CacheConfig(128, 16), attention_backend=backend
    ).cache_manager
    oracle.allocate("r", len(tokens))
    hidden_pairs = {depth: [] for depth in range(depth_count)}
    logit_pairs = []
    logit_rows = []
    mismatches = 0
    margins = []
    for position, token in enumerate(tokens):
        hidden = model.prelude(torch.tensor([token], device=device))
        for depth in range(depth_count):
            hidden, _ = model.recurrent(hidden, ["r"], [depth], [position], oracle)
            if position >= len(prefix):
                hidden_pairs[depth].append((seen[depth, position], hidden[0]))
        if position >= len(prefix):
            actual = verified_logits[position - len(prefix)]
            expected = model.coda(hidden)[0]
            logit_pairs.append((actual, expected))
            mismatches += int(actual.argmax() != expected.argmax())
            top_two = expected.float().topk(2).values
            margin = (top_two[0] - top_two[1]).item()
            margins.append(margin)
            logit_rows.append(
                {
                    "position": position,
                    "reuse_top1": int(actual.argmax()),
                    "oracle_top1": int(expected.argmax()),
                    "oracle_margin": margin,
                    "error": error_statistics(actual, expected),
                }
            )

    def summarize(pairs):
        return error_statistics(
            torch.cat([actual.reshape(-1) for actual, _ in pairs]),
            torch.cat([expected.reshape(-1) for _, expected in pairs]),
        )

    kv_pairs = {depth: [] for depth in range(depth_count)}
    for depth in range(depth_count):
        for layer in range(model.config.num_hidden_layers):
            kv_pairs[depth].extend(
                zip(
                    reuse_cache.read(layer, "r", depth, len(tokens)),
                    oracle.read(layer, "r", depth, len(tokens)),
                )
            )
    return {
        "positions_compared": k + 1,
        "candidate_ids": candidates,
        "hidden_by_depth": {str(depth): summarize(pairs) for depth, pairs in hidden_pairs.items()},
        "kv_by_depth": {str(depth): summarize(pairs) for depth, pairs in kv_pairs.items()},
        "logits": summarize(logit_pairs),
        "logit_rows": logit_rows,
        "argmax_mismatches": mismatches,
        "minimum_native_top1_top2_margin": min(margins),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args(argv)
    if args.prefix_tokens < 1:
        parser.error("--prefix-tokens must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA with BF16 support is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    base = tokenizer.encode(
        "Explain why the sky appears blue during the day in simple words. ",
        add_special_tokens=False,
    )
    tokens = (base * ((args.prefix_tokens + 1 + len(base) - 1) // len(base)))[
        : args.prefix_tokens + 1
    ]
    model = OuroForCausalLM.from_pretrained(str(args.model), device="cuda", dtype=torch.bfloat16)
    results = {
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": args.model_revision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn_4": importlib.metadata.version("flash-attn-4"),
        "dtype": "bfloat16",
        "backend": "flash_attn_4",
        "prefix_tokens": args.prefix_tokens,
        "cases": [],
    }
    for k in args.ks:
        case = compare_fixed_chain(
            model,
            backend="flash_attn_4",
            prefix=tokens[:-1],
            current=tokens[-1],
            k=k,
        )
        results["cases"].append({"k": k, **case})
        print(json.dumps({"k": k, "logits": case["logits"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
