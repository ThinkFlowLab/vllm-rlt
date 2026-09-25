import argparse
import json
from dataclasses import asdict

import torch

from vllm_rlt import LLM, SamplingParams, SchedulerConfig
from vllm_rlt.entrypoints.runtime_args import add_runtime_args, runtime_configs
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Ouro inference with loop-level continuous batching"
    )
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B")
    parser.add_argument("--revision")
    parser.add_argument("--prompt", action="append", help="Text prompt; repeat for a batch")
    parser.add_argument(
        "--toy", action="store_true", help="Use a tiny RANDOM model and token ID prompts"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--attention-backend",
        type=str.lower,
        choices=["torch", "triton", "flash_attn", "flash_attn_2", "flash_attn_3", "flash_attn_4"],
        default="torch",
    )
    parser.add_argument("--mode", choices=["refill", "no_refill"], default="refill")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument("--min-loops", type=int, default=2)
    parser.add_argument("--exit-threshold", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--dynamic-depth-temp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, help="Override automatic KV sizing")
    parser.add_argument("--block-size", type=int, default=16)
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.toy and args.prompt:
        parser.error("--toy uses built-in token ID prompts; omit --prompt")
    if not args.toy and not args.prompt:
        parser.error("provide --prompt or use --toy for a CPU smoke test")
    torch.manual_seed(args.seed)
    model = (
        OuroForCausalLM(OuroConfig.tiny()).to(device=args.device, dtype=getattr(torch, args.dtype))
        if args.toy
        else args.model
    )
    llm = LLM(
        model,
        revision=args.revision,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        **runtime_configs(args),
        scheduler_config=SchedulerConfig(
            policy=getattr(args, "scheduling_policy", "fcfs"),
            enable_preemption=getattr(args, "enable_preemption", False),
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            mode=args.mode,
            prefill_chunk_size=getattr(args, "prefill_chunk_size", 128),
            max_prefill_batches_before_decode=getattr(args, "max_prefill_batches_before_decode", 1),
            admission_scan_limit=getattr(args, "admission_scan_limit", 64),
            max_admission_bypasses=getattr(args, "max_admission_bypasses", 8),
            min_coda_batch_size=getattr(args, "min_coda_batch_size", 1),
        ),
        attention_backend=args.attention_backend,
    )
    params = SamplingParams(
        max_tokens=args.max_tokens,
        max_loops=args.max_loops,
        min_loops=args.min_loops,
        exit_threshold=args.exit_threshold,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        dynamic_depth_temp=args.dynamic_depth_temp,
        ignore_eos=args.toy,
    )
    outputs = llm.generate([[1, 2, 3], [4, 5]] if args.toy else args.prompt, params)
    for output in outputs:
        print(json.dumps(asdict(output), ensure_ascii=False))


if __name__ == "__main__":
    main()
