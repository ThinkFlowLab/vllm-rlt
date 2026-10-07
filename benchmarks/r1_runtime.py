"""Matched engine regression probe; timings exclude loading, warmup and profiling."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path


def main():
    runner_sha_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mode", choices=("sync", "async", "graph"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--sync-debug", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_tokens, args.warmups, args.repetitions) < 1:
        parser.error("--max-tokens, --warmups and --repetitions must be positive")
    if args.output.exists() or (args.profile and args.profile.exists()):
        parser.error("output artifacts already exist; choose new paths")
    if args.sync_debug and (args.device != "cuda" or args.profile):
        parser.error("--sync-debug requires CUDA and a separate run without profiling")
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    source_before = {
        str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((checkout / "vllm_rlt").rglob("*.py"))
    }
    import torch

    from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.models import OuroForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.model is not None:
        model = OuroForCausalLM.from_pretrained(
            str(args.model), device=args.device, dtype=torch.bfloat16
        )
    else:
        if args.device != "cpu":
            parser.error("CUDA qualification requires an explicit checkpoint")
        from tests.helpers import tiny_ouro_config

        model = OuroForCausalLM(tiny_ouro_config())
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=96, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=args.batch, max_num_batched_tokens=64, prefill_chunk_size=16
        ),
        execution_config=ExecutionConfig(
            async_scheduling=args.mode != "sync",
            multi_stream=True,
            static_buffers=args.mode == "graph",
            pad_to_power_of_two=args.mode == "graph",
            cuda_graphs=args.mode == "graph",
        ),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton" if args.device == "cuda" else "torch",
    )
    dtype = next(model.parameters()).dtype
    assert {parameter.dtype for parameter in model.parameters()} == {dtype}
    assert engine.cache_manager.key_cache.dtype == dtype
    if args.device == "cuda":
        assert dtype == torch.bfloat16
    prompts = [
        [504, 3575, 282, 4649, 314],
        [17872, 42, 1812, 314, 216, 33, 34, 8055, 216, 39, 47, 19842, 42],
    ]
    if args.model is None:
        prompts = [[2, 3, 4, 5], [6, 7, 8]]
    parameters = dict(
        max_tokens=args.max_tokens,
        min_loops=2,
        exit_threshold=args.threshold,
        ignore_eos=True,
        temperature=args.temperature,
    )

    def run_round(index):
        for row in range(args.batch):
            engine.add_request(
                str(row), prompts[row % len(prompts)], SamplingParams(**parameters, seed=1729 + row)
            )
        engine.model_runner.synchronize()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        first, gaps, last, host, outputs, steps = {}, [], {}, [], {}, []
        start = time.perf_counter()
        for _ in range(args.max_tokens * 12):
            if not engine.has_unfinished_requests():
                break
            before = time.perf_counter()
            emitted = engine.step()
            now = time.perf_counter()
            host.append(now - before)
            batch = engine.last_schedule
            if batch is not None:
                steps.append((batch.stage.value, [i.request.request_id for i in batch.items]))
            for output in emitted:
                rid = output.request_id
                first.setdefault(rid, now - start)
                if rid in last:
                    gaps.append(now - last[rid])
                last[rid] = now
                if output.finished:
                    outputs[rid] = dict(
                        tokens=output.token_ids,
                        depths=output.exit_depths,
                        reason=output.finish_reason,
                    )
        else:
            raise RuntimeError("engine exceeded the bounded step count")
        engine.model_runner.synchronize()
        elapsed = time.perf_counter() - start
        assert not engine.has_unfinished_requests()
        assert len(outputs) == args.batch
        assert all(len(o["tokens"]) == args.max_tokens for o in outputs.values())
        assert all(o["reason"] == "length" for o in outputs.values())
        assert engine.cache_manager.num_used_blocks == 0
        memory = (
            dict(
                allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
            )
            if args.device == "cuda"
            else {}
        )
        return dict(
            index=index,
            seconds=elapsed,
            tokens_per_second=args.batch * args.max_tokens / elapsed,
            ttft_ms={rid: value * 1000 for rid, value in first.items()},
            itl_ms=statistics.median(gaps) * 1000 if gaps else None,
            tpot_ms=statistics.mean(gaps) * 1000 if gaps else None,
            host_step_us=statistics.median(host) * 1e6,
            outputs=outputs,
            steps=steps,
            peak_memory=memory,
            used_kv_blocks_after=engine.cache_manager.num_used_blocks,
        )

    warmups = [run_round(i) for i in range(args.warmups)]
    sync_warnings = []
    if args.profile:

        def instrument(original):
            def execute(batch):
                with torch.profiler.record_function(f"r1.{batch.stage.value}"):
                    return original(batch)

            return execute

        for name in ("execute", "submit"):
            setattr(engine.model_runner, name, instrument(getattr(engine.model_runner, name)))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            with torch.profiler.record_function("r1.request"):
                runs = [run_round(0)]
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.profile))
    elif args.sync_debug:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as sync_warnings:
            warnings.simplefilter("always")
            runs = [run_round(i) for i in range(args.repetitions)]
    else:
        runs = [run_round(i) for i in range(args.repetitions)]
    assert all(run["outputs"] == warmups[-1]["outputs"] for run in runs + warmups)
    source = {
        str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((checkout / "vllm_rlt").rglob("*.py"))
    }
    assert source == source_before, "execution source changed during the run"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == runner_sha_before
    result = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        source_sha256=source,
        runner_sha256=runner_sha_before,
        controls={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        prompts=prompts,
        sampling=parameters,
        python=sys.version,
        packages={n: importlib.metadata.version(n) for n in ("torch", "triton", "transformers")},
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu={
            name: str(getattr(torch.cuda.get_device_properties(0), name, "unavailable"))
            for name in ("name", "uuid", "major", "minor", "total_memory")
        }
        if args.device == "cuda"
        else None,
        weight_dtype=str(dtype),
        kv_dtype=str(engine.cache_manager.key_cache.dtype),
        graph_stats={
            name: getattr(engine.model_runner.graphs, name)
            for name in ("captures", "replays", "fallbacks")
        }
        if engine.model_runner.graphs is not None
        else None,
        warmups=warmups,
        measured=[] if args.profile or args.sync_debug else runs,
        diagnostic=runs if args.profile or args.sync_debug else [],
        sync_warning_count=len(sync_warnings),
        sync_warning_sites=sorted({f"{w.filename}:{w.lineno}:{w.message}" for w in sync_warnings}),
    )
    engine.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
