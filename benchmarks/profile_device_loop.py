"""Decode-loop timing/profiling harness for native vs. self-speculative Ouro decoding.

Drives the production ``LLMEngine.step()`` path; it does not reimplement decoding.
Model loading, tokenization and warmup are outside every timed or traced region.

Measurements (``--measure``):
  decode  Requests are prefilled first; the clock starts at the first step boundary
          where every request has its first token (synchronized), and stops when all
          finish (synchronized). Only tokens committed inside the window are counted.
  e2e     The clock covers the whole generate loop, prefill included. No
          synchronization is inserted between the two timing boundaries.

Other modes:
  --profile-region  After warmup, trace ``--profile-steps`` decode-window engine steps
                    with labeled ranges. ``--profiler nsys`` brackets them with
                    cudaProfilerStart/Stop (use ``nsys profile --capture-range=cudaProfilerApi``);
                    ``--profiler torch`` records a Kineto/CUPTI trace (.json.gz) directly.
                    Not for timing.
  --graph-events    After warmup, run one decode window with a CUDA event pair around every
                    ``CUDAGraph.replay()`` (no profiler). Summed replay time vs. wall time bounds
                    the GPU time spent outside graphs (host gaps plus small eager kernels).
  --count-syncs     After warmup, run one decode window under
                    ``torch.cuda.set_sync_debug_mode("warn")`` and attribute every
                    reported synchronizing call to its innermost vllm_rlt source line.

Example:
  python -m benchmarks.profile_device_loop --model artifacts/models/Ouro-1.4B \
      --backend flash_attn_4 --mode spec --k 4 --concurrency 1 --output-dir out/
"""

import argparse
import contextlib
import functools
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.ouro import OuroForCausalLM
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.speculative import SpeculativeRunner

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "benchmarks" / "fixtures" / "device_loop_prompts.json"
PACKAGES = (
    "torch",
    "triton",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "transformers",
    "nvidia-cudnn-cu13",
    "nvidia-nccl-cu13",
)


# --------------------------------------------------------------------------- identity


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(args):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    model_dir = Path(args.model)
    config_path = model_dir / "config.json"
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        lsf_job_id=os.environ.get("LSB_JOBID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        python=sys.version.split()[0],
        platform=platform.platform(),
        git_sha=_run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        git_dirty=bool(
            _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
        ),
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        packages=packages,
        gpu=dict(
            name=props.name,
            capability=f"{props.major}.{props.minor}",
            sm_count=props.multi_processor_count,
            total_memory=props.total_memory,
            uuid=str(getattr(props, "uuid", "")),
        )
        if props
        else None,
        driver=_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        model_path=str(model_dir.resolve()),
        model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_path.is_file()
        else None,
        model_revision=args.model_revision,
    )


# --------------------------------------------------------------------------- workload


def build_prompts(tokenizer, count, length):
    fixture = json.loads(FIXTURE.read_text())
    texts = fixture["prompts"]
    prompts = []
    for i in range(count):
        ids, j = [], i
        while len(ids) < length:
            ids.extend(tokenizer.encode(texts[j % len(texts)] + "\n\n", add_special_tokens=False))
            j += 1
        prompts.append(ids[:length])
    digest = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    return prompts, dict(fixture_id=fixture["fixture_id"], prompt_token_sha256=digest)


def make_engine(model, args, prompt_len):
    b, k = args.concurrency, args.k if args.mode == "spec" else 0
    # One prefill batch covers every request, so the decode window starts matched.
    budget = args.max_num_batched_tokens or max(128, b * prompt_len, b * (k + 1))
    capacity = prompt_len + args.max_tokens + k + 1
    # LAST_EXITED keeps one KV plane per recurrence depth.
    planes = model.config.total_ut_steps
    blocks = args.num_blocks or b * planes * (-(-capacity // args.block_size)) + 8
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=blocks, block_size=args.block_size),
        scheduler_config=SchedulerConfig(
            max_num_seqs=b, max_num_batched_tokens=budget, prefill_chunk_size=prompt_len
        ),
        attention_backend=args.backend,
        exit_config=ExitConfig("ouro"),
        # Synchronous in both cases; "graphs" (native only) captures decode recurrent cores
        # and is a launch-overhead reference, not a speculative configuration.
        execution_config=ExecutionConfig(static_buffers=True, cuda_graphs=True)
        if args.execution == "graphs"
        else ExecutionConfig(),
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=args.k, draft_loops=args.draft_loops, target_loops=4
        )
        if args.mode == "spec"
        else None,
    )
    return engine, dict(num_blocks=blocks, max_num_batched_tokens=budget, prefill_chunk=prompt_len)


def sampling_params(args):
    # Fixed depth 4 and exit_threshold=1.0 in both modes: native and spec do the same work
    # per committed token at full depth, so spec vs. native isolates speculation.
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        max_loops=4,
        exit_threshold=1.0,
        ignore_eos=not args.natural_eos,
    )


# --------------------------------------------------------------------------- NVTX


_NVTX_TARGETS = (
    (LLMEngine, "step", "engine.step"),
    (LLMEngine, "_update_speculative", "spec.commit"),
    (LLMEngine, "_update", "native.update"),
    (Scheduler, "schedule", "sched.schedule"),
    (SpeculativeRunner, "execute", "spec.round"),
    (ModelRunner, "execute", "native.execute"),
    (KVCacheManager, "_prepare_batch", "kv.prepare_batch"),
    (KVCacheManager, "truncate_suffix", "spec.kv_update"),
    (OuroForCausalLM, "prelude", "model.prelude"),
    (OuroForCausalLM, "coda", "model.coda"),
)


@contextlib.contextmanager
def label_range(label):
    """NVTX (nsys) plus record_function (torch.profiler) range; no synchronization."""
    torch.cuda.nvtx.range_push(label)
    try:
        with torch.profiler.record_function(label):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


def install_nvtx():
    """Wrap production methods in labeled CPU ranges (they do not imply GPU durations)."""

    def wrap(fn, label):
        @functools.wraps(fn)
        def inner(*a, **kw):
            with label_range(label):
                return fn(*a, **kw)

        return inner

    for cls, name, label in _NVTX_TARGETS:
        setattr(cls, name, wrap(getattr(cls, name), label))

    core = SpeculativeRunner._core

    @functools.wraps(core)
    def core_marked(self, hidden, ids, positions, depth, *, packed=False):
        with label_range(f"spec.{'verify' if packed else 'draft'}.core.d{depth}"):
            return core(self, hidden, ids, positions, depth, packed=packed)

    SpeculativeRunner._core = core_marked


# --------------------------------------------------------------------------- sync counting


class SyncRecorder:
    """Attribute set_sync_debug_mode warnings to the innermost vllm_rlt frame."""

    def __init__(self):
        self.sites = Counter()
        self.total = 0

    @contextlib.contextmanager
    def active(self):
        previous = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if "synchroniz" not in str(message):
                return previous(message, category, filename, lineno, file, line)
            self.total += 1
            site = "<outside vllm_rlt>"
            for frame in reversed(traceback.extract_stack()[:-1]):
                if "vllm_rlt" in frame.filename and "benchmarks" not in frame.filename:
                    site = f"{Path(frame.filename).relative_to(REPO)}:{frame.lineno} {frame.line}"
                    break
            self.sites[site] += 1

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            torch.cuda.set_sync_debug_mode("warn")
            try:
                yield
            finally:
                torch.cuda.set_sync_debug_mode("default")
                warnings.showwarning = previous


# --------------------------------------------------------------------------- graph events


class GraphReplayTimer:
    """Record an event pair around each CUDAGraph.replay on the current stream."""

    def __init__(self):
        self.pairs = []

    @contextlib.contextmanager
    def active(self):
        original = torch.cuda.CUDAGraph.replay
        pairs = self.pairs

        def replay(graph):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            original(graph)
            end.record()
            pairs.append((start, end))

        torch.cuda.CUDAGraph.replay = replay
        try:
            yield
        finally:
            torch.cuda.CUDAGraph.replay = original

    def total_ms(self):
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in self.pairs)


# --------------------------------------------------------------------------- run


def graph_counters(engine):
    """Capture/replay/fallback totals of every graph cache the engine owns."""
    owners = {"native": engine.model_runner, "spec": engine.speculative_runner}
    counters = {}
    for prefix, owner in owners.items():
        for name in ("graphs", "coda_graphs"):
            g = getattr(owner, name, None) if owner is not None else None
            if g is not None:
                counters[f"{prefix}.{name}"] = dict(
                    captures=g.captures, replays=g.replays, fallbacks=g.fallbacks
                )
    return counters


def run_once(
    engine, prompts, params, args, tag, *, measure, profile=False, syncs=None, replay_timer=None
):
    runner = engine.speculative_runner
    ids = [f"{tag}-{i}" for i in range(len(prompts))]
    for rid, prompt in zip(ids, prompts):
        engine.add_request(rid, list(prompt), params)
    requests = [engine.scheduler.requests[rid] for rid in ids]
    finished = {}
    steps = []

    def step():
        t0 = time.perf_counter()
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
        batch = engine.last_schedule
        steps.append(
            (
                batch.stage.value if batch else None,
                sum(i.token_count for i in batch.items) if batch else 0,
                time.perf_counter() - t0,
            )
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    boundary_tokens = 0
    stats0 = None
    if measure == "decode" or profile or syncs is not None or replay_timer is not None:
        # Prefill phase: run until every request owns its first token.
        while engine.has_unfinished_requests() and any(
            not r.generated_token_ids and r.request_id not in finished for r in requests
        ):
            step()
        prefill_steps = len(steps)
        torch.cuda.synchronize()
        start = time.perf_counter()
        boundary_tokens = sum(len(r.generated_token_ids) for r in requests)
    else:
        prefill_steps = None
    stats0 = dict(vars(runner.stats)) if runner else None
    graphs0 = graph_counters(engine)
    window_first_step = len(steps)

    if profile:
        # torch-cuda records only CUPTI activity (no CPU op events): lower host overhead,
        # for comparing GPU busy time with the profiled window's wall time.
        torch_profiler = args.profiler in ("torch", "torch-cuda")
        if torch_profiler:
            activities = [torch.profiler.ProfilerActivity.CUDA]
            if args.profiler == "torch":
                activities.insert(0, torch.profiler.ProfilerActivity.CPU)
            tracer = torch.profiler.profile(activities=activities)
            tracer.start()
        else:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.synchronize()
        profile_t0 = time.perf_counter()
        try:
            for _ in range(args.profile_steps):
                if not engine.has_unfinished_requests():
                    break
                step()
            torch.cuda.synchronize()
            profile_wall_s = time.perf_counter() - profile_t0
        finally:
            if torch_profiler:
                tracer.stop()
                tracer.export_chrome_trace(str(args.trace_path))
            else:
                torch.cuda.cudart().cudaProfilerStop()
        profiled_steps = len(steps) - window_first_step
    if replay_timer is not None:
        with replay_timer.active():
            while engine.has_unfinished_requests():
                step()
    if syncs is not None:
        with syncs.active():
            while engine.has_unfinished_requests():
                step()
    while engine.has_unfinished_requests():
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    outputs = [finished[rid] for rid in ids]
    committed = sum(len(o.token_ids) for o in outputs) - boundary_tokens
    window = steps[window_first_step:]
    record = dict(
        tag=tag,
        measure=measure,
        elapsed_s=elapsed,
        committed_tokens=committed,
        tokens_per_s=committed / elapsed,
        prompt_tokens=[len(p) for p in prompts],
        output_tokens=[len(o.token_ids) for o in outputs],
        finish_reasons=[o.finish_reason for o in outputs],
        boundary_tokens=boundary_tokens,
        prefill_steps=prefill_steps,
        window_steps=len(window),
        window_stage_counts=dict(Counter(s for s, _, _ in window)),
        window_step_host_s=[round(t, 7) for _, _, t in window],
        window_step_rows=[n for _, n, _ in window],
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        output_sha256=hashlib.sha256(
            json.dumps([o.token_ids for o in outputs]).encode()
        ).hexdigest(),
        token_ids=[o.token_ids for o in outputs],
    )
    if runner:
        s1 = vars(runner.stats)
        # stats.rounds counts request-rounds (one per request per spec step).
        record["spec"] = {k: s1[k] - stats0[k] for k in s1}
        record["spec"]["engine_rounds"] = record["window_stage_counts"].get("speculative", 0)
        d = record["spec"]["drafted_tokens"]
        record["spec"]["acceptance_rate"] = record["spec"]["accepted_tokens"] / d if d else None
    graphs1 = graph_counters(engine)
    record["graphs"] = {k: {c: v[c] - graphs0[k][c] for c in v} for k, v in graphs1.items()}
    if profile:
        record["profiled_steps"] = profiled_steps
        record["profile_wall_s"] = profile_wall_s
    return record


def summarize(records):
    tps = [r["tokens_per_s"] for r in records]
    el = [r["elapsed_s"] for r in records]
    return dict(
        n=len(records),
        tokens_per_s_median=statistics.median(tps),
        tokens_per_s_min=min(tps),
        tokens_per_s_max=max(tps),
        tokens_per_s_stdev=statistics.stdev(tps) if len(tps) > 1 else 0.0,
        elapsed_s_median=statistics.median(el),
        outputs_identical=len({r["output_sha256"] for r in records}) == 1,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", default="574fa66cb8bf5abdc979642d01cf2b79b16bfab1")
    p.add_argument("--backend", default="flash_attn_4")
    p.add_argument("--mode", choices=["native", "spec"], required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--draft-loops", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--natural-eos", action="store_true", help="honor EOS (default: ignore_eos)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int)
    p.add_argument("--max-num-batched-tokens", type=int)
    p.add_argument("--measure", choices=["decode", "e2e"], default="decode")
    p.add_argument("--execution", choices=["eager", "graphs"], default="eager")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--profile-region", action="store_true")
    p.add_argument("--profile-steps", type=int, default=12)
    p.add_argument("--profiler", choices=["nsys", "torch", "torch-cuda"], default="nsys")
    p.add_argument("--count-syncs", action="store_true")
    p.add_argument("--graph-events", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--name", help="output file stem (default derived from the configuration)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    load_s = time.perf_counter() - t0
    prompts, workload = build_prompts(tokenizer, args.concurrency, args.prompt_len)
    engine, engine_cfg = make_engine(model, args, args.prompt_len)
    params = sampling_params(args)
    if args.profile_region:
        install_nvtx()

    kind = (
        "profile"
        if args.profile_region
        else "syncs"
        if args.count_syncs
        else "graphevents"
        if args.graph_events
        else args.measure
    )
    stem = args.name or (
        f"{args.mode}_{args.backend}_b{args.concurrency}"
        + (f"_k{args.k}" if args.mode == "spec" else "")
        + ("_graphs" if args.execution == "graphs" else "")
        + f"_p{args.prompt_len}_o{args.max_tokens}_{'eos' if args.natural_eos else 'fixed'}_{kind}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace_path = args.output_dir / f"{stem}.trace.json.gz"
    warm = []
    for i in range(args.warmup):
        warm.append(run_once(engine, prompts, params, args, f"warm{i}", measure=args.measure))
    result = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        engine=engine_cfg,
        workload=workload,
        environment=environment(args),
        load_s=load_s,
        warmup=[dict(elapsed_s=r["elapsed_s"], tokens_per_s=r["tokens_per_s"]) for r in warm],
        status="ok",
    )
    try:
        if args.profile_region:
            result["runs"] = [
                run_once(engine, prompts, params, args, "profile", measure="decode", profile=True)
            ]
        elif args.graph_events:
            timer = GraphReplayTimer()
            record = run_once(
                engine, prompts, params, args, "graphevents", measure="decode", replay_timer=timer
            )
            record["graph_replays"] = len(timer.pairs)
            record["graph_replay_ms"] = timer.total_ms()
            wall_ms = record["elapsed_s"] * 1e3
            record["outside_graph_fraction"] = 1 - record["graph_replay_ms"] / wall_ms
            result["runs"] = [record]
        elif args.count_syncs:
            recorder = SyncRecorder()
            record = run_once(
                engine, prompts, params, args, "syncs", measure="decode", syncs=recorder
            )
            record["sync_total"] = recorder.total
            record["sync_sites"] = dict(recorder.sites.most_common())
            steps = record["window_steps"]
            record["sync_per_window_step"] = recorder.total / steps if steps else None
            result["runs"] = [record]
        else:
            runs = [
                run_once(engine, prompts, params, args, f"rep{i}", measure=args.measure)
                for i in range(args.repeats)
            ]
            result["runs"] = runs
            result["summary"] = summarize(runs)
    except Exception as error:  # record failures in the raw output, then re-raise
        result["status"] = f"failed: {type(error).__name__}: {error}"
        (args.output_dir / f"{stem}.json").write_text(json.dumps(result, indent=1))
        raise
    (args.output_dir / f"{stem}.json").write_text(json.dumps(result, indent=1))
    run = result["runs"][0]
    line = f"{stem}: {run['committed_tokens']} tok"
    if "summary" in result:
        s = result["summary"]
        line += (
            f", median {s['tokens_per_s_median']:.1f} tok/s "
            f"[{s['tokens_per_s_min']:.1f}, {s['tokens_per_s_max']:.1f}] n={s['n']}"
            f", identical={s['outputs_identical']}"
        )
    if "spec" in run:
        line += f", accept={run['spec']['acceptance_rate']:.3f}"
    if "graph_replay_ms" in run:
        line += (
            f", wall {run['elapsed_s'] * 1e3:.1f} ms, graph replay {run['graph_replay_ms']:.1f} ms "
            f"({run['graph_replays']} replays), outside graphs {run['outside_graph_fraction']:.1%}"
        )
    if "sync_total" in run:
        line += f", syncs={run['sync_total']} ({run['sync_per_window_step']:.1f}/step)"
    print(line, flush=True)


if __name__ == "__main__":
    main()
