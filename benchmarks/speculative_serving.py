"""SSE client for matched Ouro native/speculative serving measurements.

Run against separately started native and speculative servers with identical
arguments. This client records all attempts, including HTTP and stream errors.
"""

import argparse
import asyncio
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from benchmarks.speculative import make_prompts


class SSECollector:
    """Account for choice events, usage and transport delivery bursts."""

    def __init__(self):
        self.buffer = b""
        self.choice_times = []
        self.chunk_choice_events = []
        self.completion_tokens = None
        self.done = False
        self.text = ""

    @property
    def first_token_time(self):
        return self.choice_times[0] if self.choice_times else None

    def feed(self, data, timestamp):
        self.buffer += data
        choice_events = 0
        while b"\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            if not frame.startswith(b"data: "):
                continue
            payload = frame[6:]
            if payload == b"[DONE]":
                self.done = True
                continue
            event = json.loads(payload)
            choices = event.get("choices", [])
            if choices:
                self.choice_times.append(timestamp)
                self.text += choices[0].get("text", "")
                choice_events += 1
            if event.get("usage"):
                self.completion_tokens = event["usage"]["completion_tokens"]
        if choice_events:
            self.chunk_choice_events.append(choice_events)


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_requests(records, elapsed_seconds=None):
    good = [row for row in records if row["error"] is None]
    metrics = {
        "successful_requests": len(good),
        "failed_requests": len(records) - len(good),
        "completion_tokens": sum(row["tokens"] for row in good),
        "choice_events": sum(row["choice_events"] for row in good),
    }
    for field in ("ttft_ms", "e2e_ms"):
        values = [row[field] for row in good]
        metrics[f"{field[:-3]}_p50_ms"] = percentile(values, 50)
        metrics[f"{field[:-3]}_p95_ms"] = percentile(values, 95)
    intervals = [value for row in good for value in row["choice_interval_ms"]]
    metrics["choice_interval_p50_ms"] = percentile(intervals, 50)
    metrics["choice_interval_p95_ms"] = percentile(intervals, 95)
    metrics["tokens_per_choice_event"] = (
        metrics["completion_tokens"] / metrics["choice_events"]
        if metrics["choice_events"]
        else None
    )
    metrics["tokens_per_second"] = (
        metrics["completion_tokens"] / elapsed_seconds if elapsed_seconds else None
    )
    return metrics


async def send_request(
    session, semaphore, url, body, index, scheduled_at, benchmark_start, *, mode="native"
):
    delay = scheduled_at - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    async with semaphore:
        started = time.perf_counter()
        collector = SSECollector()
        error = None
        try:
            async with session.post(url, json=body) as response:
                if response.status != 200:
                    error = f"HTTP {response.status}: {(await response.text())[:200]}"
                else:
                    async for data, _ in response.content.iter_chunks():
                        collector.feed(data, time.perf_counter())
                    if not collector.done:
                        error = "stream ended without [DONE]"
                    elif collector.completion_tokens is None:
                        error = "stream has no usage count"
                    elif mode == "native" and collector.completion_tokens != len(
                        collector.choice_times
                    ):
                        error = "native usage count differs from choice events"
                    elif mode == "speculative" and not (
                        1 <= len(collector.choice_times) <= collector.completion_tokens
                    ):
                        error = "invalid speculative choice event count"
                    elif body["ignore_eos"] and collector.completion_tokens != body["max_tokens"]:
                        error = "completion length differs from fixed workload"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()
        times = collector.choice_times
        return {
            "request_index": index,
            "scheduled_offset_ms": (scheduled_at - benchmark_start) * 1000,
            "start_offset_ms": (started - benchmark_start) * 1000,
            "end_offset_ms": (ended - benchmark_start) * 1000,
            "error": error,
            "ttft_ms": (times[0] - started) * 1000 if times and error is None else None,
            "e2e_ms": (times[-1] - started) * 1000 if times and error is None else None,
            "choice_interval_ms": [
                (later - earlier) * 1000 for earlier, later in zip(times, times[1:])
            ],
            "chunk_choice_events": collector.chunk_choice_events,
            "choice_events": len(times),
            "tokens": collector.completion_tokens or 0,
            "text": collector.text,
        }


async def run_workload(args, prompts):
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(args.base_url.rstrip("/") + "/health") as response:
            if response.status != 200:
                raise RuntimeError(f"server is not ready: HTTP {response.status}")
        semaphore = asyncio.Semaphore(args.max_concurrency)
        endpoint = args.base_url.rstrip("/") + "/v1/completions"
        body = {
            "model": args.served_model,
            "prompt": prompts[0],
            "max_tokens": args.output_tokens,
            "temperature": 0,
            "min_loops": 2,
            "max_loops": 4,
            "exit_threshold": 1.0,
            "ignore_eos": not args.natural_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        warmup_start = time.perf_counter()
        warmup = await send_request(
            session, semaphore, endpoint, body, -1, warmup_start, warmup_start, mode=args.mode
        )
        if warmup["error"]:
            raise RuntimeError(f"warmup failed: {warmup['error']}")
        benchmark_start = time.perf_counter()
        tasks = []
        for index, prompt in enumerate(prompts):
            scheduled_at = benchmark_start
            if math.isfinite(args.request_rate):
                scheduled_at += index / args.request_rate
            tasks.append(
                asyncio.create_task(
                    send_request(
                        session,
                        semaphore,
                        endpoint,
                        {**body, "prompt": prompt},
                        index,
                        scheduled_at,
                        benchmark_start,
                        mode=args.mode,
                    )
                )
            )
        records = await asyncio.gather(*tasks)
        elapsed = max(row["end_offset_ms"] for row in records) / 1000
        return records, summarize_requests(records, elapsed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--served-model", default="ouro")
    parser.add_argument("--mode", required=True, choices=["native", "speculative"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=math.inf)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--natural-eos", action="store_true", help="Allow EOS before max_tokens")
    args = parser.parse_args(argv)
    if args.num_requests < 1 or args.max_concurrency < 1 or args.request_rate <= 0:
        parser.error("request count, concurrency and request rate must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    prompt_ids = make_prompts(tokenizer, args.prompt_tokens, args.num_requests)
    prompts = [tokenizer.decode(row, skip_special_tokens=True) for row in prompt_ids]
    records, summary = asyncio.run(run_workload(args, prompts))
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": args.model_revision,
        "gpu_and_driver": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "mode": args.mode,
        "base_url": args.base_url,
        "num_requests": args.num_requests,
        "max_concurrency": args.max_concurrency,
        "request_rate": args.request_rate if math.isfinite(args.request_rate) else "inf",
        "requested_prompt_tokens": args.prompt_tokens,
        "actual_prompt_tokens": [len(tokenizer.encode(prompt)) for prompt in prompts],
        "output_tokens": args.output_tokens,
        "natural_eos": args.natural_eos,
        "summary": summary,
        "requests": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    if summary["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
