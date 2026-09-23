"""SSE accounting for matched speculative serving measurements."""

import asyncio
import json
import subprocess
import sys
import time

from aiohttp import ClientSession, web


def test_sse_collector_counts_choice_events_across_transport_chunks():
    from benchmarks.speculative_serving import SSECollector

    collector = SSECollector()
    first = json.dumps({"choices": [{"text": "A", "finish_reason": None}]})
    second = json.dumps({"choices": [{"text": "B", "finish_reason": "length"}]})
    usage = json.dumps({"choices": [], "usage": {"completion_tokens": 2}})
    collector.feed(f"data: {first}\n\ndata: {second[:10]}".encode(), 1.0)
    collector.feed(f"{second[10:]}\n\ndata: {usage}\n\ndata: [DONE]\n\n".encode(), 1.1)

    assert collector.first_token_time == 1.0
    assert collector.choice_times == [1.0, 1.1]
    assert collector.completion_tokens == 2
    assert collector.done
    assert collector.text == "AB"
    assert collector.chunk_choice_events == [1, 1]


def test_summary_counts_failed_requests_and_latency_percentiles():
    from benchmarks.speculative_serving import summarize_requests

    result = summarize_requests(
        [
            {
                "error": None,
                "ttft_ms": 10.0,
                "e2e_ms": 30.0,
                "choice_interval_ms": [5.0],
                "tokens": 2,
                "choice_events": 2,
            },
            {
                "error": None,
                "ttft_ms": 20.0,
                "e2e_ms": 40.0,
                "choice_interval_ms": [10.0],
                "tokens": 2,
                "choice_events": 1,
            },
            {
                "error": "HTTP 500",
                "ttft_ms": None,
                "e2e_ms": None,
                "choice_interval_ms": [],
                "tokens": 0,
                "choice_events": 0,
            },
        ]
    )
    assert result["successful_requests"] == 2
    assert result["failed_requests"] == 1
    assert result["ttft_p50_ms"] == 15.0
    assert result["e2e_p50_ms"] == 35.0
    assert result["tokens_per_choice_event"] == 4 / 3
    assert result["choice_interval_p50_ms"] == 7.5


def test_serving_cli_exposes_arrival_workload_options():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.speculative_serving", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--base-url" in result.stdout
    assert "--request-rate" in result.stdout
    assert "--max-concurrency" in result.stdout
    assert "--model-revision" in result.stdout
    assert "--natural-eos" in result.stdout


def test_incomplete_http_stream_is_counted_as_failure():
    from benchmarks.speculative_serving import send_request, summarize_requests

    async def exercise():
        async def handler(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"A"}]}\n\n')
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as session:
                start = time.perf_counter()
                row = await send_request(
                    session,
                    asyncio.Semaphore(1),
                    f"http://127.0.0.1:{port}/v1/completions",
                    {"max_tokens": 1, "ignore_eos": True},
                    0,
                    start,
                    start,
                )
            assert row["error"] == "stream ended without [DONE]"
            assert summarize_requests([row])["failed_requests"] == 1
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_speculative_stream_accepts_one_choice_for_multiple_tokens():
    from benchmarks.speculative_serving import send_request

    async def exercise():
        async def handler(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"AB"}]}\n\n')
            await response.write(b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n')
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as session:
                start = time.perf_counter()
                row = await send_request(
                    session,
                    asyncio.Semaphore(1),
                    f"http://127.0.0.1:{port}/v1/completions",
                    {"max_tokens": 2, "ignore_eos": True},
                    0,
                    start,
                    start,
                    mode="speculative",
                )
            assert row["error"] is None
            assert row["choice_events"] == 1
            assert row["tokens"] == 2
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
