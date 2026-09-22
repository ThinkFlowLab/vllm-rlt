"""Resident state and mapped-host descriptor ownership under delayed execution."""

from dataclasses import replace

import pytest
import torch

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.request import Stage


def engine(*, asynchronous=True, static=False, graph=False, shared=False):
    torch.manual_seed(19)
    config = replace(OuroConfig.tiny(), head_dim=64)
    return LLMEngine(
        OuroForCausalLM(config).to(device="cuda", dtype=torch.bfloat16),
        cache_config=CacheConfig(128, 2, "shared" if shared else "last_exited"),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=3),
        exit_config=ExitConfig("trace", depths_by_request={"a": [4, 1, 3, 2, 4, 1]}),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous,
            multi_stream=True,
            static_buffers=static,
            pad_to_power_of_two=static,
            cuda_graphs=graph,
        ),
        attention_backend="triton",
    )


def drain(e):
    result = {}
    for _ in range(200):
        if not e.has_unfinished_requests():
            return result
        for o in e.step():
            if o.finished:
                result[o.request_id] = o
    pytest.fail("failed to drain")


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
@pytest.mark.parametrize(
    "static,graph,shared", [(False, False, False), (True, True, False), (True, False, True)]
)
def test_resident_state_trace_refill_abort_and_slot_reuse(use_uva, static, graph, shared):
    e = engine(static=static, graph=graph, shared=shared)
    e.model_runner.async_state.use_uva = use_uva
    reference = engine(asynchronous=False, shared=shared)
    params = SamplingParams(max_tokens=6, min_loops=1, ignore_eos=True)
    for round_index in range(3):
        for target in (e, reference):
            for i in range(3):
                target.add_request(str(i), [i + 2, round_index + 5], params, trace_id="a")
        if round_index == 1:
            for _ in range(4):
                e.step()
            e.abort_request("1")
            reference.abort_request("1")
            for target in (e, reference):
                target.add_request("1", [7, 8, 9], params, trace_id="a")
        actual, expected = drain(e), drain(reference)
        assert actual == expected
        assert e.cache_manager.num_used_blocks == 0
        assert e.model_runner.state_slots == {}
        assert e.model_runner.async_state.owners == {}
        assert len(set(e.model_runner.free_state_slots)) == 3


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_bank_is_not_reused_until_its_final_gpu_reader_finishes(use_uva):
    e = engine()
    state = e.model_runner.async_state
    state.use_uva = use_uva
    e.add_request("a", [2], SamplingParams(max_tokens=2, min_loops=1), trace_id="a")
    e.step()
    request = e.scheduler.requests["a"]
    # Force wraparound while the first bank is still referenced by queued work.
    bank = state.banks[state.index]
    stream = torch.cuda.Stream()
    expected = torch.tensor([17, 23, 29], dtype=torch.int64)
    bank.host[0].copy_(expected)
    actual = torch.empty(3, dtype=torch.int64, device="cuda")
    from vllm_rlt.kernels.routing import gather_kernel

    with torch.cuda.stream(stream):
        slots = torch.zeros(1, dtype=torch.int64, device="cuda")
        descriptor = bank.host if use_uva else bank.host.to("cuda", non_blocking=True)
        # Warm this signature before the delay so JIT compilation cannot consume it.
        gather_kernel[(1, 1)](descriptor, slots, actual, 1, 3, 3, 3, 256)
        torch.cuda._sleep(150_000_000)
        gather_kernel[(1, 1)](descriptor, slots, actual, 1, 3, 3, 3, 256)
        bank.record_done()
    assert not bank.done.query()
    reused, _ = state.prepare([request], [0], [0], 1)
    assert reused is bank and bank.done.query()
    assert torch.equal(actual.cpu(), expected)
    e.abort_request("a")


@pytest.mark.gpu
def test_execution_error_retires_routing_readers_before_request_reuse(monkeypatch):
    e = engine()
    params = SamplingParams(max_tokens=3, min_loops=1, ignore_eos=True)
    e.add_request("a", [2], params, trace_id="a")
    execute = e.model_runner._execute

    def fail(batch, prepared=None):
        result = execute(batch, prepared)
        if batch.stage == Stage.RECURRENT:
            torch.cuda._sleep(10_000_000)
            raise RuntimeError("injected failure after GPU submission")
        return result

    monkeypatch.setattr(e.model_runner, "_execute", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        drain(e)
    assert not e.model_runner.async_state.owners
    assert e.cache_manager.num_used_blocks == 0
    monkeypatch.setattr(e.model_runner, "_execute", execute)
    e.add_request("a", [3], params, trace_id="a")
    assert drain(e)["a"].prompt_token_ids == [3]


@pytest.mark.gpu
@pytest.mark.parametrize("use_uva", [False, True])
def test_batched_exit_copies_only_target_positions_and_depths(use_uva):
    e = engine()
    runner, cache = e.model_runner, e.cache_manager
    runner.async_state.use_uva = use_uva
    cache.key_cache.fill_(-11)
    cache.value_cache.fill_(-12)
    for i in range(3):
        e.add_request(str(i), [2, 3], SamplingParams(max_tokens=2, min_loops=1), trace_id="a")
    e.scheduler._admit()
    requests = list(e.scheduler.requests.values())
    bank, _ = runner.async_state.prepare(requests, [0, 1, 2], [1, 1, 1], 3)
    bank.transfer()
    for layer in range(cache.num_layers):
        values = (
            torch.arange(3 * cache.num_kv_heads * cache.head_dim, device="cuda")
            .reshape(3, cache.num_kv_heads, cache.head_dim)
            .to(cache.key_cache.dtype)
            + layer
        )
        cache.write(layer, [r.request_id for r in requests], [0, 1, 2], [1, 1, 1], values, -values)
    expected_k, expected_v = cache.key_cache.clone(), cache.value_cache.clone()
    ready = torch.cuda.Event()
    ready.record()
    for i, request in enumerate(requests):
        request.loops_done = i + 1
        runner.events[request.request_id] = ready
        src = cache.get_block_table(request.request_id, i)[0]
        for depth in range(i + 1, 4):
            dst = cache.get_block_table(request.request_id, depth)[0]
            expected_k[dst, :, 1].copy_(expected_k[src, :, 1])
            expected_v[dst, :, 1].copy_(expected_v[src, :, 1])
    runner.finalize_many(requests)
    runner.synchronize()
    assert torch.equal(cache.key_cache, expected_k)
    assert torch.equal(cache.value_cache, expected_v)
    for request in requests:
        e.abort_request(request.request_id)
