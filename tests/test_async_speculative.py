"""Cross-round device progress, reservation and lifetime contracts."""

import pytest
import torch

from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.request import Request, Stage


def tiny(device="cpu", dtype=torch.float32):
    torch.manual_seed(123)
    return OuroForCausalLM(OuroConfig.tiny()).to(device=device, dtype=dtype)


def make_engine(model, *, asynchronous=True, k=3, budget=16, prefix=False, multi_stream=True):
    return LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(256, 2, enable_prefix_caching=prefix, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=budget, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(async_scheduling=asynchronous, multi_stream=multi_stream),
        speculative_config=SpeculativeConfig(k),
    )


def drain(engine):
    outputs = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            outputs[output.request_id] = output
    pytest.fail("async speculative engine failed to drain")


def bootstrap(engine, rid="r", max_tokens=24, ignore_eos=True):
    engine.add_request(rid, [2, 3, 4], SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))
    for _ in range(30):
        if engine.step():
            return engine.scheduler.requests[rid]
    pytest.fail("bootstrap did not produce a token")


def test_scheduler_uses_upper_bound_and_releases_rejected_budget():
    scheduler = Scheduler(SchedulerConfig(), None, SpeculativeConfig(3))
    r = Request("r", [2, 3], SamplingParams(max_tokens=8), generated_token_ids=[4])
    r.num_output_placeholders = 4
    item = scheduler._make_scheduled_item(r, Stage.SPECULATIVE, 16)
    assert (item.token_start, item.token_count) == (6, 3)
    # One committed token instead of the four reserved: recover three slots.
    r.num_output_placeholders = 0
    r.generated_token_ids.append(5)
    item = scheduler._make_scheduled_item(r, Stage.SPECULATIVE, 16)
    assert (item.token_start, item.token_count) == (3, 4)


def test_async_spec_rejects_cpu_before_cache_allocation():
    with pytest.raises(ValueError, match="CUDA and Triton"):
        make_engine(tiny())


@pytest.mark.gpu
@pytest.mark.parametrize("k,budget", [(1, 1), (2, 7), (4, 16), (8, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_async_matches_sync_with_ragged_limits_and_prefix_reuse(k, budget, dtype):
    model = tiny("cuda", dtype)
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (1, 13, 8)]
    prompts = [[2, 3, 4, 5], [2, 3, 4, 7], [9]]
    results = []
    for asynchronous in (False, True):
        e = make_engine(model, asynchronous=asynchronous, k=k, budget=budget, prefix=True)
        for repetition in range(2):
            for i, (prompt, p) in enumerate(zip(prompts, params)):
                e.add_request(str(i), prompt, p)
            output = drain(e)
            results.append([output[str(i)].token_ids for i in range(3)])
            assert all(o.finished for o in output.values())
        if asynchronous:
            assert not e.speculative_runner.states
            assert all(not bank.leased for bank in e.speculative_runner.banks)
    assert all(r == results[0] for r in results)


@pytest.mark.gpu
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_second_round_executes_before_cpu_commit_at_every_rejection(accepted, monkeypatch):
    model = tiny("cuda")
    e = make_engine(model)
    request = bootstrap(e)
    start = request.position
    calls = 0

    def controlled(hidden):
        nonlocal calls
        phase = calls % 4
        calls += 1
        logits = torch.full((len(hidden), model.config.vocab_size), -100.0, device="cuda")
        if phase < 3:
            logits[:, 10 + phase] = 100
        else:
            for row in range(4):
                logits[row, 10 + row if row < accepted else 20 + row] = 100
        return logits

    monkeypatch.setattr(model, "coda", controlled)

    # A CPU read of candidate/acceptance values would fail during submission.
    def forbidden(*args, **kwargs):
        raise AssertionError("host read in cross-round GPU submission")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        e.step()
        e.step()
    assert len(e._pending_speculative) == 2
    assert len(request.generated_token_ids) == 1
    assert e.speculative_runner.submitted_before_collect == 1
    state = e.speculative_runner.states["r"]
    e.speculative_runner.synchronize()
    assert int(state.position) == start + 2 * (accepted + 1)
    # Collecting the older round must not rewind the newer device state.
    out = e._collect_speculative()[0]
    assert out.token_ids[1:] == list(range(10, 10 + accepted)) + [20 + accepted]
    assert int(state.position) == start + 2 * (accepted + 1)
    e.abort_request("r")
    drain(e)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_eos_discards_queued_round_and_request_id_can_be_reused(monkeypatch):
    model = tiny("cuda")
    e = make_engine(model)
    bootstrap(e, ignore_eos=False)
    coda = model.coda
    monkeypatch.setattr(
        model,
        "coda",
        lambda h: torch.zeros(
            len(h),
            model.config.vocab_size,
            device="cuda",
        ),
    )
    e.step()
    e.step()
    out = e._collect_speculative()[0]
    assert out.finished and out.token_ids[-1] == 0 and len(out.token_ids) == 2
    # Reuse while an old ticket still exists. It must never touch the new request.
    monkeypatch.setattr(model, "coda", coda)
    e.add_request("r", [7, 8], SamplingParams(max_tokens=5, ignore_eos=True))
    output = drain(e)["r"]
    assert len(output.token_ids) == 5
    assert e.speculative_runner.discarded_rounds == 1
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_cancel_and_partial_submission_failure_reclaim_resources(monkeypatch):
    model = tiny("cuda")
    e = make_engine(model)
    bootstrap(e)
    e.step()
    e.step()
    assert e.abort_request("r").finish_reason == "abort"
    drain(e)
    assert not e.speculative_runner.states
    assert e.cache_manager.num_used_blocks == 0
    bootstrap(e)

    def fail(*args, **kwargs):
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(model, "coda", fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.step()
    assert not e.scheduler.requests
    assert not e.speculative_runner.states
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_async_rejects_random_sampling():
    e = make_engine(tiny("cuda"))
    with pytest.raises(ValueError, match="greedy"):
        e.add_request("r", [2], SamplingParams(temperature=0.8))


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["torch", "flash_attn_4"])
def test_async_rejects_other_backends_before_loading_attention(backend):
    with pytest.raises(ValueError, match="CUDA and Triton"):
        LLMEngine(
            tiny("cuda"),
            attention_backend=backend,
            execution_config=ExecutionConfig(async_scheduling=True),
            speculative_config=SpeculativeConfig(3),
        )


@pytest.mark.gpu
def test_prefill_arrivals_interleave_with_inflight_rounds():
    model = tiny("cuda", torch.bfloat16)
    e = make_engine(model, budget=5, prefix=True)
    bootstrap(e, max_tokens=17)
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2
    prompts = [[2, 3, 4], [2, 3, 4, 8, 9], [7]]
    lengths = [17, 7, 9]
    for i in (1, 2):
        e.add_request(str(i), prompts[i], SamplingParams(max_tokens=lengths[i], ignore_eos=True))
    actual = drain(e)
    reference = make_engine(model, asynchronous=False, budget=5, prefix=True)
    for rid, prompt, length in zip(["r", "1", "2"], prompts, lengths):
        reference.add_request(rid, prompt, SamplingParams(max_tokens=length, ignore_eos=True))
    expected = drain(reference)
    assert {rid: o.token_ids for rid, o in actual.items()} == {
        rid: o.token_ids for rid, o in expected.items()
    }
    assert all(o.finished for o in actual.values())
    assert not e.speculative_runner.states
    assert all(not bank.leased for bank in e.speculative_runner.banks)
    e.close()
    reference.close()


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_close_drains_both_submitted_rounds(multi_stream):
    e = make_engine(tiny("cuda"), multi_stream=multi_stream)
    bootstrap(e)
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2
    e.close()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.states
    assert all(not bank.leased for bank in e.speculative_runner.banks)
    e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_round_kv_matches_serial_replay_at_every_depth(dtype):
    model = tiny("cuda", dtype)
    e = make_engine(model)
    request = bootstrap(e)
    e.step()
    e.step()
    results = [ticket.collect()[0][0] for ticket in e._pending_speculative]
    tokens = request.prompt_token_ids + request.generated_token_ids
    tokens += [token for result in results for token in result.token_ids]
    # Last emitted token has not been forwarded; all earlier tokens have KV.
    tokens = tokens[:-1]
    oracle = make_engine(model, asynchronous=False).cache_manager
    assert oracle.allocate("serial", len(tokens))
    with torch.inference_mode():
        for position, token in enumerate(tokens):
            hidden = model.prelude(torch.tensor([token], device="cuda"))
            for depth in range(model.config.total_ut_steps):
                hidden, _ = model.recurrent(hidden, ["serial"], [depth], [position], oracle)
    allocation = e.cache_manager._get_allocation("r")
    pos = torch.arange(len(tokens), device="cuda")
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
    for depth in range(model.config.total_ut_steps):
        pages = torch.tensor(allocation.block_tables[depth], device="cuda")
        blocks, offsets = pages[pos // 2], pos % 2
        for layer in range(model.config.num_hidden_layers):
            expected_k, expected_v = oracle.read(layer, "serial", depth, len(tokens))
            torch.testing.assert_close(
                e.cache_manager.key_cache[blocks, layer, offsets], expected_k, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                e.cache_manager.value_cache[blocks, layer, offsets],
                expected_v,
                atol=atol,
                rtol=rtol,
            )
    e.close()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.states
    assert all(not bank.leased for bank in e.speculative_runner.banks)
    e.close()
