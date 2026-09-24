"""Fixed-depth oracle, rollback, lifecycle and distribution tests for speculation."""

import pickle
from collections import Counter
from contextlib import nullcontext

import pytest
import torch

from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.models.reference import dense_reference
from vllm_rlt.request import Request, Stage
from vllm_rlt.worker.sampling import probabilities, rejection_sample


def model(seed=123, dtype=torch.float32):
    torch.manual_seed(seed)
    return OuroForCausalLM(OuroConfig.tiny()).to(dtype=dtype)


def engine(m, k=3, **kwargs):
    return LLMEngine(m, speculative_config=SpeculativeConfig(k), **kwargs)


def drain(e):
    final = {}
    for _ in range(2000):
        if not e.has_unfinished_requests():
            return final
        for output in e.step():
            if output.finished:
                final[output.request_id] = output
    pytest.fail("speculative scheduler did not drain")


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("budget", [1, 3, 32])
def test_greedy_matches_native_with_ragged_rounds_and_limits(k, budget):
    m = model()
    prompts = [[2], [3, 4, 5, 6, 7], [9, 3]]
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in [1, 13, 8]]
    expected = LLM(m).generate(prompts, params)
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(k),
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=budget, prefill_chunk_size=2),
    )
    actual = llm.generate(prompts, params)
    assert [o.token_ids for o in actual] == [o.token_ids for o in expected]
    assert [o.exit_depths for o in actual] == [o.exit_depths for o in expected]
    assert llm.engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch"),
        pytest.param("cuda", "triton", marks=pytest.mark.gpu),
        pytest.param("cuda", "flash_attn_4", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reused_hidden_logits_and_all_kv_match_serial_oracle(dtype, device, backend):
    if device == "cuda" and dtype == torch.float32 and backend == "flash_attn_4":
        pytest.skip("FA4 requires 16-bit operands")
    m = model(dtype=dtype).to(device)
    if backend == "flash_attn_4":
        torch.manual_seed(123)
        m = OuroForCausalLM(OuroConfig.tiny(hidden_size=256, head_dim=64)).to(
            device=device, dtype=dtype
        )
    e = engine(m, k=3, cache_config=CacheConfig(128, 2), attention_backend=backend)
    cache = e.cache_manager
    prefix = [3, 4, 5]
    current = 7
    assert cache.allocate("r", 16)
    # Native token-by-token full-depth prefix.
    for pos, token in enumerate(prefix):
        h = m.prelude(torch.tensor([token], device=device))
        for d in range(4):
            h, _ = m.recurrent(h, ["r"], [d], [pos], cache)
    r = Request("r", prefix, SamplingParams(ignore_eos=True), generated_token_ids=[current])
    runner = e.speculative_runner
    core = runner._core
    seen = {}
    work = Counter()

    def record(hidden, ids, positions, depth, **kwargs):
        work[depth] += len(ids)
        out = core(hidden, ids, positions, depth, **kwargs)
        for row, pos in enumerate(positions):
            seen[depth, pos] = out[row].clone()
        return out

    runner._core = record
    head_outputs = []
    hook = m.lm_head.register_forward_hook(
        lambda module, inputs, output: head_outputs.append(output.detach().clone())
    )
    try:
        runner.execute(SchedulerOutput(Stage.SPECULATIVE, [ScheduledItem(r, 3, 4)]))
    finally:
        hook.remove()
    # Include the actual batched LM-head GEMM in the numeric comparison.
    verified_logits = head_outputs[-1]
    # The shallow head outputs determine each subsequent input in this round.
    tokens = prefix + [current]
    for pos in range(3, 6):
        tokens.append(int(m.coda(seen[1, pos]).argmax()))
    oracle = LLMEngine(m, cache_config=CacheConfig(128, 2), attention_backend=backend).cache_manager
    oracle.allocate("r", 16)
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
    max_logit_error = 0.0
    for pos, token in enumerate(tokens):
        h = m.prelude(torch.tensor([token], device=device))
        for d in range(4):
            h, _ = m.recurrent(h, ["r"], [d], [pos], oracle)
            if pos >= 3:
                torch.testing.assert_close(seen[d, pos], h[0], atol=atol, rtol=rtol)
        if pos >= 3:
            actual, expected = verified_logits[pos - 3], m.coda(h)[0]
            max_logit_error = max(max_logit_error, float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            assert actual.argmax() == expected.argmax()
    for d in range(4):
        for layer in range(m.config.num_hidden_layers):
            for actual, expected in zip(
                cache.read(layer, "r", d, 7), oracle.read(layer, "r", d, 7)
            ):
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    # Every input/depth computed exactly once: no shallow recomputation in verify.
    assert work == Counter({0: 4, 1: 4, 2: 4, 3: 4})
    if dtype == torch.float32:
        independent = dense_reference(m, tokens, 4)
        for d in range(4):
            torch.testing.assert_close(
                torch.stack([seen[d, p] for p in range(3, 7)]),
                independent[d][0][3:],
                atol=4e-5,
                rtol=4e-5,
            )
    assert max_logit_error < (1e-4 if dtype == torch.float32 else 0.025)


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_each_rejection_position_and_bonus_commit_correct_frontier(rejected, monkeypatch):
    m = model()
    e = engine(m, k=3)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    request = e.scheduler.requests["r"]
    start = request.position
    # Force draft IDs 10,11,12 and target accept up to the selected boundary.
    coda = m.coda
    calls = 0

    def controlled(h):
        nonlocal calls
        logits = torch.full((len(h), m.config.vocab_size), -100.0)
        if calls < 3:
            logits[:, 10 + calls] = 100
        else:
            for i in range(4):
                logits[i, 10 + i if i < rejected else 20 + i] = 100
        calls += 1
        return logits

    monkeypatch.setattr(m, "coda", controlled)
    out = e.step()[0]
    n = rejected + 1
    assert out.token_ids[-n:] == list(range(10, 10 + rejected)) + [20 + rejected]
    for plane in e.cache_manager._get_allocation("r").written:
        assert all(w.prefix == start + n and not w.pending for w in plane)
    monkeypatch.setattr(m, "coda", coda)
    # The next round must overwrite stale rejected slots and agree with a fresh
    # native engine on the committed prefix, including when the page is partial.
    expected = LLM(m).generate(
        [out.prompt_token_ids + out.token_ids],
        SamplingParams(max_tokens=12 - len(out.token_ids), ignore_eos=True),
    )[0]
    final = drain(e)["r"]
    assert final.token_ids[len(out.token_ids) :] == expected.token_ids


def test_eos_in_accepted_prefix_never_delivers_following_tokens(monkeypatch):
    m = model()
    e = engine(m)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=12))
    while not e.step():
        pass
    # All later proposals/target rows choose EOS. Only first EOS is committed.
    monkeypatch.setattr(m, "coda", lambda h: torch.zeros(len(h), m.config.vocab_size))
    out = e.step()[0]
    assert out.finished and out.finish_reason == "stop"
    assert len(out.token_ids) == 2 and out.token_ids[-1] == 0
    assert e.speculative_runner.stats.committed_tokens == 1
    assert e.cache_manager.num_used_blocks == 0


def test_truncation_removes_sparse_validity_without_destroying_prefix():
    e = engine(model(), cache_config=CacheConfig(64, 2))
    cache = e.cache_manager
    cache.allocate("r", 8)
    k = torch.randn(4, 2, 8)
    # Initialize [0,1,2] and a sparse stale position 6 at every layer/depth.
    for d in range(4):
        for layer in range(2):
            cache.write(layer, ["r"] * 4, [d] * 4, [0, 1, 2, 6], k, k)
    before = cache.read(0, "r", 0, 2)[0].clone()
    cache.truncate_suffix("r", 2)
    torch.testing.assert_close(cache.read(0, "r", 0, 2)[0], before, rtol=0, atol=0)
    for plane in cache._get_allocation("r").written:
        assert all(w.prefix == 2 and not w.pending for w in plane)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(0, "r", 0, 3)


@pytest.mark.parametrize("prefix_cache,incremental", [(False, True), (True, False), (True, True)])
def test_prefix_replay_incremental_growth_cancel_and_reuse(prefix_cache, incremental):
    m = model()
    kwargs = dict(
        cache_config=CacheConfig(
            128, 2, enable_prefix_caching=prefix_cache, incremental_allocation=incremental
        )
    )
    llm = LLM(m, speculative_config=SpeculativeConfig(4), **kwargs)
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    prompts = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 7]]
    expected = LLM(m).generate(prompts, params)
    for _ in range(2):
        assert [o.token_ids for o in llm.generate(prompts, params)] == [
            o.token_ids for o in expected
        ]
    e = llm.engine
    e.add_request("reuse", prompts[0], params)
    while not e.step():
        pass
    e.step()
    assert e.abort_request("reuse").finish_reason == "abort"
    e.add_request("reuse", prompts[1], params)
    assert drain(e)["reuse"].token_ids == expected[1].token_ids
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("top_k,top_p", [(-1, 1.0), (5, 1.0), (7, 0.8)])
def test_sampling_seeded_replay_and_request_isolation(top_k, top_p):
    m = model()
    prompts = [[2, 3], [7, 8, 9]]
    params = SamplingParams(
        max_tokens=10, temperature=0.8, top_k=top_k, top_p=top_p, ignore_eos=True, seed=42
    )

    def run(prompts):
        return LLM(m, speculative_config=SpeculativeConfig(3)).generate(prompts, params)

    batched = run(prompts)
    assert [o.token_ids for o in run(prompts)] == [o.token_ids for o in batched]
    assert [run([p])[0].token_ids for p in prompts] == [o.token_ids for o in batched]


@pytest.mark.parametrize(
    "p,q",
    [
        ([0.1, 0.6, 0.3], [0.7, 0.2, 0.1]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
    ],
)
def test_rejection_sampling_recovers_target_distribution(p, q):
    p, q = torch.tensor(p), torch.tensor(q)
    g = torch.Generator().manual_seed(431)
    counts = torch.zeros(3)
    for _ in range(12000):
        candidate = int(torch.multinomial(q, 1, generator=g))
        token, _ = rejection_sample(candidate, p, q, g)
        counts[token] += 1
    torch.testing.assert_close(counts / counts.sum(), p, atol=0.016, rtol=0)


def test_probability_filters_match_expected_and_keep_boundary_ties():
    logits = torch.tensor([0.50, 0.25, 0.15, 0.10]).log()
    p = probabilities(logits, SamplingParams(temperature=1, top_p=0.8))
    torch.testing.assert_close(p, torch.tensor([0.50, 0.25, 0.15, 0]) / 0.9)
    p = probabilities(logits, SamplingParams(temperature=0.5, top_k=2))
    torch.testing.assert_close(p, torch.tensor([0.8, 0.2, 0, 0]))
    p = probabilities(torch.zeros(4), SamplingParams(temperature=1, top_k=1))
    torch.testing.assert_close(p, torch.full((4,), 0.25))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cache_config=CacheConfig(layout="shared")),
        dict(execution_config=ExecutionConfig(async_scheduling=True)),
        dict(scheduler_config=SchedulerConfig(mode="no_refill")),
    ],
)
def test_unsupported_modes_fail_before_allocating(kwargs):
    with pytest.raises(ValueError, match="speculative"):
        engine(model(), **kwargs)


def test_rejects_adaptive_or_wrong_target_requests():
    e = engine(model())
    for params in [SamplingParams(exit_threshold=0.5), SamplingParams(max_loops=3)]:
        with pytest.raises(ValueError, match="fixed target depth"):
            e.add_request("bad", [2], params)
    assert not e.has_unfinished_requests()


def test_dynamic_arrival_budget_and_failure_reclamation(monkeypatch):
    e = engine(model(), scheduler_config=SchedulerConfig(max_num_batched_tokens=5))
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    e.add_request("a", [2, 3], params)
    while not e.step():
        pass
    e.add_request("b", [4, 5, 6], params)
    for _ in range(4):
        e.step()
        assert e.last_schedule.num_tokens <= 5

    def fail(*args, **kwargs):
        raise RuntimeError("injected verify failure")

    monkeypatch.setattr(e.speculative_runner, "execute", fail)
    with pytest.raises(RuntimeError, match="injected"):
        while e.has_unfinished_requests():
            e.step()
    for rid in list(e.scheduler.requests):
        e.abort_request(rid)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "device,backend,graphs",
    [
        ("cpu", "torch", False),
        pytest.param("cuda", "triton", False, marks=pytest.mark.gpu),
        pytest.param("cuda", "triton", True, marks=pytest.mark.gpu),
    ],
)
def test_priority_preempts_only_between_speculative_rounds_and_resumes(device, backend, graphs):
    m = (
        model()
        if device == "cpu"
        else OuroForCausalLM(OuroConfig.tiny(hidden_size=256, head_dim=64)).to(
            "cuda", torch.bfloat16
        )
    )
    prompts = {"low": [2, 3, 4], "high": [7, 8, 9]}
    params = {
        "low": SamplingParams(max_tokens=10, ignore_eos=True, priority=10),
        "high": SamplingParams(max_tokens=7, ignore_eos=True, priority=0),
    }
    expected = {
        rid: LLM(m, attention_backend=backend).generate([prompt], params[rid])[0]
        for rid, prompt in prompts.items()
    }
    e = engine(
        m,
        k=3,
        cache_config=CacheConfig(64, 16 if graphs else 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            policy="priority",
            enable_preemption=True,
        ),
        attention_backend=backend,
        execution_config=ExecutionConfig(cuda_graphs=graphs, cuda_graph_max_batch_size=8),
    )
    e.add_request("low", prompts["low"], params["low"])
    while len(e.scheduler.requests["low"].generated_token_ids) < 2:
        e.step()
    e.add_request("high", prompts["high"], params["high"])
    actual = drain(e)
    assert e.preemption.preemptions > 0
    assert e.preemption.resumptions > 0
    assert {rid: out.token_ids for rid, out in actual.items()} == {
        rid: out.token_ids for rid, out in expected.items()
    }
    assert {rid: out.exit_depths for rid, out in actual.items()} == {
        rid: out.exit_depths for rid, out in expected.items()
    }
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "torch"), pytest.param("cuda", "triton", marks=pytest.mark.gpu)],
)
def test_prefill_interleaves_between_draft_and_verify_without_changing_outputs(device, backend):
    m = (
        model()
        if device == "cpu"
        else OuroForCausalLM(OuroConfig.tiny(hidden_size=256, head_dim=64)).to(
            "cuda", torch.bfloat16
        )
    )
    prompts = {"first": [2, 3], "second": [7, 8, 9]}
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    expected = {
        rid: LLM(m, attention_backend=backend).generate([prompt], params)[0].token_ids
        for rid, prompt in prompts.items()
    }
    e = LLMEngine(
        m,
        speculative_config=SpeculativeConfig(3, interleave_round=True),
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=2
        ),
        attention_backend=backend,
    )
    e.add_request("first", prompts["first"], params)
    while not e.scheduler.requests["first"].generated_token_ids:
        e.step()
    assert e.step() == []
    assert e.last_schedule.stage == Stage.SPECULATIVE
    assert "first" in e.preemption.inflight_ids
    e.add_request("second", prompts["second"], params)
    assert e.step() == []
    assert e.last_schedule.stage == Stage.PREFILL
    assert e.step()[0].request_id == "first"
    assert e.last_schedule.stage == Stage.SPECULATIVE
    assert not e.preemption.inflight_ids
    actual = drain(e)
    assert {rid: out.token_ids for rid, out in actual.items()} == expected
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_cuda_graph_replay_with_intra_round_prefill():
    torch.manual_seed(123)
    m = OuroForCausalLM(OuroConfig.tiny(hidden_size=256, head_dim=64)).to("cuda", torch.bfloat16)
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    expected = LLM(m, speculative_config=SpeculativeConfig(3), attention_backend="triton").generate(
        [[2, 3], [7, 8, 9]], params
    )
    e = LLMEngine(
        m,
        speculative_config=SpeculativeConfig(3, interleave_round=True),
        execution_config=ExecutionConfig(cuda_graphs=True, cuda_graph_max_batch_size=8),
        cache_config=CacheConfig(128, 16),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        attention_backend="triton",
    )
    e.add_request("first", [2, 3], params)
    while not e.scheduler.requests["first"].generated_token_ids:
        e.step()
    assert e.step() == []
    e.add_request("second", [7, 8, 9], params)
    e.step()
    assert e.last_schedule.stage == Stage.PREFILL
    e.step()
    assert e.last_schedule.stage == Stage.SPECULATIVE
    actual = drain(e)
    assert [actual[rid].token_ids for rid in ("first", "second")] == [
        output.token_ids for output in expected
    ]
    assert [actual[rid].exit_depths for rid in ("first", "second")] == [
        output.exit_depths for output in expected
    ]
    assert e.speculative_runner.graphs.replays > 0
    assert e.cache_manager.num_used_blocks == 0


def test_priority_waits_for_draft_verification_before_preempting():
    m = model()
    params = {
        "low": SamplingParams(max_tokens=8, ignore_eos=True, priority=10),
        "high": SamplingParams(max_tokens=5, ignore_eos=True, priority=0),
    }
    e = LLMEngine(
        m,
        speculative_config=SpeculativeConfig(3, interleave_round=True),
        cache_config=CacheConfig(64, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            policy="priority",
            enable_preemption=True,
        ),
    )
    e.add_request("low", [2, 3, 4], params["low"])
    while not e.scheduler.requests["low"].generated_token_ids:
        e.step()
    e.step()  # Draft is complete; verification still owns provisional KV.
    e.add_request("high", [7, 8, 9], params["high"])
    assert e.step()[0].request_id == "low"
    assert e.preemption.preemptions == 0
    actual = drain(e)
    assert e.preemption.preemptions > 0
    assert e.preemption.resumptions > 0
    for rid, prompt in (("low", [2, 3, 4]), ("high", [7, 8, 9])):
        expected = LLM(m).generate([prompt], params[rid])[0]
        assert actual[rid].token_ids == expected.token_ids
        assert actual[rid].exit_depths == expected.exit_depths
    assert e.cache_manager.num_used_blocks == 0


def test_abort_during_draft_does_not_reuse_provisional_kv():
    m = model()
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    e = LLMEngine(
        m,
        speculative_config=SpeculativeConfig(3, interleave_round=True),
        cache_config=CacheConfig(64, 2),
    )
    e.add_request("r", [2, 3], params)
    while not e.scheduler.requests["r"].generated_token_ids:
        e.step()
    assert e.step() == []
    e.abort_request("r")
    e.add_request("r", [7, 8, 9], params)
    actual = drain(e)["r"]
    expected = LLM(m).generate([[7, 8, 9]], params)[0]
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("temperature", [0, 0.8])
@pytest.mark.parametrize(
    "device,backend,graphs",
    [
        ("cpu", "torch", False),
        pytest.param("cuda", "triton", False, marks=pytest.mark.gpu),
        pytest.param("cuda", "triton", True, marks=pytest.mark.gpu),
    ],
)
def test_committed_round_migrates_kv_and_sampling_state(temperature, device, backend, graphs):
    if device == "cuda":
        if torch.cuda.device_count() < 2:
            pytest.skip("migration requires two visible GPUs")
        config = OuroConfig.tiny(hidden_size=256, head_dim=64)
        torch.manual_seed(123)
        m = OuroForCausalLM(config).to("cuda:0", torch.bfloat16)
        target_model = OuroForCausalLM(config).to("cuda:1", torch.bfloat16)
        target_model.load_state_dict(m.state_dict())
    else:
        m = target_model = model()
    cache = CacheConfig(128, 16) if device == "cuda" else CacheConfig(64, 2)
    spec = SpeculativeConfig(3, interleave_round=True)
    execution = ExecutionConfig(cuda_graphs=graphs, cuda_graph_max_batch_size=8)
    params = SamplingParams(
        max_tokens=12,
        ignore_eos=True,
        temperature=temperature,
        top_k=7,
        seed=42,
    )
    expected = LLM(
        m, speculative_config=spec, attention_backend=backend, execution_config=execution
    ).generate([[2, 3, 4]], params)[0]
    source = LLMEngine(
        m,
        speculative_config=spec,
        cache_config=cache,
        attention_backend=backend,
        execution_config=execution,
    )
    target = LLMEngine(
        target_model,
        speculative_config=spec,
        cache_config=cache,
        attention_backend=backend,
        execution_config=execution,
    )
    source.add_request("r", [2, 3, 4], params)
    while not source.scheduler.requests["r"].generated_token_ids:
        source.step()
    assert source.step() == []  # Drafted KV is still provisional.
    with pytest.raises(RuntimeError, match="committed speculative round boundary"):
        source.export_request("r")
    source.step()
    packet = pickle.loads(pickle.dumps(source.export_request("r")))
    assert packet.state["keys"].device.type == "cpu"
    assert not source.has_unfinished_requests()
    assert source.cache_manager.num_used_blocks == 0
    with torch.cuda.device(1) if device == "cuda" else nullcontext():
        assert target.import_request(packet)
        actual = drain(target)["r"]
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert target.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        SpeculativeConfig(k)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
def test_gpu_sampling_and_greedy_match_replay_with_ragged_requests(backend):
    torch.manual_seed(123)
    m = OuroForCausalLM(OuroConfig.tiny(hidden_size=256, head_dim=64)).to(
        device="cuda", dtype=torch.bfloat16
    )
    prompts = [[2, 3, 4], [7, 8], [9]]
    params = [
        SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.8, top_k=7, top_p=0.8, seed=42)
        for n in [8, 5, 3]
    ]

    def run(speculative, sampling):
        return LLM(
            m,
            speculative_config=SpeculativeConfig(3) if speculative else None,
            cache_config=CacheConfig(128, 16),
            attention_backend=backend,
        ).generate(prompts, sampling)

    a, b = run(True, params), run(True, params)
    assert [o.token_ids for o in a] == [o.token_ids for o in b]
    greedy = [SamplingParams(max_tokens=n, ignore_eos=True) for n in [8, 5, 3]]
    a, b = run(True, greedy), run(False, greedy)
    assert [o.token_ids for o in a] == [o.token_ids for o in b]


@pytest.mark.gpu
@pytest.mark.parametrize("max_graph_rows", [2, 8])
def test_speculative_cuda_graph_matches_eager_with_ragged_replay(max_graph_rows):
    m = model(dtype=torch.bfloat16).to("cuda")
    prompts = [[2, 3, 4], [7, 8]]
    params = SamplingParams(max_tokens=9, ignore_eos=True)

    def run(graphs):
        llm = LLM(
            m,
            speculative_config=SpeculativeConfig(3),
            cache_config=CacheConfig(128, 16),
            attention_backend="triton",
            execution_config=ExecutionConfig(
                cuda_graphs=graphs, cuda_graph_max_batch_size=max_graph_rows
            ),
        )
        outputs = [llm.generate(prompts, params) for _ in range(2)]
        assert llm.engine.cache_manager.num_used_blocks == 0
        return llm.engine.speculative_runner, [
            [(out.token_ids, out.exit_depths) for out in batch] for batch in outputs
        ]

    _, eager = run(False)
    runner, graphed = run(True)
    assert graphed == eager
    assert runner.graphs.captures > 0
    assert runner.graphs.replays > runner.graphs.captures
    assert runner.coda_graphs.captures > 0
    if max_graph_rows == 2:
        assert runner.graphs.fallbacks > 0
    else:
        assert any(rows > 2 for rows, _, _, _ in runner.graphs.entries)
