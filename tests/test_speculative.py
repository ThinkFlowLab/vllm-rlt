"""Fixed-depth oracle, rollback, lifecycle and distribution tests for speculation."""

from collections import Counter

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
from vllm_rlt.worker.speculative import greedy_accept


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


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8])
def test_greedy_accept_commits_target_prefix_through_first_mismatch(k):
    candidates = list(range(100, 100 + k))
    for rejected in range(k + 1):
        # Rows before `rejected` agree; row `rejected` is the correction, or the bonus.
        targets = candidates[:rejected] + [7] + list(range(200, 200 + k - rejected))
        tokens, accepted = greedy_accept(candidates, targets)
        assert accepted == rejected
        assert tokens == candidates[:rejected] + [7]


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
        dict(execution_config=ExecutionConfig(cuda_graphs=True)),
        dict(scheduler_config=SchedulerConfig(enable_preemption=True)),
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
