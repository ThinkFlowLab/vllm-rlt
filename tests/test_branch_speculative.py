"""Optional single-fallback branch speculation: equality, lifecycle and cleanup."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.reference import dense_reference
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
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, Stage
from vllm_rlt.worker.branch_speculative import BranchSpeculativeRunner
from vllm_rlt.worker.speculative import SpeculativeRunner


def model(seed=123, dtype=torch.float32):
    torch.manual_seed(seed)
    return OuroForCausalLM(tiny_ouro_config()).to(dtype=dtype)


def branch_engine(m, k=3, margin=1.0, draft_loops=2, **kwargs):
    return LLMEngine(
        m,
        speculative_config=SpeculativeConfig(
            k, draft_loops=draft_loops, target_loops=4, alternate_prob_gap_threshold=margin
        ),
        **kwargs,
    )


def drain(e):
    final = {}
    for _ in range(2000):
        if not e.has_unfinished_requests():
            return final
        for output in e.step():
            if output.finished:
                final[output.request_id] = output
    pytest.fail("branch speculative scheduler did not drain")


def install_controlled_coda(monkeypatch, m, shallow, deep):
    """Feed fixed logits per coda call: 3 shallow calls, then one deep call."""
    calls = {"count": 0}

    def coda(hidden):
        index = calls["count"]
        calls["count"] += 1
        logits = torch.full((len(hidden), m.config.vocab_size), -100.0)
        tokens = shallow[index] if index < len(shallow) else deep
        for row, token in enumerate(tokens):
            logits[row, token] = 100.0
            if index < len(shallow):
                logits[row, (token + 1) % m.config.vocab_size] = 39.0
        return logits

    monkeypatch.setattr(m, "coda", coda)
    return calls


# --------------------------------------------------------------------------- #
# configuration and runner selection
# --------------------------------------------------------------------------- #
def test_alternate_prob_gap_threshold_none_keeps_plain_runner():
    m = model()
    plain = LLMEngine(m, speculative_config=SpeculativeConfig(3))
    assert type(plain.speculative_runner) is SpeculativeRunner
    enabled = branch_engine(m)
    assert type(enabled.speculative_runner) is BranchSpeculativeRunner


@pytest.mark.parametrize("margin", [None, 0.0, 0.5, 1.0])
def test_valid_alternate_prob_gap_threshold(margin):
    config = SpeculativeConfig(3, alternate_prob_gap_threshold=margin)
    assert config.alternate_prob_gap_threshold == margin


@pytest.mark.parametrize("margin", [-0.1, 1.5, float("nan"), float("inf"), True, "0.2"])
def test_invalid_alternate_prob_gap_threshold(margin):
    with pytest.raises(ValueError):
        SpeculativeConfig(3, alternate_prob_gap_threshold=margin)


def test_enabled_fallback_rejects_sampling_before_enqueue():
    e = branch_engine(model())
    with pytest.raises(ValueError, match="greedy"):
        e.add_request("r", [2, 3], SamplingParams(max_tokens=8, temperature=0.8))
    assert not e.has_unfinished_requests()
    e.add_request("greedy", [2, 3], SamplingParams(max_tokens=8, temperature=0.0))
    assert e.has_unfinished_requests()


# --------------------------------------------------------------------------- #
# equality with native and the plain runner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("draft_loops", [1, 2])
def test_enabled_fallback_matches_native_and_plain(draft_loops):
    torch.manual_seed(0)
    m = OuroForCausalLM(tiny_ouro_config(vocab_size=8))
    prompts = [[2], [3, 4, 5, 6, 7], [7, 3]]
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in [2, 32, 24]]
    common = dict(
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=32),
    )
    native = LLM(m).generate(prompts, params)
    plain = LLM(
        m,
        speculative_config=SpeculativeConfig(3, draft_loops=draft_loops, target_loops=4),
        **common,
    ).generate(prompts, params)
    branch_llm = LLM(
        m,
        speculative_config=SpeculativeConfig(
            3, draft_loops=draft_loops, target_loops=4, alternate_prob_gap_threshold=1.0
        ),
        **common,
    )
    branch = branch_llm.generate(prompts, params)
    assert branch_llm.engine.speculative_runner.stats.alternate_selected > 0
    assert branch_llm.engine.cache_manager.num_used_blocks == 0
    assert [o.token_ids for o in native] == [o.token_ids for o in plain]
    assert [o.token_ids for o in native] == [o.token_ids for o in branch]
    assert [o.exit_depths for o in native] == [o.exit_depths for o in branch]


def test_enabled_fallback_packs_multiple_requests(monkeypatch):
    m = model()
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(3, alternate_prob_gap_threshold=1.0),
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=32, max_num_seqs=4),
    )
    runner = llm.engine.speculative_runner
    seen = []
    original = runner._core

    def record(hidden, ids, positions, depth, **kwargs):
        seen.append(tuple(ids))
        return original(hidden, ids, positions, depth, **kwargs)

    monkeypatch.setattr(runner, "_core", record)
    prompts = [[2], [3, 4], [9, 3, 4]]
    llm.generate(prompts, [SamplingParams(max_tokens=6, ignore_eos=True)] * 3)
    packed = [ids for ids in seen if len({rid.split("::alt")[0] for rid in ids}) > 1]
    assert packed
    assert runner.stats.fork_rounds > 0


def test_forced_later_fork_matches_native(monkeypatch):
    m = model()
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(3, alternate_prob_gap_threshold=1.0),
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=32),
    )
    runner = llm.engine.speculative_runner
    original = runner._gate_decision

    def later(logits, offset):
        trigger, second = original(logits, offset)
        return offset == 1, second

    monkeypatch.setattr(runner, "_gate_decision", later)
    prompts = [[2, 3], [4, 5, 6]]
    params = [SamplingParams(max_tokens=8, ignore_eos=True)] * 2
    expected = LLM(m).generate(prompts, params)
    actual = llm.generate(prompts, params)
    assert [o.token_ids for o in actual] == [o.token_ids for o in expected]
    assert runner.stats.fork_rounds > 0


# --------------------------------------------------------------------------- #
# deterministic branch outcomes through controlled logits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "suffix_target, expected, rollback",
    [(15, [11, 13, 15, 16], 0), (42, [11, 13, 42], 1)],
)
def test_branch_hit_commits_alternate(monkeypatch, suffix_target, expected, rollback):
    m = model()
    e = branch_engine(m)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    install_controlled_coda(
        monkeypatch,
        m,
        shallow=[[10], [12, 13], [14, 15]],
        deep=[11, 39, 38, 37, 13, suffix_target, 16],
    )
    out = e.step()[0]
    stats = e.speculative_runner.stats
    assert out.token_ids[-len(expected) :] == expected
    assert stats.alternate_selected == 1
    assert stats.primary_rejected_at_fork == 1
    assert stats.rollback_rounds == rollback
    assert stats.alt_drafted == 3
    e.abort_request("r")
    assert e.cache_manager.num_used_blocks == 0


def test_branch_discarded_when_primary_accepted(monkeypatch):
    m = model()
    e = branch_engine(m)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    install_controlled_coda(
        monkeypatch,
        m,
        shallow=[[10], [12, 13], [14, 15]],
        deep=[10, 12, 14, 16, 0, 0, 0],
    )
    out = e.step()[0]
    stats = e.speculative_runner.stats
    assert out.token_ids[-4:] == [10, 12, 14, 16]
    assert stats.fork_rounds == 1
    assert stats.alternate_selected == 0
    assert stats.rollback_rounds == 0
    e.abort_request("r")
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("first_target, expected", [(10, [10, 39]), (20, [20])])
def test_branch_rejected_elsewhere_rolls_back(monkeypatch, first_target, expected):
    m = model()
    e = branch_engine(m)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    install_controlled_coda(
        monkeypatch,
        m,
        shallow=[[10], [12, 13], [14, 15]],
        deep=[first_target, 39, 38, 37, 0, 0, 0],
    )
    out = e.step()[0]
    stats = e.speculative_runner.stats
    assert out.token_ids[-len(expected) :] == expected
    assert stats.alternate_selected == 0
    assert stats.primary_rejected_elsewhere == 1
    assert stats.rollback_rounds == 1
    e.abort_request("r")
    assert e.cache_manager.num_used_blocks == 0


# --------------------------------------------------------------------------- #
# lifecycle, capacity and collision
# --------------------------------------------------------------------------- #
def test_cache_lifecycle_balanced_after_fallback_run():
    m = model()
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(3, alternate_prob_gap_threshold=1.0),
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=32),
    )
    prompts = [[2], [3, 4, 5], [9, 3, 4, 5]]
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in [8, 6, 4]]
    llm.generate(prompts, params)
    cache = llm.engine.cache_manager
    assert cache.num_used_blocks == 0
    assert not cache._allocations
    assert all(ref == 0 for ref in cache._refs)


def test_injected_copy_failure_frees_branch(monkeypatch):
    m = model()
    e = branch_engine(
        m,
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=2),
    )
    runner = e.speculative_runner

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected stem-copy failure")

    monkeypatch.setattr(runner, "_copy_shallow_stem", fail)
    e.add_request("a", [1, 2, 3], SamplingParams(max_tokens=8, ignore_eos=True))
    e.add_request("b", [4, 5, 6], SamplingParams(max_tokens=8, ignore_eos=True))
    with pytest.raises(RuntimeError, match="injected"):
        while e.has_unfinished_requests():
            e.step()
    for rid in list(e.scheduler.requests):
        e.abort_request(rid)
    cache = e.cache_manager
    assert not [rid for rid in cache._allocations if "::alt" in rid]
    assert cache.num_used_blocks == 0


def test_colliding_public_request_id_keeps_outputs():
    m = model()
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(3, alternate_prob_gap_threshold=1.0),
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=32, max_num_seqs=2),
    )
    prompts = [[2, 3], [4, 5, 6]]
    params = [SamplingParams(max_tokens=8, ignore_eos=True)] * 2
    expected = LLM(m).generate(prompts, params)
    e = llm.engine
    e.add_request("r", prompts[0], params[0])
    e.add_request("r::alt", prompts[1], params[1])
    final = drain(e)
    assert final["r"].token_ids == expected[0].token_ids
    assert final["r::alt"].token_ids == expected[1].token_ids
    assert e.cache_manager.num_used_blocks == 0


def test_insufficient_spare_capacity_uses_primary():
    m = model()
    # One admitted request needs all four physical blocks, so a fork cannot be
    # opened without evicting another request's reservation.
    e = branch_engine(
        m,
        cache_config=CacheConfig(num_blocks=4, block_size=16),
        scheduler_config=SchedulerConfig(max_num_seqs=1),
    )
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=8, ignore_eos=True))
    final = drain(e)["r"]
    expected = LLM(m).generate([[2, 3, 4]], [SamplingParams(max_tokens=8, ignore_eos=True)])[0]
    assert final.token_ids == expected.token_ids
    assert e.speculative_runner.stats.fork_rounds == 0
    assert e.cache_manager.num_used_blocks == 0


# --------------------------------------------------------------------------- #
# ordinary EOS stop
# --------------------------------------------------------------------------- #
def test_enabled_fallback_eos_stop(monkeypatch):
    m = model()
    e = branch_engine(m)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=12))
    while not e.step():
        pass
    monkeypatch.setattr(m, "coda", lambda h: torch.zeros(len(h), m.config.vocab_size))
    out = e.step()[0]
    assert out.finished and out.finish_reason == "stop"
    assert out.token_ids[-1] == 0
    assert e.speculative_runner.stats.committed_tokens == 1
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "execution",
    [ExecutionConfig(cuda_graphs=True), ExecutionConfig(async_scheduling=True)],
)
def test_fallback_rejects_graphs_and_async(execution):
    with pytest.raises(ValueError, match="fallback branch.*eager"):
        branch_engine(model(), execution_config=execution)


# --------------------------------------------------------------------------- #
# fixed-input oracle for the primary and alternate chains
# --------------------------------------------------------------------------- #
def serial_replay(m, backend, tokens, target_loops):
    """Replay one token chain into a fresh cache one row/depth at a time."""
    device = next(m.parameters()).device
    oracle = LLMEngine(m, cache_config=CacheConfig(128, 2), attention_backend=backend).cache_manager
    assert oracle.allocate("oracle", len(tokens))
    hiddens, logits = [], []
    for pos, token in enumerate(tokens):
        h = m.prelude(torch.tensor([token], device=device))
        for depth in range(target_loops):
            h, _ = m.recurrent(h, ["oracle"], [depth], [pos], oracle)
        hiddens.append(h[0].clone())
        logits.append(m.coda(h)[0].clone())
    return oracle, hiddens, logits


def fork_round(m, monkeypatch, *, backend, draft_loops, fork, prefix_len, block_size=2):
    """Run one branch execute, forcing the fork at ``fork`` and capturing live state."""
    device = next(m.parameters()).device
    target_loops = 4
    engine = LLMEngine(
        m,
        speculative_config=SpeculativeConfig(
            3, draft_loops=draft_loops, target_loops=target_loops, alternate_prob_gap_threshold=1.0
        ),
        cache_config=CacheConfig(256, block_size),
        attention_backend=backend,
    )
    cache, runner = engine.cache_manager, engine.speculative_runner
    prompt = [10 + index for index in range(prefix_len)]
    current = 7
    assert cache.allocate("r", prefix_len + 8)
    # Native token-by-token full-depth prefix writes every layer and depth.
    for pos, token in enumerate(prompt):
        h = m.prelude(torch.tensor([token], device=device))
        for depth in range(target_loops):
            h, _ = m.recurrent(h, ["r"], [depth], [pos], cache)

    original_gate = runner._gate_decision

    def gate(logits, offset):
        trigger, second = original_gate(logits, offset)
        return offset == fork, second

    monkeypatch.setattr(runner, "_gate_decision", gate)
    request = Request("r", prompt, SamplingParams(ignore_eos=True), generated_token_ids=[current])
    token_count = 4
    captured = {}
    original_verify = runner._verify_plan

    def verify(plan, primary_rows, alt_rows):
        assert plan.fork == fork
        assert plan.alt_id is not None and plan.alt_id in cache._allocations
        end = plan.p0 + plan.token_count
        captured.update(
            fork=plan.fork,
            p0=plan.p0,
            token_count=plan.token_count,
            primary_cand=list(plan.primary_cand),
            alt_cand=dict(plan.alt_cand),
            primary_rows=primary_rows.detach().clone(),
            alt_rows=alt_rows.detach().clone(),
            primary_kv={
                (layer, depth): tuple(
                    tensor.detach().clone() for tensor in cache.read(layer, plan.rid, depth, end)
                )
                for layer in range(cache.num_layers)
                for depth in range(target_loops)
            },
            alt_kv={
                (layer, depth): tuple(
                    tensor.detach().clone() for tensor in cache.read(layer, plan.alt_id, depth, end)
                )
                for layer in range(cache.num_layers)
                for depth in range(target_loops)
            },
        )
        return original_verify(plan, primary_rows, alt_rows)

    monkeypatch.setattr(runner, "_verify_plan", verify)
    runner.execute(
        SchedulerOutput(Stage.SPECULATIVE, [ScheduledItem(request, prefix_len, token_count)])
    )
    captured["stats"] = runner.stats
    return captured, prompt, current


def assert_fork_chains(
    m, backend, captured, prompt, current, *, fork, dtype, token_count=4, target_loops=4
):
    p0 = captured["p0"]
    k_drafts = token_count - 1
    primary_tokens = prompt + [current] + captured["primary_cand"]
    alt_tokens = (
        prompt
        + [current]
        + captured["primary_cand"][:fork]
        + [captured["alt_cand"][offset] for offset in range(fork, token_count - 1)]
    )
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
    logit_tol = 1e-4 if dtype == torch.float32 else 0.025
    max_error = 0.0
    chains = (
        (
            "primary_kv",
            captured["primary_rows"],
            primary_tokens,
            range(p0, p0 + token_count),
        ),
        (
            "alt_kv",
            captured["alt_rows"],
            alt_tokens,
            range(p0 + fork + 1, p0 + token_count),
        ),
    )
    for kv_key, rows, tokens, positions in chains:
        oracle, _, logits = serial_replay(m, backend, tokens, target_loops)
        for row, pos in zip(rows, positions):
            torch.testing.assert_close(row, logits[pos], atol=atol, rtol=rtol)
            assert int(row.argmax()) == int(logits[pos].argmax())
            max_error = max(max_error, float((row - logits[pos]).abs().max()))
        for (layer, depth), (key, value) in captured[kv_key].items():
            expected_key, expected_value = oracle.read(layer, "oracle", depth, len(tokens))
            torch.testing.assert_close(
                key[p0 : p0 + token_count],
                expected_key[p0 : p0 + token_count],
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(
                value[p0 : p0 + token_count],
                expected_value[p0 : p0 + token_count],
                atol=atol,
                rtol=rtol,
            )
        if dtype == torch.float32:
            independent = dense_reference(m, tokens, target_loops)[-1][2]
            for row, pos in zip(rows, positions):
                torch.testing.assert_close(row, independent[pos], atol=4e-5, rtol=4e-5)
    assert max_error < logit_tol

    # The outcome must follow from the captured target logits, not the patch.
    accepted_before_fork = all(
        int(captured["primary_rows"][offset].argmax()) == captured["primary_cand"][offset]
        for offset in range(fork)
    )
    hit = accepted_before_fork and (
        int(captured["primary_rows"][fork].argmax()) == captured["alt_cand"][fork]
    )
    stats = captured["stats"]
    assert stats.fork_rounds == 1
    assert stats.alt_drafted == k_drafts - fork
    assert stats.alternate_selected == int(hit)


@pytest.mark.parametrize("draft_loops,fork", [(1, 0), (1, 1), (2, 0), (2, 1)])
@pytest.mark.parametrize(
    "device,backend,dtype",
    [
        ("cpu", "torch", torch.float32),
        pytest.param("cuda", "triton", torch.float32, marks=pytest.mark.gpu),
        pytest.param("cuda", "triton", torch.bfloat16, marks=pytest.mark.gpu),
        pytest.param("cuda", "flash_attn_4", torch.bfloat16, marks=pytest.mark.gpu),
    ],
)
def test_branch_paths_match_serial_oracle(monkeypatch, dtype, device, backend, draft_loops, fork):
    if backend == "flash_attn_4":
        torch.manual_seed(123)
        m = OuroForCausalLM(tiny_ouro_config(hidden_size=256, head_dim=64)).to(
            device=device, dtype=dtype
        )
    else:
        m = model(dtype=dtype).to(device)
    captured, prompt, current = fork_round(
        m, monkeypatch, backend=backend, draft_loops=draft_loops, fork=fork, prefix_len=4
    )
    assert_fork_chains(m, backend, captured, prompt, current, fork=fork, dtype=dtype)


def test_branch_stem_copy_crosses_private_page_boundary(monkeypatch):
    m = model()
    captured, prompt, current = fork_round(
        m, monkeypatch, backend="torch", draft_loops=2, fork=1, prefix_len=5
    )
    # Prompt [10..14]: pages 0-1 are shared full pages; position 4 starts the
    # private page and the fork stem spans the page boundary at position 6.
    assert_fork_chains(m, "torch", captured, prompt, current, fork=1, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# prefix caching, incremental allocation, cancellation and later admission
# --------------------------------------------------------------------------- #
def snapshot_blocks(cache, groups):
    return {
        (block, layer, offset): (
            cache.key_cache[block, layer, offset].clone(),
            cache.value_cache[block, layer, offset].clone(),
        )
        for page in groups
        for block in page
        for layer in range(cache.num_layers)
        for offset in range(cache.block_size)
    }


def blocks_unchanged(cache, snapshot):
    return all(
        torch.equal(cache.key_cache[block, layer, offset], key)
        and torch.equal(cache.value_cache[block, layer, offset], value)
        for (block, layer, offset), (key, value) in snapshot.items()
    )


def test_branch_prefix_cache_incremental_cancel_and_later_admission():
    m = model()
    llm = LLM(
        m,
        speculative_config=SpeculativeConfig(3, alternate_prob_gap_threshold=1.0),
        cache_config=CacheConfig(128, 2, enable_prefix_caching=True, incremental_allocation=True),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=32),
    )
    params = SamplingParams(max_tokens=9, ignore_eos=True)
    prompts = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 7]]
    expected = LLM(m).generate(prompts, params)
    for _ in range(2):
        assert [o.token_ids for o in llm.generate(prompts, params)] == [
            o.token_ids for o in expected
        ]
    engine = llm.engine
    cache = engine.cache_manager
    shared = cache.lookup_prefix(prompts[0])
    assert shared and cache.prefix_hits > 0
    before = snapshot_blocks(cache, shared)

    engine.add_request("a", prompts[0], params)
    while not engine.step():
        pass
    engine.step()
    assert engine.speculative_runner.stats.fork_rounds > 0
    assert engine.abort_request("a").finish_reason == "abort"
    # The private stem copy never overwrites shared full prompt pages.
    assert blocks_unchanged(cache, before)

    engine.add_request("b", prompts[1], params)
    assert drain(engine)["b"].token_ids == expected[1].token_ids
    assert cache.num_used_blocks == 0
    assert not cache._allocations
    # Flush retained prefix entries to confirm every block reference is balanced.
    while cache._prefixes:
        _, blocks = cache._prefixes.popitem(last=False)
        cache._drop_refs(blocks)
    assert all(reference == 0 for reference in cache._refs)
    assert cache.num_free_blocks == cache.num_blocks
