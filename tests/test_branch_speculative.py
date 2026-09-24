"""Optional single-fallback branch speculation: equality, lifecycle and cleanup."""

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
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.worker.branch_speculative import BranchSpeculativeRunner
from vllm_rlt.worker.speculative import SpeculativeRunner


def model(seed=123, dtype=torch.float32):
    torch.manual_seed(seed)
    return OuroForCausalLM(OuroConfig.tiny()).to(dtype=dtype)


def branch_engine(m, k=3, margin=1.0, draft_loops=2, **kwargs):
    return LLMEngine(
        m,
        speculative_config=SpeculativeConfig(
            k, draft_loops=draft_loops, target_loops=4, fallback_margin=margin
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
def test_fallback_margin_none_keeps_plain_runner():
    m = model()
    plain = LLMEngine(m, speculative_config=SpeculativeConfig(3))
    assert type(plain.speculative_runner) is SpeculativeRunner
    enabled = branch_engine(m)
    assert type(enabled.speculative_runner) is BranchSpeculativeRunner


@pytest.mark.parametrize("margin", [None, 0.0, 0.5, 1.0])
def test_valid_fallback_margin(margin):
    config = SpeculativeConfig(3, fallback_margin=margin)
    assert config.fallback_margin == margin


@pytest.mark.parametrize("margin", [-0.1, 1.5, float("nan"), float("inf"), True, "0.2"])
def test_invalid_fallback_margin(margin):
    with pytest.raises(ValueError):
        SpeculativeConfig(3, fallback_margin=margin)


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
    m = OuroForCausalLM(OuroConfig.tiny(vocab_size=8))
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
            3, draft_loops=draft_loops, target_loops=4, fallback_margin=1.0
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
        speculative_config=SpeculativeConfig(3, fallback_margin=1.0),
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
        speculative_config=SpeculativeConfig(3, fallback_margin=1.0),
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
def test_branch_hit_commits_alternate(monkeypatch):
    m = model()
    e = branch_engine(m)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    install_controlled_coda(
        monkeypatch,
        m,
        shallow=[[10], [12, 13], [14, 15]],
        deep=[11, 39, 38, 37, 13, 15, 16],
    )
    out = e.step()[0]
    stats = e.speculative_runner.stats
    assert out.token_ids[-4:] == [11, 13, 15, 16]
    assert stats.alternate_selected == 1
    assert stats.primary_rejected_at_fork == 1
    assert stats.rollback_rounds == 0
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


def test_branch_rejected_elsewhere_rolls_back(monkeypatch):
    m = model()
    e = branch_engine(m)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=12, ignore_eos=True))
    while not e.step():
        pass
    install_controlled_coda(
        monkeypatch,
        m,
        shallow=[[10], [12, 13], [14, 15]],
        deep=[10, 39, 38, 37, 0, 0, 0],
    )
    out = e.step()[0]
    stats = e.speculative_runner.stats
    assert out.token_ids[-2:] == [10, 39]
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
        speculative_config=SpeculativeConfig(3, fallback_margin=1.0),
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
        speculative_config=SpeculativeConfig(3, fallback_margin=1.0),
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
