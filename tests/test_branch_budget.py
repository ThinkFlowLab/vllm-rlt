"""Branch verification respects the scheduler row budget."""

import pytest

from tests.test_branch_speculative import branch_engine, drain, model
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig


@pytest.mark.parametrize("budget", [1, 2, 5, 9, 12])
@pytest.mark.parametrize("threshold", [None, 1.0])
def test_verification_rows_stay_within_budget(monkeypatch, budget, threshold):
    engine = branch_engine(
        model(),
        k=4,
        margin=threshold,
        cache_config=CacheConfig(512, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=budget, max_num_seqs=4),
    )
    runner = engine.speculative_runner
    original = runner._core
    verification_rows = []

    def record(hidden, ids, positions, depth, **kwargs):
        if kwargs.get("packed"):
            verification_rows.append(len(ids))
            assert len(ids) <= budget
        return original(hidden, ids, positions, depth, **kwargs)

    monkeypatch.setattr(runner, "_core", record)
    limits = [2, 7, 13]
    for index, limit in enumerate(limits):
        engine.add_request(
            f"r{index}",
            [2, 3, 4][: index + 1],
            SamplingParams(max_tokens=limit, ignore_eos=True),
        )
    outputs = drain(engine)
    assert verification_rows
    assert [len(outputs[f"r{i}"].token_ids) for i in range(3)] == limits
    if threshold is not None and budget >= 5:
        assert runner.stats.fork_rounds > 0
    assert not engine.cache_manager._allocations
    assert engine.cache_manager.num_used_blocks == 0


def test_capacity_fallback_preserves_evictable_prefix():
    engine = branch_engine(
        model(),
        cache_config=CacheConfig(12, 2, enable_prefix_caching=True),
    )
    cache = engine.cache_manager
    # One retained page consumes four depth planes. Its references are
    # evictable, so num_free_blocks differs from truly unclaimed blocks.
    assert cache.allocate("seed", 3)
    cache.mark_imported_prefix("seed", 3)
    cache.publish_prefix("seed", [2, 3, 4], 3)
    cache.free("seed")
    assert cache.allocate("r", 4)
    before = tuple(cache._prefixes.items())
    refs = tuple(cache._refs)
    assert before and not cache._free_blocks
    assert cache.num_free_blocks == 4
    assert not engine.speculative_runner._create_alt("r", "r::alt", 0)
    assert tuple(cache._prefixes.items()) == before
    assert tuple(cache._refs) == refs
    assert "r::alt" not in cache._allocations
    cache.free("r")
    assert cache.num_used_blocks == 0
