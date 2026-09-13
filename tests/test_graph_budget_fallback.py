"""Budget fallback requires confirmed completion, restoration and owned-resource release."""

from dataclasses import asdict

import pytest
import test_recurrent_graph as fixtures
import torch

from vllm_lt.worker import recurrent_graph as graph
from vllm_lt.worker.model_runner import ModelRunner

model = fixtures.model
pytestmark = pytest.mark.usefixtures("forbid_cuda")


def low_limits(field):
    return {**asdict(graph.GraphLimits()), field: 1}


def compact_matches(model, runner, reference):
    for cache in (runner.cache_manager, reference):
        assert cache.allocate("request", 3)
    for position in range(3):
        hidden = model.prelude(torch.tensor([position + 3]))
        actual = runner._recurrent(hidden, ["request"], [0], [position])
        expected = model.recurrent(hidden, ["request"], [0], [position], reference)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(runner.cache_manager.key_cache, reference.key_cache)
    assert torch.equal(runner.cache_manager.value_cache, reference.value_cache)
    assert fixtures.prefixes(runner.cache_manager) == fixtures.prefixes(reference)
    assert runner._graph_snapshot()["budget_decline"]["calls"] == 3


@pytest.mark.parametrize("field", list(asdict(graph.GraphLimits())))
def test_budget_decline_restores_resources_before_compact(model, monkeypatch, field):
    cache, reference = fixtures.make_cache(model), fixtures.make_cache(model)
    cache._free_blocks[:] = cache._free_blocks[13:] + cache._free_blocks[:13]
    reference._free_blocks[:] = cache._free_blocks
    before = cache.key_cache.clone(), cache.value_cache.clone(), tuple(cache._free_blocks)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    early = field in ("common_payload_bytes", "cpu_staging_bytes")
    if early:
        monkeypatch.setattr(
            cache, "_allocate_metadata_storage", lambda **kw: pytest.fail("allocated over budget")
        )
    elif field == "setup_timeout_s":
        clock = [10**12]
        monkeypatch.setattr(graph.time, "perf_counter_ns", lambda: clock[0])
        body = graph.RecurrentGraphExecutor._tensor_body

        def slow(executor, bucket):
            output = body(executor, bucket)
            clock[0] += 2 * 10**9
            return output

        monkeypatch.setattr(graph.RecurrentGraphExecutor, "_tensor_body", slow)
    else:
        measured = (
            field.removeprefix("graph_retained_")
            if field.startswith("graph_")
            else field.removeprefix("setup_")
        )
        runtime.after_memory = {**runtime.memory(), measured: 2}
        runtime.memory_calls = 0
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True, limits=low_limits(field))
    decline = runner._graph_snapshot()["budget_decline"]
    assert decline["error"]["limit_name"] == field and decline["completion_confirmed"]
    assert runner._decode_executor is None
    if early:
        assert decline["attempt_setup"] is None and runtime.events == []
    else:
        assert decline["attempt_setup"]["scratch"]["restored"]
        assert decline["attempt_failure"]["secondary"] == []
    assert not cache._allocations and tuple(cache._free_blocks) == before[2]
    assert torch.equal(cache.key_cache, before[0]) and torch.equal(cache.value_cache, before[1])
    assert all(g.reset_done for g in runtime.graphs)
    assert all(ref() is None for ref in runtime.pool_refs)
    compact_matches(model, runner, reference)
    runner._close_recurrent_graph()
    assert runner._graph_snapshot()["status"] == "closed"


@pytest.mark.parametrize("failure", ["construction", "capture"])
def test_device_errors_never_decline_to_compact(model, monkeypatch, failure):
    primary = MemoryError("allocation") if failure == "construction" else ValueError("capture")
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)

    create = runtime.new_graph

    def fail():
        if failure == "construction":
            raise primary
        value = create()
        value.end_failure = primary
        return value

    monkeypatch.setattr(runtime, "new_graph", fail)
    runner = ModelRunner(model, cache)
    with pytest.raises(type(primary)) as caught:
        runner._enable_recurrent_graph(use_graphs=True)
    assert caught.value is primary and runner._graph_declined is None
    assert runner._decode_executor.status == "failed"
    assert runner._decode_executor.setup_record["scratch"]["restored"]
    runner._close_recurrent_graph()
    assert all(ref() is None for ref in runtime.pool_refs)


@pytest.mark.parametrize("secondary", ["synchronize", "restore", "close", "prior_secondary"])
def test_budget_decline_refused_after_uncertain_or_failed_cleanup(model, monkeypatch, secondary):
    class LegacyBudgetError(graph.CaptureBudgetExceeded):
        def __getattribute__(self, name):
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    monkeypatch.setattr(graph, "CaptureBudgetExceeded", LegacyBudgetError)
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    runtime.after_memory = {
        "allocated_bytes": 2,
        "reserved_bytes": 2,
        "peak_allocated_bytes": 2,
        "peak_reserved_bytes": 2,
    }
    primary = []
    settle = graph.RecurrentGraphExecutor.settle_failure

    def failed_settle(executor, error):
        primary.append(error)
        if secondary in ("synchronize", "prior_secondary"):
            runtime.main.failure = RuntimeError("completion uncertain")
        elif secondary == "restore":

            def broken_restore():
                raise RuntimeError("scratch restoration failed")

            monkeypatch.setattr(executor, "_restore_scratch", broken_restore)
        result = settle(executor, error)
        if secondary == "prior_secondary":
            # A later close could succeed; the recorded earlier uncertainty must
            # still prevent eager fallback and preserve quarantine.
            runtime.main.failure = None
        return result

    monkeypatch.setattr(graph.RecurrentGraphExecutor, "settle_failure", failed_settle)
    if secondary == "close":

        def broken_reset():
            raise RuntimeError("graph reset failed")

        create = runtime.new_graph

        def graph_with_broken_reset():
            value = create()
            value.reset = broken_reset
            return value

        monkeypatch.setattr(runtime, "new_graph", graph_with_broken_reset)
    runner = ModelRunner(model, cache)
    with pytest.raises(graph.CaptureBudgetExceeded) as raised:
        runner._enable_recurrent_graph(
            use_graphs=True, limits=low_limits("graph_retained_allocated_bytes")
        )
    assert raised.value is primary[0]
    assert runner._graph_declined is None and runner._decode_executor is not None
    assert runner._decode_executor.status == "failed"
    assert cache._quarantine_reason is not None
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.allocate("new", 2)
    with pytest.raises(RuntimeError):
        runner._recurrent(torch.zeros(1, model.config.hidden_size), ["new"], [0], [0])


@pytest.mark.parametrize("confirmed", [False, True])
def test_repeated_settlement_keeps_cascaded_failure_without_retrying_device(confirmed):
    executor = object.__new__(graph.RecurrentGraphExecutor)
    primary = {"type": "RuntimeError", "message": "first failure"}
    executor.failure = {"primary": primary, "secondary": [], "completion_confirmed": confirmed}
    result = executor.settle_failure(ValueError("later failure"))
    assert result is confirmed
    assert executor.failure == {
        "primary": primary,
        "secondary": [{"type": "ValueError", "message": "later failure"}],
        "completion_confirmed": confirmed,
    }
