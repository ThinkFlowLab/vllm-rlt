"""CPU checks for the capture boundary; kernel substitutes do not qualify replay."""

import pytest
import test_recurrent_graph as fixtures
import torch
from test_recurrent_graph import fake_runtime, make_cache, prefixes

from vllm_lt.worker import capture_resources
from vllm_lt.worker.model_runner import ModelRunner
from vllm_lt.worker.recurrent_graph import _CudaRuntime

model = fixtures.model
pytestmark = pytest.mark.usefixtures("forbid_cuda")


def prepare(cache, ids=("a",), depths=(0,), positions=(0,), *, row_count=4):
    storage = cache._allocate_metadata_storage(row_count)
    host = cache._prepare_host_batch(ids, depths, positions)
    ticket = cache._begin_decode_traversal(host)
    batch = cache._prepare_into(storage, host)
    cache._bind_decode_traversal(ticket, batch)
    return storage, batch, ticket


def test_commit_waits_for_completion_and_preserves_sparse_and_already_written_positions(model):
    cache = make_cache(model)
    cache.allocate("a", 4)
    for tracker in cache._allocations["a"].written[0]:
        tracker.add(0)
        tracker.add(2)
    before = prefixes(cache)
    storage, batch, ticket = prepare(cache, positions=(1,))
    assert prefixes(cache) == before and ticket.state == "bound"
    with pytest.raises(RuntimeError, match="finish before releasing"):
        cache._release_prepared(batch)
    with pytest.raises(RuntimeError, match="confirmed"):
        cache._commit_decode_traversal(ticket, completion_confirmed=False)
    assert prefixes(cache) == before
    cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert all(p.prefix == 3 and not p.pending for p in cache._allocations["a"].written[0])
    with pytest.raises(RuntimeError, match="already finished"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    cache._release_prepared(batch)
    assert not storage.in_use
    _, rewrite, again = prepare(cache, positions=(1,))
    cache._commit_decode_traversal(again, completion_confirmed=True)
    assert all(p.prefix == 3 for p in cache._allocations["a"].written[0])
    cache._release_prepared(rewrite)


def test_commit_revalidates_all_owners_before_first_prefix_mutation(model):
    cache = make_cache(model)
    for name in ("a", "b"):
        cache.allocate(name, 2)
    storage, _, ticket = prepare(cache, ids=("a", "b"), depths=(0, 0), positions=(0, 0))
    old_b = cache._allocations["b"]
    cache.free("b")
    cache.allocate("b", 2)
    with pytest.raises(RuntimeError, match="stale"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert all(p.prefix == 0 for p in cache._allocations["a"].written[0])
    assert all(p.prefix == 0 for p in old_b.written[0])
    assert storage.failed and ticket.state == "failed"
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.free("a")


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_host_commit_failure_quarantines_without_a_rollback_claim(
    model, monkeypatch, error_type
):
    cache = make_cache(model)
    for name in ("a", "b"):
        cache.allocate(name, 2)
    storage, batch, ticket = prepare(cache, ids=("a", "b"), depths=(0, 0), positions=(0, 0))

    def fail(position):
        raise error_type("injected later tracker failure")

    monkeypatch.setattr(cache._allocations["b"].written[0][0], "add", fail)
    with pytest.raises(error_type, match="later tracker"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert all(p.prefix == 1 for p in cache._allocations["a"].written[0])
    assert cache._allocations["b"].written[0][0].prefix == 0
    assert ticket.state == "failed" and storage.failed
    cache._abort_decode_traversal(ticket, completion_confirmed=True)
    assert not storage.in_use and storage.transaction is None
    for operation in (
        lambda: cache.allocate("new", 1),
        lambda: cache.free("a"),
        lambda: cache._release_prepared(batch),
    ):
        with pytest.raises(RuntimeError, match="quarantined"):
            operation()


def test_tensor_body_uses_only_device_metadata_and_zeroes_inactive_rows(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    storage = cache._allocate_metadata_storage(4)
    batch = cache._prepare_into(storage, cache._prepare_host_batch([], [], []))
    view = cache._make_tensor_decode_view(storage)
    original = cache.key_cache.clone(), cache.value_cache.clone()

    def forbidden(*args, **kwargs):
        pytest.fail("tensor body read host ownership or prefix state")

    for name in (
        "_require_live_batch",
        "_require_prefix",
        "_require_usable",
        "_get_allocation",
        "_write_prepared",
        "_attend_prepared",
    ):
        monkeypatch.setattr(cache, name, forbidden)
    hidden, gates = model._recurrent_tensor(
        torch.full((4, model.config.hidden_size), torch.nan), view
    )
    for value in (hidden, gates):
        assert torch.equal(value, torch.zeros_like(value)) and not torch.signbit(value).any()
    assert torch.equal(cache.key_cache, original[0]) and torch.equal(cache.value_cache, original[1])
    assert batch.live_rows == ()


def test_host_validation_runs_once_and_metadata_uses_one_packed_copy(model, monkeypatch):
    cache = make_cache(model)
    cache.allocate("a", 2)
    checks, copies = [], []
    validate = cache._validate_decode_host
    copy = torch.Tensor.copy_
    monkeypatch.setattr(
        cache, "_validate_decode_host", lambda host: (checks.append(host), validate(host))[-1]
    )

    def copied(target, source, **kwargs):
        copies.append((target, source, kwargs))
        return copy(target, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copied)
    storage, batch, ticket = prepare(cache)
    cache._commit_decode_traversal(ticket, completion_confirmed=True)
    cache._release_prepared(batch)
    assert len(checks) == len(copies) == 1
    assert copies[0][0] is storage.packed and copies[0][1] is storage.packed_staging
    assert all(
        t.untyped_storage().data_ptr() == storage.packed.data_ptr()
        for t in storage.tensors.values()
    )


def test_capture_stream_reused_only_after_release_and_never_shared_by_live_runtimes(monkeypatch):
    monkeypatch.setattr(capture_resources, "_IDLE_STREAMS", {})
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    a, b, c = (_CudaRuntime("cuda:0") for _ in range(3))
    first, second = a.new_stream(), b.new_stream()
    assert first is not second
    with pytest.raises(RuntimeError, match="already leased"):
        a.new_stream()
    with pytest.raises(RuntimeError, match="does not belong"):
        a.release_stream(second)
    a.release_stream(first)
    assert c.new_stream() is first
    assert b._leased_stream is second
    other_device = _CudaRuntime("cuda:1")
    assert other_device.new_stream() not in (first, second)
    c.release_stream(first)
    b.release_stream(second)


def enable(model, monkeypatch):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    return cache, runtime, runner, runner._decode_executor


def released(runtime):
    assert all(ref() is None for ref in runtime.pool_refs)
    assert len(runtime.pool_releases) == len(runtime.pool_refs)
    assert all(ready for _, ready in runtime.pool_releases)


def test_pool_rejects_unqualified_allocator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    with pytest.raises(RuntimeError, match="native CUDA allocator"):
        _CudaRuntime("cuda:0").new_pool()


def test_shared_owner_releases_after_all_resets_and_captured_outputs(model, monkeypatch):
    _, runtime, runner, executor = enable(model, monkeypatch)
    snapshot = executor.snapshot()
    pool_ids = []
    for key, bucket in snapshot["buckets"].items():
        owner = bucket["pool_owner"]
        assert owner == {
            "kind": "torch.cuda.MemPool",
            "id": bucket["pool_id"],
            "release_policy": "synchronize-reset-drop-captured-outputs-owner-last",
        }
        assert list(executor.buckets[int(key)]["graph"].pool()) == owner["id"]
        pool_ids.append(tuple(owner["id"]))
    assert len(set(pool_ids)) == 1
    assert all(ref() is not None for ref in runtime.pool_refs)
    assert all(ref() is not None for graph in runtime.graphs for ref in graph.output_refs)
    before_close = len(runtime.events)
    runner._close_recurrent_graph()
    events = runtime.events[before_close:]
    assert events[:2] == [("synchronize", "setup"), ("synchronize", "main")]
    resets = [i for i, event in enumerate(events) if event[0] == "reset"]
    frees = [i for i, event in enumerate(events) if event[0] == "pool_release"]
    assert len(resets) == 2 and len(frees) == 1 and max(resets) < min(frees)
    assert events[-1] == ("release_stream", "setup")
    released(runtime)
    assert executor.snapshot()["buckets"] == {}


@pytest.mark.parametrize("failure_site", ["synchronize", "second_reset"])
def test_failed_close_retains_shared_owner_and_outputs_until_confirmed_close(
    model, monkeypatch, failure_site
):
    _, runtime, _, executor = enable(model, monkeypatch)
    pointer = executor.buckets[4]["tensors"]["hidden_in"].data_ptr()
    primary = RuntimeError("close completion/reset failed")
    reset = runtime.graphs[1].reset

    def fail():
        raise primary

    if failure_site == "synchronize":
        runtime.main.failure = primary
    else:
        monkeypatch.setattr(runtime.graphs[1], "reset", fail)
    with pytest.raises(RuntimeError) as caught:
        executor.close()
    assert caught.value is primary
    assert executor.status == "failed"
    assert executor.buckets[4]["tensors"]["hidden_in"].data_ptr() == pointer
    assert executor.failure["primary"]["message"] == str(primary)
    assert all(ref() is not None for ref in runtime.pool_refs)
    assert all(ref() is not None for graph in runtime.graphs for ref in graph.output_refs)
    assert not runtime.pool_releases
    assert len(executor.buckets) == 2 and executor.failure is not None
    assert not any(event[0] == "release_stream" for event in runtime.events)
    runtime.main.failure = None
    monkeypatch.setattr(runtime.graphs[1], "reset", reset)
    executor.close()
    assert executor.status == "closed"
    # The saved primary traceback still holds local graph/bucket references.
    # Explicit owner/output assignments must nevertheless have released pools.
    released(runtime)
