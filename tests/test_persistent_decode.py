"""CPU lifetime, publication, and failure checks for fixed decode storage."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager, _metadata_payload_bytes
from vllm_lt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Request, Stage
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.worker.model_runner import ModelRunner


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("persistent CPU tests must not discover or initialize CUDA")

    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def model():
    torch.manual_seed(37)
    return OuroForCausalLM(OuroConfig.tiny())


def cache_for(model, *, num_blocks=160, block_size=16):
    c = model.config
    cache = KVCacheManager(
        c.num_hidden_layers,
        c.num_key_value_heads,
        c.head_dim,
        num_blocks,
        block_size,
        c.total_ut_steps,
    )
    cache.key_cache.fill_(71)
    cache.value_cache.fill_(-83)
    return cache


def request(model, name, token=3):
    return Request(
        name,
        [token],
        SamplingParams(max_tokens=3, ignore_eos=True),
        stage=Stage.RECURRENT,
        hidden_state=model.prelude(torch.tensor([token]))[0],
    )


def run_request(runner, *requests):
    return runner.execute(SchedulerOutput(Stage.RECURRENT, [ScheduledItem(r) for r in requests]))


def clone_storage(storage):
    return {name: value.clone() for name, value in storage.tensors.items()}


def test_fixed_storage_payload_and_snapshot_are_detached(model):
    runner = ModelRunner(model, cache_for(model))
    assert runner._persistent_snapshot() == {"enabled": False}
    runner._enable_persistent_decode()
    snapshot = runner._persistent_snapshot()
    assert snapshot["status"] == "ready" and snapshot["generation"] == 0
    assert snapshot["capacity"] == {"row_count": 8, "table_width": 32, "max_live_rows": 4}
    assert snapshot["device_payload_bytes"] == 64 * model.config.hidden_size + 1320
    assert snapshot["cpu_staging_bytes"] == 1256
    assert snapshot["cpu_staging_bytes"] == _metadata_payload_bytes()
    assert snapshot["device_payload_bytes"] <= 256 * 1024
    assert snapshot["cpu_staging_bytes"] <= 16 * 1024
    metadata = runner._persistent["metadata"]
    assert all(not tensor.is_pinned() for tensor in metadata.staging.values())
    assert all(
        metadata.staging[k].data_ptr() != tensor.data_ptr()
        for k, tensor in metadata.tensors.items()
    )
    snapshot["counters"]["calls"] = 999
    snapshot["tensors"]["hidden_in"]["shape"][0] = 99
    assert runner._persistent_snapshot()["counters"]["calls"] == 0
    assert runner._persistent_snapshot()["tensors"]["hidden_in"]["shape"][0] == 8
    with pytest.raises(RuntimeError, match="replacement"):
        runner._enable_persistent_decode()


def test_constructor_cap_dtype_and_partial_failure_leave_executor_disabled(model, monkeypatch):
    cache = cache_for(model)
    runner = ModelRunner(model, cache)
    initial = cache.num_free_blocks
    with monkeypatch.context() as patch:
        patch.setattr(model, "config", replace(model.config, hidden_size=8192))
        with pytest.raises(ValueError, match="256 KiB"):
            runner._enable_persistent_decode()
    assert runner._persistent_snapshot() == {"enabled": False}
    original = torch.empty

    def fail_core(shape, *args, **kwargs):
        if shape == (8, model.config.hidden_size):
            raise MemoryError("injected core allocation failure")
        return original(shape, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", fail_core)
        with pytest.raises(MemoryError):
            runner._enable_persistent_decode()
    assert runner._persistent_snapshot() == {"enabled": False}
    assert cache.num_free_blocks == initial and not cache._allocations
    model.bfloat16()
    with pytest.raises(ValueError, match="float32"):
        runner._enable_persistent_decode()


def test_storage_generation_shrink_and_reallocated_ownership(model):
    cache = cache_for(model)
    for name in ("a", "b", "c", "d"):
        cache.allocate(name, 33)
    storage = cache._allocate_metadata_storage()
    pointers = {k: v.data_ptr() for k, v in storage.tensors.items()}
    first = cache._prepare_into(
        storage, cache._prepare_host_batch(["a", "b", "c", "d"], [0, 1, 2, 3], [32, 31, 16, 0])
    )
    assert first.live_rows == (1, 3, 5, 7)
    with pytest.raises(RuntimeError, match="already in use"):
        cache._prepare_into(storage, cache._prepare_host_batch(["a"], [0], [0]))
    cache._release_prepared(first)
    second = cache._prepare_into(storage, cache._prepare_host_batch(["a"], [2], [1]))
    assert second.generation == 2
    assert {k: v.data_ptr() for k, v in storage.tensors.items()} == pointers
    assert second.active.tolist() == [False, True, False, False, False, False, False, False]
    assert second.position_ids.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    for row in (0, 2, 3, 4, 5, 6, 7):
        assert second.write_blocks[row] == second.write_offsets[row] == -1
        assert second.block_tables[row].tolist() == [-1] * 32
        assert second.context_lengths[row] == 0
    with pytest.raises(RuntimeError, match="generation"):
        cache._require_live_batch(first)
    altered = replace(second, position_ids=second.position_ids.clone())
    with pytest.raises(ValueError, match="borrowed tensor"):
        cache._require_live_batch(altered)
    cache.free("a")
    cache.allocate("a", 33)
    with pytest.raises(RuntimeError, match="stale prepared"):
        cache._require_live_batch(second)


def test_invalid_host_metadata_is_rejected_before_storage_mutation(model):
    cache = cache_for(model)
    cache.allocate("a", 513)
    storage = cache._allocate_metadata_storage()
    for tensor in storage.tensors.values():
        tensor.zero_()
    before = clone_storage(storage)
    with pytest.raises(ValueError, match="duplicate"):
        cache._prepare_host_batch(["a", "a"], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="capacity"):
        cache._prepare_into(storage, cache._prepare_host_batch(["a"], [0], [512]))
    stale = cache._prepare_host_batch(["a"], [0], [0])
    cache.free("a")
    cache.allocate("a", 513)
    with pytest.raises(RuntimeError, match="stale host"):
        cache._prepare_into(storage, stale)
    assert storage.generation == 0 and not storage.in_use and not storage.failed
    assert all(torch.equal(before[name], value) for name, value in storage.tensors.items())


def test_persistent_matches_allocating_padded_and_published_state_survives_reuse(
    model, monkeypatch
):
    a_cache, b_cache = cache_for(model), cache_for(model)
    a, b = ModelRunner(model, a_cache), ModelRunner(model, b_cache)
    b._enable_persistent_decode()
    requests = [request(model, name, token) for name, token in (("a", 3), ("b", 7))]
    for cache in (a_cache, b_cache):
        for r in requests:
            cache.allocate(r.request_id, 4)
    initial = b._persistent_snapshot()
    hidden = torch.stack([r.hidden_state for r in requests])
    expected = a._recurrent_padded(
        hidden, ["a", "b"], [0, 0], [0, 0], row_indices=[1, 3], row_count=8, table_width=32
    )
    observations = []
    original = model._recurrent_prepared

    def observed(hidden, batch, cache):
        observations.append((b._persistent_snapshot(), hidden.data_ptr(), batch))
        return original(hidden, batch, cache)

    monkeypatch.setattr(model, "_recurrent_prepared", observed)
    actual = run_request(b, *requests)
    assert observations[0][0]["status"] == "in_flight"
    assert observations[0][1] == initial["tensors"]["hidden_in"]["data_ptr"]
    torch.testing.assert_close(
        torch.stack([r.hidden_state for r in requests]), expected[0], atol=0, rtol=0
    )
    torch.testing.assert_close(torch.tensor(actual), expected[1].sigmoid(), atol=0, rtol=0)
    assert torch.equal(a_cache.key_cache, b_cache.key_cache)
    assert torch.equal(a_cache.value_cache, b_cache.value_cache)
    held = requests[1]
    held.stage = Stage.CODA
    held.generator = torch.Generator().manual_seed(9)
    held_state, rng = held.hidden_state.clone(), held.generator.get_state().clone()
    snapshot = b._persistent_snapshot()
    owned = {t["storage_ptr"] for t in snapshot["tensors"].values()}
    assert all(t["storage_ptr"] not in owned for t in snapshot["last_publication"].values())
    with torch.inference_mode():
        b._persistent["tensors"]["hidden_out"].fill_(999)
        b._persistent["tensors"]["gate_out"].fill_(-999)
    b_cache.free("a")
    b_cache.allocate("a", 4)
    run_request(b, request(model, "a", 11))
    assert torch.equal(held.hidden_state, held_state) and held.stage == Stage.CODA
    assert torch.equal(held.generator.get_state(), rng)
    assert held.generated_token_ids == [] and held.exit_depths == []
    after = b._persistent_snapshot()
    assert after["tensors"] == initial["tensors"]
    assert after["staging_tensors"] == initial["staging_tensors"]
    assert after["counters"] == {"calls": 2, "prepared": 2, "completed": 2, "empty": 0}
    assert after["status"] == "ready" and after["generation"] == 2


def test_direct_calls_complete_before_lease_release_and_empty_is_noop(model, monkeypatch):
    cache = cache_for(model)
    cache.allocate("a", 2)
    runner = ModelRunner(model, cache)
    runner._enable_persistent_decode()
    order = []
    monkeypatch.setattr(runner, "_synchronize_persistent", lambda: order.append("complete"))
    release = cache._release_prepared

    def checked_release(batch):
        assert order == ["complete"]
        order.append("release")
        release(batch)

    monkeypatch.setattr(cache, "_release_prepared", checked_release)
    runner._recurrent_persistent(model.prelude(torch.tensor([1])), ["a"], [0], [0])
    assert order == ["complete", "release"]
    before = runner._persistent_snapshot()
    runner._recurrent_persistent(torch.empty(0, model.config.hidden_size), [], [], [])
    after = runner._persistent_snapshot()
    assert order == ["complete", "release"]
    assert before["tensors"] == after["tensors"] and after["generation"] == 1
    assert after["counters"]["empty"] == 1 and after["last_dispatch"]["kind"] == "empty"


@pytest.mark.parametrize("reason", ["live_count", "table_width"])
def test_fallback_counts_real_compact_dispatch_without_mutating_buffers(model, monkeypatch, reason):
    cache = cache_for(model, num_blocks=200)
    runner = ModelRunner(model, cache)
    runner._enable_persistent_decode()
    count, position = (5, 0) if reason == "live_count" else (1, 512)
    ids = [str(i) for i in range(count)]
    for name in ids:
        cache.allocate(name, position + 1)
    tensors = runner._persistent["tensors"]
    for tensor in tensors.values():
        tensor.zero_()
    before = {k: t.clone() for k, t in tensors.items()}
    calls = []

    def compact(hidden, request_ids, depths, positions, owner):
        calls.append((list(request_ids), list(depths), list(positions), owner))
        # Real compact metadata construction proves the fallback width is retained.
        batch = owner._prepare_batch(request_ids, depths, positions)
        assert batch.active is None and batch.row_count == count
        return hidden + 1, torch.zeros(count)

    monkeypatch.setattr(model, "recurrent", compact)
    runner._recurrent_persistent(
        torch.zeros(count, model.config.hidden_size), ids, [0] * count, [position] * count
    )
    snapshot = runner._persistent_snapshot()
    assert len(calls) == 1 and calls[0][3] is cache
    assert snapshot["fallback_counts"][reason] == 1
    assert snapshot["generation"] == 0 and snapshot["counters"]["prepared"] == 0
    assert all(torch.equal(before[k], tensor) for k, tensor in tensors.items())
    with pytest.raises(ValueError, match="duplicate"):
        runner._recurrent_persistent(
            torch.zeros(5, model.config.hidden_size), [ids[0]] * 5, [0] * 5, [position] * 5
        )
    assert runner._persistent_snapshot() == snapshot


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_padded_vs_persistent_generation_matches_trajectories_and_cleanup(model, mode):
    results = []
    for persistent in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(80, 16),
            scheduler_config=SchedulerConfig(mode=mode, max_num_batched_tokens=3),
        )
        runner = engine.model_runner
        if persistent:
            runner._enable_persistent_decode()
        else:
            runner._recurrent = lambda hidden, ids, depths, positions: runner._recurrent_padded(
                hidden,
                ids,
                depths,
                positions,
                row_indices=[2 * i + 1 for i in range(len(ids))],
                row_count=8,
                table_width=32,
            )
        for i, prompt in enumerate(([1, 2, 3], [7], [4, 5])):
            engine.add_request(
                str(i),
                list(prompt),
                SamplingParams(max_tokens=4, ignore_eos=True, exit_threshold=(0, 1, 0.7)[i]),
            )
        outputs, trace = {}, []
        for _ in range(150):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = (output.token_ids, output.exit_depths)
            trace.append(
                (
                    engine.last_schedule.stage,
                    [i.request.request_id for i in engine.last_schedule.items],
                )
            )
        assert set(outputs) == {"0", "1", "2"}
        assert not engine.has_unfinished_requests() and engine.cache_manager.num_used_blocks == 0
        results.append((outputs, trace))
        if persistent:
            state = runner._persistent_snapshot()
            assert state["status"] == "ready" and state["counters"]["completed"] > 0
            assert state["fallback_counts"] == {"live_count": 0, "table_width": 0}
        else:
            del runner._recurrent
    assert results[0] == results[1]


def failing_engine(model):
    engine = LLMEngine(model, cache_config=CacheConfig(40, 16))
    engine.model_runner._enable_persistent_decode()
    engine.add_request("a", [3], SamplingParams(max_tokens=3, ignore_eos=True))
    return engine


@pytest.mark.parametrize("completion_fails", [False, True])
def test_execution_failure_settles_before_abort_and_preserves_original(
    model, monkeypatch, completion_fails
):
    engine = failing_engine(model)
    runner, cache = engine.model_runner, engine.cache_manager
    events = []
    original = RuntimeError("original core failure")
    free = cache.free

    def fail(batch):
        events.append("execute")
        raise original

    def complete():
        events.append("complete")
        if completion_fails:
            raise RuntimeError("stream failed")

    def release(request_id):
        events.append("free")
        assert events == ["execute", "complete", "free"]
        free(request_id)

    monkeypatch.setattr(runner, "_execute_batch", fail)
    monkeypatch.setattr(runner, "_synchronize_persistent", complete)
    monkeypatch.setattr(cache, "free", release)
    with pytest.raises(RuntimeError) as raised:
        engine.step()
    assert raised.value is original
    assert events == ["execute", "complete"] + ([] if completion_fails else ["free"])
    snapshot = runner._persistent_snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["failure"]["completion_confirmed"] is not completion_fails
    assert cache.num_used_blocks == (4 if completion_fails else 0)
    before = deepcopy(cache._free_blocks)
    allocation = dict(cache._allocations)
    for operation in (
        lambda: engine.add_request("new", [3]),
        engine.step,
        lambda: runner._enable_persistent_decode(),
    ):
        with pytest.raises(RuntimeError):
            operation()
    if completion_fails:
        for operation in (
            lambda: engine.abort_request("a"),
            lambda: cache.free("a"),
            lambda: cache.allocate("new", 1),
            lambda: cache._prepare_batch(["a"], [0], [0]),
        ):
            with pytest.raises(RuntimeError, match="quarantined"):
                operation()
        assert cache._free_blocks == before and cache._allocations == allocation


def test_cleanup_error_does_not_mask_primary_and_quarantines_remaining_pages(model, monkeypatch):
    engine = failing_engine(model)
    primary = ValueError("execute primary")

    def execute(batch):
        raise primary

    def abort(request_id):
        raise RuntimeError("abort secondary")

    monkeypatch.setattr(engine.model_runner, "_execute_batch", execute)
    monkeypatch.setattr(engine.scheduler, "abort", abort)
    with pytest.raises(ValueError) as raised:
        engine.step()
    assert raised.value is primary
    failure = engine.model_runner._persistent_snapshot()["failure"]
    assert failure["primary"]["message"] == "execute primary"
    assert failure["secondary"] == [{"type": "RuntimeError", "message": "abort secondary"}]
    assert failure["completion_confirmed"] and engine.cache_manager.num_used_blocks == 4
    with pytest.raises(RuntimeError, match="quarantined"):
        engine.cache_manager.free("a")


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_update_failure_after_lease_release_still_settles_finalization(
    model, monkeypatch, error_type
):
    engine = failing_engine(model)
    runner = engine.model_runner
    # A synthetic execute has already passed the gate readback and has no lease;
    # _update can nevertheless submit later KV propagation work before failing.
    monkeypatch.setattr(runner, "execute", lambda batch: None)
    events = []
    original = error_type("finalization failed")

    def update(batch, result):
        assert runner._persistent_lease is None
        events.append("finalization")
        raise original

    monkeypatch.setattr(engine, "_update", update)
    monkeypatch.setattr(runner, "_synchronize_persistent", lambda: events.append("complete"))
    free = engine.cache_manager.free
    monkeypatch.setattr(
        engine.cache_manager, "free", lambda name: (events.append("free"), free(name))
    )
    with pytest.raises(error_type) as raised:
        engine.step()
    assert raised.value is original and events == ["finalization", "complete", "free"]
    assert runner._persistent_snapshot()["status"] == "failed"


def test_default_compact_path_does_not_allocate_persistent_storage(model, monkeypatch):
    cache = cache_for(model)
    cache.allocate("a", 2)
    runner = ModelRunner(model, cache)

    def forbidden():
        pytest.fail("default execution allocated persistent tensors")

    monkeypatch.setattr(cache, "_allocate_metadata_storage", forbidden)
    run_request(runner, request(model, "a"))
    assert runner._persistent_snapshot() == {"enabled": False}


def test_persistent_prefix_guard_still_requires_each_layer_write(model):
    cache = cache_for(model)
    cache.allocate("a", 2)
    storage = cache._allocate_metadata_storage()
    batch = cache._prepare_into(storage, cache._prepare_host_batch(["a"], [2], [0]))
    x = torch.full((8, cache.num_kv_heads, cache.head_dim), torch.nan)
    x[1] = 3
    with pytest.raises(RuntimeError, match="uninitialized KV history"):
        cache._attend_prepared(0, batch, x)
    cache._write_prepared(0, batch, x, x)
    out = cache._attend_prepared(0, batch, x)
    assert torch.equal(out[1], x[1])
    assert torch.equal(out[[0, 2, 3, 4, 5, 6, 7]], torch.zeros_like(out[[0, 2, 3, 4, 5, 6, 7]]))
    with pytest.raises(RuntimeError, match="layer 1"):
        cache._attend_prepared(1, batch, x)
    cache._write_prepared(1, batch, x, x)
    assert torch.isfinite(cache._attend_prepared(1, batch, x)).all()
    cache._release_prepared(batch)
    with pytest.raises(RuntimeError, match="generation"):
        cache._write_prepared(0, batch, x, x)


def test_execute_readback_precedes_lease_release_without_extra_sync(model, monkeypatch):
    cache = cache_for(model)
    cache.allocate("a", 2)
    runner = ModelRunner(model, cache)
    runner._enable_persistent_decode()
    events = []
    cpu = torch.Tensor.cpu
    release = cache._release_prepared

    def readback(tensor, *args, **kwargs):
        assert runner._persistent_snapshot()["status"] == "in_flight"
        events.append("actual_gate_readback")
        return cpu(tensor, *args, **kwargs)

    def complete(batch):
        assert events == ["actual_gate_readback"]
        events.append("release")
        release(batch)

    def unexpected_sync():
        pytest.fail("execute added a synchronization beyond the actual gate readback")

    monkeypatch.setattr(torch.Tensor, "cpu", readback)
    monkeypatch.setattr(cache, "_release_prepared", complete)
    monkeypatch.setattr(runner, "_synchronize_persistent", unexpected_sync)
    run_request(runner, request(model, "a"))
    assert events == ["actual_gate_readback", "release"]
    assert runner._persistent_snapshot()["status"] == "ready"


def test_direct_completion_failure_is_not_retried_and_quarantines(model, monkeypatch):
    cache = cache_for(model)
    cache.allocate("a", 2)
    runner = ModelRunner(model, cache)
    runner._enable_persistent_decode()
    attempts = []
    primary = RuntimeError("completion could not be established")

    def fail_completion():
        attempts.append(1)
        raise primary

    monkeypatch.setattr(runner, "_synchronize_persistent", fail_completion)
    with pytest.raises(RuntimeError) as raised:
        runner._recurrent_persistent(model.prelude(torch.tensor([3])), ["a"], [0], [0])
    assert raised.value is primary and len(attempts) == 1
    assert runner._persistent_snapshot()["failure"]["completion_confirmed"] is False
    assert cache.num_used_blocks == 4
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.free("a")


def test_reentrant_prepare_is_rejected_without_corrupting_outer_generation(model, monkeypatch):
    cache = cache_for(model)
    cache.allocate("a", 2)
    runner = ModelRunner(model, cache)
    runner._enable_persistent_decode()
    core = model._recurrent_prepared

    def reentrant(hidden, batch, owner):
        assert runner._persistent_snapshot()["status"] == "in_flight"
        with pytest.raises(RuntimeError, match="in-flight lease"):
            runner._recurrent_persistent(hidden[:1], ["a"], [0], [0])
        return core(hidden, batch, owner)

    monkeypatch.setattr(model, "_recurrent_prepared", reentrant)
    runner._recurrent_persistent(model.prelude(torch.tensor([3])), ["a"], [0], [0])
    snapshot = runner._persistent_snapshot()
    assert snapshot["status"] == "ready" and snapshot["generation"] == 1
    assert snapshot["counters"] == {"calls": 1, "prepared": 1, "completed": 1, "empty": 0}
