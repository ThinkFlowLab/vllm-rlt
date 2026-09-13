"""CPU transaction/lifetime tests; fake capture does not qualify CUDA replay."""

import sys
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_lt.core.kv_cache_manager as kv_module
import vllm_lt.worker.recurrent_graph as graph_module
from vllm_lt.config import CacheConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Request, Stage
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.worker.model_runner import ModelRunner
from vllm_lt.worker.recurrent_graph import RecurrentGraphExecutor

pytestmark = pytest.mark.usefixtures("forbid_cuda")


@pytest.fixture
def model():
    torch.manual_seed(83)
    return OuroForCausalLM(OuroConfig.tiny())


def make_cache(model):
    c = model.config
    cache = KVCacheManager(c.num_hidden_layers, c.num_key_value_heads, c.head_dim, 160, 16, 4)
    cache.key_cache.fill_(71)
    cache.value_cache.fill_(-83)
    return cache


def prefixes(cache):
    return {
        name: [[(p.prefix, sorted(p.pending)) for p in depth] for depth in alloc.written]
        for name, alloc in cache._allocations.items()
    }


class FakeStream:
    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name
        self.failure = None

    def wait_stream(self, other):
        self.runtime.events.append(("wait", self.name, other.name))

    def synchronize(self):
        self.runtime.events.append(("synchronize", self.name))
        if self.failure is not None:
            raise self.failure


class FakePool:
    def __init__(self, runtime, index):
        self.runtime, self.id = runtime, (0, index)
        runtime.events.append(("pool_create", self.id))

    def __del__(self):
        graphs = [g for g in self.runtime.graphs if g.pool_id == self.id]
        ready = all(g.reset_done and all(ref() is None for ref in g.output_refs) for g in graphs)
        self.runtime.pool_releases.append((self.id, ready))
        self.runtime.events.append(("pool_release", self.id))


class FakeGraph:
    def __init__(self, runtime, index):
        self.runtime, self.index = runtime, index
        self.body = None
        self.reset_done = False
        self.replay_failure = self.end_failure = None
        self.pool_id, self.output_refs = None, []

    def capture_begin(self, *, pool, capture_error_mode):
        assert pool[0] == 0 and pool[1] > 0 and capture_error_mode == "global"
        self.pool_id = pool
        assert self.runtime.current is self.runtime.side
        self.runtime.capture = self
        self.runtime.events.append(("capture_begin", self.index))

    def capture_end(self):
        self.runtime.capture = None
        self.runtime.events.append(("capture_end", self.index))
        if self.end_failure is not None:
            raise self.end_failure

    def replay(self):
        self.runtime.events.append(("replay", self.index))
        if self.replay_failure is not None:
            raise self.replay_failure
        assert self.body is not None
        self.body()

    def pool(self):
        return self.pool_id

    def raw_cuda_graph_exec(self):
        return self.index + 100

    def reset(self):
        self.runtime.events.append(("reset", self.index))
        self.reset_done = True
        self.body = None


class FakeRuntime:
    def __init__(self):
        self.events, self.graphs = [], []
        self.main, self.side = FakeStream(self, "main"), FakeStream(self, "setup")
        self.current, self.capture = self.main, None
        self.after_memory = None
        self.memory_calls = 0
        self.pool_refs, self.pool_releases = [], []

    def current_stream(self):
        return self.current

    def new_stream(self):
        return self.side

    def release_stream(self, stream):
        assert stream is self.side
        self.events.append(("release_stream", stream.name))

    @contextmanager
    def stream_context(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old

    def new_graph(self):
        graph = FakeGraph(self, len(self.graphs) + 1)
        self.graphs.append(graph)
        return graph

    def new_pool(self):
        pool = FakePool(self, len(self.pool_refs) + 1)
        self.pool_refs.append(weakref.ref(pool))
        return pool

    def reset_peaks(self):
        self.events.append(("reset_peaks",))

    def memory(self):
        self.memory_calls += 1
        if self.memory_calls > 1 and self.after_memory is not None:
            return self.after_memory
        return dict.fromkeys(
            ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"), 0
        )


_original_tensor_body = RecurrentGraphExecutor._tensor_body


def fake_runtime(monkeypatch, cache=None):
    """Replace raw kernels and stream/graph plumbing, retaining real tensor/body/cache code."""

    def scatter(keys, values, blocks, offsets, k, v, active):
        live = active.nonzero().flatten()
        keys[blocks[live], offsets[live]] = k[live]
        values[blocks[live], offsets[live]] = v[live]

    monkeypatch.setitem(
        sys.modules, "vllm_lt.kernels.triton_kv_write", SimpleNamespace(masked_kv_write=scatter)
    )
    monkeypatch.setattr(kv_module, "triton_paged_attention", torch_paged_attention)
    if cache is not None:
        monkeypatch.setattr(cache, "backend", "triton")
    runtime = FakeRuntime()
    monkeypatch.setattr(graph_module, "_make_runtime", lambda device: runtime)
    original = _original_tensor_body

    def body(executor, bucket):
        runtime = executor.runtime
        if runtime.capture is not None:
            runtime.capture.body = lambda: original(executor, bucket)
        outputs = original(executor, bucket)
        if runtime.capture is not None:
            runtime.capture.output_refs = [weakref.ref(value) for value in outputs]
        return outputs

    monkeypatch.setattr(RecurrentGraphExecutor, "_tensor_body", body)
    return runtime


@pytest.mark.parametrize("use_graphs", [False, True])
def test_torch_backend_falls_back_without_cuda_setup(model, use_graphs):
    cache, reference = make_cache(model), make_cache(model)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=use_graphs)
    assert runner._graph_snapshot()["setup"]["skip_reason"] == "backend"
    for current in (cache, reference):
        current.allocate("a", 2)
    hidden = model.prelude(torch.tensor([3]))
    actual = runner._recurrent(hidden, ["a"], [0], [0])
    expected = model.recurrent(hidden, ["a"], [0], [0], reference)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert runner._decode_executor.fallback_counts["backend"] == 1
    runner._close_recurrent_graph()


def test_empty_invalid_prefix_and_shape_fallback_do_not_use_bucket(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=False)
    executor = runner._decode_executor
    initial = executor.snapshot()["buckets"]
    out = runner._recurrent(torch.empty(0, model.config.hidden_size), [], [], [])
    assert out[0].shape == (0, model.config.hidden_size) and out[1].shape == (0,)
    assert executor.counters["empty"] == 1
    assert executor.last_dispatch["row_count"] == executor.last_dispatch["table_width"] == 0
    assert executor.snapshot()["buckets"] == initial
    cache.allocate("long", 513)
    with pytest.raises(RuntimeError, match="uninitialized"):
        runner._recurrent(torch.zeros(1, model.config.hidden_size), ["long"], [0], [1])
    assert executor.status == "ready" and executor.snapshot()["buckets"] == initial
    for tracker in cache._allocations["long"].written[0]:
        tracker.prefix = 512
    # Seeded storage is finite; this checks routing and prefix completion, not numerical fidelity.
    runner._recurrent(torch.randn(1, model.config.hidden_size), ["long"], [0], [512])
    assert executor.last_dispatch["bucket_id"] is None
    assert executor.fallback_counts["table_width"] == 1
    cache.free("long")
    for name in "abcde":
        cache.allocate(name, 2)
    runner._recurrent(torch.randn(5, model.config.hidden_size), list("abcde"), [0] * 5, [0] * 5)
    assert executor.fallback_counts["live_count"] == 1
    assert executor.counters["prepared"] == 0
    assert executor.snapshot()["buckets"] == initial
    executor.close()


def test_request_publication_waits_for_gate_readback_and_commit(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=False)
    cache.allocate("a", 2)
    request = Request(
        "a",
        [3],
        SamplingParams(max_tokens=2, ignore_eos=True),
        stage=Stage.RECURRENT,
        hidden_state=model.prelude(torch.tensor([3]))[0],
    )
    initial = request.hidden_state
    events = []
    tolist = torch.Tensor.tolist
    commit = cache._commit_decode_traversal

    def readback(value):
        if value.dtype == torch.float32 and value.shape == (1,):
            events.append("readback")
        return tolist(value)

    def publish(ticket, *, completion_confirmed):
        assert request.hidden_state is initial
        assert events == ["readback"]
        events.append("commit")
        return commit(ticket, completion_confirmed=completion_confirmed)

    monkeypatch.setattr(torch.Tensor, "tolist", readback)
    monkeypatch.setattr(cache, "_commit_decode_traversal", publish)
    gates = runner.execute(SchedulerOutput(Stage.RECURRENT, [ScheduledItem(request)]))
    assert len(gates) == 1 and events == ["readback", "commit"]
    assert request.hidden_state is not initial and runner._decode_executor.ticket is None
    assert all(p.prefix == 1 for p in cache._allocations["a"].written[0])
    runner._close_recurrent_graph()


@pytest.mark.parametrize("confirmed", [False, True])
def test_partial_replay_failure_retains_primary_and_resources_until_safe_close(
    model, monkeypatch, confirmed
):
    cache = make_cache(model)
    runtime = fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    executor = runner._decode_executor
    cache.allocate("a", 2)
    before = prefixes(cache)
    primary = RuntimeError("replay primary")
    runtime.graphs[0].replay_failure = primary
    if not confirmed:
        runtime.main.failure = RuntimeError("stream secondary")
    with pytest.raises(RuntimeError) as raised:
        runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    assert raised.value is primary and prefixes(cache) == before
    assert executor.failure["completion_confirmed"] is confirmed
    assert executor.buckets[4]["metadata"].in_use is not confirmed
    with pytest.raises(RuntimeError):
        runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    if not confirmed:
        with pytest.raises(RuntimeError, match="stream secondary"):
            executor.close()
        assert not runtime.graphs[0].reset_done
        runtime.main.failure = None
    executor.close()
    assert (
        executor.status == "closed" and executor.failure["primary"]["message"] == "replay primary"
    )
    assert executor.failure["completion_confirmed"]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("confirmed", [False, True])
def test_engine_update_failure_preserves_commit_and_only_frees_after_completion(
    model, monkeypatch, error_type, confirmed
):
    engine = LLMEngine(model, cache_config=CacheConfig(num_blocks=160))
    cache, runner = engine.cache_manager, engine.model_runner
    runtime = fake_runtime(monkeypatch, cache)
    engine._enable_recurrent_graph(use_graphs=True)
    engine.add_request("a", [3], SamplingParams(max_tokens=3, ignore_eos=True))
    for _ in range(10):
        engine.step()
        if engine.last_schedule.stage == Stage.PRELUDE:
            break
    else:
        pytest.fail("tiny engine never reached decode")
    allocation = cache._allocations["a"]
    assert all(p.prefix == 1 for p in allocation.written[0])
    primary = error_type("update primary")

    def fail_update(batch, result):
        assert batch.stage == Stage.RECURRENT and len(result) == 1
        assert runner._decode_executor.ticket is None
        assert runner._graph_snapshot()["last_dispatch"]["ticket_state"] == "committed"
        assert all(p.prefix == 2 for p in allocation.written[0])
        if not confirmed:
            runtime.main.failure = RuntimeError("update completion secondary")
        raise primary

    monkeypatch.setattr(engine, "_update", fail_update)
    with pytest.raises(error_type) as raised:
        engine.step()
    assert raised.value is primary
    # Successful traversal prefixes are never rolled back by a later route failure.
    assert all(p.prefix == 2 for p in allocation.written[0])
    assert ("a" in cache._allocations) is not confirmed
    assert runner._graph_snapshot()["failure"]["completion_confirmed"] is confirmed
    with pytest.raises(RuntimeError):
        engine.step()
    if not confirmed:
        with pytest.raises(RuntimeError, match="quarantined"):
            engine.abort_request("a")
        runtime.main.failure = None
    runner._close_recurrent_graph()


@pytest.mark.parametrize("output", ["hidden_out", "gate_out"])
def test_setup_rejects_finite_wrong_replay_and_engine_close_releases_it(model, monkeypatch, output):
    engine = LLMEngine(model, cache_config=CacheConfig(num_blocks=160))
    runtime = fake_runtime(monkeypatch, engine.cache_manager)
    replay = FakeGraph.replay

    def corrupt(graph):
        replay(graph)
        engine.model_runner.decode_executor.buckets[4]["tensors"][output][1] += 0.01

    monkeypatch.setattr(FakeGraph, "replay", corrupt)
    with pytest.raises(RuntimeError, match="replay differs from warmup"):
        engine._enable_recurrent_graph(use_graphs=True)
    executor = engine.model_runner.decode_executor
    assert executor.status == "failed"
    assert not engine.cache_manager._allocations
    engine.close()
    engine.close()
    assert executor.status == "closed"
    assert all(ref() is None for ref in runtime.pool_refs)
    assert ("release_stream", "setup") in runtime.events


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("pool", "cache pool identity"),
        ("storage", "bucket storage identity"),
        ("view", "view metadata"),
    ],
)
def test_changed_bucket_ownership_rejected_before_replay(model, monkeypatch, mutation, message):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    executor = RecurrentGraphExecutor(model, cache, use_graphs=True)
    executor.setup()
    cache.allocate("a", 2)
    bucket = executor.buckets[4]
    if mutation == "pool":
        cache.key_cache = cache.key_cache.clone()
    elif mutation == "storage":
        bucket["tensors"]["hidden_in"] = bucket["tensors"]["hidden_in"].clone()
    else:
        bucket["metadata"].tensors["active"] = bucket["metadata"].tensors["active"].clone()
    replays = executor.counters["replays"]
    with pytest.raises(RuntimeError, match=message):
        executor.recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    assert executor.counters["replays"] == replays
    executor.close()


@torch.inference_mode()
def test_setup_checks_full_signatures_but_dispatch_only_checks_identity(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    executor = RecurrentGraphExecutor(model, cache, use_graphs=False)
    executor.setup()
    bucket = executor.buckets[4]
    original_shape = bucket["tensors"]["hidden_in"].shape
    bucket["tensors"]["hidden_in"].transpose_(0, 1)
    with pytest.raises(RuntimeError, match="storage signature"):
        executor._check_bucket(bucket, full=True)
    bucket["tensors"]["hidden_in"].transpose_(0, 1)
    assert bucket["tensors"]["hidden_in"].shape == original_shape
    cache.allocate("a", 2)
    monkeypatch.setattr(graph_module, "_signature", lambda _: pytest.fail("hot-path signature"))
    executor.recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    executor.close()


def test_empty_scheduled_recurrent_batch_never_starts_completion(model):
    runner = ModelRunner(model, make_cache(model))
    runner._enable_recurrent_graph(use_graphs=False)
    assert runner.execute(SchedulerOutput(Stage.RECURRENT, [])) == []
    assert runner.decode_executor.status == "ready"
    assert runner.decode_executor.counters["completed"] == 0
    runner._close_recurrent_graph()


def test_close_rejects_inflight_and_cleanup_errors_reach_executor(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    executor = runner.decode_executor
    cache.allocate("a", 2)
    executor.recurrent(
        torch.ones(1, model.config.hidden_size), ["a"], [0], [0], defer_completion=True
    )
    with pytest.raises(RuntimeError, match="close requires completed"):
        executor.close()
    assert executor.status == "in_flight" and executor.pool_owner is not None
    executor.settle_failure(RuntimeError("primary"))
    runner._record_cleanup_failure(RuntimeError("cleanup"))
    assert executor.failure["secondary"][-1]["message"] == "cleanup"
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.allocate("b", 2)
    executor.close()
