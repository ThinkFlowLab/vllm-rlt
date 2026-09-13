"""Eager padding must preserve live KV, request ownership and publication."""

import pytest
import torch

from vllm_lt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Request, Stage
from vllm_lt.worker.model_runner import ModelRunner


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inactive-row CPU tests must not query or initialize CUDA")

    for name in ("is_available", "device_count", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def make_cache(**overrides):
    values = dict(
        num_layers=2, num_kv_heads=1, head_dim=4, num_blocks=40, block_size=2, max_loops=4
    )
    values.update(overrides)
    cache = KVCacheManager(**values)
    cache.key_cache.fill_(71)
    cache.value_cache.fill_(-83)
    return cache


def model_cache(model, **overrides):
    return make_cache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        **overrides,
    )


def test_padding_metadata_has_one_host_map_and_invalid_inactive_addresses():
    cache = make_cache()
    cache.allocate("a", 5)
    cache.allocate("b", 3)
    compact = cache._prepare_batch(["a", "b"], [3, 1], [4, 2])
    indices = [3, 1]
    padded = cache._pad_prepared(compact, row_indices=indices, row_count=5, table_width=4)
    indices[:] = [0, 2]
    assert compact.active is None and compact.live_rows == (0, 1)
    assert padded.rows is compact.rows and padded.allocations is compact.allocations
    assert padded.row_count == 5 and padded.live_rows == (3, 1)
    assert padded.active.dtype == torch.bool
    assert padded.active.tolist() == [False, True, False, True, False]
    assert padded.position_ids.tolist() == [0, 2, 0, 4, 0]
    assert padded.context_lengths.tolist() == [0, 3, 0, 5, 0]
    for row in (0, 2, 4):
        assert padded.write_blocks[row] == padded.write_offsets[row] == -1
        assert padded.block_tables[row].tolist() == [-1] * 4
    assert all(
        written.prefix == 0 and not written.pending
        for allocation in cache._allocations.values()
        for layers in allocation.written
        for written in layers
    )


@pytest.mark.parametrize(
    "indices,rows,width",
    [
        ([0, 0], 4, 3),
        ([0, 4], 4, 3),
        ([-1, 2], 4, 3),
        ([True, 2], 4, 3),
        ([0.0, 2], 4, 3),
        ([0], 4, 3),
        ([0, 2], -1, 3),
        ([0, 2], True, 3),
        ([0, 2], 4, 2),
        ([0, 2], 4, -1),
        ([0, 2], 4, 3.0),
    ],
)
def test_invalid_layout_fails_without_cache_mutation(indices, rows, width):
    cache = make_cache()
    cache.allocate("a", 5)
    batch = cache._prepare_batch(["a", "a"], [0, 1], [4, 0])
    keys, values = cache.key_cache.clone(), cache.value_cache.clone()
    with pytest.raises(ValueError):
        cache._pad_prepared(batch, row_indices=indices, row_count=rows, table_width=width)
    assert torch.equal(keys, cache.key_cache) and torch.equal(values, cache.value_cache)
    assert cache.read(0, "a", 0)[0].shape[0] == 0


@pytest.mark.parametrize(
    "live_rows,row_count", [((), 0), ((), 4), ((0,), 1), ((2,), 4), ((3, 1), 4), ((2, 0, 1), 3)]
)
def test_only_live_rows_write_and_attend_with_poisoned_padding(live_rows, row_count):
    torch.manual_seed(23)
    compact_cache, padded_cache = make_cache(), make_cache()
    ids = [f"r{i}" for i in range(len(live_rows))]
    for cache in (compact_cache, padded_cache):
        for name in ids:
            cache.allocate(name, 2)
    plain = compact_cache._prepare_batch(ids, [0] * len(ids), [0] * len(ids))
    padded = padded_cache._pad_prepared(
        padded_cache._prepare_batch(ids, [0] * len(ids), [0] * len(ids)),
        row_indices=live_rows,
        row_count=row_count,
        table_width=2,
    )
    keys, values, query = (
        torch.randn(len(ids), 1, 4),
        torch.randn(len(ids), 1, 4),
        torch.randn(len(ids), 2, 4),
    )
    pk, pv, pq = (torch.full((row_count, heads, 4), torch.nan) for heads in (1, 1, 2))
    for physical, k, v, q in zip(live_rows, keys, values, query):
        pk[physical], pv[physical], pq[physical] = k, v, q
    inactive = [row for row in range(row_count) if row not in live_rows]
    # Both negative and positive out-of-range addresses must stay unconsumed.
    if inactive:
        padded.write_blocks[inactive[::2]] = padded_cache.num_blocks + 99
        padded.write_offsets[inactive[::2]] = padded_cache.block_size + 99
        padded.block_tables[inactive[::2]] = padded_cache.num_blocks + 99
    for layer in range(2):
        compact_cache._write_prepared(layer, plain, keys, values)
        padded_cache._write_prepared(layer, padded, pk, pv)
        expected = compact_cache._attend_prepared(layer, plain, query)
        actual = padded_cache._attend_prepared(layer, padded, pq)
        assert torch.equal(actual[list(live_rows)], expected)
        assert torch.equal(actual[inactive], torch.zeros_like(actual[inactive]))
        assert not torch.signbit(actual[inactive]).any()
        assert torch.equal(compact_cache.key_cache, padded_cache.key_cache)
        assert torch.equal(compact_cache.value_cache, padded_cache.value_cache)
    assert padded_cache.num_used_blocks == compact_cache.num_used_blocks


def test_fragmentation_mixed_depth_partial_pages_and_per_layer_history_survive_padding():
    torch.manual_seed(7)
    cache = make_cache(num_blocks=36)
    for name, capacity in (("left", 4), ("guard", 2), ("right", 4)):
        cache.allocate(name, capacity)
    cache.free("left")
    cache.free("right")
    cache.allocate("a", 5)
    cache.allocate("b", 3)
    assert any(
        table != tuple(range(table[0], table[0] + len(table)))
        for table in cache._allocations["a"].block_tables
    )
    guard = [page for table in cache._allocations["guard"].block_tables for page in table]
    before = cache.key_cache[guard].clone(), cache.value_cache[guard].clone()
    # Mixed request/depth prefix fill crosses two block boundaries.
    ids, depths, positions = ["a"] * 4 + ["b"] * 2, [3] * 4 + [1] * 2, [0, 1, 2, 3, 0, 1]
    history = torch.randn(6, 1, 4)
    cache.write(0, ids, depths, positions, history, history + 2)
    cache.write(1, ids[:-1], depths[:-1], positions[:-1], history[:-1], history[:-1] + 2)
    batch = cache._pad_prepared(
        cache._prepare_batch(["b", "a"], [1, 3], [2, 4]),
        row_indices=[3, 1],
        row_count=5,
        table_width=4,
    )
    x = torch.full((5, 1, 4), torch.nan)
    x[[3, 1]] = torch.randn(2, 1, 4)
    cache._write_prepared(0, batch, x, x)
    assert torch.isfinite(cache._attend_prepared(0, batch, x)).all()
    cache._write_prepared(1, batch, x, x)
    with pytest.raises(RuntimeError, match="layer 1.*depth 1"):
        cache._attend_prepared(1, batch, x)
    cache.write(1, ["b"], [1], [1], history[-1:], history[-1:] + 2)
    assert torch.isfinite(cache._attend_prepared(1, batch, x)).all()
    cache.finalize_token("b", 2, 1)
    assert cache._allocations["b"].written[2][0].pending == {2}
    assert torch.equal(cache.key_cache[guard], before[0])
    assert torch.equal(cache.value_cache[guard], before[1])


@pytest.mark.parametrize("operation", ["write", "attend", "pad", "recurrent"])
def test_stale_padded_descriptor_cannot_touch_recycled_pages(operation):
    model = OuroForCausalLM(OuroConfig.tiny())
    cache = model_cache(model, num_blocks=4)
    cache.allocate("a", 2)
    batch = cache._pad_prepared(
        cache._prepare_batch(["a"], [0], [0]), row_indices=[1], row_count=3, table_width=1
    )
    cache.free("a")
    cache.allocate("a", 2)
    before = cache.key_cache.clone(), cache.value_cache.clone()
    tensor = torch.ones(3, cache.num_kv_heads, cache.head_dim)
    with pytest.raises(RuntimeError, match="stale"):
        if operation == "write":
            cache._write_prepared(0, batch, tensor, tensor)
        elif operation == "attend":
            cache._attend_prepared(0, batch, tensor)
        elif operation == "pad":
            cache._pad_prepared(batch, row_indices=[0], row_count=1, table_width=1)
        else:
            model._recurrent_prepared(torch.ones(3, model.config.hidden_size), batch, cache)
    assert torch.equal(cache.key_cache, before[0]) and torch.equal(cache.value_cache, before[1])
    assert cache.read(0, "a", 0)[0].shape[0] == 0


def test_padded_duplicate_queries_are_read_only_and_duplicate_writes_fail():
    cache = make_cache()
    cache.allocate("a", 2)
    x = torch.ones(1, 1, 4)
    cache.write(0, ["a"], [0], [0], x, x * 3)
    with pytest.raises(ValueError, match="duplicate"):
        cache._prepare_batch(["a", "a"], [0, 0], [0, 0])
    batch = cache._pad_prepared(
        cache._prepare_batch(["a", "a"], [0, 0], [0, 0], for_write=False),
        row_indices=[3, 1],
        row_count=4,
        table_width=1,
    )
    q = torch.full((4, 1, 4), torch.nan)
    q[[3, 1]] = 0
    out = cache._attend_prepared(0, batch, q)
    assert torch.equal(out[[3, 1]], torch.full((2, 1, 4), 3.0))
    with pytest.raises(ValueError, match="read-only"):
        cache._write_prepared(0, batch, q, q)


@pytest.mark.parametrize("row_count,live_rows", [(0, ()), (4, ()), (1, (0,)), (4, (2,))])
def test_prepared_model_sanitizes_poison_inputs_and_biased_inactive_gates(row_count, live_rows):
    torch.manual_seed(11)
    model = OuroForCausalLM(OuroConfig.tiny())
    model.model.early_exit_gate.bias.fill_(3)
    cache = model_cache(model)
    ids = ["a"] if live_rows else []
    if ids:
        cache.allocate("a", 2)
    batch = cache._pad_prepared(
        cache._prepare_batch(ids, [0] * len(ids), [0] * len(ids)),
        row_indices=live_rows,
        row_count=row_count,
        table_width=1,
    )
    hidden = torch.full((row_count, model.config.hidden_size), torch.nan)
    hidden[list(live_rows)] = 1
    observed_inputs = []
    hook = model.model.layers[0].register_forward_pre_hook(
        lambda module, args: observed_inputs.append(args[0].clone())
    )
    try:
        out, gates = model._recurrent_prepared(hidden, batch, cache)
    finally:
        hook.remove()
    inactive = [row for row in range(row_count) if row not in live_rows]
    assert torch.isfinite(out).all() and torch.isfinite(gates).all()
    assert torch.equal(out[inactive], torch.zeros_like(out[inactive]))
    assert torch.equal(gates[inactive], torch.zeros_like(gates[inactive]))
    assert not torch.signbit(out[inactive]).any() and not torch.signbit(gates[inactive]).any()
    if live_rows:
        assert torch.isfinite(observed_inputs[0]).all()
        assert torch.isnan(hidden[inactive]).all()
    else:
        assert not observed_inputs and cache.num_used_blocks == 0


def test_all_active_model_path_does_not_launch_padding_masks(monkeypatch):
    model = OuroForCausalLM(OuroConfig.tiny())
    cache = model_cache(model)
    cache.allocate("a", 2)
    hidden = model.prelude(torch.tensor([1]))

    def forbidden(*args, **kwargs):
        pytest.fail("all-active traversal launched an unnecessary mask")

    monkeypatch.setattr(torch.Tensor, "masked_fill", forbidden)
    model.recurrent(hidden, ["a"], [0], [0], cache)
    padded = cache._pad_prepared(
        cache._prepare_batch(["a"], [1], [0]), row_indices=[0], row_count=1, table_width=2
    )
    model._recurrent_prepared(hidden, padded, cache)


def test_runner_gathers_scheduler_order_without_touching_coda_or_rng(monkeypatch):
    torch.manual_seed(41)
    model = OuroForCausalLM(OuroConfig.tiny())
    cache, reference_cache = model_cache(model), model_cache(model)
    runner = ModelRunner(model, cache)
    requests = [
        Request(
            name,
            [token],
            SamplingParams(temperature=0.7, seed=i),
            stage=Stage.RECURRENT,
            hidden_state=model.prelude(torch.tensor([token]))[0],
        )
        for i, (name, token) in enumerate((("b", 7), ("a", 3)))
    ]
    held = Request(
        "held",
        [9],
        SamplingParams(),
        stage=Stage.CODA,
        hidden_state=torch.randn(model.config.hidden_size),
    )
    for request in [*requests, held]:
        cache.allocate(request.request_id, 2)
        request.generator = torch.Generator().manual_seed(19)
    for request in requests:
        reference_cache.allocate(request.request_id, 2)
    held_hidden = held.hidden_state.clone()
    generator_states = [request.generator.get_state().clone() for request in [*requests, held]]
    expected, expected_gates = model.recurrent(
        torch.stack([r.hidden_state for r in requests]),
        [r.request_id for r in requests],
        [0, 0],
        [0, 0],
        reference_cache,
    )
    original = runner._recurrent_padded
    physical_outputs = []
    core = model._recurrent_prepared

    def record_core(*args):
        result = core(*args)
        physical_outputs.append(result)
        return result

    monkeypatch.setattr(model, "_recurrent_prepared", record_core)
    monkeypatch.setattr(
        runner,
        "_recurrent",
        lambda *args: original(*args, row_indices=[3, 1], row_count=5, table_width=2),
    )
    actual_gates = runner.execute(
        SchedulerOutput(Stage.RECURRENT, [ScheduledItem(r) for r in requests])
    )
    torch.testing.assert_close(
        torch.stack([r.hidden_state for r in requests]), expected, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        torch.tensor(actual_gates), expected_gates.sigmoid(), atol=1e-6, rtol=1e-5
    )
    assert len(actual_gates) == 2
    with torch.inference_mode():
        physical_outputs[0][0].fill_(999)
    torch.testing.assert_close(
        torch.stack([r.hidden_state for r in requests]), expected, atol=1e-6, rtol=1e-5
    )
    assert torch.equal(held.hidden_state, held_hidden) and held.stage == Stage.CODA
    for request, before in zip([*requests, held], generator_states):
        assert torch.equal(request.generator.get_state(), before)
        assert request.loops_done == 0 and request.remaining_probability == 1.0
        assert request.generated_token_ids == [] and request.exit_depths == []
    assert cache.read(0, "held", 0)[0].shape[0] == 0
    # Recycle another request's pages and execute again while held stays in coda.
    cache.free("a")
    cache.allocate("a", 2)
    original(
        model.prelude(torch.tensor([11])),
        ["a"],
        [0],
        [0],
        row_indices=[4],
        row_count=5,
        table_width=2,
    )
    assert torch.equal(held.hidden_state, held_hidden) and held.stage == Stage.CODA
    assert torch.equal(held.generator.get_state(), generator_states[-1])
    assert cache.read(0, "held", 0)[0].shape[0] == 0


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_padded_generation_preserves_actual_histories_sampling_and_cleanup(mode, monkeypatch):
    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    results = []
    traces = []
    samples = []
    for padded in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(64, 2),
            scheduler_config=SchedulerConfig(mode=mode, max_num_batched_tokens=3),
        )
        if padded:
            monkeypatch.setattr(
                engine.model_runner,
                "_recurrent",
                lambda *args: engine.model_runner._recurrent_padded(
                    *args,
                    row_indices=[2 * i + 1 for i in range(len(args[1]))],
                    row_count=8,
                    table_width=4,
                ),
            )
        sample = engine.model_runner._sample
        sample_states = []

        def record_sample(logits, request):
            token = sample(logits, request)
            sample_states.append((request.request_id, token, request.generator.get_state().clone()))
            return token

        monkeypatch.setattr(engine.model_runner, "_sample", record_sample)
        for i, prompt in enumerate(([1, 2, 3], [7], [4, 5])):
            engine.add_request(
                str(i),
                prompt,
                SamplingParams(
                    max_tokens=4,
                    temperature=0.8,
                    top_k=9,
                    top_p=0.8,
                    seed=i + 5,
                    exit_threshold=(0, 1, 0.7)[i],
                    ignore_eos=True,
                ),
            )
        outputs, trace = {}, []
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = (output.token_ids, output.exit_depths)
            trace.append(
                (
                    engine.last_schedule.stage,
                    [item.request.request_id for item in engine.last_schedule.items],
                )
            )
        assert not engine.has_unfinished_requests() and engine.cache_manager.num_used_blocks == 0
        assert set(outputs) == {"0", "1", "2"}
        results.append(outputs)
        traces.append(trace)
        samples.append(sample_states)
    assert results[0] == results[1] and traces[0] == traces[1]
    assert len(samples[0]) == len(samples[1]) == 12
    for (aid, atoken, astate), (bid, btoken, bstate) in zip(*samples):
        assert (aid, atoken) == (bid, btoken) and torch.equal(astate, bstate)


def test_padded_two_to_four_transition_matches_compact_and_complete_populated_kv():
    torch.manual_seed(29)
    model = OuroForCausalLM(OuroConfig.tiny())
    cache = model_cache(model, num_blocks=32)
    reference_cache = model_cache(model, num_blocks=32)
    runner = ModelRunner(model, cache)
    prompt, inputs, depths = [1, 3, 5], [7, 9, 11], [2, 4, 3]
    for pool in (cache, reference_cache):
        pool.allocate("a", len(prompt) + len(inputs))
    hidden = model.prelude(torch.tensor(prompt))
    expected = hidden.clone()
    try:
        for depth in range(4):
            hidden, _ = model.recurrent(hidden, ["a"] * 3, [depth] * 3, [0, 1, 2], cache)
            expected, _ = model.recurrent(
                expected, ["a"] * 3, [depth] * 3, [0, 1, 2], reference_cache
            )
        hidden, expected = hidden[-1:], expected[-1:]
        for index in range(4):
            if index:
                hidden = model.prelude(torch.tensor([inputs[index - 1]]))
                expected = hidden.clone()
                for depth in range(depths[index - 1]):
                    hidden, _ = runner._recurrent_padded(
                        hidden,
                        ["a"],
                        [depth],
                        [index + 2],
                        row_indices=[3],
                        row_count=8,
                        table_width=4,
                    )
                    expected, _ = model.recurrent(
                        expected, ["a"], [depth], [index + 2], reference_cache
                    )
                for pool in (cache, reference_cache):
                    pool.finalize_token("a", index + 2, depths[index - 1] - 1)
            actual_logits, expected_logits = model.coda(hidden)[0], model.coda(expected)[0]
            torch.testing.assert_close(actual_logits, expected_logits, atol=0.001, rtol=0.0001)
            assert actual_logits.argmax() == expected_logits.argmax()
            for depth in range(4):
                for layer in range(cache.num_layers):
                    keys, values = cache.read(layer, "a", depth, len(prompt) + index)
                    expected_keys, expected_values = reference_cache.read(
                        layer, "a", depth, len(prompt) + index
                    )
                    torch.testing.assert_close(keys, expected_keys, atol=5e-6, rtol=1e-4)
                    torch.testing.assert_close(values, expected_values, atol=5e-6, rtol=1e-4)
    finally:
        cache.free("a")
        reference_cache.free("a")
    assert cache.num_used_blocks == reference_cache.num_used_blocks == 0
