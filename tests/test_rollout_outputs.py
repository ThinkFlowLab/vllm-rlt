"""Rollout outputs: per-token logprobs, effective seeds and stop token IDs."""

import gc
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict, deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.reference import dense_reference
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.config import SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine, Peer
from vllm_rlt.request import RequestOutput, Stage
from vllm_rlt.serving.protocol import CompletionRequest
from vllm_rlt.worker.model_runner import ReadbackSlot, Submission
from vllm_rlt.worker.sampling import probabilities
from vllm_rlt.worker.speculative import SpeculativeResult

TRACES = {"a": [4, 1, 3, 2, 4, 1, 2, 3], "b": [4, 2, 1, 4, 3, 2, 1, 4]}  # Keyed by request ID.
EXIT_MODES = [  # (async scheduling, exit config, exit_threshold) for every CPU exit policy.
    pytest.param(False, ExitConfig("ouro"), 1.0, id="ouro-full-depth"),
    pytest.param(False, ExitConfig("ouro"), 0.5, id="ouro-early-exit"),
    pytest.param(False, ExitConfig("trace", depths_by_request=TRACES), 1.0, id="trace"),
    pytest.param(True, ExitConfig("ouro_delayed"), 0.3, id="async-ouro_delayed"),
    pytest.param(True, ExitConfig("random_lookahead"), 0.3, id="async-random_lookahead"),
    pytest.param(True, ExitConfig("trace", depths_by_request=TRACES), 1.0, id="async-trace"),
]
ASYNC_EXIT_MODES = [pytest.param(*p.values[1:], id=p.id) for p in EXIT_MODES if p.values[0]]
LOGPROBS_MODES = ["raw_logprobs", "processed_logprobs"]
# How request "a" ends: max_tokens, EOS (honored or ignored), stop IDs, both, abort.
FINISHES = "length eos stop_id ignored_eos eos_with_stop_ids stop_id_at_max eos_at_max abort"


def model(seed=123, device="cpu", dtype=torch.float32, **overrides):
    torch.manual_seed(seed)
    return OuroForCausalLM(tiny_ouro_config(**overrides)).to(device, dtype)


def rollout(**overrides):
    return SamplingParams(**{"ignore_eos": True, "logprobs": 0, **overrides})


def make_engine(m, *, asynchronous=False, exit_config=None, execution=None, **kwargs):
    """Small CPU engine by default; asynchronous engines default to ouro_delayed exits."""
    kwargs.setdefault("cache_config", CacheConfig(128, 2))
    kwargs.setdefault("scheduler_config", SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=3))
    execution = ExecutionConfig(async_scheduling=asynchronous, **(execution or {}))
    exit_config = exit_config or ExitConfig("ouro_delayed" if asynchronous else "ouro")
    return LLMEngine(m, execution_config=execution, exit_config=exit_config, **kwargs)


def check_aligned(output):
    n, logprobs = len(output.token_ids), output.logprobs
    assert len(output.exit_depths) == n
    assert (logprobs is None) == (output.sampling_params.logprobs is None)
    assert logprobs is None or len(logprobs) == n
    assert all(math.isfinite(value) and value <= 0 for value in logprobs or ())


def stream(engine, until=None, streamed=None):
    """Step until ``until()`` holds, else drain, checking every output's alignment. A drained
    LLMEngine (1000-step limit) holds zero used KV blocks; PD gets a 90 s wall-clock limit."""
    streamed = defaultdict(list) if streamed is None else streamed
    done = until or (lambda: not engine.has_unfinished_requests())
    local, deadline, steps = isinstance(engine, LLMEngine), time.monotonic() + 90, 0
    while not done():
        assert engine.has_unfinished_requests()
        assert (steps < 1000) if local else (time.monotonic() < deadline)
        steps += 1
        for output in engine.step():
            check_aligned(output)
            streamed[output.request_id].append(output)
    if until is None and local:
        assert engine.cache_manager.num_used_blocks == 0
    return streamed


def final(streamed):
    return {rid: outputs[-1] for rid, outputs in streamed.items()}


def assert_same(actual, expected, atol=None):
    """Same tokens and exit depths; logprobs equal, or within ``atol`` when given."""
    assert (actual.token_ids, actual.exit_depths) == (expected.token_ids, expected.exit_depths)
    logprobs = expected.logprobs
    if atol is not None and logprobs is not None:
        logprobs = pytest.approx(logprobs, abs=atol)
    assert actual.logprobs == logprobs


def hold_coda_delivery(monkeypatch):
    """Report CODA tickets unfinished, so only the one-output bound delivers them."""
    ready = Submission.ready
    monkeypatch.setattr(Submission, "ready", lambda t: t.batch.stage != Stage.CODA and ready(t))
    return ready


def lease_cpu_logprob_slots(monkeypatch, engine, count=8):
    """Return the logprob readback pool. On CPU, which has none, install ``count`` emulated
    slots and monkeypatch ``runner.submit`` to lease one per logprob ticket (CUDA: unchanged)."""
    runner, submit = engine.model_runner, engine.model_runner.submit
    if runner.device.type == "cuda":
        return runner.logprob_slots
    runner.logprob_slots = [ReadbackSlot(torch.empty(1)) for _ in range(count)]

    def leasing_submit(batch):
        ticket = submit(batch)
        if ticket.logprobs is not None:
            ticket.logprob_slot = runner._readback_slot(ticket.logprobs, runner.logprob_slots)
        return ticket

    monkeypatch.setattr(runner, "submit", leasing_submit)
    return runner.logprob_slots


def test_valid_rollout_params_are_normalized_and_hashable():
    params = SamplingParams(seed=None, logprobs=0, stop_token_ids=[5, 3, 5])
    assert (params.seed, params.logprobs, params.stop_token_ids) == (None, 0, (5, 3, 5))
    assert {params} == {replace(params, stop_token_ids=(5, 3, 5))}
    assert SamplingParams().logprobs is None and SamplingParams(seed=2**63 - 1).seed == 2**63 - 1


def rejected(field, match, *values):
    return [pytest.param(field, value, match, id=f"{field}={value!r}") for value in values]


@pytest.mark.parametrize(
    "field,value,match",
    rejected("logprobs", "not supported yet", 1, 5)
    + rejected("logprobs", "None or 0", -1, True, 0.0, "0")
    + rejected("seed", "seed", True, -1, 1.5, "1")
    + [pytest.param("seed", 2**63, "seed", id="seed=2**63")]
    + rejected("stop_token_ids", "stop_token_ids", None, 3, "3", [True], [1.0], [-1], {3}),
)
def test_invalid_rollout_params_are_rejected(field, value, match):
    with pytest.raises(ValueError, match=match):
        SamplingParams(**{field: value})


def test_logprobs_mode_is_validated_and_forwarded_to_pd_workers(monkeypatch):
    process = Mock(side_effect=RuntimeError("stop before workers"))
    context = SimpleNamespace(Pipe=mp.Pipe, Process=process)
    monkeypatch.setattr("vllm_rlt.pd.engine.mp.get_context", lambda method: context)
    for logprobs_mode in LOGPROBS_MODES:
        llm = LLM(model(), logprobs_mode=logprobs_mode)
        for engine in (LLMEngine(model(), logprobs_mode=logprobs_mode), llm.engine):
            assert engine.model_runner.logprobs_mode == logprobs_mode
        with pytest.raises(RuntimeError, match="stop before workers"):
            PDEngine(tiny_ouro_config(), logprobs_mode=logprobs_mode)
    for logprobs_mode in ("raw_logits", "processed_logits", "logprobs", None):
        for build, source in ((LLMEngine, model), (LLM, model), (PDEngine, tiny_ouro_config)):
            with pytest.raises(ValueError, match="not supported"):
                build(source(), logprobs_mode=logprobs_mode)
    # Valid modes reach the PD worker engines; invalid ones fail before any worker starts.
    forwarded = [c.kwargs["args"][5]["engine"]["logprobs_mode"] for c in process.call_args_list]
    assert forwarded == LOGPROBS_MODES


def test_admission_rejects_out_of_vocabulary_stop_ids_and_speculative_logprobs():
    engine = LLMEngine(model())
    for stop in ([64], [3, 1000]):  # The tiny vocabulary is IDs 0..63.
        with pytest.raises(ValueError, match="stop_token_ids"):
            engine.add_request("bad", [1], SamplingParams(stop_token_ids=stop))
    speculative = LLMEngine(model(), speculative_config=SpeculativeConfig(3))
    with pytest.raises(ValueError, match="speculative"):
        speculative.add_request("r", [2], SamplingParams(logprobs=0))
    assert not engine.has_unfinished_requests() and not speculative.has_unfinished_requests()
    engine.add_request("ok", [1], SamplingParams(max_tokens=2, stop_token_ids=[63], logprobs=0))
    assert final(stream(engine))["ok"].finished


def test_engine_rejects_missing_logprob_for_requesting_row():
    engine = LLMEngine(model())
    engine.add_request("r", [2], SamplingParams(logprobs=0))
    with pytest.raises(RuntimeError, match="missing logprob"):
        engine._append_output(engine.scheduler.requests["r"], 3, 4, None)


def test_http_request_fields_are_unchanged():
    # tests/test_serving.py covers the other rollout fields.
    for value in (0, True):
        with pytest.raises(ValueError, match="logprobs"):
            CompletionRequest.parse({"model": "m", "prompt": "a", "logprobs": value}, "m")
    assert CompletionRequest.parse({"model": "m", "prompt": "a", "seed": 7}, "m").params.seed == 7


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize("finish", FINISHES.split())
@pytest.mark.parametrize("asynchronous,exit_config,threshold", EXIT_MODES)
def test_logprobs_are_defined_and_aligned_through_each_finish(
    monkeypatch, asynchronous, exit_config, threshold, finish, logprobs_mode
):
    rows = defaultdict(list)  # CODA logits rows in token order; the baseline runs first.
    common = dict(min_loops=1, exit_threshold=threshold)
    options = dict(asynchronous=asynchronous, exit_config=exit_config, logprobs_mode=logprobs_mode)
    # The unrequested row exercises mixed batches (NaN or ignored values). Admitted first, it
    # puts "a" behind row 0 in shared batches, so each row must read its own logprob.
    unrequested = rollout(max_tokens=6, temperature=0.9, seed=5, logprobs=None, **common)

    def run(params, *, eos=0, abort=False):
        engine = make_engine(model(eos_token_id=eos), **options)
        runner, sample = engine.model_runner, engine.model_runner._sample_tensor
        if params.logprobs is None:  # No logprob work when no request asks.
            unexpected = AssertionError("logprobs computed without a request")
            monkeypatch.setattr(runner, "_logprobs", Mock(side_effect=unexpected))

        def record(row, request):
            rows[request.request_id].append(row.detach().clone())
            return sample(row, request)

        runner._sample_tensor = record
        engine.add_request("b", [4], unrequested)
        engine.add_request("a", [2, 3], replace(params, **common))
        if not abort:
            return stream(engine)
        request = engine.scheduler.requests["a"]
        streamed = stream(engine, lambda: len(request.generated_token_ids) >= 2)
        streamed["a"].append(engine.abort_request("a"))
        return stream(engine, streamed=streamed)

    params = rollout(max_tokens=6, temperature=1.0, top_k=20, top_p=0.9, seed=7)
    baseline = final(run(params))["a"]
    tokens = baseline.token_ids
    assert baseline.finish_reason == "length" and len(rows["a"]) == len(tokens) == 6
    # Raw scores the token's CODA row; processed, the distribution the sampler draws from.
    processed = logprobs_mode == "processed_logprobs"
    expected = [
        (probabilities(row, params).log() if processed else row.float().log_softmax(-1))[t].item()
        for row, t in zip(rows["a"], tokens)
    ]
    assert baseline.logprobs == pytest.approx(expected, abs=1e-6)
    # Every policy except fixed full depth must exercise shallow exits here.
    shallow = threshold < 1 or exit_config.mode == "trace"
    assert any(depth < 4 for depth in baseline.exit_depths) == shallow
    # Asking for logprobs changes no token or exit depth.
    quiet = final(run(replace(params, logprobs=None)))["a"]
    assert (quiet.token_ids, quiet.exit_depths) == (tokens, baseline.exit_depths)
    # Stop on the first token that is new at index >= 2, so a stop must cut the stream.
    index = next(i for i, t in enumerate(tokens) if i >= 2 and t not in tokens[:i])
    stop, other = tokens[index], next(t for t in range(64) if t not in tokens)
    # finish -> (EOS ID, overrides, reason, length); stop IDs, unlike EOS, override ignore_eos.
    honor, cut = dict(ignore_eos=False), dict(max_tokens=index + 1)
    cases = {
        "length": (0, {}, "length", 6),
        "eos": (stop, honor, "stop", index + 1),
        "stop_id": (0, dict(stop_token_ids=[other, stop]), "stop", index + 1),
        "ignored_eos": (stop, {}, "length", 6),
        "eos_with_stop_ids": (stop, dict(honor, stop_token_ids=[other]), "stop", index + 1),
        "stop_id_at_max": (0, dict(cut, stop_token_ids=[stop]), "stop", index + 1),
        "eos_at_max": (stop, dict(honor, **cut), "stop", index + 1),
        "abort": (0, {}, "abort", None),  # After two delivered tokens.
    }
    eos, overrides, reason, length = cases[finish]
    outputs = run(replace(params, **overrides), eos=eos, abort=finish == "abort")["a"]
    n = len(outputs[-1].token_ids)
    assert outputs[-1].finish_reason == reason and (2 <= n < 6 if length is None else n == length)
    if finish != "abort":  # One token per output; a stop token keeps its depth and logprob.
        assert [len(o.token_ids) for o in outputs] == list(range(1, n + 1))
    for output in outputs:
        # Re-checked after draining: each output must own its lists, not the request's.
        check_aligned(output)
        k = len(output.token_ids)
        assert output.token_ids == tokens[:k] and output.exit_depths == baseline.exit_depths[:k]
        assert output.logprobs == baseline.logprobs[:k]


@pytest.mark.parametrize("layout", ["last_exited", "shared"])
def test_full_depth_raw_logprobs_vs_teacher_forced_reference(layout):
    m = model()
    engine = make_engine(m, cache_config=CacheConfig(128, 2, layout))
    prompts = {"a": [2, 3, 5], "s": [7]}
    engine.add_request("a", prompts["a"], rollout(max_tokens=6))
    engine.add_request("s", prompts["s"], rollout(max_tokens=6, temperature=0.8, seed=3))
    for rid, output in final(stream(engine)).items():
        assert output.exit_depths == [4] * 6
        tokens = prompts[rid] + output.token_ids
        rows = dense_reference(m, tokens, 4)[-1][2][len(prompts[rid]) - 1 : -1]
        expected = rows.float().log_softmax(-1)[torch.arange(6), torch.tensor(output.token_ids)]
        difference = (torch.tensor(output.logprobs) - expected).abs().max().item()
        # Shared KV reads earlier tokens' final-loop KV at every depth: no plain forward matches.
        assert difference < 2e-5 if layout == "last_exited" else difference > 1e-2


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_seed_none_is_resolved_at_admission_and_replays_exactly(asynchronous):
    params = rollout(max_tokens=8, temperature=1.0, seed=None)
    engine = make_engine(model(), asynchronous=asynchronous)
    torch_state, python_state = torch.get_rng_state(), random.getstate()
    for rid in ("x", "y", "unsampled"):
        engine.add_request(rid, [2, 3], params)
    # Seeds come from OS entropy, not from the torch or Python global RNG.
    assert torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == python_state
    seeds = {rid: r.sampling_params.seed for rid, r in engine.scheduler.requests.items()}
    assert all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds.values())
    # Every output, even an abort before the first sample, reports the effective params.
    outputs = {"unsampled": engine.abort_request("unsampled"), **final(stream(engine))}
    for rid, output in outputs.items():
        check_aligned(output)
        assert output.sampling_params == replace(params, seed=seeds[rid])
    assert len(set(seeds.values())) == 3 and outputs["x"].token_ids != outputs["y"].token_ids
    # Replaying the reported params reproduces the rollout exactly.
    replay = make_engine(model(), asynchronous=asynchronous)
    for rid in ("x", "y"):
        replay.add_request(rid, [2, 3], outputs[rid].sampling_params)
    for rid, output in final(stream(replay)).items():
        assert_same(output, outputs[rid])


def test_preemption_and_prefix_hits_keep_logprobs():
    params = rollout(max_tokens=7, temperature=0.8, seed=45, exit_threshold=0)
    results = []
    for suspend in (False, True):
        scheduler = SchedulerConfig(enable_preemption=True)
        engine = make_engine(model(), cache_config=CacheConfig(64, 2), scheduler_config=scheduler)
        engine.add_request("a", [1, 2, 3], params)
        request = engine.scheduler.requests["a"]
        stream(engine, lambda: len(request.generated_token_ids) >= 3)
        if suspend:
            before = list(request.logprobs)
            engine.add_request("b", [5], SamplingParams(max_tokens=1))
            engine.scheduler.selected_request_ids.clear()
            assert engine.preemption.preempt(engine.scheduler.requests["b"])
            assert request.logprobs == before and request.stage == Stage.WAITING
        results.append(final(stream(engine))["a"])
        assert engine.preemption.resumptions == int(suspend)
    assert_same(results[1], results[0])
    # Memory-pressure preemption matches an unpressured run.
    params, results = rollout(max_tokens=8, exit_threshold=0), []
    for blocks, preempt in ((128, False), (24, True)):
        cache = CacheConfig(blocks, 2, incremental_allocation=True)
        scheduler = SchedulerConfig(max_num_seqs=3, prefill_chunk_size=2, enable_preemption=preempt)
        engine = make_engine(model(15), cache_config=cache, scheduler_config=scheduler)
        for i in range(3):
            engine.add_request(str(i), [1, 2], params)
        results.append(final(stream(engine)))
        assert (engine.preemption.preemptions > 0) == preempt
    for rid, output in results[0].items():
        assert_same(results[1][rid], output, atol=1e-6)
    # A prefix-cache hit matches the cold run.
    cache = CacheConfig(64, 2, enable_prefix_caching=True)
    scheduler = SchedulerConfig(prefill_chunk_size=2)
    engine = make_engine(model(15), cache_config=cache, scheduler_config=scheduler)
    engine.add_request("cold", [1, 2, 3, 4, 5], replace(params, max_tokens=4))
    cold = final(stream(engine))["cold"]
    engine.add_request("warm", [1, 2, 3, 4, 5], replace(params, max_tokens=4))
    engine.step()
    assert engine.last_schedule.items[0].token_start == 4
    assert_same(final(stream(engine))["warm"], cold, atol=1e-6)


@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize(
    "old,new", [(0, None), (None, 0)], ids=["logprobs-then-none", "none-then-logprobs"]
)
@pytest.mark.parametrize("exit_config,threshold", ASYNC_EXIT_MODES)
def test_lagging_delivery_abort_and_id_reuse_keep_logprobs(
    monkeypatch, exit_config, threshold, old, new, logprobs_mode
):
    sampling = dict(temperature=0.9, top_p=0.95, seed=7, min_loops=1, exit_threshold=threshold)
    params = rollout(max_tokens=6, **sampling)  # T != 1 and top-p: processed differs from raw.
    options = dict(exit_config=exit_config, logprobs_mode=logprobs_mode)

    def fresh(*, asynchronous):
        engine = make_engine(model(), asynchronous=asynchronous, **options)
        engine.add_request("a", [5, 6], replace(params, logprobs=new))
        return final(stream(engine))["a"]

    expected = fresh(asynchronous=True)
    if exit_config.mode == "trace":  # Asynchronous delivery matches the synchronous path.
        assert_same(fresh(asynchronous=False), expected, atol=1e-6)
    hold_coda_delivery(monkeypatch)
    engine = make_engine(model(), asynchronous=True, **options)
    slots = lease_cpu_logprob_slots(monkeypatch, engine)
    engine.add_request("a", [2, 3], replace(params, logprobs=old))
    engine.add_request("b", [4], replace(params, seed=5))
    request = engine.scheduler.requests["a"]
    # Step until the request has delivered a token and has one undelivered CODA output.
    stream(engine, lambda: request.generated_token_ids and request.num_output_placeholders == 1)
    delivered = len(request.generated_token_ids)
    aborted = engine.abort_request("a")
    check_aligned(aborted)
    # The undelivered token is not part of the abort output.
    assert aborted.finish_reason == "abort" and len(aborted.token_ids) == delivered
    # Reuse the ID with the opposite setting while the stale ticket (and lease) is pending.
    assert engine._pending_coda and (old is None or any(slot.leased for slot in slots))
    engine.add_request("a", [5, 6], replace(params, logprobs=new))
    assert_same(final(stream(engine))["a"], expected, atol=1e-6)
    assert not engine._pending_coda and not any(slot.leased for slot in slots)


@pytest.mark.parametrize(
    "finish,max_tokens,count",
    [("round", 9, 8), ("stop_token_ids", 9, 4), ("eos", 9, 5), ("length", 4, 3)],
    ids=["length-at-round-end", "stop-id-mid-round", "eos-mid-round", "length-mid-round"],
)
def test_speculative_update_slices_logprobs(monkeypatch, finish, max_tokens, count):
    engine = LLMEngine(model(), speculative_config=SpeculativeConfig(3))
    stops = [50] if finish == "stop_token_ids" else []
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=finish != "eos", stop_token_ids=stops)
    engine.add_request("r", [2, 3], params)
    request = engine.scheduler.requests["r"]
    # Stand in for admission once the speculative runner computes logprobs.
    request.sampling_params, request.logprobs = replace(params, logprobs=0), []
    # Each round accepts all but its last draft token; token t has logprob -1 - t / 100.
    rounds = ([10, 11], [12, 50, 0, 13], [14, 15], [16, 17])
    results = deque(SpeculativeResult(t, len(t) - 1, 3, [-1 - i / 100 for i in t]) for t in rounds)
    runner = engine.speculative_runner
    monkeypatch.setattr(runner, "execute", lambda batch: [results.popleft() for _ in batch.items])
    result = stream(engine)["r"][-1]
    emitted = [10, 11, 12, 50, 0, 13, 14, 15][:count]
    reason = "length" if finish in ("round", "length") else "stop"
    assert (result.token_ids[1:], result.finish_reason) == (emitted, reason)
    assert result.exit_depths == [4] * len(result.token_ids)
    assert result.logprobs[1:] == [-1 - t / 100 for t in emitted]
    assert math.isfinite(result.logprobs[0])
    assert runner.stats.committed_tokens == len(emitted)


def test_pd_coordinator_resolves_seeds_and_mirrors_logprobs():
    engine = object.__new__(PDEngine)  # Coordinator only: no worker processes.
    engine.closed, engine.failure = False, None
    engine.requests, engine.transfers, engine.outputs = {}, {}, deque()
    engine.config, engine.exit_config = PDConfig(), ExitConfig("ouro_delayed")
    engine.model, engine.cache_manager = SimpleNamespace(config=tiny_ouro_config()), Mock()
    info, roles = dict(block_size=4, depths=4, num_blocks=1024), ("prefill", "decode")
    engine.peers = {role: Peer(role, role, None, Mock(), info=info) for role in roles}
    with pytest.raises(ValueError, match="stop_token_ids"):
        engine.add_request("bad", [1], SamplingParams(stop_token_ids=[64]))
    # P and D get these effective params (one resolved seed), which an early abort reports.
    for rid in ("unsampled", "r"):
        engine.add_request(rid, [1, 2], rollout(max_tokens=8, temperature=1.0, seed=None))
    unsampled = engine.requests["unsampled"].sampling_params
    assert engine.abort_request("unsampled").sampling_params == unsampled
    request = engine.requests["r"]
    assert type(unsampled.seed) is int and type(request.sampling_params.seed) is int
    assert request.logprobs == []
    w = next(iter(engine.transfers.values()))
    w.p, w.d, w.phase = "prefill", "decode", "decode"
    for n in (1, 2):
        mirrored = dict(logprobs=[-0.5] * n, sampling_params=request.sampling_params)
        output = RequestOutput(w.tid, [1, 2], [5] * n, [4] * n, False, **mirrored)
        engine._message(engine.peers["decode"], dict(kind="output", tid=w.tid, output=output))
    assert [o.request_id for o in engine.outputs] == ["r", "r"]
    assert engine.outputs[-1].logprobs == [-0.5, -0.5]
    aborted = engine.abort_request("r")
    assert aborted.finish_reason == "abort" and aborted.sampling_params == request.sampling_params
    assert (aborted.token_ids, aborted.exit_depths) == ([5, 5], [4, 4])
    assert aborted.logprobs == [-0.5, -0.5]


def gpu_engine(
    m, *, asynchronous, exit_config=None, seqs=3, logprobs_mode="raw_logprobs", **execution
):
    scheduler = SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=8, prefill_chunk_size=4)
    options = dict(cache_config=CacheConfig(128, 16), scheduler_config=scheduler)
    options |= dict(logprobs_mode=logprobs_mode, attention_backend="triton", execution=execution)
    exit_config = exit_config or ExitConfig("ouro_delayed")
    return make_engine(m, asynchronous=asynchronous, exit_config=exit_config, **options)


def gpu_params(rid, **overrides):
    values = dict(max_tokens=6 + int(rid), temperature=0.8, top_k=16, seed=40 + int(rid))
    return rollout(**(values | dict(min_loops=2, exit_threshold=0.2) | overrides))


@pytest.mark.gpu
@pytest.mark.parametrize("logprobs_mode", LOGPROBS_MODES)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_gpu_cuda_graph_logprobs_match_eager(asynchronous, logprobs_mode):
    # Static eager pads RECURRENT rows to a power of two while graph replay runs len(rows),
    # so GEMM reductions may round; FP32 keeps tokens and exits exact and bounds logprobs.
    m = model(321, "cuda", torch.float32, head_dim=64)
    results = []
    for graphs in (False, True):
        # One stream (no prefer_recurrent overlap): batches cannot depend on graph timing.
        static = dict(static_buffers=True, pad_to_power_of_two=True, multi_stream=False)
        options = dict(asynchronous=asynchronous, logprobs_mode=logprobs_mode, cuda_graphs=graphs)
        engine = gpu_engine(m, **options, **static)
        for i in range(3):
            engine.add_request(str(i), [2 + i, 3, 4] * (i + 1), gpu_params(str(i)))
        results.append(final(stream(engine)))
        assert not graphs or engine.model_runner.graphs.replays > 0
    eager, graphed = results
    assert graphed.keys() == eager.keys()
    for rid, output in eager.items():
        assert_same(graphed[rid], output, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True], ids=["one-stream", "multi-stream"])
def test_gpu_async_readback_matches_sync_and_returns_leases(monkeypatch, multi_stream):
    # FP32 keeps sampled tokens stable across sync/async batch compositions.
    m = model(321, "cuda", torch.float32, head_dim=64)
    traces = {str(i): [4, 1, 3, 2, 4, 1, 2, 3, 4, 2, 1, 3] for i in range(4)}
    options = dict(exit_config=ExitConfig("trace", depths_by_request=traces), seqs=4)

    def run(*, asynchronous):
        engine = gpu_engine(m, asynchronous=asynchronous, multi_stream=multi_stream, **options)
        for i in range(4):
            params = gpu_params(str(i), max_tokens=12, min_loops=1, exit_threshold=1.0)
            engine.add_request(str(i), [2, 3 + i], params)
        return engine, final(stream(engine))

    _, expected = run(asynchronous=False)
    # Delivery lags to the one-output bound: logprob leases overlap live exit-score tickets.
    hold_coda_delivery(monkeypatch)
    engine, actual = run(asynchronous=True)
    assert not any(slot.leased for slot in engine.model_runner.logprob_slots)
    for rid, output in expected.items():
        assert_same(actual[rid], output, atol=1e-4)


@pytest.mark.gpu
def test_gpu_delayed_exit_scores_and_logprobs_do_not_exhaust_readback_pools(monkeypatch):
    engine = gpu_engine(model(5, "cuda", torch.bfloat16, head_dim=64), asynchronous=True, seqs=4)
    assert len(engine.model_runner.logprob_slots) == 4 + 4
    # Lagging delivery holds a logprob lease per request while delayed exit scores hold theirs.
    hold_coda_delivery(monkeypatch)
    for round_index in range(3):
        for i in range(4):
            engine.add_request(f"{round_index}-{i}", [2, 3, i], gpu_params(str(i), max_tokens=16))
        assert len(final(stream(engine))) == 4
    assert not any(slot.leased for slot in engine.model_runner.logprob_slots)


@pytest.mark.gpu
def test_gpu_abort_with_pending_coda_returns_lease_and_reuses_id(monkeypatch):
    engine = gpu_engine(model(5, "cuda", torch.bfloat16, head_dim=64), asynchronous=True)
    slots = engine.model_runner.logprob_slots
    ready = hold_coda_delivery(monkeypatch)
    for i in range(2):
        engine.add_request(str(i), [2, 3, i], gpu_params(str(i), max_tokens=12))
    request = engine.scheduler.requests["0"]
    # Step until the request has delivered a token and has one undelivered CODA output.
    stream(engine, lambda: request.generated_token_ids and request.num_output_placeholders == 1)
    delivered = len(request.generated_token_ids)
    aborted = engine.abort_request("0")
    check_aligned(aborted)
    assert len(aborted.token_ids) == delivered and any(slot.leased for slot in slots)
    stale = [t for t in engine._pending_coda if any(i.request is request for i in t.batch.items)]
    assert stale
    # release() synchronized the aborted request, so the next step collects its ticket.
    monkeypatch.setattr(Submission, "ready", ready)
    engine.add_request("0", [2, 3], gpu_params("0", logprobs=None))
    for output in engine.step():
        check_aligned(output)
    assert not any(t is s for t in engine._pending_coda for s in stale)
    outputs = final(stream(engine))
    assert outputs["0"].logprobs is None and len(outputs["0"].token_ids) == 6
    assert len(outputs["1"].logprobs) == 12
    assert not engine._pending_coda and not any(slot.leased for slot in slots)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_execution_error_returns_undelivered_logprob_leases(monkeypatch, device):
    cuda = device == "cuda"
    m = model(5, "cuda", torch.bfloat16, head_dim=64) if cuda else model(5)
    engine = gpu_engine(m, asynchronous=True, seqs=4) if cuda else make_engine(m, asynchronous=True)
    slots = lease_cpu_logprob_slots(monkeypatch, engine)
    runner = engine.model_runner
    ready, execute = hold_coda_delivery(monkeypatch), runner._execute
    held = []

    def fail(batch, prepared=None):
        result = execute(batch, prepared)
        if batch.stage == Stage.RECURRENT and engine._pending_coda:
            held.append(sum(slot.leased for slot in slots))
            raise RuntimeError("injected failure after CODA submission")
        return result

    monkeypatch.setattr(runner, "_execute", fail)
    for i in range(3):
        engine.add_request(str(i), [2, 3, i], gpu_params(str(i), max_tokens=8))
    with pytest.raises(RuntimeError, match="injected"):
        stream(engine)
    # The failure path drops undelivered tickets without collect(); __del__ returns leases.
    assert held and held[0] > 0 and not engine._pending_coda
    gc.collect()
    assert not any(slot.leased for slot in slots)
    monkeypatch.setattr(runner, "_execute", execute)
    monkeypatch.setattr(Submission, "ready", ready)
    engine.add_request("3", [2, 3], gpu_params("3", max_tokens=4))
    check_aligned(final(stream(engine))["3"])


@pytest.mark.gpu
def test_gpu_pd_forwards_logprobs_and_abort_stays_aligned():
    pytest.importorskip("nixl")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 visible GPUs")
    exit_config = ExitConfig("trace", depths_by_request={"t": [4, 2, 3, 1] * 16})
    prefill = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=5, prefill_chunk_size=3)
    transfer = dict(transfer_chunk_bytes=4096, max_inflight_bytes=16384)
    caches = dict(prefill_cache_config=CacheConfig(128, 4), decode_cache_config=CacheConfig(128, 4))
    with PDEngine(
        tiny_ouro_config(head_dim=64),
        pd_config=PDConfig(**transfer, request_timeout=90, startup_timeout=90),  # GPUs 0, 1.
        prefill_scheduler_config=prefill,
        decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
        exit_config=exit_config,
        execution_config=ExecutionConfig(async_scheduling=True),
        attention_backend="triton",
        seed=123,
        **caches,
    ) as e:
        m = model(123, "cuda:0", torch.bfloat16, head_dim=64)
        options = dict(cache_config=CacheConfig(128, 4), scheduler_config=prefill)
        reference = make_engine(m, exit_config=exit_config, attention_backend="triton", **options)
        params = rollout(max_tokens=6, min_loops=1, temperature=0.7, top_k=32, seed=947)
        for target in (e, reference):
            target.add_request("r", [1, 2, 3, 4, 5], params, trace_id="t")
        assert_same(final(stream(e))["r"], final(stream(reference))["r"], atol=5e-2)
        e.add_request("abort", [1, 2, 3, 4, 5], replace(params, max_tokens=64), trace_id="t")
        delivered = defaultdict(list)
        stream(e, lambda: len(delivered["abort"]) >= 2, delivered)
        output = e.abort_request("abort")
        check_aligned(output)
        assert output.token_ids == delivered["abort"][-1].token_ids
        assert output.logprobs == delivered["abort"][-1].logprobs
        stream(e)
