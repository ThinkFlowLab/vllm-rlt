"""Deployment owners share execution semantics, not scheduler or KV state."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestStatus


def make_engine(*, role="mixed", atomic=False, asynchronous=False, mixed=False):
    torch.manual_seed(123)
    return StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        stage_config=StageConfig(role=role),
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            mode="mixed" if mixed else "refill",
            max_num_seqs=3,
            max_num_batched_tokens=4,
            prefill_chunk_size=3,
        ),
        execution_config=ExecutionConfig(
            atomic_prefill=atomic,
            async_scheduling=asynchronous,
            static_buffers=True,
        ),
        exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
    )


def drain(engine):
    outputs = {}
    for _ in range(500):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = (output.token_ids, output.exit_depths)
    pytest.fail("stage engine did not make progress")


class RecordingConnector:
    def __init__(self):
        self.blocked = set()
        self.submissions = []

    def exports(self, request):
        return True

    def can_schedule(self, request):
        return request.request_id not in self.blocked

    def prefill_submitted(self, request, **submission):
        self.submissions.append((request, submission))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mixed_keeps_one_resource_owner_across_all_phases(asynchronous):
    engine = make_engine(mixed=True, asynchronous=asynchronous)
    engine.add_request("r", [1, 2, 3, 4, 5], SamplingParams(max_tokens=3, ignore_eos=True))
    assert len(drain(engine)["r"][0]) == 3
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.state_slots


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_prefill_handoff_matches_local_execution_without_sampling(atomic, asynchronous):
    engine = make_engine(role="prefill", atomic=atomic, asynchronous=asynchronous)
    local = make_engine(atomic=atomic, asynchronous=asynchronous)
    connector = engine.connector = RecordingConnector()
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    for e in (engine, local):
        e.add_request("r", [1, 2, 3, 4, 5], params)
    request = engine.scheduler.requests["r"]
    original = local.scheduler.requests["r"]
    for e, r in ((engine, request), (local, original)):
        for _ in range(100):
            e.step()
            if r.num_prefilled_tokens == 5:
                break
        else:
            pytest.fail("prompt did not complete")
    assert request.status == RequestStatus.TRANSFERRING
    assert local.scheduler.pending_work(original) == WorkKey.DECODE_CODA
    assert request.generated_token_ids == []
    torch.testing.assert_close(request.hidden_state, original.hidden_state)
    assert all(r is request for r, _ in connector.submissions)
    completed = [s for _, s in connector.submissions if s["progress"].chunk_complete]
    assert [s["start"] for s in completed] == [0, 3]
    assert completed[-1]["progress"].prompt_complete
    assert engine.step() == []  # Await remote acknowledgement; never sample on P.
    engine.close()
    local.close()


def test_connector_backpressure_preserves_other_requests_and_queue_membership():
    engine = make_engine(role="prefill")
    connector = engine.connector = RecordingConnector()
    connector.blocked.add("a")
    for name in ("a", "b"):
        engine.add_request(name, [1, 2], SamplingParams(max_tokens=1))
    for _ in range(20):
        engine.step()
    assert engine.scheduler.requests["a"].num_preluded_tokens == 0
    assert engine.scheduler.requests["b"].status == RequestStatus.TRANSFERRING
    connector.blocked.clear()
    for _ in range(20):
        engine.step()
    assert engine.scheduler.requests["a"].status == RequestStatus.TRANSFERRING
    engine.close()


def test_cancelled_import_cannot_activate_reused_request_id():
    engine = make_engine(role="decode")
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=2))
    old = engine.scheduler.requests["r"]
    engine.abort_request("r")
    engine.add_request("r", [3], SamplingParams(max_tokens=2))
    current = engine.scheduler.requests["r"]
    assert not engine.activate_import(old, torch.zeros(1))
    assert current.status == RequestStatus.WAITING
    assert current.hidden_state is None
    engine.close()


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_independent_stage_resources_preserve_output_after_import(atomic, asynchronous):
    p = make_engine(role="prefill", atomic=atomic, asynchronous=asynchronous)
    d = make_engine(role="decode", atomic=atomic, asynchronous=asynchronous)
    local = make_engine(atomic=atomic, asynchronous=asynchronous)
    p.connector = RecordingConnector()
    prompt = [1, 2, 3, 4, 5]
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    source = p.reserve_transfer("r", prompt, params, receiving=False)
    target = d.reserve_transfer("r", prompt, params, receiving=True)
    assert source is not target
    assert p.scheduler is not d.scheduler
    assert p.cache_manager is not d.cache_manager
    assert p.model_runner is not d.model_runner
    assert d.step() == []
    for _ in range(100):
        p.step()
        if source.status == RequestStatus.TRANSFERRING:
            break
    else:
        pytest.fail("source prompt did not complete")
    # CPU transport simulation: copy payload only, never the live Request.
    # This checks import semantics, not NIXL or CUDA stream ordering.
    source_tables = p.cache_manager.plane_block_tables("r")
    target_tables = d.cache_manager.plane_block_tables("r")
    for left, right in zip(source_tables, target_tables):
        for src, dst in zip(left, right):
            d.cache_manager.key_cache[dst].copy_(p.cache_manager.key_cache[src])
            d.cache_manager.value_cache[dst].copy_(p.cache_manager.value_cache[src])
    assert d.activate_import(target, source.hidden_state.clone())
    assert d.scheduler.requests["r"] is target
    assert p.release_transfer(source)
    local.add_request("r", prompt, params)
    assert drain(d) == drain(local)
    assert p.cache_manager.num_used_blocks == d.cache_manager.num_used_blocks == 0
    for engine in (p, d, local):
        engine.close()


def test_transfer_reservation_reuses_identity_and_requires_retirement_before_id_reuse():
    engine = make_engine(role="decode")
    params = SamplingParams(max_tokens=2)
    request = engine.reserve_transfer("generation-1", [1, 2], params, receiving=True)
    assert engine.cache_manager.has_transfer_lease(request.request_id)
    with pytest.raises(ValueError, match="duplicate request ID"):
        engine.reserve_transfer("generation-1", [1, 2], params, receiving=True)
    assert engine.scheduler.requests[request.request_id] is request
    assert engine.release_transfer(request)
    newer = engine.reserve_transfer("generation-1", [3], params, receiving=True)
    assert newer is not request
    assert not engine.activate_import(request, torch.zeros(1))
    assert not engine.release_transfer(request)
    assert engine.cache_manager.has_transfer_lease(newer.request_id)
    assert engine.release_transfer(newer)
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


def test_connector_submission_failure_keeps_remote_resources_pinned(monkeypatch):
    engine = make_engine(role="prefill")
    engine.connector = RecordingConnector()
    request = engine.reserve_transfer("r", [1, 2], SamplingParams(max_tokens=1), receiving=False)

    def fail(batch):
        raise RuntimeError("submission failed")

    monkeypatch.setattr(engine.executor, "submit", fail)
    with pytest.raises(RuntimeError, match="submission failed"):
        engine.step()
    assert engine.scheduler.requests["r"] is request
    assert engine.cache_manager.has_transfer_lease("r")
    assert engine.release_transfer(request)  # Only after the owner stops remote access.
    engine.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mixed_stage_routes_per_request_and_keeps_local_decode_running(asynchronous):
    engine = make_engine(mixed=True, asynchronous=asynchronous)
    connector = engine.connector = RecordingConnector()
    connector.exports = lambda request: request.request_id == "remote"
    connector.blocked.add("remote")
    for rid in ("remote", "local"):
        engine.add_request(rid, [1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
    local_outputs = []
    for _ in range(100):
        local_outputs.extend(engine.step())
        if "local" not in engine.scheduler.requests:
            break
    assert local_outputs[-1].finished
    assert local_outputs[-1].request_id == "local"
    assert engine.scheduler.requests["remote"].num_prefilled_tokens == 0
    assert not connector.submissions  # Local loop/chunk boundaries never export.
    connector.blocked.clear()
    for _ in range(100):
        engine.step()
        if engine.scheduler.requests["remote"].status == RequestStatus.TRANSFERRING:
            break
    else:
        pytest.fail("remote request did not reach its handoff boundary")
    assert all(r.request_id == "remote" for r, _ in connector.submissions)
    assert engine.scheduler.requests["remote"].generated_token_ids == []
    engine.close()


def test_decode_capability_rejects_unreserved_prompt():
    engine = make_engine(role="decode")
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
    with pytest.raises(ValueError, match="reserve_transfer"):
        engine.step()
    assert not engine.cache_manager.has_allocation("r")
    engine.close()


def test_prefill_requires_explicit_export_route():
    engine = make_engine(role="prefill")
    engine.connector = RecordingConnector()
    engine.connector.exports = lambda request: False
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
    with pytest.raises(ValueError, match="outgoing route"):
        engine.step()
    assert engine.scheduler.requests["r"].num_prefilled_tokens == 0
    engine.close()


def test_prefill_does_not_hide_a_lost_runnable_request():
    engine = make_engine(role="prefill")
    engine.connector = RecordingConnector()
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
    engine.scheduler.waiting.clear()
    with pytest.raises(RuntimeError, match="no progress"):
        engine.step()
    engine.close()


def test_prefill_admission_waits_for_transfer_to_release_kv():
    torch.manual_seed(123)
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        stage_config=StageConfig(role="prefill"),
        cache_config=CacheConfig(4, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
    )
    engine.connector = RecordingConnector()
    engine.add_request("first", [1, 2], SamplingParams(max_tokens=1))
    engine.step()
    first = engine.scheduler.requests["first"]
    assert first.status == RequestStatus.TRANSFERRING
    engine.add_request("second", [1, 2], SamplingParams(max_tokens=1))
    assert engine.step() == []
    assert engine.scheduler.requests["second"].status == RequestStatus.WAITING
    engine.release_transfer(first)
    engine.step()
    assert engine.scheduler.requests["second"].status == RequestStatus.TRANSFERRING
    engine.close()
