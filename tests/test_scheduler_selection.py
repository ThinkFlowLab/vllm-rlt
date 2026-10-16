"""Selection protection is batch-local state, not Scheduler-global state.

Preemption must not suspend a request the Scheduler has already placed in the
batch under construction. That exclusion belongs to one batch construction, so
it must not survive the batch, reach a direct `take_works()` caller such as PD
prefill worker, or outlive the request whose ID it names.
"""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, RequestStatus


def model():
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config())


def engine(max_num_seqs=2, atomic_prefill=True, **scheduler_options):
    return LLMEngine(
        model(),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            enable_preemption=True,
            **scheduler_options,
        ),
        execution_config=ExecutionConfig(atomic_prefill=atomic_prefill),
    )


def params(max_tokens=5):
    return SamplingParams(max_tokens=max_tokens, ignore_eos=True)


def test_work_queue_is_the_scheduler_owned_queue():
    e = engine()
    e.add_request("r", [1, 2], params())
    e.scheduler._admit()
    request = e.scheduler.requests["r"]
    e.scheduler.work_queues[WorkKey.PROMPT_ATOMIC].clear()

    e.scheduler.enqueue_work(request, WorkKey.DECODE_RECURRENT)

    assert e.scheduler.pending_work(request) == WorkKey.DECODE_RECURRENT
    assert list(e.scheduler.work_queues[WorkKey.DECODE_RECURRENT]) == ["r"]
    batch = e.scheduler.take_works((WorkKey.DECODE_RECURRENT,))
    assert batch.requests[0].work == WorkKey.DECODE_RECURRENT


def test_enqueue_work_requires_a_registered_request_and_one_pending_item():
    e = engine()
    e.add_request("r", [1, 2], params())
    e.scheduler._admit()
    request = e.scheduler.requests["r"]
    e.scheduler.work_queues[WorkKey.PROMPT_ATOMIC].clear()

    e.scheduler.enqueue_work(request, WorkKey.DECODE_RECURRENT)

    with pytest.raises(ValueError, match="already has pending work"):
        e.scheduler.enqueue_work(request, WorkKey.DECODE_RECURRENT)
    with pytest.raises(ValueError, match="already has pending work"):
        e.scheduler.enqueue_work(request, WorkKey.PROMPT_ATOMIC)

    replacement = Request("r", [3, 4], params())
    replacement.status = RequestStatus.ACTIVE
    with pytest.raises(ValueError, match="registered request"):
        e.scheduler.enqueue_work(replacement, WorkKey.DECODE_RECURRENT)

    assert e.scheduler.pending_work(request) == WorkKey.DECODE_RECURRENT
    assert list(e.scheduler.work_queues[WorkKey.PROMPT_ATOMIC]) == []
    assert list(e.scheduler.work_queues[WorkKey.DECODE_RECURRENT]) == ["r"]


def test_take_works_does_not_select_a_request_from_two_queues():
    e = engine(max_num_seqs=2, atomic_prefill=False, mode="mixed")
    e.add_request("r", [1, 2], params())
    e.scheduler._admit()
    e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE].clear()
    e.scheduler.work_queues[WorkKey.DECODE_PRELUDE].append("r")
    e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE].append("r")

    batch = e.scheduler.take_works((WorkKey.DECODE_PRELUDE, WorkKey.PROMPT_PRELUDE))

    assert batch is not None
    assert len(batch.requests) == 1
    assert batch.requests[0].work == WorkKey.DECODE_PRELUDE
    assert not e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE]


def test_batch_selection_is_never_preempted_by_a_later_item(monkeypatch):
    """Selecting A must stop B's capacity failure from evicting A."""
    e = engine()
    for name in ("a", "b"):
        e.add_request(name, [1, 2], params())
    e.scheduler._admit()
    monkeypatch.setattr(e.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "b")
    batch = e.scheduler.take_works((WorkKey.PROMPT_ATOMIC,))
    assert [scheduled_request.request.request_id for scheduled_request in batch.requests] == ["a"]
    assert e.preemption.preemptions == 0
    assert e.scheduler.pending_work(e.scheduler.requests["a"]) is None


def test_direct_take_does_not_inherit_an_earlier_batch_selection(monkeypatch):
    """The PD prefill worker calls take_works() directly, without schedule()."""
    e = engine()
    for name in ("a", "b"):
        e.add_request(name, [1, 2], params())
    e.scheduler._admit()
    # A first batch selects both requests.
    assert e.scheduler.take_works((WorkKey.PROMPT_ATOMIC,)) is not None

    seen = []

    def spy(requester, *, priority_only=False, excluded=frozenset()):
        seen.append((requester.request_id, excluded))
        return False

    # The Scheduler holds the callback it was given, so replace that reference.
    monkeypatch.setattr(e.scheduler, "preempt_callback", spy)
    for name in ("a", "b"):
        e.scheduler.enqueue_work(e.scheduler.requests[name], WorkKey.PROMPT_ATOMIC)
    monkeypatch.setattr(e.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "b")
    assert e.scheduler.take_works((WorkKey.PROMPT_ATOMIC,)) is not None
    # Only "a", selected in this batch, is excluded. The first batch's
    # selection is not still in force, and the set is handed over frozen.
    assert seen == [("b", frozenset({"a"}))]


def test_request_selected_in_an_earlier_batch_is_still_preemptable():
    """A previously scheduled request remains a legitimate preemption victim."""
    e = engine(max_num_seqs=1)
    e.add_request("victim", [1, 2], params())
    e.step()
    assert e.scheduler.requests["victim"].status == RequestStatus.ACTIVE
    e.add_request("requester", [3, 4], params())
    assert e.preemption.preempt(e.scheduler.requests["requester"])
    assert "victim" in e.preemption.snapshots


def test_reused_request_id_is_not_protected_by_the_previous_request():
    """Cancelling a request must not leave its ID protected for a new one."""
    e = engine(max_num_seqs=1)
    e.add_request("r", [1, 2], params())
    e.step()
    e.abort_request("r")
    e.add_request("r", [3, 4], params())
    e.step()
    e.add_request("requester", [5, 6], params())
    assert e.preemption.preempt(e.scheduler.requests["requester"])
    assert "r" in e.preemption.snapshots


@pytest.mark.parametrize("boundary", [WorkKey.DECODE_PRELUDE, WorkKey.DECODE_CODA])
@pytest.mark.parametrize("atomic_prefill", [True, False], ids=["atomic", "split"])
def test_refill_runs_boundary_restored_during_admission(boundary, atomic_prefill):
    """A sole resumed boundary must run in the admission step that restores it."""
    e = engine(max_num_seqs=1, atomic_prefill=atomic_prefill)
    e.add_request("resumed", [1, 2], params(max_tokens=3))
    request = e.scheduler.requests["resumed"]
    for _ in range(20):
        if e.scheduler.pending_work(request) == boundary:
            break
        e.step()
    else:
        pytest.fail(f"request did not reach {boundary.value}")

    generated_before = len(request.generated_token_ids)
    e.add_request("pressure", [3, 4], params(max_tokens=1))
    assert e.preemption.preempt(e.scheduler.requests["pressure"])
    assert e.preemption.snapshots["resumed"]["resume_work"] == boundary
    assert request.status == RequestStatus.WAITING
    e.abort_request("pressure")

    outputs = e.step()

    assert e.preemption.resumptions == 1
    assert e.scheduler.admission_slots == {"resumed"}
    assert e.last_schedule.works == frozenset((boundary,))
    if boundary == WorkKey.DECODE_PRELUDE:
        assert not outputs
        assert e.scheduler.pending_work(request) == WorkKey.DECODE_RECURRENT
        assert len(request.generated_token_ids) == generated_before
    else:
        assert [output.request_id for output in outputs] == ["resumed"]
        assert e.scheduler.pending_work(request) == WorkKey.DECODE_PRELUDE
        assert len(request.generated_token_ids) == generated_before + 1


def test_refill_restored_coda_waits_for_recurrent_batchmate():
    """Post-admission CODA keeps the same batching rule as an existing CODA."""
    e = engine(
        max_num_seqs=2,
        min_coda_batch_size=2,
        max_prefill_batches_before_decode=100,
    )
    e.add_request("core", [1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    core = e.scheduler.requests["core"]
    for _ in range(20):
        if e.scheduler.pending_work(core) == WorkKey.DECODE_RECURRENT:
            break
        e.step()
    else:
        pytest.fail("core request did not reach recurrent execution")

    e.add_request(
        "restored-coda",
        [3, 4],
        SamplingParams(max_tokens=1, ignore_eos=True, priority=10),
    )
    e.step()
    restored = e.scheduler.requests["restored-coda"]
    assert e.scheduler.pending_work(core) == WorkKey.DECODE_RECURRENT
    assert e.scheduler.pending_work(restored) == WorkKey.DECODE_CODA

    e.add_request("pressure", [5, 6], params(max_tokens=1))
    assert e.preemption.preempt(e.scheduler.requests["pressure"])
    assert e.preemption.snapshots["restored-coda"]["resume_work"] == WorkKey.DECODE_CODA
    e.abort_request("pressure")

    e.step()

    assert e.preemption.resumptions == 1
    assert WorkKey.DECODE_RECURRENT in e.last_schedule.works
    assert e.scheduler.pending_work(restored) == WorkKey.DECODE_CODA

    for _ in range(20):
        e.step()
        if e.last_schedule.works == frozenset((WorkKey.DECODE_CODA,)):
            break
    else:
        pytest.fail("CODA requests did not form a batch")
    assert {
        scheduled_request.request.request_id for scheduled_request in e.last_schedule.requests
    } == {
        "core",
        "restored-coda",
    }
    assert not e.has_unfinished_requests()
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0


def test_mixed_coda_batch_gate_ignores_prompt_recurrent_work():
    e = engine(
        max_num_seqs=2,
        atomic_prefill=False,
        mode="mixed",
        min_coda_batch_size=2,
        max_prefill_batches_before_decode=1,
    )
    e.add_request("prompt", [1, 2], params())
    e.add_request("coda", [3, 4], params(max_tokens=1))
    e.scheduler._admit()
    prompt = e.scheduler.requests["prompt"]
    coda = e.scheduler.requests["coda"]
    e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE].clear()
    prompt.num_preluded_tokens = len(prompt.prompt_token_ids)
    e.scheduler.enqueue_work(prompt, WorkKey.PROMPT_RECURRENT)
    e.scheduler.enqueue_work(coda, WorkKey.DECODE_CODA)

    batch = e.scheduler.schedule()

    assert batch.works == frozenset((WorkKey.DECODE_CODA,))


def test_mixed_decode_due_requires_decode_recurrent_before_skipping_admission():
    e = engine(
        max_num_seqs=2,
        atomic_prefill=False,
        mode="mixed",
        max_prefill_batches_before_decode=1,
    )
    e.add_request("prompt-core", [1, 2], params())
    e.scheduler._admit()
    prompt = e.scheduler.requests["prompt-core"]
    e.scheduler.work_queues[WorkKey.PROMPT_PRELUDE].clear()
    prompt.num_preluded_tokens = len(prompt.prompt_token_ids)
    e.scheduler.enqueue_work(prompt, WorkKey.PROMPT_RECURRENT)
    e.scheduler.policy.prefill_batches_since_recurrent = 1
    e.add_request("waiting", [3, 4], params())

    batch = e.scheduler.schedule()

    assert e.scheduler.requests["waiting"].status == RequestStatus.ACTIVE
    assert batch.works == frozenset((WorkKey.PROMPT_PRELUDE,))
    assert batch.requests[0].request.request_id == "waiting"


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_repeated_pressure_reclaims_earlier_batches(mode):
    """Pressure must keep reclaiming earlier batches, not only the newest one."""
    e = LLMEngine(
        model(),
        cache_config=CacheConfig(24, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, prefill_chunk_size=2, mode=mode, enable_preemption=True
        ),
    )
    for name in range(3):
        e.add_request(str(name), [1, 2], SamplingParams(max_tokens=8, ignore_eos=True))
    outputs = {}
    for _ in range(1000):
        if not e.has_unfinished_requests():
            break
        for output in e.step():
            if output.finished:
                outputs[output.request_id] = output
    else:
        pytest.fail("engine did not drain")
    assert sorted(outputs) == ["0", "1", "2"]
    assert outputs["0"].token_ids == outputs["1"].token_ids == outputs["2"].token_ids
    assert e.preemption.preemptions > 0
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0
