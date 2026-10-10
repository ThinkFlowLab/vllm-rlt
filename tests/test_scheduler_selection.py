"""Selection protection is batch-local state, not Scheduler-global state.

Preemption must not suspend a request the Scheduler has already placed in the
batch under construction. That exclusion belongs to one batch construction, so
it must not survive the batch, reach a direct `_take()` caller such as the PD
prefill worker, or outlive the request whose ID it names.
"""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage


def model():
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config())


def engine(max_num_seqs=2):
    return LLMEngine(
        model(),
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, enable_preemption=True),
    )


def params(max_tokens=5):
    return SamplingParams(max_tokens=max_tokens, ignore_eos=True)


def test_batch_selection_is_never_preempted_by_a_later_item(monkeypatch):
    """Selecting A must stop B's capacity failure from evicting A."""
    e = engine()
    for name in ("a", "b"):
        e.add_request(name, [1, 2], params())
    e.scheduler._admit()
    monkeypatch.setattr(e.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "b")
    batch = e.scheduler._take(Stage.PREFILL)
    assert [item.request.request_id for item in batch.items] == ["a"]
    assert e.preemption.preemptions == 0
    assert e.scheduler.requests["a"].stage == Stage.PREFILL


def test_direct_take_does_not_inherit_an_earlier_batch_selection(monkeypatch):
    """The PD prefill worker calls _take() directly, without schedule()."""
    e = engine()
    for name in ("a", "b"):
        e.add_request(name, [1, 2], params())
    e.scheduler._admit()
    # A first batch selects both requests.
    assert e.scheduler._take(Stage.PREFILL) is not None

    seen = []

    def spy(requester, *, priority_only=False, excluded=frozenset()):
        seen.append((requester.request_id, excluded))
        return False

    # The Scheduler holds the callback it was given, so replace that reference.
    monkeypatch.setattr(e.scheduler, "preempt_callback", spy)
    for name in ("a", "b"):
        e.scheduler.enqueue(e.scheduler.requests[name], Stage.PREFILL)
    monkeypatch.setattr(e.cache_manager, "ensure_capacity", lambda rid, frontier: rid != "b")
    assert e.scheduler._take(Stage.PREFILL) is not None
    # Only "a", selected in this batch, is excluded. The first batch's
    # selection is not still in force, and the set is handed over frozen.
    assert seen == [("b", frozenset({"a"}))]


def test_request_selected_in_an_earlier_batch_is_still_preemptable():
    """A previously scheduled request remains a legitimate preemption victim."""
    e = engine(max_num_seqs=1)
    e.add_request("victim", [1, 2], params())
    e.step()
    assert e.scheduler.requests["victim"].stage != Stage.WAITING
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
