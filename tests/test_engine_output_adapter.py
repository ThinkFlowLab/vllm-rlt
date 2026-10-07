"""The temporary adapters expose host values without changing runner calls."""

from types import SimpleNamespace
from unittest.mock import Mock

from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_rlt.engine.output_adapter import (
    adapt_async_delivery,
    adapt_async_submit,
    adapt_speculative_execute,
    adapt_sync_execute,
)
from vllm_rlt.request import Request, Stage
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.output import Progress


def batch(stage):
    request = Request("a", [2], SamplingParams(max_tokens=2), generation=7)
    item = ScheduledItem(
        request, 0, 1, request_id="a", generation=7, position=0, loops_done=2, output_index=0
    )
    return SchedulerOutput(stage, [item], seq=11)


def test_sync_adapters_preserve_row_identity_and_host_values():
    prefill = adapt_sync_execute(batch(Stage.PREFILL), None)
    assert prefill.progress == Progress.COMPLETED
    assert prefill.prefill_ranges == ((0, 1),)
    assert prefill.completion == (None,)
    signal = adapt_sync_execute(batch(Stage.RECURRENT), [0.25]).exit_signals[0]
    assert (
        signal.request_id,
        signal.generation,
        signal.position,
        signal.signal_depth,
        signal.source_seq,
        signal.score,
    ) == ("a", 7, 0, 3, 11, 0.25)
    coda = adapt_sync_execute(batch(Stage.CODA), [5])
    assert coda.sampled_token_ids == (5,)
    speculative = adapt_speculative_execute(
        batch(Stage.SPECULATIVE),
        [SimpleNamespace(token_ids=[6, 7], accepted_count=1, draft_count=2)],
    )
    assert speculative.speculative[0].token_ids == (6, 7)
    assert speculative.speculative[0].accepted_count == 1
    assert speculative.speculative[0].draft_count == 2


def test_async_adapters_separate_submission_from_delivery():
    event = object()
    prefill = adapt_async_submit(batch(Stage.PREFILL), {"a": event})
    assert prefill.progress == Progress.SUBMITTED
    assert prefill.completion == (event,)
    submitted = adapt_async_submit(batch(Stage.CODA), {})
    assert submitted.sampled_token_ids == ()
    ticket = SimpleNamespace(batch=batch(Stage.CODA), collect=Mock(return_value=[9]))
    delivered = adapt_async_delivery(ticket)
    assert (delivered.seq, delivered.stage, delivered.progress) == (
        11,
        Stage.CODA,
        Progress.DELIVERED,
    )
    assert delivered.sampled_token_ids == (9,)
    ticket.collect.assert_called_once_with()
