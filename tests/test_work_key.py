import pytest

from vllm_rlt.core.execution import ExecutionBatch, Operation, StateSpan, WorkKey
from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
from vllm_rlt.request import Request, RequestStatus, SamplingParams
from vllm_rlt.worker.executor import build_execution_batch


def test_work_key_encodes_only_supported_operation_pairs():
    assert {
        WorkKey.PROMPT_ATOMIC: Operation.PRELUDE,
        WorkKey.PROMPT_PRELUDE: Operation.PRELUDE,
        WorkKey.PROMPT_RECURRENT: Operation.RECURRENT,
        WorkKey.DECODE_PRELUDE: Operation.PRELUDE,
        WorkKey.DECODE_RECURRENT: Operation.RECURRENT,
        WorkKey.DECODE_CODA: Operation.CODA,
        WorkKey.SPECULATIVE: None,
    } == {work: work.operation for work in WorkKey}


def test_request_status_has_lifecycle_states_only():
    assert {status.value for status in RequestStatus} == {
        "waiting",
        "receiving",
        "transferring",
        "active",
        "finished",
    }


def test_scheduler_output_reports_split_prefill_from_scheduled_work():
    request = Request("r", [1, 2], SamplingParams())
    for work in WorkKey:
        batch = SchedulerOutput([ScheduledRequest(request, work)])
        assert batch.split_prefilling is (
            work in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
        )


def test_direct_execution_batch_infers_split_prefilling_from_chunk_spans():
    request = Request("r", [1, 2], SamplingParams())
    chunk = StateSpan(request, 0, 1, 0, chunk_start=0)
    ordinary = StateSpan(request, 0, 1, 0)

    assert ExecutionBatch(Operation.PRELUDE, (chunk,)).split_prefilling
    assert not ExecutionBatch(Operation.PRELUDE, (ordinary,)).split_prefilling
    with pytest.raises(ValueError, match="chunk spans require split prefill"):
        ExecutionBatch(Operation.PRELUDE, (chunk,), split_prefilling=False)
    assert ExecutionBatch(Operation.PRELUDE, (ordinary,), split_prefilling=True).split_prefilling


@pytest.mark.parametrize(
    ("prompt_work", "decode_work", "operation", "loop_count"),
    [
        (WorkKey.PROMPT_PRELUDE, WorkKey.DECODE_PRELUDE, Operation.PRELUDE, 0),
        (WorkKey.PROMPT_RECURRENT, WorkKey.DECODE_RECURRENT, Operation.RECURRENT, 1),
    ],
)
def test_execution_batch_builds_from_compatible_work_keys(
    prompt_work, decode_work, operation, loop_count
):
    prompt = Request("prompt", [1, 2], SamplingParams())
    decode = Request("decode", [3, 4], SamplingParams())
    depth = 1 if operation == Operation.RECURRENT else None
    batch = SchedulerOutput(
        [
            ScheduledRequest(
                prompt,
                prompt_work,
                token_count=1,
                depth=depth,
                chunk_start=0,
            ),
            ScheduledRequest(decode, decode_work, token_count=1, depth=depth),
        ],
    )

    execution_batch = build_execution_batch(batch, total_depths=4)

    assert execution_batch.operation == operation
    assert execution_batch.loop_count == loop_count
    assert execution_batch.split_prefilling
    assert execution_batch.has_chunks
    assert batch.is_mixed


def test_execution_batch_reports_atomic_boundary_mismatch_by_request():
    prompt = Request("prompt", [1, 2], SamplingParams())
    decode = Request("decode", [3, 4], SamplingParams())
    batch = SchedulerOutput(
        [
            ScheduledRequest(prompt, WorkKey.PROMPT_ATOMIC),
            ScheduledRequest(decode, WorkKey.DECODE_PRELUDE),
        ],
    )

    with pytest.raises(ValueError, match="request decode work DECODE_PRELUDE"):
        build_execution_batch(batch, total_depths=4)
