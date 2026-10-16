"""Stage-local execution. Placement never changes at a loop boundary."""

from vllm_rlt.core.execution import ExecutionBatch, Operation, StateSpan, WorkKey
from vllm_rlt.core.scheduler import SchedulerOutput


def build_execution_batch(scheduled_batch: SchedulerOutput, total_depths: int) -> ExecutionBatch:
    """Bind selected Request references to immutable operation/position/depth rows.

    This creates scalar execution descriptors, not new Requests or KV metadata.
    The runner prepares its backend-specific metadata once from these bindings.
    """
    spans = []
    batch_operation = None
    batch_loop_count = None
    for scheduled_request in scheduled_batch.requests:
        request = scheduled_request.request
        work = scheduled_request.work
        operation = work.operation
        if operation is None:
            raise ValueError(f"work {work.name} is handled by the speculative runner")
        chunk = work in (WorkKey.PROMPT_PRELUDE, WorkKey.PROMPT_RECURRENT)
        prompt_input = work in (WorkKey.PROMPT_ATOMIC, WorkKey.PROMPT_PRELUDE)
        depth = (
            scheduled_request.depth if scheduled_request.depth is not None else request.loops_done
        )
        if operation != Operation.RECURRENT:
            depth = 0
        position = (
            scheduled_request.token_start
            if chunk or work == WorkKey.PROMPT_ATOMIC or scheduled_request.depth is not None
            else request.position
        )
        loop_count = (
            total_depths if work == WorkKey.PROMPT_ATOMIC else int(operation == Operation.RECURRENT)
        )
        if batch_operation is None:
            batch_operation = operation
            batch_loop_count = loop_count
        elif operation != batch_operation or loop_count != batch_loop_count:
            raise ValueError(
                f"request {request.request_id} work {work.name} has "
                f"operation={operation.value}, loop_count={loop_count}; "
                f"batch expects operation={batch_operation.value}, "
                f"loop_count={batch_loop_count}"
            )
        spans.append(
            StateSpan(
                request,
                position,
                scheduled_request.token_count,
                depth,
                chunk_start=scheduled_request.chunk_start if chunk else None,
                retain_chunk=chunk and (work == WorkKey.PROMPT_PRELUDE or depth + 1 < total_depths),
                retain_last=scheduled_request.completes_prompt if chunk else True,
                emit_gate=work == WorkKey.DECODE_RECURRENT,
                input_tokens=tuple(
                    request.prompt_token_ids[position : position + scheduled_request.token_count]
                )
                if prompt_input and not chunk
                else None,
            )
        )
    if batch_operation is None:
        raise ValueError("cannot build an execution batch without scheduled work")
    return ExecutionBatch(
        batch_operation,
        tuple(spans),
        batch_loop_count,
        split_prefilling=scheduled_batch.split_prefilling,
    )


class Executor:
    """Dispatch local batches, preserving the runner's device-resident paths.

    The round controller shares the model runtime and RNG with the runner.
    It expands a speculative round locally, without a stage handoff.
    """

    def __init__(self, runner, *, speculation=None):
        self.runner = runner
        self.speculation = speculation

    def submit(self, scheduled_batch):
        if scheduled_batch.is_speculative:
            if self.speculation is None:
                raise ValueError("speculative submission requires a configured round controller")
            return self.speculation.submit(scheduled_batch)
        batch = build_execution_batch(scheduled_batch, self.runner.model.config.total_ut_steps)
        return self.runner.submit(scheduled_batch, batch=batch)
