"""Temporary wrappers around the existing runner calls and readbacks."""

from dataclasses import dataclass

from vllm_rlt.request import Stage
from vllm_rlt.worker.output import ExitSignal, ModelRunnerOutput, Progress, SpeculativeTokens


@dataclass(frozen=True, slots=True)
class SignalHandle:
    ticket: object
    row: int
    generation: int
    position: int
    signal_depth: int
    source_seq: int


def adapt_sync_execute(batch, values) -> ModelRunnerOutput:
    output = ModelRunnerOutput(batch.seq, batch.stage, Progress.COMPLETED)
    if batch.stage == Stage.PREFILL:
        output.prefill_ranges = tuple((i.token_start, i.token_count) for i in batch.items)
        output.completion = (None,) * len(batch.items)
    elif batch.stage == Stage.RECURRENT and values is not None:
        output.exit_signals = [
            ExitSignal(
                item.request_id,
                item.generation,
                item.position,
                item.loops_done + 1,
                batch.seq,
                score,
            )
            for item, score in zip(batch.items, values, strict=True)
        ]
    elif batch.stage == Stage.CODA:
        output.sampled_token_ids = tuple(values)
    return output


def adapt_speculative_execute(batch, results) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        batch.seq,
        batch.stage,
        Progress.COMPLETED,
        speculative=tuple(
            SpeculativeTokens(tuple(r.token_ids), r.accepted_count, r.draft_count) for r in results
        ),
    )


def adapt_async_submit(batch, events) -> ModelRunnerOutput:
    output = ModelRunnerOutput(batch.seq, batch.stage, Progress.SUBMITTED)
    if batch.stage == Stage.PREFILL:
        output.prefill_ranges = tuple((i.token_start, i.token_count) for i in batch.items)
        output.completion = tuple(events.get(i.request_id) for i in batch.items)
    return output


def adapt_async_delivery(ticket) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        ticket.batch.seq,
        ticket.batch.stage,
        Progress.DELIVERED,
        sampled_token_ids=tuple(ticket.collect()),
    )
