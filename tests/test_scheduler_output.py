"""Scheduler decisions returned to EngineCore after runner progress."""

from unittest.mock import Mock

import pytest

from vllm_rlt import SamplingParams, SchedulerConfig
from vllm_rlt.core.scheduler import ScheduledItem, Scheduler, SchedulerOutput
from vllm_rlt.request import FinishReason, Request, Stage
from vllm_rlt.worker.output import ModelRunnerOutput, Progress


def coda_batch():
    scheduler = Scheduler(SchedulerConfig(), Mock(), total_ut_steps=4, eos_token_ids=(2,))
    request = Request("a", [1], SamplingParams(max_tokens=3))
    scheduler.add_request(request)
    scheduler.queues[Stage.WAITING].clear()
    request.stage = Stage.CODA
    request.loops_done = 4
    item = ScheduledItem(
        request,
        request_id="a",
        generation=request.generation,
        position=request.position,
        loops_done=4,
        output_index=0,
    )
    return scheduler, request, SchedulerOutput(Stage.CODA, [item], seq=1)


@pytest.mark.parametrize("progress", [Progress.COMPLETED, Progress.DELIVERED])
def test_coda_eos_returns_finish_without_queuing_next_stage(progress):
    scheduler, request, batch = coda_batch()
    if progress == Progress.DELIVERED:
        request.num_output_placeholders = 1
    result = ModelRunnerOutput(1, Stage.CODA, progress, sampled_token_ids=(2,))
    update = scheduler.update_from_output(batch, result)
    assert update.finished == [("a", request.generation, FinishReason.STOP)]
    assert update.output_rows == batch.items
    assert not scheduler.queues[Stage.PRELUDE]
    assert request.generated_token_ids == [2]
    assert request.exit_depths == [4]


def test_scheduler_rejects_output_from_another_batch():
    scheduler, _, batch = coda_batch()
    with pytest.raises(ValueError, match="does not match"):
        scheduler.update_from_output(
            batch, ModelRunnerOutput(2, Stage.CODA, Progress.COMPLETED, sampled_token_ids=(3,))
        )
