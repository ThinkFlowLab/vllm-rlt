"""Stage selection and fairness state for loop-level scheduling."""

from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.request import Stage

if TYPE_CHECKING:
    from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput


class NoRefillPhase(Enum):
    FILL = auto()
    CORE = auto()
    CODA = auto()


class SchedulingPolicy:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.prefill_batches_since_recurrent = 0

    @property
    def decode_due(self) -> bool:
        return self.prefill_batches_since_recurrent >= self.config.max_prefill_batches_before_decode

    def record_batch(self, stage: Stage) -> None:
        """Count nonempty scheduled batches, not attempts or GPU completions."""
        if stage == Stage.PREFILL:
            self.prefill_batches_since_recurrent += 1
        elif stage == Stage.RECURRENT:
            self.prefill_batches_since_recurrent = 0


class RefillPolicy(SchedulingPolicy):
    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        q = scheduler.queues
        # Give independent core work an overlap opportunity before refilling.
        if prefer_recurrent and q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        if q[Stage.PRELUDE]:
            return scheduler._take(Stage.PRELUDE)
        if q[Stage.CODA] and (
            len(q[Stage.CODA]) >= self.config.min_coda_batch_size or not q[Stage.RECURRENT]
        ):
            return scheduler._take(Stage.CODA)
        if self.decode_due and q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        scheduler._admit()
        if q[Stage.PREFILL]:
            return scheduler._take(Stage.PREFILL)
        if q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        return None


class NoRefillPolicy(SchedulingPolicy):
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.phase = NoRefillPhase.FILL

    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        q = scheduler.queues
        if self.phase == NoRefillPhase.CORE:
            if q[Stage.RECURRENT]:
                return scheduler._take(Stage.RECURRENT)
            self.phase = NoRefillPhase.CODA
        if self.phase == NoRefillPhase.CODA:
            if q[Stage.CODA]:
                return scheduler._take(Stage.CODA)
            self.phase = NoRefillPhase.FILL
        # Bound prefill runs so continuous arrivals cannot starve decode.
        if self.decode_due:
            if q[Stage.PRELUDE]:
                return scheduler._take(Stage.PRELUDE)
            if q[Stage.RECURRENT]:
                self.phase = NoRefillPhase.CORE
                return scheduler._take(Stage.RECURRENT)
        scheduler._admit()
        if q[Stage.PREFILL]:
            return scheduler._take(Stage.PREFILL)
        if q[Stage.CODA]:  # First output after full-depth prompt prefill.
            return scheduler._take(Stage.CODA)
        if q[Stage.PRELUDE]:
            return scheduler._take(Stage.PRELUDE)
        if q[Stage.RECURRENT]:
            self.phase = NoRefillPhase.CORE
            return scheduler._take(Stage.RECURRENT)
        return None
