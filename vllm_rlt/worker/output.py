"""Host-visible results at the Scheduler / runner boundary."""

from dataclasses import dataclass, field
from enum import Enum, auto

from vllm_rlt.request import Stage


class Progress(Enum):
    COMPLETED = auto()
    SUBMITTED = auto()
    DELIVERED = auto()


@dataclass(frozen=True, slots=True)
class ExitSignal:
    request_id: str
    generation: int
    position: int
    signal_depth: int
    source_seq: int
    score: float


@dataclass(frozen=True, slots=True)
class SpeculativeTokens:
    token_ids: tuple[int, ...]
    accepted_count: int
    draft_count: int


@dataclass(slots=True)
class ModelRunnerOutput:
    seq: int
    stage: Stage
    progress: Progress
    prefill_ranges: tuple[tuple[int, int], ...] = ()
    completion: tuple[object | None, ...] = ()
    exit_signals: list[ExitSignal] = field(default_factory=list)
    sampled_token_ids: tuple[int, ...] = ()
    speculative: tuple[SpeculativeTokens, ...] = ()
