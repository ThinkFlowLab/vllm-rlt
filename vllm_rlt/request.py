from dataclasses import dataclass, field, replace
from enum import Enum
from secrets import randbits

import torch

from vllm_rlt.sampling_params import SamplingParams


class Stage(str, Enum):
    WAITING = "waiting"
    RECEIVING = "receiving"
    PREFILL = "prefill"
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    SPECULATIVE = "speculative"
    CODA = "coda"
    FINISHED = "finished"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    stage: Stage = Stage.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    exit_depths: list[int] = field(default_factory=list)
    # Scheduling advances before coda's CPU output delivery.
    num_output_placeholders: int = 0
    input_token_tensor: torch.Tensor | None = field(default=None, repr=False)
    num_prefilled_tokens: int = 0
    # Host progress; in async mode an event confirms submitted GPU work is done.
    loops_done: int = 0
    pending_exit_depth: int | None = None
    admission_bypasses: int = 0
    remaining_probability: float = 1.0
    hidden_state: torch.Tensor | None = field(default=None, repr=False)
    generator: torch.Generator | None = field(default=None, repr=False)
    finish_reason: FinishReason | None = None
    exit_trace: tuple[int, ...] = field(default=(), repr=False)
    weight_version: int = 0
    log_probs: list[torch.Tensor] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.sampling_params.seed is None:
            self.sampling_params = replace(self.sampling_params, seed=randbits(63))

    @property
    def num_scheduled_outputs(self) -> int:
        return len(self.generated_token_ids) + self.num_output_placeholders

    @property
    def position(self) -> int:
        return len(self.prompt_token_ids) + self.num_scheduled_outputs - 1

    @property
    def input_token_id(self) -> int:
        return self.generated_token_ids[-1]


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    exit_depths: list[int]
    finished: bool
    finish_reason: str | None = None
    text: str = ""
    log_probs: list[float] | None = None
    weight_version: int = 0
    sampling_params: SamplingParams | None = None

    @classmethod
    def from_request(cls, request: Request):
        return cls(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            token_ids=list(request.generated_token_ids),
            exit_depths=list(request.exit_depths),
            finished=request.stage == Stage.FINISHED,
            finish_reason=request.finish_reason.value
            if request.finish_reason is not None
            else None,
            log_probs=(
                torch.stack(request.log_probs[: len(request.generated_token_ids)]).tolist()
                if request.generated_token_ids
                else []
            )
            if request.sampling_params.logprobs is not None
            else None,
            weight_version=request.weight_version,
            sampling_params=request.sampling_params,
        )
