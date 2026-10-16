"""Execution operations and row/state bindings, independent of scheduling policy.

The scheduler selects request snapshots; Executor binds them into an ExecutionBatch. The runner
uses operations and storage bindings only. Request identity is retained for
state ownership and cancellation, never to infer a submitted loop position.
"""

from dataclasses import dataclass, replace
from enum import Enum

from vllm_rlt.request import Request


class Operation(Enum):
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    CODA = "coda"


class WorkKey(Enum):
    PROMPT_ATOMIC = ("prompt_atomic", Operation.PRELUDE)
    PROMPT_PRELUDE = ("prompt_prelude", Operation.PRELUDE)
    PROMPT_RECURRENT = ("prompt_recurrent", Operation.RECURRENT)
    DECODE_PRELUDE = ("decode_prelude", Operation.PRELUDE)
    DECODE_RECURRENT = ("decode_recurrent", Operation.RECURRENT)
    DECODE_CODA = ("decode_coda", Operation.CODA)
    SPECULATIVE = ("speculative", None)

    @property
    def operation(self) -> Operation | None:
        return self.value[1]


@dataclass(frozen=True)
class StateSpan:
    request: Request
    token_start: int
    token_count: int
    depth: int
    # None addresses the request's single resident hidden row. Otherwise the
    # span addresses [token_start-chunk_start, ...] in its resident chunk.
    chunk_start: int | None = None
    retain_chunk: bool = False
    retain_last: bool = True
    emit_gate: bool = False
    # For chunk PRELUDE, None reads the immutable request prompt range directly.
    # Explicit tokens override that source (also used by direct execution callers).
    input_tokens: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ExecutionBatch:
    operation: Operation
    spans: tuple[StateSpan, ...]
    # Number of recurrent operations following PRELUDE, or starting at RECURRENT.
    # PRELUDE + N loops retains activations locally until the submission boundary.
    loop_count: int = 0
    # Runtime work label; direct batches infer it from chunk spans when omitted.
    split_prefilling: bool | None = None

    def __post_init__(self):
        if self.loop_count < 0:
            raise ValueError("loop_count must be nonnegative")
        if self.split_prefilling is None:
            object.__setattr__(self, "split_prefilling", self.has_chunks)
        elif self.has_chunks and not self.split_prefilling:
            raise ValueError("chunk spans require split prefill")
        if self.operation == Operation.RECURRENT and self.loop_count == 0:
            object.__setattr__(self, "loop_count", 1)
        if self.operation == Operation.CODA and self.loop_count:
            raise ValueError("CODA cannot include recurrent work")
        for span in self.spans:
            if span.token_start < 0 or span.token_count <= 0 or span.depth < 0:
                raise ValueError("state spans require nonnegative addresses and positive rows")

    def at_loop(self, offset: int):
        """Bind a loop view without changing request progress or storage ownership."""
        if not 0 <= offset < self.loop_count:
            raise ValueError("loop offset outside execution range")
        return replace(
            self,
            operation=Operation.RECURRENT,
            loop_count=1,
            spans=tuple(replace(span, depth=span.depth + offset) for span in self.spans),
        )

    def validate_depths(self, total_depths: int):
        if any(span.depth + self.loop_count > total_depths for span in self.spans):
            raise ValueError("execution range exceeds model loop depth")

    @property
    def num_rows(self):
        return sum(span.token_count for span in self.spans)

    @property
    def has_chunks(self):
        return any(span.chunk_start is not None for span in self.spans)

    @property
    def all_chunks(self):
        return bool(self.spans) and all(span.chunk_start is not None for span in self.spans)

    @property
    def emit_gate(self):
        return any(span.emit_gate for span in self.spans)

    def addresses(self):
        ids, depths, positions = [], [], []
        for span in self.spans:
            ids.extend([span.request.request_id] * span.token_count)
            depths.extend([span.depth] * span.token_count)
            positions.extend(range(span.token_start, span.token_start + span.token_count))
        return ids, depths, positions
