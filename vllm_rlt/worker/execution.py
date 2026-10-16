"""Shared model execution for request, chunk and temporary speculative state.

Bindings are borrowed device tensors. The caller owns their lifetime and stream
ordering; execution never commits request progress or speculative candidates.
Metadata may be lazy so continuous loops retain activations without round trips.
"""

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import torch

from vllm_rlt.core.execution import Operation


@dataclass(frozen=True)
class ForwardBatch:
    """One compatible device batch, independent of scheduling stage.

    PRELUDE can include a sequence of RECURRENT views. RECURRENT starts from supplied hidden.
    CODA projects supplied hidden. KV views carry allocation identity and may
    use host-planned or device-owned positions. Graph executors must match the
    operation and gate mode. All tensors/results are borrowed, not snapshots.
    """

    operation: Operation
    hidden: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    metadata: Iterable = ()
    boundary_kv: object = None
    generators: object = None
    hidden_buffer: torch.Tensor | None = None
    compute_gate: bool = False
    graphs: object = None
    compute_logits: bool = True


@dataclass(frozen=True)
class ForwardResult:
    hidden: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class ExecutionResult(NamedTuple):
    hidden: torch.Tensor
    gate_logits: torch.Tensor | None


class ExecutionRuntime:
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache

    def forward(self, batch: ForwardBatch) -> ForwardResult:
        hidden, gate = batch.hidden, None
        try:
            if batch.operation == Operation.CODA:
                if hidden is None:
                    raise ValueError("CODA requires hidden state")
                if batch.graphs is not None:
                    logits = (
                        batch.graphs.run(hidden, batch.boundary_kv)
                        if batch.boundary_kv is not None
                        else batch.graphs.run(hidden)
                    )
                elif batch.boundary_kv is not None:
                    logits = self.model.coda_prepared(
                        hidden,
                        batch.boundary_kv,
                        self.cache,
                        compute_logits=batch.compute_logits,
                    )
                else:
                    logits = self.model.coda(hidden) if batch.compute_logits else None
                return ForwardResult(hidden=hidden, logits=logits)
            if batch.operation == Operation.PRELUDE:
                if batch.tokens is None:
                    raise ValueError("PRELUDE requires token IDs")
                if batch.boundary_kv is not None:
                    hidden = self.model.prelude_prepared(
                        batch.tokens[: len(batch.boundary_kv.position_ids)],
                        batch.boundary_kv,
                        self.cache,
                        generators=batch.generators,
                    )
                    if batch.hidden_buffer is not None:
                        batch.hidden_buffer.zero_()
                        batch.hidden_buffer[: len(hidden)].copy_(hidden)
                        hidden = batch.hidden_buffer
                else:
                    hidden = self.model.prelude(batch.tokens)
            elif batch.operation != Operation.RECURRENT:
                raise ValueError(f"unsupported operation {batch.operation}")
            if hidden is None:
                raise ValueError("RECURRENT requires hidden state")
            for kv in batch.metadata:
                if batch.graphs is not None:
                    hidden, gate = batch.graphs.run(hidden, kv)
                else:
                    hidden, gate = self.model.recurrent_prepared(
                        hidden, kv, self.cache, compute_gate=batch.compute_gate
                    )
            return ForwardResult(hidden, gate)
        finally:
            if isinstance(batch.metadata, Generator):
                batch.metadata.close()

    def advance(
        self,
        hidden=None,
        metadata=(),
        *,
        input_tokens=None,
        input_kv=None,
        generators=None,
        hidden_buffer=None,
        compute_gate=False,
        graphs=None,
    ) -> ExecutionResult:
        """Continuous PRELUDE/RECURRENT adapter; preserves the existing two-value API."""
        result = self.forward(
            ForwardBatch(
                Operation.PRELUDE if input_tokens is not None else Operation.RECURRENT,
                hidden=hidden,
                tokens=input_tokens,
                metadata=metadata,
                boundary_kv=input_kv,
                generators=generators,
                hidden_buffer=hidden_buffer,
                compute_gate=compute_gate,
                graphs=graphs,
            )
        )
        return ExecutionResult(result.hidden, result.gate_logits)
