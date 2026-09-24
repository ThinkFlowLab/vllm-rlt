"""Synchronous block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to one execute call. Requests expose only committed
outputs; the engine applies returned tokens and owns KV commit/truncation.
"""

from dataclasses import dataclass, field

import torch

from vllm_rlt.core.scheduler import ScheduledItem
from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.sampling import (
    draw,
    generator_for,
    probabilities,
    rejection_sample,
    sample_logits,
)


@dataclass(frozen=True)
class SpeculativeResult:
    token_ids: list[int]
    accepted_count: int
    draft_count: int


@dataclass
class SpeculativeStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    verified_rows: int = 0


@dataclass
class DraftedItem:
    item: ScheduledItem
    candidates: list[torch.Tensor] = field(default_factory=list)
    proposals: list[torch.Tensor] = field(default_factory=list)
    states: list[torch.Tensor] = field(default_factory=list)


class SpeculativeRunner:
    def __init__(self, model, cache, config, execution):
        self.model, self.cache, self.config = model, cache, config
        self.device = next(model.parameters()).device
        self.stats = SpeculativeStats()
        self.graphs = (
            RecurrentGraphs(model, cache, execution, False) if execution.cuda_graphs else None
        )
        self.coda_graphs = CodaGraphs(model, execution) if execution.cuda_graphs else None

    def _core(self, hidden, ids, positions, depth, *, packed=False):
        batch = self.cache._prepare_batch(
            ids,
            [depth] * len(ids),
            positions,
            packed_prefill=packed and self.cache.attention_info.get("generation") == 4,
        )
        if self.graphs is not None:
            hidden, _ = self.graphs.run(hidden, batch)
        else:
            hidden, _ = self.model.recurrent_prepared(hidden, batch, self.cache, compute_gate=False)
        return hidden

    def _coda(self, hidden):
        return (
            self.coda_graphs.run(hidden)
            if self.coda_graphs is not None
            else self.model.coda(hidden)
        )

    @torch.inference_mode()
    def draft(self, batch) -> list[DraftedItem]:
        items = batch.items
        drafted = [DraftedItem(item) for item in items]
        # Each item includes the final shallow input needed for the bonus row.
        # Drafting batches independent requests at every autoregressive step.
        for offset in range(max(item.token_count for item in items)):
            active = [i for i, item in enumerate(items) if offset < item.token_count]
            ids = [items[i].request.request_id for i in active]
            positions = [items[i].token_start + offset for i in active]
            tokens = (
                torch.tensor(
                    [items[i].request.input_token_id for i in active],
                    device=self.device,
                    dtype=torch.long,
                )
                if offset == 0
                else torch.stack([drafted[i].candidates[-1] for i in active])
            )
            hidden = self.model.prelude(tokens)
            for depth in range(self.config.draft_loops):
                hidden = self._core(hidden, ids, positions, depth)
            drafting = []
            for row, i in enumerate(active):
                drafted[i].states.append(hidden[row])
                if offset + 1 < items[i].token_count:
                    drafting.append((row, i))
            if not drafting:
                continue
            logits = self._coda(torch.stack([hidden[row] for row, _ in drafting]))
            for row, (_, i) in enumerate(drafting):
                request = items[i].request
                if request.sampling_params.temperature == 0:
                    candidate = logits[row].argmax()
                else:
                    q = probabilities(logits[row], request.sampling_params)
                    drafted[i].proposals.append(q)
                    candidate = draw(q, generator_for(request, self.device))
                drafted[i].candidates.append(candidate)
        return drafted

    @torch.inference_mode()
    def verify(self, drafted: list[DraftedItem]) -> list[SpeculativeResult]:
        # Group contiguous positions per request for causal ragged attention.
        ids, positions = [], []
        for entry in drafted:
            item = entry.item
            ids.extend([item.request.request_id] * item.token_count)
            positions.extend(range(item.token_start, item.token_start + item.token_count))
        hidden = torch.stack([h for entry in drafted for h in entry.states])
        for depth in range(self.config.draft_loops, self.config.target_loops):
            hidden = self._core(hidden, ids, positions, depth, packed=True)
        logits = self._coda(hidden)
        results, start = [], 0
        for entry in drafted:
            item = entry.item
            rows = logits[start : start + item.token_count]
            start += item.token_count
            request = item.request
            greedy = request.sampling_params.temperature == 0
            drafts = (
                torch.stack(entry.candidates)
                if entry.candidates
                else torch.empty(0, device=self.device, dtype=torch.long)
            )
            if greedy:
                ids = torch.cat((rows.argmax(dim=-1), drafts)).tolist()
                targets, candidates = ids[: item.token_count], ids[item.token_count :]
            else:
                candidates = drafts.tolist()
            tokens, accepted = [], 0
            for offset, candidate in enumerate(candidates):
                if greedy:
                    token = targets[offset]
                    accept = token == candidate
                else:
                    p = probabilities(rows[offset], request.sampling_params)
                    token, accept = rejection_sample(
                        candidate, p, entry.proposals[offset], generator_for(request, self.device)
                    )
                tokens.append(token)
                if not accept:
                    break
                accepted += 1
            else:
                if greedy:
                    tokens.append(targets[-1])
                else:
                    tokens.append(int(sample_logits(rows[-1], request).item()))
            results.append(SpeculativeResult(tokens, accepted, len(entry.candidates)))
        self.stats.rounds += len(drafted)
        self.stats.drafted_tokens += sum(len(entry.candidates) for entry in drafted)
        self.stats.verified_rows += len(ids)
        return results

    def execute(self, batch):
        return self.verify(self.draft(batch))
