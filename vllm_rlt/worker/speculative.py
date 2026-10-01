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


def greedy_accept(candidates, targets):
    """Return (tokens, accepted) for one request from host candidate and target IDs.

    targets[j] is the target argmax of verification row j (len(candidates) + 1 rows).
    Accepted candidates equal their targets, so the committed tokens are the target
    prefix through the first mismatch: the correction, or the bonus if all match.
    """
    accepted = 0
    for candidate, target in zip(candidates, targets):
        if candidate != target:
            break
        accepted += 1
    return list(targets[: accepted + 1]), accepted


class SpeculativeRunner:
    def __init__(self, model, cache, config, execution):
        self.model, self.cache, self.config = model, cache, config
        self.device = next(model.parameters()).device
        self.stats = SpeculativeStats()
        self.graphs = (
            RecurrentGraphs(model, cache, execution, False) if execution.cuda_graphs else None
        )
        self.coda_graphs = CodaGraphs(model, execution) if execution.cuda_graphs else None

    def _packed(self, packed):
        return packed and self.cache.attention_info.get("generation") == 4

    def _cores(self, hidden, ids, positions, depths, *, packed=False):
        # Depth-independent metadata is prepared and copied once for all depths.
        batches = self.cache._prepare_batches(
            ids,
            [[depth] * len(ids) for depth in depths],
            positions,
            packed_prefill=self._packed(packed),
        )
        for depth, batch in zip(depths, batches):
            hidden = self._core(hidden, ids, positions, depth, packed=packed, batch=batch)
        return hidden

    def _core(self, hidden, ids, positions, depth, *, packed=False, batch=None):
        if batch is None:
            batch = self.cache._prepare_batch(
                ids, [depth] * len(ids), positions, packed_prefill=self._packed(packed)
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
        inputs = self.cache._stage([item.request.input_token_id for item in items], torch.long)
        for offset in range(max(item.token_count for item in items)):
            active = [i for i, item in enumerate(items) if offset < item.token_count]
            ids = [items[i].request.request_id for i in active]
            positions = [items[i].token_start + offset for i in active]
            hidden = self.model.prelude(inputs)
            hidden = self._cores(hidden, ids, positions, range(self.config.draft_loops))
            drafting = []
            for row, i in enumerate(active):
                drafted[i].states.append(hidden[row])
                if offset + 1 < items[i].token_count:
                    drafting.append((row, i))
            if not drafting:
                break
            logits = self._coda(torch.stack([hidden[row] for row, _ in drafting]))
            inputs = logits.argmax(-1)
            for row, (_, i) in enumerate(drafting):
                request = items[i].request
                if request.sampling_params.temperature != 0:
                    q = probabilities(logits[row], request.sampling_params)
                    drafted[i].proposals.append(q)
                    inputs[row] = draw(q, generator_for(request, self.device))
                drafted[i].candidates.append(inputs[row])
        return drafted

    @torch.inference_mode()
    def verify(self, drafted: list[DraftedItem]) -> list[SpeculativeResult]:
        ids, positions = [], []
        for entry in drafted:
            item = entry.item
            ids.extend([item.request.request_id] * item.token_count)
            positions.extend(range(item.token_start, item.token_start + item.token_count))
        hidden = torch.stack([h for entry in drafted for h in entry.states])
        hidden = self._cores(
            hidden,
            ids,
            positions,
            range(self.config.draft_loops, self.config.target_loops),
            packed=True,
        )
        logits = self._coda(hidden)
        greedy = any(entry.item.request.sampling_params.temperature == 0 for entry in drafted)
        pending = [candidate.reshape(1) for entry in drafted for candidate in entry.candidates]
        if greedy:
            pending.append(logits.argmax(-1))
        values = torch.cat(pending).tolist() if pending else []
        draft_count = sum(len(entry.candidates) for entry in drafted)
        targets = values[draft_count:]
        results, start, candidate_start = [], 0, 0
        for entry in drafted:
            item = entry.item
            rows = logits[start : start + item.token_count]
            request = item.request
            count = len(entry.candidates)
            candidates = values[candidate_start : candidate_start + count]
            candidate_start += count
            if request.sampling_params.temperature == 0:
                tokens, accepted = greedy_accept(
                    candidates, targets[start : start + item.token_count]
                )
            else:
                tokens, accepted = [], 0
                for offset, candidate in enumerate(candidates):
                    p = probabilities(rows[offset], request.sampling_params)
                    token, accept = rejection_sample(
                        candidate, p, entry.proposals[offset], generator_for(request, self.device)
                    )
                    tokens.append(token)
                    if not accept:
                        break
                    accepted += 1
                else:
                    tokens.append(int(sample_logits(rows[-1], request).item()))
            start += item.token_count
            results.append(SpeculativeResult(tokens, accepted, count))
        self.stats.rounds += len(drafted)
        self.stats.drafted_tokens += draft_count
        self.stats.verified_rows += len(ids)
        return results

    def execute(self, batch):
        return self.verify(self.draft(batch))
