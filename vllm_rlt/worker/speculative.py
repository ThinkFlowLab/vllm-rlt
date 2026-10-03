"""Synchronous block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to one execute call. Requests expose only committed
outputs; the engine applies returned tokens and owns KV commit/truncation.
"""

from dataclasses import dataclass

import torch

from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.sampling import (
    draw,
    generator_for,
    probabilities,
    rejection_sample,
)


@dataclass(frozen=True)
class SpeculativeResult:
    token_ids: list[int]
    accepted_count: int
    draft_count: int
    # Aligned with token_ids when the request asked for logprobs; else None.
    logprobs: list[float] | None = None


@dataclass
class SpeculativeStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    verified_rows: int = 0


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
    def execute(self, batch):
        items = batch.items
        proposals = [[] for _ in items]
        states = [[] for _ in items]
        # Draft IDs stay on the device: each offset's IDs are the next offset's
        # prelude input, and all of them are read back once after verification.
        drafted = []
        inputs = self.cache._stage([item.request.input_token_id for item in items], torch.long)
        # Each item includes the final shallow input needed for the bonus row.
        # Drafting batches independent requests at every autoregressive step.
        for offset in range(max(item.token_count for item in items)):
            # The rows still drafting at the previous offset are exactly this
            # offset's active rows, in the same order, so their IDs are the inputs.
            active = [i for i, item in enumerate(items) if offset < item.token_count]
            ids = [items[i].request.request_id for i in active]
            positions = [items[i].token_start + offset for i in active]
            hidden = self.model.prelude(inputs)
            hidden = self._cores(hidden, ids, positions, range(self.config.draft_loops))
            drafting = []
            for row, i in enumerate(active):
                states[i].append(hidden[row])
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
                    proposals[i].append(q)
                    # Per-request RNG consumption order is unchanged; only the read moves.
                    inputs[row] = draw(q, generator_for(request, self.device))
            drafted.append(([i for _, i in drafting], inputs))
        # Group contiguous positions per request for causal ragged attention.
        ids, positions = [], []
        for item in items:
            ids.extend([item.request.request_id] * item.token_count)
            positions.extend(range(item.token_start, item.token_start + item.token_count))
        hidden = torch.stack([h for request_states in states for h in request_states])
        hidden = self._cores(
            hidden,
            ids,
            positions,
            range(self.config.draft_loops, self.config.target_loops),
            packed=True,
        )
        logits = self._coda(hidden)
        # One device-to-host read returns every draft ID and, for greedy requests,
        # every verification target argmax.
        greedy = any(item.request.sampling_params.temperature == 0 for item in items)
        pending = [draft for _, draft in drafted] + ([logits.argmax(-1)] if greedy else [])
        values = torch.cat(pending).tolist() if pending else []
        candidates = [[] for _ in items]
        start = 0
        for rows, _ in drafted:
            for i, candidate in zip(rows, values[start : start + len(rows)]):
                candidates[i].append(candidate)
            start += len(rows)
        targets = values[start:]
        results, start = [], 0
        for i, item in enumerate(items):
            rows = logits[start : start + item.token_count]
            request = item.request
            if request.sampling_params.temperature == 0:
                tokens, accepted = greedy_accept(
                    candidates[i], targets[start : start + item.token_count]
                )
                start += item.token_count
                results.append(SpeculativeResult(tokens, accepted, len(candidates[i])))
                continue
            start += item.token_count
            # Sampling keeps sequential rejection; RNG consumption order is unchanged.
            tokens, accepted = [], 0
            for offset, candidate in enumerate(candidates[i]):
                p = probabilities(rows[offset], request.sampling_params)
                token, accept = rejection_sample(
                    candidate, p, proposals[i][offset], generator_for(request, self.device)
                )
                tokens.append(token)
                if not accept:
                    break
                accepted += 1
            else:
                token = draw(
                    probabilities(rows[-1], request.sampling_params),
                    generator_for(request, self.device),
                )
                tokens.append(int(token.item()))
            results.append(SpeculativeResult(tokens, accepted, len(candidates[i])))
        self.stats.rounds += len(items)
        self.stats.drafted_tokens += sum(len(c) for c in candidates)
        self.stats.verified_rows += len(ids)
        return results
