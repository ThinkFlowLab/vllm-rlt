"""Synchronous block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to one execute call. Requests expose only committed
outputs; the engine applies returned tokens and owns KV commit/truncation.
"""

from dataclasses import dataclass, field

import torch

from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.sampler import RngRegistry, Sampler


@dataclass(frozen=True)
class SpeculativeResult:
    """One request's committed tokens from one round.

    ``token_ids[:accepted_count]`` are accepted draft tokens; ``token_ids[accepted_count]``
    is the correction, or the bonus when ``accepted_count == draft_count``. Token ``j``
    comes from verification row ``j``.
    """

    token_ids: list[int]
    accepted_count: int
    draft_count: int
    # Aligned with token_ids when the request asked for logprobs; else None. Each is the
    # target's logprob of the committed token on its verification row, never the draft's.
    logprobs: list[float] | None = None


@dataclass
class SpeculativeStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    committed_tokens: int = 0
    verified_rows: int = 0


@dataclass
class _CommittedScores:
    """Committed tokens of one round to score, read back together at the end.

    Raw-scored tokens name their verification row in the round's logits; processed-scored
    tokens bring the FP32 processed logits their p was normalized from. Spans map a
    result index to its tokens' contiguous slice within its group.
    """

    raw_rows: list[int] = field(default_factory=list)
    raw_tokens: list[int] = field(default_factory=list)
    raw_spans: dict[int, slice] = field(default_factory=dict)
    processed_logits: list[torch.Tensor] = field(default_factory=list)
    processed_tokens: list[int] = field(default_factory=list)
    processed_spans: dict[int, slice] = field(default_factory=dict)

    def add_raw(self, index, first_row, tokens):
        start = len(self.raw_tokens)
        self.raw_spans[index] = slice(start, start + len(tokens))
        self.raw_rows.extend(range(first_row, first_row + len(tokens)))
        self.raw_tokens.extend(tokens)

    def add_processed(self, index, processed_logits, tokens):
        start = len(self.processed_tokens)
        self.processed_spans[index] = slice(start, start + len(tokens))
        self.processed_logits.extend(processed_logits)
        self.processed_tokens.extend(tokens)

    def resolve(self, logits) -> dict[int, list[float]]:
        """Map each added result index to its tokens' FP32 logprobs.

        At most two batched scoring ops and one host readback; none if nothing was added.
        """
        parts = []
        if self.raw_tokens:
            rows = torch.tensor(self.raw_rows, device=logits.device)
            tokens = torch.tensor(self.raw_tokens, device=logits.device)
            parts.append(Sampler.raw_logprobs(logits[rows], tokens))
        if self.processed_tokens:
            tokens = torch.tensor(self.processed_tokens, device=logits.device)
            parts.append(Sampler.raw_logprobs(torch.stack(self.processed_logits), tokens))
        if not parts:
            return {}
        values = torch.cat(parts).tolist()
        raw, processed = values[: len(self.raw_tokens)], values[len(self.raw_tokens) :]
        scores = {index: raw[span] for index, span in self.raw_spans.items()}
        scores.update((index, processed[span]) for index, span in self.processed_spans.items())
        return scores


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


def rejection_sample(candidate, target, proposal, generator):
    """Return (token, accepted) for normalized, actually sampled p and q.

    Comparing u*q to p avoids division by tiny q. A zero residual after a real
    rejection is a numerical error, never a reason to silently sample from p.
    """
    mass = proposal[candidate]
    if not bool(mass > 0):
        raise ValueError("candidate has zero proposal probability")
    uniform = torch.rand((), device=target.device, generator=generator)
    if bool(uniform * mass < target[candidate]):
        return candidate, True
    residual = (target - proposal).clamp_min(0)
    total = residual.sum()
    if not bool(torch.isfinite(total) & (total > 0)):
        raise RuntimeError("invalid speculative residual probability mass")
    return int(Sampler.draw(residual / total, generator).item()), False


class SpeculativeRunner:
    def __init__(
        self,
        model,
        cache,
        config,
        execution,
        rng: RngRegistry | None = None,
        logprobs_mode="raw_logprobs",
    ):
        self.model, self.cache, self.config = model, cache, config
        self.device = next(model.parameters()).device
        # The engine injects one registry per engine so both sampling paths share
        # each request's RNG slot.
        self.rng = rng if rng is not None else RngRegistry()
        self.logprobs_mode = logprobs_mode
        self.stats = SpeculativeStats()
        self.graphs = (
            RecurrentGraphs(model, cache, execution, False) if execution.cuda_graphs else None
        )
        self.coda_graphs = CodaGraphs(model, execution) if execution.cuda_graphs else None

    def _packed(self, packed):
        return packed and self.cache.attention_capabilities.packed_prefill

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

    def _acquire_generator(self, request, logits):
        """Resolve this request's generator from the shared engine registry.

        The generator has to live on the distribution's device, so it follows the
        logits row rather than the device the model was loaded on.
        """
        return self.rng.acquire(request.request_id, request.sampling_params, logits.device)

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
                    q = Sampler.probabilities(logits[row], request.sampling_params)
                    proposals[i].append(q)
                    # Per-request RNG consumption order is unchanged; only the read moves.
                    inputs[row] = Sampler.draw(q, self._acquire_generator(request, logits[row]))
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
        committed, scores, start = [], _CommittedScores(), 0
        for i, item in enumerate(items):
            rows = logits[start : start + item.token_count]
            params = item.request.sampling_params
            # Greedy rows report raw values in both modes (vLLM). Processed scores of
            # sampled rows reuse the processed logits p was normalized from.
            score_processed = (
                params.logprobs is not None
                and params.temperature != 0
                and self.logprobs_mode == "processed_logprobs"
            )
            if params.temperature == 0:
                tokens, accepted = greedy_accept(
                    candidates[i], targets[start : start + item.token_count]
                )
            else:
                tokens, accepted, processed = self._sample_round(
                    item.request,
                    rows,
                    candidates[i],
                    proposals[i],
                    score_processed=score_processed,
                )
            # Token j was committed on verification row j.
            if score_processed:
                scores.add_processed(i, processed, tokens)
            elif params.logprobs is not None:
                scores.add_raw(i, start, tokens)
            start += item.token_count
            committed.append((tokens, accepted))
        logprobs = scores.resolve(logits)
        results = [
            SpeculativeResult(tokens, accepted, len(candidates[i]), logprobs.get(i))
            for i, (tokens, accepted) in enumerate(committed)
        ]
        self.stats.rounds += len(items)
        self.stats.drafted_tokens += sum(len(c) for c in candidates)
        self.stats.verified_rows += len(ids)
        return results

    def _sample_round(self, request, rows, candidates, proposals, *, score_processed=False):
        """Return (tokens, accepted, processed) by sequential rejection over ``rows``.

        RNG consumption order is unchanged. With ``score_processed``, ``processed`` holds
        the FP32 processed logits of the row each committed token was drawn on, one per
        token; otherwise it is None. A correction is drawn from the residual (p - q)+,
        but the round's output is distributed as p, so every token is scored under p.
        """
        params = request.sampling_params
        processed = [] if score_processed else None

        def target(row):
            if processed is None:
                return Sampler.probabilities(row, params)
            processed.append(Sampler.processed_logits(row, params))
            return processed[-1].softmax(-1)

        tokens, accepted = [], 0
        for offset, candidate in enumerate(candidates):
            token, accept = rejection_sample(
                candidate,
                target(rows[offset]),
                proposals[offset],
                self._acquire_generator(request, rows[offset]),
            )
            tokens.append(token)
            if not accept:
                return tokens, accepted, processed
            accepted += 1
        token = Sampler.draw(target(rows[-1]), self._acquire_generator(request, rows[-1]))
        tokens.append(int(token.item()))
        return tokens, accepted, processed
