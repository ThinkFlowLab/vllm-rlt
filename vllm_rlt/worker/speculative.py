"""Submitted block self-speculation with depth-specific KV/activation reuse.

Temporary tensors belong to a submission until collection. Requests expose only committed
outputs; the engine applies returned tokens and owns KV commit/truncation.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm_rlt.core.execution import Operation
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.execution import ExecutionRuntime, ForwardBatch
from vllm_rlt.worker.sampler import RngRegistry, Sampler


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


@dataclass
class SpeculativeSubmission:
    runner: "SpeculativeRunner"
    batch: SchedulerOutput
    logits: torch.Tensor | None
    drafted: list
    proposals: list
    event: object = None
    cached: list[SpeculativeResult] | None = None

    @property
    def has_result(self):
        return True

    def ready(self):
        return self.event is None or self.event.query()

    def collect(self):
        if self.cached is None:
            if self.event is not None:
                self.event.synchronize()
            stream = self.runner.compute_stream
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                self.cached = self.runner._collect_round(self)
            self.logits = None
            self.drafted.clear()
            self.proposals.clear()
        return self.cached


class SpeculativeRunner:
    def __init__(
        self,
        model,
        cache,
        config,
        execution,
        rng: RngRegistry | None = None,
        *,
        runtime=None,
        compute_stream=None,
    ):
        self.model, self.cache, self.config = model, cache, config
        self.runtime = runtime if runtime is not None else ExecutionRuntime(model, cache)
        self.device = next(model.parameters()).device
        self.compute_stream = compute_stream
        # The engine injects one registry per engine so both sampling paths share
        # each request's RNG slot.
        self.rng = rng if rng is not None else RngRegistry()
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
        return self.runtime.advance(hidden, batches, graphs=self.graphs).hidden

    def _core(self, hidden, ids, positions, depth, *, packed=False, batch=None):
        if batch is None:
            batch = self.cache._prepare_batch(
                ids, [depth] * len(ids), positions, packed_prefill=self._packed(packed)
            )
        hidden, _ = self.runtime.advance(hidden, (batch,), graphs=self.graphs)
        return hidden

    def _acquire_generator(self, request, logits):
        """Resolve this request's generator from the shared engine registry.

        The generator has to live on the distribution's device, so it follows the
        logits row rather than the device the model was loaded on.
        """
        return self.rng.acquire(request.request_id, request.sampling_params, logits.device)

    def _coda(self, hidden):
        return self.runtime.forward(
            ForwardBatch(
                Operation.CODA,
                hidden=hidden,
                graphs=self.coda_graphs,
            )
        ).logits

    @torch.inference_mode()
    def submit(self, batch):
        try:
            if self.compute_stream is not None:
                self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
            with (
                torch.cuda.stream(self.compute_stream)
                if self.compute_stream is not None
                else nullcontext()
            ):
                return self._submit_round(batch)
        except Exception:
            # Failed submissions may not have recorded a completion event.
            if self.device.type == "cuda":
                (self.compute_stream or torch.cuda.current_stream(self.device)).synchronize()
            raise

    def _submit_round(self, batch):
        scheduled_requests = batch.requests
        proposals = [[] for _ in scheduled_requests]
        states = [[] for _ in scheduled_requests]
        # Draft IDs stay on the device: each offset's IDs are the next offset's
        # prelude input, and all of them are read back once after verification.
        drafted = []
        inputs = self.cache._stage(
            [scheduled_request.request.input_token_id for scheduled_request in scheduled_requests],
            torch.long,
        )
        # Each scheduled request includes the final shallow input needed for the bonus row.
        # Drafting batches independent requests at every autoregressive step.
        for offset in range(
            max(scheduled_request.token_count for scheduled_request in scheduled_requests)
        ):
            # The rows still drafting at the previous offset are exactly this
            # offset's active rows, in the same order, so their IDs are the inputs.
            active = [
                i
                for i, scheduled_request in enumerate(scheduled_requests)
                if offset < scheduled_request.token_count
            ]
            ids = [scheduled_requests[i].request.request_id for i in active]
            positions = [scheduled_requests[i].token_start + offset for i in active]
            hidden = self.runtime.forward(ForwardBatch(Operation.PRELUDE, tokens=inputs)).hidden
            hidden = self._cores(hidden, ids, positions, range(self.config.draft_loops))
            drafting = []
            for row, i in enumerate(active):
                states[i].append(hidden[row])
                if offset + 1 < scheduled_requests[i].token_count:
                    drafting.append((row, i))
            if not drafting:
                break
            logits = self._coda(torch.stack([hidden[row] for row, _ in drafting]))
            inputs = logits.argmax(-1)
            for row, (_, i) in enumerate(drafting):
                request = scheduled_requests[i].request
                if request.sampling_params.temperature != 0:
                    q = Sampler.probabilities(logits[row], request.sampling_params)
                    proposals[i].append(q)
                    # Per-request RNG consumption order is unchanged; only the read moves.
                    inputs[row] = Sampler.draw(q, self._acquire_generator(request, logits[row]))
            drafted.append(([i for _, i in drafting], inputs))
        # Group contiguous positions per request for causal ragged attention.
        ids, positions = [], []
        for scheduled_request in scheduled_requests:
            ids.extend([scheduled_request.request.request_id] * scheduled_request.token_count)
            positions.extend(
                range(
                    scheduled_request.token_start,
                    scheduled_request.token_start + scheduled_request.token_count,
                )
            )
        hidden = torch.stack([h for request_states in states for h in request_states])
        hidden = self._cores(
            hidden,
            ids,
            positions,
            range(self.config.draft_loops, self.config.target_loops),
            packed=True,
        )
        logits = self._coda(hidden)
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        self.stats.rounds += len(scheduled_requests)
        self.stats.drafted_tokens += sum(len(rows) for rows, _ in drafted)
        self.stats.verified_rows += len(ids)
        return SpeculativeSubmission(self, batch, logits, drafted, proposals, event)

    @torch.inference_mode()
    def _collect_round(self, submission):
        scheduled_requests = submission.batch.requests
        logits, drafted, proposals = submission.logits, submission.drafted, submission.proposals
        # One device-to-host read returns every draft ID and, for greedy requests,
        # every verification target argmax.
        greedy = any(
            scheduled_request.request.sampling_params.temperature == 0
            for scheduled_request in scheduled_requests
        )
        pending = [draft for _, draft in drafted] + ([logits.argmax(-1)] if greedy else [])
        values = torch.cat(pending).tolist() if pending else []
        candidates = [[] for _ in scheduled_requests]
        start = 0
        for rows, _ in drafted:
            for i, candidate in zip(rows, values[start : start + len(rows)]):
                candidates[i].append(candidate)
            start += len(rows)
        targets = values[start:]
        results, start = [], 0
        for i, scheduled_request in enumerate(scheduled_requests):
            rows = logits[start : start + scheduled_request.token_count]
            request = scheduled_request.request
            if request.sampling_params.temperature == 0:
                tokens, accepted = greedy_accept(
                    candidates[i], targets[start : start + scheduled_request.token_count]
                )
                start += scheduled_request.token_count
                results.append(SpeculativeResult(tokens, accepted, len(candidates[i])))
                continue
            start += scheduled_request.token_count
            # Sampling keeps sequential rejection; RNG consumption order is unchanged.
            tokens, accepted = [], 0
            for offset, candidate in enumerate(candidates[i]):
                p = Sampler.probabilities(rows[offset], request.sampling_params)
                token, accept = rejection_sample(
                    candidate,
                    p,
                    proposals[i][offset],
                    self._acquire_generator(request, rows[offset]),
                )
                tokens.append(token)
                if not accept:
                    break
                accepted += 1
            else:
                token = Sampler.draw(
                    Sampler.probabilities(rows[-1], request.sampling_params),
                    self._acquire_generator(request, rows[-1]),
                )
                tokens.append(int(token.item()))
            results.append(SpeculativeResult(tokens, accepted, len(candidates[i])))
        return results
