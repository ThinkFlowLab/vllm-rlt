"""Stage execution; the scheduler never invokes a whole-token model forward."""

import torch

from vllm_lt.core.scheduler import SchedulerOutput
from vllm_lt.request import Request, Stage


class ModelRunner:
    def __init__(self, model, cache_manager):
        self.model = model.eval()
        self.cache_manager = cache_manager
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def execute(self, batch: SchedulerOutput):
        requests = [item.request for item in batch.items]
        if batch.stage == Stage.PREFILL:
            ids, positions, tokens = [], [], []
            for item in batch.items:
                start, count, request = item.token_start, item.token_count, item.request
                ids.extend([request.request_id] * count)
                positions.extend(range(start, start + count))
                tokens.extend(request.prompt_token_ids[start : start + count])
            hidden = self.model.prelude(torch.tensor(tokens, device=self.device, dtype=torch.long))
            # Every prompt token reaches the model's full depth, independently of decode policy.
            for depth in range(self.model.config.total_ut_steps):
                hidden, _ = self.model.recurrent(
                    hidden, ids, [depth] * len(ids), positions, self.cache_manager
                )
            offset = 0
            for item in batch.items:
                offset += item.token_count
                item.request.hidden_state = hidden[offset - 1].clone()
            return None
        if batch.stage == Stage.PRELUDE:
            token_ids = torch.tensor(
                [r.input_token_id for r in requests], device=self.device, dtype=torch.long
            )
            hidden = self.model.prelude(token_ids)
            for request, state in zip(requests, hidden):
                request.hidden_state = state
            return None
        hidden = torch.stack([r.hidden_state for r in requests])
        if batch.stage == Stage.RECURRENT:
            hidden, gate_logits = self._recurrent(
                hidden,
                [r.request_id for r in requests],
                [r.loops_done for r in requests],
                [r.position for r in requests],
            )
            for request, state in zip(requests, hidden):
                request.hidden_state = state
            # This is explicitly synchronous. A stock gate cannot act as the paper's lookahead gate.
            return gate_logits.float().sigmoid().cpu().tolist()
        if batch.stage == Stage.CODA:
            logits = self.model.coda(hidden)
            return [self._sample(row, request) for row, request in zip(logits, requests)]
        raise ValueError(f"unsupported execution stage {batch.stage}")

    def _recurrent(self, hidden, request_ids, depths, positions):
        """Private decode seam; normal generation always uses compact rows."""
        return self.model.recurrent(hidden, request_ids, depths, positions, self.cache_manager)

    def _recurrent_padded(
        self,
        hidden,
        request_ids,
        depths,
        positions,
        *,
        row_indices,
        row_count,
        table_width,
    ):
        """Exercise padding eagerly, returning only live rows in scheduler order.

        No scheduler item, coda input or request generator is manufactured for
        padding. Gathered output owns storage independently of the physical
        traversal, which may outlive publication to a paused request.
        """
        if hidden.ndim != 2 or hidden.shape != (len(request_ids), self.model.config.hidden_size):
            raise ValueError("padded recurrent requires compact live hidden inputs")
        batch = self.cache_manager._prepare_batch(request_ids, depths, positions)
        batch = self.cache_manager._pad_prepared(
            batch, row_indices=row_indices, row_count=row_count, table_width=table_width
        )
        live = torch.tensor(batch.live_rows, device=self.device, dtype=torch.long)
        physical = hidden.new_zeros((row_count, hidden.shape[1]))
        physical.index_copy_(0, live, hidden)
        physical, gates = self.model._recurrent_prepared(physical, batch, self.cache_manager)
        return physical.index_select(0, live), gates.index_select(0, live)

    def _sample(self, logits: torch.Tensor, request: Request) -> int:
        params = request.sampling_params
        if params.temperature == 0:
            return int(logits.argmax().item())
        logits = logits.float() / params.temperature
        if params.top_k > 0:
            threshold = logits.topk(min(params.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if params.top_p < 1:
            sorted_logits, indices = logits.sort(descending=True)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
        if request.generator is None:
            request.generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return int(torch.multinomial(logits.softmax(-1), 1, generator=request.generator).item())
