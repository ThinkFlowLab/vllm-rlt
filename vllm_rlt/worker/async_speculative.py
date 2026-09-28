"""Bounded cross-round greedy speculation with device-owned token positions."""

import math
from dataclasses import dataclass

import torch

from vllm_rlt.core.kv_cache_manager import _DeviceKVBatch
from vllm_rlt.worker.speculative import SpeculativeResult, SpeculativeRunner


@dataclass
class DeviceRequest:
    request: object
    allocation: object
    token: torch.Tensor
    position: torch.Tensor
    submitted: int = 0
    delivered: int = 0
    event: object = None


class RoundBank:
    def __init__(self, runner, scheduler):
        cache = runner.cache
        width = math.ceil(runner.model.config.max_position_embeddings / cache.block_size)
        shape = (scheduler.max_num_seqs, cache.max_loops, width)
        self.host_tables = torch.empty(shape, dtype=torch.int32, pin_memory=True)
        self.tables = torch.empty(shape, dtype=torch.int32, device=cache.device)
        size = scheduler.max_num_batched_tokens + 2 * scheduler.max_num_seqs
        self.host_result = torch.empty(size, dtype=torch.int64, pin_memory=True)
        self.result = torch.empty(size, dtype=torch.int64, device=cache.device)
        self.leased = False
        self.copy_done = None
        self.compute_done = None
        self.keepalive = []


@dataclass
class RoundTicket:
    batch: object
    bank: RoundBank
    states: tuple[DeviceRequest, ...]
    rounds: tuple[int, ...]
    rows: int
    cached: object = None

    def ready(self):
        return self.bank.copy_done.query()

    def collect(self):
        if self.cached is None:
            self.bank.copy_done.synchronize()
            b = len(self.batch.items)
            values = self.bank.host_result[: self.rows + 2 * b].tolist()
            results, start = [], 0
            for i, item in enumerate(self.batch.items):
                accepted = values[self.rows + i]
                results.append(
                    SpeculativeResult(
                        values[start : start + accepted + 1],
                        accepted,
                        item.token_count - 1,
                    )
                )
                start += item.token_count
            self.cached = (results, values[self.rows + b :])
        return self.cached

    def retire(self):
        if self.cached is None:
            raise RuntimeError("collect a speculative ticket before retiring its bank")
        self.bank.compute_done.synchronize()
        self.bank.keepalive.clear()
        self.bank.leased = False


class AsyncSpeculativeRunner(SpeculativeRunner):
    """Two submitted rounds may share a request; only GPU state joins them.

    Every request's work runs on one compute stream. CPU token_start is only a
    reservation upper bound and must never be used as the actual write position.
    """

    def __init__(self, model, cache, config, scheduler, *, multi_stream=True):
        super().__init__(model, cache, config)
        self.stream = torch.cuda.Stream(device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device) if multi_stream else self.stream
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.banks = [RoundBank(self, scheduler) for _ in range(2)]
        self.states = {}
        self.submitted_before_collect = 0
        self.discarded_rounds = 0
        self.readback_waits = 0

    def _state(self, request, bank):
        rid = request.request_id
        allocation = self.cache._get_allocation(rid)
        state = self.states.get(rid)
        if state is not None:
            if state.request is not request or state.allocation is not allocation:
                raise RuntimeError("speculative request state reused before retirement")
            return state
        # Bootstrap only: all later input IDs and positions stay on the GPU.
        host = torch.tensor([request.input_token_id, request.position], pin_memory=True)
        values = host.to(self.device, non_blocking=True)
        bank.keepalive.extend((host, values))
        state = DeviceRequest(request, allocation, values[0], values[1])
        self.states[rid] = state
        return state

    def _core_device(self, hidden, positions, tables, depth, allocations):
        table = tables[:, depth].contiguous()
        blocks = table.gather(1, (positions // self.cache.block_size)[:, None]).squeeze(1)
        metadata = _DeviceKVBatch(
            self.cache,
            allocations,
            positions,
            blocks.long(),
            positions % self.cache.block_size,
            table,
            (positions + 1).int(),
        )
        hidden, _ = self.model.recurrent_prepared(
            hidden,
            metadata,
            self.cache,
            compute_gate=False,
        )
        return hidden

    @torch.inference_mode()
    def submit_round(self, batch):
        bank = next((b for b in self.banks if not b.leased), None)
        if bank is None:
            raise RuntimeError("speculative banks exhausted; retire an earlier round")
        bank.leased = True
        items = batch.items
        allocations = tuple(
            (item.request.request_id, self.cache._get_allocation(item.request.request_id))
            for item in items
        )
        # Each bank owns a stable page-table snapshot, including during growth.
        table_view = bank.host_tables.numpy()
        table_view.fill(0)
        width = 1
        for i, (item, (_, allocation)) in enumerate(zip(items, allocations)):
            upper = item.token_start + item.token_count
            if upper > allocation.max_tokens or upper <= 0:
                raise ValueError("speculative round exceeds reserved context")
            pages = math.ceil(upper / self.cache.block_size)
            width = max(width, pages)
            for depth, table in enumerate(allocation.block_tables):
                if len(table) < pages:
                    raise ValueError("speculative round has unallocated KV pages")
                table_view[i, depth, :pages] = table[:pages]
        # If Python fails halfway through submission, engine.synchronize() must
        # drain this stream before any allocation or bank can be released.
        with torch.cuda.stream(self.stream):
            states = tuple(self._state(item.request, bank) for item in items)
            rounds = tuple(s.submitted for s in states)
            if any(s.submitted > s.delivered for s in states):
                self.submitted_before_collect += 1
            bank.tables.copy_(bank.host_tables, non_blocking=True)
            tables = bank.tables[: len(items), :, :width]
            starts = [s.position.clone() for s in states]
            candidates = [[] for _ in items]
            hidden_rows = [[] for _ in items]
            for offset in range(max(item.token_count for item in items)):
                active = [i for i, item in enumerate(items) if offset < item.token_count]
                tokens = torch.stack(
                    [states[i].token if offset == 0 else candidates[i][-1] for i in active]
                )
                positions = torch.stack([starts[i] + offset for i in active])
                row_tables = torch.stack([tables[i] for i in active])
                hidden = self.model.prelude(tokens)
                for depth in range(self.config.draft_loops):
                    hidden = self._core_device(hidden, positions, row_tables, depth, allocations)
                drafting = []
                for row, i in enumerate(active):
                    hidden_rows[i].append(hidden[row])
                    if offset + 1 < items[i].token_count:
                        drafting.append((row, i))
                if drafting:
                    proposed = self.model.coda(
                        torch.stack([hidden[row] for row, _ in drafting])
                    ).argmax(-1)
                    for row, (_, i) in enumerate(drafting):
                        candidates[i].append(proposed[row])
            positions = torch.cat(
                [
                    start + torch.arange(item.token_count, device=self.device)
                    for start, item in zip(starts, items)
                ]
            )
            row_tables = torch.cat(
                [tables[i : i + 1].expand(item.token_count, -1, -1) for i, item in enumerate(items)]
            )
            hidden = torch.stack([h for rows in hidden_rows for h in rows])
            for depth in range(self.config.draft_loops, self.config.target_loops):
                hidden = self._core_device(hidden, positions, row_tables, depth, allocations)
            target = self.model.coda(hidden).argmax(-1)
            count = len(target)
            bank.result[:count].copy_(target)
            start = 0
            for i, (item, state) in enumerate(zip(items, states)):
                rows = target[start : start + item.token_count]
                if candidates[i]:
                    matches = torch.stack(candidates[i]) == rows[:-1]
                    accepted = matches.long().cumprod(0).sum()
                else:
                    accepted = rows.new_zeros(())
                # All target IDs through the first rejection equal the accepted
                # candidates followed by the correction (or final bonus).
                state.token.copy_(rows.gather(0, accepted.reshape(1)).squeeze(0))
                state.position.copy_(starts[i] + accepted + 1)
                bank.result[count + i].copy_(accepted)
                bank.result[count + len(items) + i].copy_(state.position)
                state.submitted += 1
                start += item.token_count
            bank.compute_done = torch.cuda.Event()
            bank.compute_done.record(self.stream)
            for state in states:
                state.event = bank.compute_done
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(bank.compute_done)
            n = count + 2 * len(items)
            bank.host_result[:n].copy_(bank.result[:n], non_blocking=True)
            bank.copy_done = torch.cuda.Event()
            bank.copy_done.record(self.copy_stream)
        self.stats.rounds += len(items)
        self.stats.drafted_tokens += sum(item.token_count - 1 for item in items)
        self.stats.verified_rows += count
        return RoundTicket(batch, bank, states, rounds, count)

    def request_ready(self, request):
        state = self.states.get(request.request_id)
        return state is None or state.event is None or state.event.query()

    def release(self, request):
        state = self.states.get(request.request_id)
        if state is not None:
            if state.request is not request:
                raise RuntimeError("releasing a different speculative request generation")
            if state.event is not None:
                state.event.synchronize()
            del self.states[request.request_id]

    def synchronize(self):
        self.stream.synchronize()
        self.copy_stream.synchronize()
