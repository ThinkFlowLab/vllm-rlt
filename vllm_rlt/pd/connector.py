"""PD transfer adapter. Request computation remains in StageEngine."""

import time
from collections import deque
from dataclasses import dataclass, field, replace

import torch

from vllm_rlt.core.execution import WorkKey
from vllm_rlt.request import Request

from .transport import kv_segments, partition_segments


@dataclass
class RequestTransfer:
    tid: str
    request: Request
    slot: int
    peer: str
    outgoing: bool = False
    trace_id: str | None = None
    target_tables: tuple = ()
    target_slot: int = 0
    cached_tokens: int = 0
    target_cached_tokens: int = 0
    chunks: deque = field(default_factory=deque)
    transfer_boundary_count: int = 0
    transfer_boundary_first: tuple[int, int] | None = None
    transfer_boundary_last: tuple[int, int] | None = None
    transfer_overlaps: int = 0
    received: set = field(default_factory=set)
    next_sequence: int = 0
    expected: int | None = None
    committed: bool = False
    cancelled: bool = False
    active: bool = False
    compute_done: bool = False
    compute_released: bool = False
    last_event: object = None
    acknowledged: bool = False
    started: float = field(default_factory=time.monotonic)


class PDStageConnector:
    def __init__(self, engine, hidden, transport, config, *, notify):
        self.engine = engine
        self.cache = engine.cache_manager
        self.hidden = hidden
        self.transport = transport
        self.config = config
        self.notify = notify
        self.transfers = {}
        self.free_slots = list(reversed(range(len(hidden))))
        self.transfer_cursor = 0
        self.prefill_tokens = 0

    def _remove(self, transfer):
        self.engine.release_transfer(transfer.request)
        self._forget(transfer)

    def _forget(self, transfer):
        self.free_slots.append(transfer.slot)
        del self.transfers[transfer.tid]
        self.notify("released", tid=transfer.tid, role=self.engine.stage_config.role)

    def output_finished(self, request_id):
        """The engine has retired a completed decode request; release its staging slot."""
        self._forget(self.transfers[request_id])

    def exports(self, request):
        transfer = self.transfers.get(request.request_id)
        return transfer is not None and transfer.request is request and transfer.outgoing

    def can_schedule(self, request):
        if not self.exports(request):
            return True
        # Ready handoffs must reach the transport before another prompt can
        # block this owner thread on GPU submission backpressure (PR #122).
        if any(
            transfer.outgoing and transfer.chunks and transfer.chunks[0][0].query()
            for transfer in self.transfers.values()
        ):
            return False
        transfer = self.transfers.get(request.request_id)
        return bool(
            transfer is not None
            and transfer.request is request
            and not transfer.cancelled
            and not transfer.chunks
            and (transfer.last_event is None or transfer.last_event.query())
            and self.transport.inflight_bytes < self.config.max_inflight_bytes
        )

    def prefill_submitted(self, request, *, work, start, progress, event, stream):
        transfer = self.transfers[request.request_id]
        if transfer.request is not request:
            raise RuntimeError("stale transfer generation")
        if work == WorkKey.PROMPT_PRELUDE:
            transfer.transfer_overlaps += self.transport.busy(transfer.tid)
        if not progress.chunk_complete:
            transfer.last_event = event
            return
        stream = stream or torch.cuda.current_stream(self.cache.device)
        with torch.cuda.stream(stream):
            if event is not None:
                stream.wait_event(event)
            if progress.prompt_complete:
                self.hidden[transfer.slot].copy_(request.hidden_state)
            ready = torch.cuda.Event()
            ready.record(stream)
        transfer.last_event = ready
        self.prefill_tokens += request.num_prefilled_tokens - start
        self.queue_chunk(
            transfer, start, request.num_prefilled_tokens, ready, progress.prompt_complete
        )
        transfer.compute_done = progress.prompt_complete

    def queue_chunk(self, transfer, start, end, event, final):
        peer = self.transport.peers[transfer.peer]
        tables = self.cache.plane_block_tables(transfer.tid)
        start = max(start, transfer.target_cached_tokens)
        if start < end:
            boundary = (start, end)
            if transfer.transfer_boundary_first is None:
                transfer.transfer_boundary_first = boundary
            transfer.transfer_boundary_last = boundary
            transfer.transfer_boundary_count += 1
        source = list(kv_segments(self.transport.info, tables, start, end)) if start < end else []
        target = list(kv_segments(peer, transfer.target_tables, start, end)) if start < end else []
        if final:
            size = self.transport.info["hidden_bytes"]
            source.append(
                (
                    self.transport.info["hidden_ptr"] + transfer.slot * size,
                    size,
                    self.transport.info["device"],
                )
            )
            target.append((peer["hidden_ptr"] + transfer.target_slot * size, size, peer["device"]))
        for local, remote, size in partition_segments(
            source, target, self.config.transfer_chunk_bytes, self.config.max_transfer_descriptors
        ):
            transfer.chunks.append((event, transfer.next_sequence, local, remote, size))
            transfer.next_sequence += 1

    def reserve(
        self,
        tid,
        tokens,
        params,
        *,
        receiving,
        peer,
        trace_id=None,
        target_tables=(),
        target_slot=0,
        target_cached_tokens=0,
    ):
        """Reserve one local Request and a connector-owned staging slot per generation."""
        if tid in self.transfers:
            raise RuntimeError("duplicate transfer generation")
        if not self.free_slots:
            self.notify("rejected", tid=tid, role=self.engine.stage_config.role)
            return
        # P owns only prompt KV and never samples. D owns all sampling/RNG.
        actual_params = replace(params, max_tokens=1) if not receiving else params
        request = self.engine.reserve_transfer(
            tid,
            tokens,
            actual_params,
            receiving=receiving,
            trace_id=trace_id,
        )
        if request is None:
            self.notify("rejected", tid=tid, role=self.engine.stage_config.role)
            return
        hit = request.num_prefilled_tokens
        transfer = RequestTransfer(
            tid,
            request,
            self.free_slots.pop(),
            peer,
            outgoing=not receiving,
            trace_id=trace_id,
        )
        transfer.cached_tokens = hit
        self.transfers[tid] = transfer
        if receiving:
            self.notify(
                "reserved",
                tid=tid,
                tables=self.cache.plane_block_tables(tid),
                slot=transfer.slot,
                cached_tokens=hit,
            )
        else:
            transfer.target_tables, transfer.target_slot = target_tables, target_slot
            transfer.target_cached_tokens = target_cached_tokens
            if hit > transfer.target_cached_tokens:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.cache.device))
                self.queue_chunk(transfer, transfer.target_cached_tokens, hit, event, False)
            self.notify("prefill_started", tid=tid)

    def cancel(self, transfer_id):
        transfer = self.transfers.get(transfer_id)
        if transfer is None:
            self.notify("released", tid=transfer_id, role=self.engine.stage_config.role)
            return
        transfer.cancelled = True
        if transfer.outgoing:
            transfer.chunks.clear()
            self.engine.suspend_transfer(transfer.request)
        elif transfer.active:
            self._remove(transfer)  # Activation proves all remote writes have ended.

    def cancel_safe(self, transfer_id):
        """Coordinator confirms remote writers stopped; resources may now be retired."""
        transfer = self.transfers.get(transfer_id)
        if transfer is not None:
            if not transfer.cancelled:
                raise RuntimeError("unexpected cancellation acknowledgement")
            if transfer.outgoing:
                raise RuntimeError("source cancellation must drain local and remote accesses")
            self._remove(transfer)

    def commit(self, transfer_id, count):
        transfer = self.transfers.get(transfer_id)
        if transfer is None or transfer.cancelled:
            return
        if transfer.outgoing:
            raise RuntimeError("commit requires an incoming transfer")
        if transfer.expected is not None and transfer.expected != count:
            raise RuntimeError("conflicting transfer commit")
        if type(count) is not int or count <= 0:
            raise RuntimeError("invalid transfer commit count")
        transfer.expected = count

    def acknowledge(self, transfer_id):
        transfer = self.transfers.get(transfer_id)
        if transfer is not None:
            if not transfer.outgoing:
                raise RuntimeError("acknowledgement requires an outgoing transfer")
            transfer.acknowledged = True

    def progress(self):
        self.cache.poll_prefixes()
        _, notifications = self.transport.poll()
        for peer, tid, sequence in notifications:
            transfer = self.transfers.get(tid)
            if transfer is None or transfer.cancelled:
                continue  # Old generations cannot activate a reused request ID.
            if (
                transfer.outgoing
                or peer != transfer.peer
                or type(sequence) is not int
                or sequence < 0
            ):
                raise RuntimeError("invalid transfer completion")
            if transfer.active and sequence not in transfer.received:
                raise RuntimeError("late KV write after decode activation")
            transfer.received.add(sequence)
        transfers = list(self.transfers.values())
        if transfers:
            start = self.transfer_cursor % len(transfers)
            transfers = transfers[start:] + transfers[:start]
            self.transfer_cursor += 1
        if self.engine.scheduler.config.policy == "priority":
            # Only import activation is priority-ranked; exports retain round robin.
            imports = iter(
                sorted(
                    (transfer for transfer in transfers if not transfer.outgoing),
                    key=lambda transfer: (
                        transfer.request.sampling_params.priority,
                        transfer.started,
                    ),
                )
            )
            transfers = [transfer if transfer.outgoing else next(imports) for transfer in transfers]
        for transfer in transfers:
            if transfer.outgoing:
                if transfer.cancelled:
                    if not self.transport.busy(transfer.tid) and (
                        transfer.last_event is None or transfer.last_event.query()
                    ):
                        # The hidden staging copy may outlive the model event.
                        # Do not reuse its slot until both local and remote users finish.
                        self._remove(transfer)
                    continue
                if (
                    transfer.compute_done
                    and not transfer.compute_released
                    and self.engine.release_transfer_compute(
                        transfer.request, ready_event=transfer.last_event
                    )
                ):
                    transfer.compute_released = True
                    self.notify(
                        "prefill_complete",
                        tid=transfer.tid,
                        boundaries=(
                            transfer.transfer_boundary_first,
                            transfer.transfer_boundary_last,
                            transfer.transfer_boundary_count,
                        ),
                        overlaps=transfer.transfer_overlaps,
                    )
                # Round robin: at most one bounded transfer per request per tick.
                if transfer.chunks:
                    event, seq, local, remote, size = transfer.chunks[0]
                    if event.query() and self.transport.submit(
                        transfer.tid, seq, transfer.peer, local, remote, size
                    ):
                        transfer.chunks.popleft()
                if transfer.compute_done and not transfer.chunks and not transfer.committed:
                    transfer.committed = True
                    self.notify("commit", tid=transfer.tid, count=transfer.next_sequence)
                if (
                    transfer.acknowledged
                    and transfer.committed
                    and transfer.compute_released
                    and not self.transport.busy(transfer.tid)
                ):
                    self._remove(transfer)
            elif not transfer.cancelled and not transfer.active and transfer.expected is not None:
                if any(seq >= transfer.expected for seq in transfer.received):
                    raise RuntimeError("unexpected transfer sequence")
                if (
                    len(transfer.received) == transfer.expected
                    and self.engine.import_capacity_available()
                ):
                    if not self.engine.activate_import(
                        transfer.request,
                        self.hidden[transfer.slot],
                        cached_tokens=transfer.cached_tokens,
                    ):
                        raise RuntimeError("stale import activation")
                    transfer.active = True
                    self.notify(
                        "activated", tid=transfer.tid, seconds=time.monotonic() - transfer.started
                    )
