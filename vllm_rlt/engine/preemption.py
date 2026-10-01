"""Lossless pressure preemption for recurrent-depth KV.

CPU snapshots retain exact KV planes, hidden state and request RNG ownership.
Full-depth replay of generated tokens is deliberately avoided: it changes RLT
semantics. Snapshots are bounded by the admitted request population.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

import torch

from vllm_rlt.request import Request, Stage


@dataclass(frozen=True)
class RequestMigration:
    request: Request
    state: dict
    rng_state: torch.Tensor | None
    signature: tuple


class PreemptionManager:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}
        self.inflight_ids = set()
        self.preemptions = self.resumptions = 0

    def discard_snapshot(self, request_id: str) -> None:
        """Discard saved CPU state when a request will not be resumed."""
        self.snapshots.pop(request_id, None)

    def _is_preemption_candidate(self, request, requester, *, priority_only):
        """Check safety before considering a victim's priority.

        Pending outputs, transfer leases and this batch's selected requests must
        remain alive. priority_only additionally forbids evicting equal/higher
        priority work to admit a new arrival.
        """
        e = self.engine
        if (
            request is requester
            or request.stage in (Stage.WAITING, Stage.RECEIVING)
            or request.num_output_placeholders
            or request.request_id in e.scheduler.selected_request_ids
            or request.request_id in self.inflight_ids
        ):
            return False
        if e.cache_manager._get_allocation(request.request_id).transfer_leases:
            return False
        return not priority_only or (
            request.sampling_params.priority > requester.sampling_params.priority
        )

    def _select_preemption_victim(self, requester, *, priority_only):
        """Prefer lower priority, then a larger current-position KV footprint.

        Priority 10 loses to priority 0. The footprint tie-breaker uses
        required_blocks(position + 1), not exact reclaimable physical pages.
        Capacity-pressure preemption retains this ranking even in FCFS mode;
        priority_only restricts eligibility only for priority admission.
        """
        candidates = (
            request
            for request in self.engine.scheduler.requests.values()
            if self._is_preemption_candidate(request, requester, priority_only=priority_only)
        )
        return max(
            candidates,
            key=lambda request: (
                request.sampling_params.priority,
                self.engine.cache_manager.required_blocks(request.position + 1),
            ),
            default=None,
        )

    def _signature(self):
        e, cache = self.engine, self.engine.cache_manager
        return (
            e.model.config,
            cache.layout,
            cache.block_size,
            cache.dtype,
            cache.key_cache.shape[1:],
            cache.attention_info,
            cache.device.type,
        )

    def _capture(self, request):
        e = self.engine
        e.model_runner.synchronize()
        cache = e.cache_manager
        allocation = cache._get_allocation(request.request_id)
        blocks = [b for table in allocation.block_tables for b in table]
        keys = torch.empty((len(blocks), *cache.key_cache.shape[1:]), dtype=cache.dtype)
        values = torch.empty_like(keys)
        for row, block in enumerate(blocks):
            keys[row].copy_(cache.key_cache[block])
            values[row].copy_(cache.value_cache[block])
        return dict(
            stage=request.stage,
            maximum=allocation.max_tokens,
            pages=len(allocation.block_tables[0]),
            written=deepcopy(allocation.written),
            keys=keys,
            values=values,
            hidden=None if request.hidden_state is None else request.hidden_state.cpu().clone(),
            token=None
            if request.input_token_tensor is None
            else request.input_token_tensor.cpu().clone(),
        )

    def _release(self, request):
        e = self.engine
        e.model_runner.release(request.request_id)
        e.cache_manager.poll_prefixes()
        e.cache_manager.free(request.request_id)
        for queue in e.scheduler.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        request.hidden_state = request.input_token_tensor = None

    def export_request(self, request_id: str) -> RequestMigration:
        """Detach a committed speculative request as a CPU-owned migration packet."""
        e = self.engine
        request = e.scheduler.requests[request_id]
        if (
            request.stage != Stage.SPECULATIVE
            or request_id in self.inflight_ids
            or request_id in e.scheduler.selected_request_ids
            or request.num_output_placeholders
            or e.cache_manager._get_allocation(request_id).transfer_leases
        ):
            raise RuntimeError("migration requires a committed speculative round boundary")
        state = self._capture(request)
        saved = replace(
            request,
            prompt_token_ids=list(request.prompt_token_ids),
            generated_token_ids=list(request.generated_token_ids),
            exit_depths=list(request.exit_depths),
            hidden_state=None,
            input_token_tensor=None,
            generator=None,
        )
        packet = RequestMigration(
            saved,
            state,
            None if request.generator is None else request.generator.get_state().clone(),
            self._signature(),
        )
        self._release(request)
        e.scheduler.requests.pop(request_id)
        return packet

    def import_request(self, packet: RequestMigration) -> bool:
        """Restore a migration packet; False leaves it intact for a later retry."""
        e = self.engine
        request = packet.request
        if packet.signature != self._signature():
            raise ValueError("migration source and target model/cache differ")
        if request.request_id in e.scheduler.requests:
            raise ValueError("duplicate migrated request ID")
        if not self._restore_state(request, packet.state):
            return False
        if packet.rng_state is not None:
            request.generator = torch.Generator(device=e.cache_manager.device)
            request.generator.set_state(packet.rng_state)
        e.scheduler.requests[request.request_id] = request
        e.scheduler.enqueue(request, packet.state["stage"])
        return True

    def preempt(self, requester, *, priority_only=False):
        """Suspend one other request, preserving its recurrent execution state.

        Returns False if no safe victim exists. True means the victim's device
        resources have been released and it has been moved to WAITING; the
        requester must still retry its own allocation.
        """
        victim = self._select_preemption_victim(requester, priority_only=priority_only)
        if victim is None:
            return False
        self.snapshots[victim.request_id] = self._capture(victim)
        self._release(victim)
        self.engine.scheduler.enqueue(victim, Stage.WAITING)
        self.preemptions += 1
        return True

    def resume(self, request):
        """Restore pages/state and re-enqueue the saved stage.

        None means no snapshot; False means insufficient capacity; True means
        restoration and enqueue both succeeded. Scheduler translates this
        callback contract into a named ResumeResult.
        """
        state = self.snapshots.get(request.request_id)
        if state is None:
            return None
        if not self._restore_state(request, state):
            return False
        self.engine.scheduler.enqueue(request, state["stage"])
        del self.snapshots[request.request_id]
        self.resumptions += 1
        return True

    def _restore_state(self, request, state):
        cache = self.engine.cache_manager
        frontier = min(state["maximum"], state["pages"] * cache.block_size)
        if not cache.allocate(request.request_id, state["maximum"], initial_tokens=frontier):
            return False
        allocation = cache._get_allocation(request.request_id)
        blocks = [b for table in allocation.block_tables for b in table]
        for row, block in enumerate(blocks):
            cache.key_cache[block].copy_(state["keys"][row])
            cache.value_cache[block].copy_(state["values"][row])
        allocation.written = deepcopy(state["written"])
        request.hidden_state = None if state["hidden"] is None else state["hidden"].to(cache.device)
        request.input_token_tensor = (
            None if state["token"] is None else state["token"].to(cache.device)
        )
        # Restored buffers become visible to every runner stream before resumption.
        if cache.device.type == "cuda":
            torch.cuda.current_stream(cache.device).synchronize()
        return True
