"""Lossless pressure preemption for recurrent-depth KV.

CPU snapshots retain exact KV planes, hidden state and request RNG ownership.
Full-depth replay of generated tokens is deliberately avoided: it changes RLT
semantics. Snapshots are bounded by the admitted request population.
"""

from copy import deepcopy

import torch

from vllm_rlt.request import Stage


class PreemptionManager:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}
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

    def preempt(self, requester, *, priority_only=False):
        """Suspend one other request, preserving its recurrent execution state.

        Returns False if no safe victim exists. True means the victim's device
        resources have been released and it has been moved to WAITING; the
        requester must still retry its own allocation.
        """
        victim = self._select_preemption_victim(requester, priority_only=priority_only)
        if victim is None:
            return False
        e = self.engine
        e.model_runner.synchronize()
        cache = e.cache_manager
        allocation = cache._get_allocation(victim.request_id)
        blocks = [b for table in allocation.block_tables for b in table]
        # Copy views one page at a time: a pressure recovery must not allocate
        # another request-sized temporary on an already full GPU.
        keys = torch.empty((len(blocks), *cache.key_cache.shape[1:]), dtype=cache.dtype)
        values = torch.empty_like(keys)
        for row, block in enumerate(blocks):
            keys[row].copy_(cache.key_cache[block])
            values[row].copy_(cache.value_cache[block])
        snapshot = dict(
            stage=victim.stage,
            maximum=allocation.max_tokens,
            pages=len(allocation.block_tables[0]),
            written=deepcopy(allocation.written),
            keys=keys,
            values=values,
            hidden=None if victim.hidden_state is None else victim.hidden_state.cpu().clone(),
            token=None
            if victim.input_token_tensor is None
            else victim.input_token_tensor.cpu().clone(),
        )
        e.model_runner.release(victim.request_id)
        cache.poll_prefixes()
        cache.free(victim.request_id)
        for q in e.scheduler.queues.values():
            while victim.request_id in q:
                q.remove(victim.request_id)
        victim.hidden_state = victim.input_token_tensor = None
        self.snapshots[victim.request_id] = snapshot
        e.scheduler.enqueue(victim, Stage.WAITING)
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
        e, cache = self.engine, self.engine.cache_manager
        frontier = min(state["maximum"], state["pages"] * cache.block_size)
        if not cache.allocate(request.request_id, state["maximum"], initial_tokens=frontier):
            return False
        allocation = cache._get_allocation(request.request_id)
        blocks = [b for table in allocation.block_tables for b in table]
        for row, block in enumerate(blocks):
            cache.key_cache[block].copy_(state["keys"][row])
            cache.value_cache[block].copy_(state["values"][row])
        allocation.written = state["written"]
        request.hidden_state = None if state["hidden"] is None else state["hidden"].to(cache.device)
        request.input_token_tensor = (
            None if state["token"] is None else state["token"].to(cache.device)
        )
        # Restored buffers become visible to every runner stream before resumption.
        if cache.device.type == "cuda":
            torch.cuda.current_stream(cache.device).synchronize()
        e.scheduler.enqueue(request, state["stage"])
        del self.snapshots[request.request_id]
        self.resumptions += 1
        return True
