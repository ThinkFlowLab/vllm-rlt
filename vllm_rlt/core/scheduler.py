"""CPU scheduling at stage/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.scheduling_policy import NoRefillPolicy, RefillPolicy
from vllm_rlt.request import FinishReason, Request, Stage


@dataclass(frozen=True)
class ScheduledItem:
    request: Request
    # Used by prefill only; decode always schedules one position per request.
    token_start: int = 0
    token_count: int = 1


@dataclass(frozen=True)
class SchedulerOutput:
    stage: Stage
    items: list[ScheduledItem]

    @property
    def num_tokens(self) -> int:
        return sum(item.token_count for item in self.items)


class ResumeResult(Enum):
    NOT_PREEMPTED = auto()
    RESTORED = auto()
    BLOCKED = auto()


@dataclass(frozen=True)
class AdmissionPlan:
    """One candidate's allocation inputs and admission budget, not a reservation."""

    capacity_tokens: int
    initial_tokens: int
    prefix_blocks: tuple[tuple[int, ...], ...]
    cached_tokens: int
    required_blocks: int
    reserved_growth_blocks: int
    admission_headroom_blocks: int
    cached_claim_blocks: int

    @property
    def total_budget_blocks(self) -> int:
        return (
            self.required_blocks
            + self.reserved_growth_blocks
            + self.admission_headroom_blocks
            + self.cached_claim_blocks
        )


class Scheduler:
    def __init__(self, config: SchedulerConfig, cache_manager):
        self.config = config
        self.cache_manager = cache_manager
        self.requests: dict[str, Request] = {}
        self.queues: dict[Stage, deque[str]] = {s: deque() for s in Stage}
        self.selected_request_ids: set[str] = set()
        # Bound by Engine when preemption is enabled. These callbacks can copy
        # device state and mutate queues; they are operations, not predicates.
        self.preempt_callback = None
        self.resume_callback = None
        policy_cls = NoRefillPolicy if config.mode == "no_refill" else RefillPolicy
        self.policy = policy_cls(config)

    def add_request(self, request: Request):
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        self.requests[request.request_id] = request
        self.queues[Stage.WAITING].append(request.request_id)

    def enqueue(self, request: Request, stage: Stage):
        request.stage = stage
        self.queues[stage].append(request.request_id)

    def finish(self, request: Request, reason: FinishReason):
        # A delayed EOS can stop a request already queued for its next stage.
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        self.cache_manager.free(request.request_id)
        request.stage = Stage.FINISHED
        request.finish_reason = reason
        request.hidden_state = None
        request.input_token_tensor = None
        request.num_output_placeholders = 0
        request.generator = None
        self.requests.pop(request.request_id)

    def abort(self, request_id: str) -> Request:
        request = self.requests[request_id]
        self.finish(request, FinishReason.ABORT)
        return request

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def _order_waiting_requests(self) -> deque[str]:
        """Return the live waiting queue in admission order.

        Priority is supplied by the caller, not estimated from prompt length.
        Lower values go first: -5, 0, 10. Equal values retain current queue order.
        FCFS preserves queue order, but admission may still bypass blocked work.
        """
        waiting = self.queues[Stage.WAITING]
        if self.config.policy == "priority":
            ordered = sorted(waiting, key=lambda rid: self.requests[rid].sampling_params.priority)
            waiting.clear()
            waiting.extend(ordered)
        return waiting

    def _ensure_active_slot(self, requester: Request, active_count: int) -> int | None:
        """Return the updated active count, or None when admission must stop.

        A full engine may make room for a higher-priority arrival. The callback
        suspends another request and moves it to WAITING; it does not admit the
        requester. This operation can synchronize the runner and copy KV to CPU.
        """
        if active_count < self.config.max_num_seqs:
            return active_count
        if (
            self.config.policy == "priority"
            and self.preempt_callback is not None
            and self.preempt_callback(requester, priority_only=True)
        ):
            return active_count - 1
        return None

    def _try_resume(self, request: Request) -> ResumeResult:
        """Translate the existing restoration callback's three-way result.

        None: no snapshot, continue fresh admission.
        True: resources/state restored and the original stage already enqueued.
        False: snapshot exists, but restoration cannot obtain capacity yet.
        """
        restored = self.resume_callback(request) if self.resume_callback is not None else None
        if restored is None:
            return ResumeResult.NOT_PREEMPTED
        return ResumeResult.RESTORED if restored else ResumeResult.BLOCKED

    def _reserved_growth_blocks(self, *, reserve_outputs: bool) -> int:
        """Budget future growth of existing requests without allocating it.

        Example: a request needs 12 blocks eventually but owns 4. Without
        preemption, reserve the remaining 8 even though they still look free.
        With preemption, protect in-progress prompts; decode growth may instead
        reclaim capacity by suspending another request later.
        """
        cache = self.cache_manager
        reserved = 0
        for request in self.requests.values():
            active = request.stage not in (Stage.WAITING, Stage.RECEIVING)
            if request.stage != Stage.PREFILL and not (reserve_outputs and active):
                continue
            tokens = len(request.prompt_token_ids)
            if reserve_outputs:
                tokens += request.sampling_params.max_tokens - 1
            allocated = (
                len(cache._get_allocation(request.request_id).block_tables[0])
                * cache.storage_depths
            )
            reserved += max(0, cache.required_blocks(tokens) - allocated)
        return reserved

    def _plan_admission(self, request: Request, active_count: int) -> AdmissionPlan:
        """Describe prefix reuse, physical allocation and logical admission cost.

        With 8 prompt tokens and 5 outputs, capacity is 12: the last sampled
        output is never fed back into the model. Incremental allocation may
        acquire fewer pages now, while budgeting the remaining growth.

        Prefix lookup can publish completed prefixes and update cache LRU order;
        planning does not allocate request pages, but is not a pure cache read.
        All block counts include the cache's storage-depth multiplier.
        """
        cache = self.cache_manager
        capacity = len(request.prompt_token_ids) + request.sampling_params.max_tokens - 1
        prefix = cache.lookup_prefix(request.prompt_token_ids)
        cached_tokens = len(prefix) * cache.block_size
        initial_tokens = min(
            len(request.prompt_token_ids), cached_tokens + self.config.prefill_chunk_size
        )
        if not cache.incremental_allocation:
            initial_tokens = capacity
        reserve_outputs = cache.incremental_allocation and self.preempt_callback is None
        budget_tokens = (
            len(request.prompt_token_ids)
            if cache.incremental_allocation and not reserve_outputs
            else capacity
        )
        required = cache.required_blocks(budget_tokens) - len(prefix) * cache.storage_depths
        reserved = self._reserved_growth_blocks(reserve_outputs=reserve_outputs)
        # Allow an otherwise idle engine to start a large request.
        admission_headroom_blocks = cache.watermark_blocks if active_count else 0
        # num_free_blocks includes evictable prefix-only blocks. Reusing those
        # blocks makes them non-evictable: do not count the same capacity twice.
        # Example: free=8 fresh + 4 cached, demand=12 with a 4-block prefix.
        # New demand is 8, but the cached claim is 4: admission costs 12, not 8.
        cached_claim = sum(cache._refs[b] == 1 for group in prefix for b in group)
        return AdmissionPlan(
            capacity_tokens=capacity,
            initial_tokens=initial_tokens,
            prefix_blocks=prefix,
            cached_tokens=cached_tokens,
            required_blocks=required,
            reserved_growth_blocks=reserved,
            admission_headroom_blocks=admission_headroom_blocks,
            cached_claim_blocks=cached_claim,
        )

    def _commit_admission(self, request: Request, plan: AdmissionPlan) -> bool:
        """Allocate pages and enter PREFILL only when the whole budget fits."""
        cache = self.cache_manager
        if plan.total_budget_blocks > cache.num_free_blocks:
            return False
        if not cache.allocate(
            request.request_id,
            plan.capacity_tokens,
            initial_tokens=plan.initial_tokens,
            prefix=plan.prefix_blocks,
        ):
            return False
        request.num_prefilled_tokens = plan.cached_tokens
        self.enqueue(request, Stage.PREFILL)
        return True

    def _admit(self):
        """Admit waiting work: order -> slot -> restore/allocate -> defer/enter.

        A blocked long request may let a short one pass. Each successful fresh
        admission increments earlier blocked requests' bypass counts; reaching
        the limit stops further bypasses so active work can drain.
        """
        active_count = sum(
            r.stage not in (Stage.WAITING, Stage.RECEIVING) for r in self.requests.values()
        )
        waiting = self._order_waiting_requests()
        deferred = []
        scan_count = min(len(waiting), self.config.admission_scan_limit)
        for _ in range(scan_count):
            if not waiting:
                break
            # Peek before popping: preemption may append its victim to WAITING.
            requester = self.requests[waiting[0]]
            available_count = self._ensure_active_slot(requester, active_count)
            if available_count is None:
                break
            active_count = available_count
            request_id = waiting.popleft()
            request = self.requests[request_id]

            resume_result = self._try_resume(request)
            if resume_result == ResumeResult.RESTORED:
                active_count += 1
                continue
            if resume_result == ResumeResult.BLOCKED:
                deferred.append(request_id)
                continue

            plan = self._plan_admission(request, active_count)
            if not self._commit_admission(request, plan):
                deferred.append(request_id)
                if request.admission_bypasses >= self.config.max_admission_bypasses:
                    break
                continue

            active_count += 1
            for blocked_id in deferred:
                self.requests[blocked_id].admission_bypasses += 1
            if any(
                self.requests[rid].admission_bypasses >= self.config.max_admission_bypasses
                for rid in deferred
            ):
                break
        # [A, B] deferred must precede the untouched tail in the same order.
        waiting.extendleft(reversed(deferred))

    def _make_scheduled_item(
        self, request: Request, stage: Stage, token_budget: int
    ) -> ScheduledItem:
        """Bound one request's work without advancing its completed progress.

        Prompt length 10, prefilled 4, chunk 4, budget 6 -> range [4, 8).
        Decode stages contribute one position; its loop may differ by request.
        """
        if stage == Stage.PREFILL:
            start = request.num_prefilled_tokens
            count = min(
                token_budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
            )
            return ScheduledItem(request, start, count)
        return ScheduledItem(request)

    def _ensure_execution_capacity(self, request: Request, frontier: int) -> bool:
        """Grow KV for this step; if needed, preempt one other request and retry.

        frontier is an exclusive token count, not a loop count. Another loop at
        the same token position normally uses existing capacity. The callback
        excludes requests already selected into this batch.
        """
        cache = self.cache_manager
        if cache.ensure_capacity(request.request_id, frontier):
            return True
        if self.preempt_callback is None or not self.preempt_callback(request):
            return False
        return cache.ensure_capacity(request.request_id, frontier)

    def _take(self, stage: Stage) -> SchedulerOutput | None:
        """Build a stage batch: describe work -> ensure capacity -> select.

        Requests that cannot grow return to the tail. Scan at most the initial
        queue length so a blocked request cannot cycle forever in this call.
        Successful items leave the queue; the engine later updates progress and
        enqueues their next stage after handling execution results.
        """
        token_budget = self.config.max_num_batched_tokens
        items = []
        queue = self.queues[stage]
        remaining = len(queue)
        while queue and token_budget and len(items) < self.config.max_num_seqs and remaining:
            remaining -= 1
            request = self.requests[queue.popleft()]
            item = self._make_scheduled_item(request, stage, token_budget)
            if stage in (Stage.PREFILL, Stage.PRELUDE, Stage.RECURRENT):
                frontier = (
                    item.token_start + item.token_count
                    if stage == Stage.PREFILL
                    else request.position + 1
                )
                if not self._ensure_execution_capacity(request, frontier):
                    queue.append(request.request_id)
                    continue
            # Selecting A must prevent B's later capacity check from evicting A.
            self.selected_request_ids.add(request.request_id)
            items.append(item)
            token_budget -= item.token_count
        if not items:
            return None
        self.policy.record_batch(stage)
        return SchedulerOutput(stage, items)

    def schedule(self, *, prefer_recurrent=False) -> SchedulerOutput | None:
        """Choose existing work or an admission opportunity, then build a batch.

        An unfinished request is not necessarily runnable (e.g. PD RECEIVING).
        Conversely, one long active request must not prevent new requests from
        filling available slots. The stage policy decides when to call _admit;
        merely checking whether requests exist must not allocate or preempt.
        """
        self.selected_request_ids.clear()
        if not self.requests:
            return None
        return self.policy.schedule(self, prefer_recurrent=prefer_recurrent)
