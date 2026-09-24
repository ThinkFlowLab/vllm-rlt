"""CPU scheduling at stage/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.scheduling_policy import NoRefillPolicy, RefillPolicy, SpeculativePolicy
from vllm_rlt.request import FinishReason, Request, Stage


@dataclass
class PrefillTask:
    task_id: int
    request: Request
    token_start: int
    token_count: int
    depth: int
    hidden_state: object = None
    event: object = None


@dataclass(frozen=True)
class ScheduledItem:
    request: Request
    # PREFILL and SPECULATIVE use a contiguous span; ordinary decode uses one row.
    token_start: int = 0
    token_count: int = 1
    prefill_task: PrefillTask | None = None


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
    def __init__(self, config: SchedulerConfig, cache_manager, speculative_config=None):
        self.config = config
        self.cache_manager = cache_manager
        self.speculative_config = speculative_config
        self.requests: dict[str, Request] = {}
        self.queues: dict[Stage, deque[str]] = {s: deque() for s in Stage}
        self.selected_request_ids: set[str] = set()
        self.prefill_tasks: dict[int, PrefillTask] = {}
        self.prefill_ready: deque[int] = deque()
        self._next_prefill_task_id = 0
        # Bound by Engine when preemption is enabled. These callbacks can copy
        # device state and mutate queues; they are operations, not predicates.
        self.preempt_callback = None
        self.resume_callback = None
        policy_cls = NoRefillPolicy if config.mode == "no_refill" else RefillPolicy
        self.policy = (SpeculativePolicy if speculative_config else policy_cls)(config)

    def add_request(self, request: Request):
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        self.requests[request.request_id] = request
        self.queues[Stage.WAITING].append(request.request_id)

    def enqueue(self, request: Request, stage: Stage):
        request.stage = stage
        if stage == Stage.PREFILL and self.config.wavefront_prefill:
            if request.request_id not in self.queues[Stage.PREFILL]:
                self.queues[Stage.PREFILL].append(request.request_id)
            if not any(
                task.request.request_id == request.request_id
                for task in self.prefill_tasks.values()
            ):
                self._enqueue_prefill_task(
                    request,
                    request.num_prefilled_tokens,
                    min(
                        self.config.prefill_chunk_size,
                        self.config.max_num_batched_tokens,
                        len(request.prompt_token_ids) - request.num_prefilled_tokens,
                    ),
                    0,
                )
            return
        self.queues[stage].append(request.request_id)

    def _enqueue_prefill_task(
        self, request: Request, token_start: int, token_count: int, depth: int, hidden_state=None
    ) -> PrefillTask:
        if token_count <= 0:
            raise ValueError("prefill task must contain at least one token")
        task = PrefillTask(
            self._next_prefill_task_id,
            request,
            token_start,
            token_count,
            depth,
            hidden_state,
        )
        self._next_prefill_task_id += 1
        self.prefill_tasks[task.task_id] = task
        self.prefill_ready.append(task.task_id)
        if request.request_id not in self.queues[Stage.PREFILL]:
            self.queues[Stage.PREFILL].append(request.request_id)
        return task

    def _remove_prefill_request(self, request_id: str) -> None:
        queue = self.queues[Stage.PREFILL]
        while request_id in queue:
            queue.remove(request_id)

    def _prefill_task_ready(self, task: PrefillTask) -> bool:
        """Check hidden-state and same-depth causal-prefix readiness."""
        if task.hidden_state is None and task.depth:
            return False
        allocation = self.cache_manager._get_allocation(task.request.request_id)
        plane = self.cache_manager._plane(task.depth)
        return all(
            allocation.written[plane][layer].prefix >= task.token_start
            for layer in range(self.cache_manager.num_layers)
        )

    def _clear_prefill_tasks(self, request_id: str) -> None:
        removed = {
            task_id
            for task_id, task in self.prefill_tasks.items()
            if task.request.request_id == request_id
        }
        for task_id in removed:
            self.prefill_tasks.pop(task_id, None)
        if removed:
            self.prefill_ready = deque(
                task_id for task_id in self.prefill_ready if task_id not in removed
            )

    def advance_prefill_task(self, task: PrefillTask) -> bool:
        """Retire one depth task and enqueue its dependency successors.

        Returns whether the request's entire prompt has reached the final depth.
        The next chunk at depth zero can run as soon as the current chunk's depth
        zero completes; deeper tasks retain the previous depth's hidden state.
        """
        current = self.prefill_tasks.pop(task.task_id, None)
        if current is not task:
            raise RuntimeError("stale prefill task completion")
        request = task.request
        if task.depth + 1 < self.cache_manager.max_loops:
            self._enqueue_prefill_task(
                request,
                task.token_start,
                task.token_count,
                task.depth + 1,
                task.hidden_state,
            )
        if task.depth == 0:
            next_start = task.token_start + task.token_count
            if next_start < len(request.prompt_token_ids):
                self._enqueue_prefill_task(
                    request,
                    next_start,
                    min(
                        self.config.prefill_chunk_size,
                        self.config.max_num_batched_tokens,
                        len(request.prompt_token_ids) - next_start,
                    ),
                    0,
                )
        if task.depth != self.cache_manager.max_loops - 1:
            return False
        request.prefill_completed_chunks[task.token_start] = (
            task.token_start + task.token_count
        )
        while request.num_prefilled_tokens in request.prefill_completed_chunks:
            request.num_prefilled_tokens = request.prefill_completed_chunks.pop(
                request.num_prefilled_tokens
            )
        if request.num_prefilled_tokens == len(request.prompt_token_ids):
            self._remove_prefill_request(request.request_id)
            return True
        return False

    def finish(self, request: Request, reason: FinishReason):
        # A delayed EOS can stop a request already queued for its next stage.
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        if self.config.wavefront_prefill:
            self._clear_prefill_tasks(request.request_id)
            self._remove_prefill_request(request.request_id)
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
        if self.config.wavefront_prefill:
            self._enqueue_prefill_task(
                request,
                plan.cached_tokens,
                min(
                    self.config.prefill_chunk_size,
                    self.config.max_num_batched_tokens,
                    len(request.prompt_token_ids) - plan.cached_tokens,
                ),
                0,
            )
            request.stage = Stage.PREFILL
        else:
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
        self, request: Request, stage: Stage, token_budget: int, task: PrefillTask | None = None
    ) -> ScheduledItem:
        """Bound one request's work without advancing its completed progress.

        Prompt length 10, prefilled 4, chunk 4, budget 6 -> range [4, 8).
        Decode stages contribute one position; its loop may differ by request.
        """
        if stage == Stage.PREFILL and task is not None:
            if task.token_count > token_budget:
                raise ValueError("prefill task exceeds the available token budget")
            return ScheduledItem(
                request,
                task.token_start,
                task.token_count,
                task,
            )
        if stage == Stage.PREFILL:
            start = request.num_prefilled_tokens
            count = min(
                token_budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
            )
            return ScheduledItem(request, start, count)
        if stage == Stage.SPECULATIVE:
            # K candidates plus one bonus distribution. At the output limit,
            # K=0 is an ordinary fixed-depth step and needs no extra KV slot.
            remaining = request.sampling_params.max_tokens - len(request.generated_token_ids)
            count = min(self.speculative_config.num_speculative_tokens + 1, remaining, token_budget)
            return ScheduledItem(request, request.position, count)
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
        if stage == Stage.PREFILL and self.config.wavefront_prefill:
            for request_id in tuple(self.queues[Stage.PREFILL]):
                if not any(
                    task.request.request_id == request_id for task in self.prefill_tasks.values()
                ):
                    request = self.requests[request_id]
                    if request.num_prefilled_tokens < len(request.prompt_token_ids):
                        self._enqueue_prefill_task(
                            request,
                            request.num_prefilled_tokens,
                            min(
                                self.config.prefill_chunk_size,
                                self.config.max_num_batched_tokens,
                                len(request.prompt_token_ids) - request.num_prefilled_tokens,
                            ),
                            0,
                        )
            queue = self.prefill_ready
        else:
            queue = self.queues[stage]
        remaining = len(queue)
        while queue and token_budget and len(items) < self.config.max_num_seqs and remaining:
            remaining -= 1
            entry = queue.popleft()
            task = (
                self.prefill_tasks[entry]
                if stage == Stage.PREFILL and self.config.wavefront_prefill
                else None
            )
            request = task.request if task is not None else self.requests[entry]
            if task is not None and not self._prefill_task_ready(task):
                queue.append(entry)
                continue
            if task is not None and task.token_count > token_budget:
                queue.append(entry)
                continue
            item = self._make_scheduled_item(request, stage, token_budget, task)
            if stage in (Stage.PREFILL, Stage.PRELUDE, Stage.RECURRENT, Stage.SPECULATIVE):
                frontier = (
                    item.token_start + item.token_count
                    if stage in (Stage.PREFILL, Stage.SPECULATIVE)
                    else request.position + 1
                )
                if not self._ensure_execution_capacity(request, frontier):
                    queue.append(entry)
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
