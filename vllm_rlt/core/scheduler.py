"""CPU scheduling at stage/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.scheduling_policy import NoRefillPolicy, RefillPolicy, SpeculativePolicy
from vllm_rlt.request import FinishReason, Request, Stage
from vllm_rlt.worker.output import ExitSignal, ModelRunnerOutput, Progress

_VALID_PROGRESS = {
    Stage.PREFILL: frozenset((Progress.COMPLETED, Progress.SUBMITTED)),
    Stage.PRELUDE: frozenset((Progress.COMPLETED, Progress.SUBMITTED)),
    Stage.RECURRENT: frozenset((Progress.COMPLETED, Progress.SUBMITTED)),
    Stage.CODA: frozenset((Progress.COMPLETED, Progress.SUBMITTED, Progress.DELIVERED)),
    Stage.SPECULATIVE: frozenset((Progress.COMPLETED,)),
}


@dataclass(frozen=True)
class ScheduledItem:
    request: Request
    # PREFILL and SPECULATIVE use a contiguous span; ordinary decode uses one row.
    token_start: int = 0
    token_count: int = 1
    # Snapshot at selection, before async submission can advance host state.
    request_id: str | None = None
    generation: int = 0
    position: int | None = None
    loops_done: int | None = None
    output_index: int | None = None


@dataclass(frozen=True)
class SchedulerOutput:
    stage: Stage
    items: list[ScheduledItem]
    # Zero is reserved for manually constructed batches in existing callers.
    seq: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(item.token_count for item in self.items)


@dataclass(slots=True)
class SchedulerUpdate:
    exited: list[tuple[str, int]] = field(default_factory=list)
    finished: list[tuple[str, int, FinishReason]] = field(default_factory=list)
    output_rows: list[ScheduledItem] = field(default_factory=list)
    retain_signal: list[tuple[int, str, int, int, int]] = field(default_factory=list)
    next_prelude: list[int] = field(default_factory=list)
    speculative_committed: list[tuple[int, int]] = field(default_factory=list)


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
    def __init__(
        self,
        config: SchedulerConfig,
        cache_manager,
        speculative_config=None,
        *,
        total_ut_steps=0,
        exit_mode="ouro",
        eos_token_ids=(),
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.speculative_config = speculative_config
        self.total_ut_steps = total_ut_steps
        self.exit_mode = exit_mode
        self.eos_token_ids = frozenset(eos_token_ids)
        self.requests: dict[str, Request] = {}
        self._next_generation = 1
        self._next_seq = 1
        self.queues: dict[Stage, deque[str]] = {s: deque() for s in Stage}
        self.selected_request_ids: set[str] = set()
        # Bound by Engine when preemption is enabled. These callbacks can copy
        # device state and mutate queues; they are operations, not predicates.
        self.preempt_callback = None
        self.resume_callback = None
        policy_cls = NoRefillPolicy if config.mode == "no_refill" else RefillPolicy
        self.policy = (SpeculativePolicy if speculative_config else policy_cls)(config)

    def add_request(self, request: Request):
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        request.generation = self._next_generation
        self._next_generation += 1
        self.requests[request.request_id] = request
        self.queues[Stage.WAITING].append(request.request_id)

    def get_live(self, request_id: str, generation: int) -> Request | None:
        """Return the current registration only when its generation matches."""
        request = self.requests.get(request_id)
        return request if request is not None and request.generation == generation else None

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
        snapshots = dict(
            request_id=request.request_id,
            generation=request.generation,
            position=request.position,
            loops_done=request.loops_done,
            output_index=request.num_scheduled_outputs if stage == Stage.CODA else None,
        )
        if stage == Stage.PREFILL:
            start = request.num_prefilled_tokens
            count = min(
                token_budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
            )
            return ScheduledItem(request, start, count, **snapshots)
        if stage == Stage.SPECULATIVE:
            # K candidates plus one bonus distribution. At the output limit,
            # K=0 is an ordinary fixed-depth step and needs no extra KV slot.
            remaining = request.sampling_params.max_tokens - len(request.generated_token_ids)
            count = min(self.speculative_config.num_speculative_tokens + 1, remaining, token_budget)
            return ScheduledItem(request, request.position, count, **snapshots)
        return ScheduledItem(request, **snapshots)

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
        """Select work whose KV capacity is available.

        Scan only the initial queue; blocked requests return to its tail.
        Selected work advances through update_from_output.
        """
        token_budget = self.config.max_num_batched_tokens
        items = []
        queue = self.queues[stage]
        remaining = len(queue)
        while queue and token_budget and len(items) < self.config.max_num_seqs and remaining:
            remaining -= 1
            request = self.requests[queue.popleft()]
            item = self._make_scheduled_item(request, stage, token_budget)
            if stage in (Stage.PREFILL, Stage.PRELUDE, Stage.RECURRENT, Stage.SPECULATIVE):
                frontier = (
                    item.token_start + item.token_count
                    if stage in (Stage.PREFILL, Stage.SPECULATIVE)
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
        output = SchedulerOutput(stage, items, self._next_seq)
        self._next_seq += 1
        return output

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

    def _should_exit(self, request: Request) -> bool:
        params = request.sampling_params
        maximum = params.max_loops or self.total_ut_steps
        reached_threshold = (
            params.exit_threshold < 1.0
            and request.loops_done >= params.min_loops
            and 1.0 - request.remaining_probability >= params.exit_threshold
        )
        return request.loops_done >= maximum or reached_threshold

    def _delayed_signal(self, request: Request, score: float, signal_depth: int) -> bool:
        params = request.sampling_params
        if self.exit_mode == "ouro_delayed":
            request.remaining_probability *= 1.0 - score
            score = 1.0 - request.remaining_probability
            eligible_depth = signal_depth
        else:
            eligible_depth = signal_depth + 1
        return (
            params.exit_threshold < 1
            and eligible_depth >= params.min_loops
            and score >= params.exit_threshold
        )

    def _trace_exit(self, request: Request) -> bool:
        return request.loops_done >= request.exit_trace[request.num_scheduled_outputs]

    def _recurrent_completed(self, request: Request, signal: ExitSignal | None) -> bool:
        request.loops_done += 1
        if self.exit_mode == "trace":
            return self._trace_exit(request)
        if signal is None:
            raise RuntimeError("missing recurrent exit signal")
        if self.exit_mode == "ouro":
            request.remaining_probability *= 1.0 - signal.score
            return self._should_exit(request)
        maximum = request.sampling_params.max_loops or self.total_ut_steps
        if request.loops_done >= maximum or request.pending_exit_depth == request.loops_done:
            return True
        if self._delayed_signal(request, signal.score, request.loops_done):
            request.pending_exit_depth = request.loops_done + 1
        return False

    def _recurrent_submitted(
        self, request: Request, signal: ExitSignal | None, update: SchedulerUpdate, row: int
    ) -> bool:
        request.loops_done += 1
        maximum = request.sampling_params.max_loops or self.total_ut_steps
        should_exit = (
            self._trace_exit(request)
            if self.exit_mode == "trace"
            else request.loops_done >= maximum
        )
        if signal is not None and not should_exit:
            if signal.position != request.position or signal.signal_depth != request.loops_done - 1:
                raise RuntimeError("stale lookahead signal")
            should_exit = self._delayed_signal(request, signal.score, signal.signal_depth)
        if should_exit:
            request.pending_exit_depth = request.loops_done
        elif (
            self.exit_mode in ("ouro_delayed", "random_lookahead")
            and request.sampling_params.exit_threshold < 1
            and request.loops_done + 1 < maximum
        ):
            update.retain_signal.append(
                (row, request.request_id, request.generation, request.position, request.loops_done)
            )
        return should_exit

    def _record_token(self, request: Request, token: int, depth: int) -> FinishReason | None:
        request.generated_token_ids.append(token)
        request.exit_depths.append(depth)
        params = request.sampling_params
        if token in self.eos_token_ids and not params.ignore_eos:
            return FinishReason.STOP
        if len(request.generated_token_ids) >= params.max_tokens:
            return FinishReason.LENGTH
        return None

    def update_from_output(
        self, batch: SchedulerOutput, result: ModelRunnerOutput
    ) -> SchedulerUpdate:
        """Apply host-visible execution progress without waiting on the runner."""
        if result.seq != batch.seq or result.stage != batch.stage:
            raise ValueError("runner output does not match scheduler batch")
        stage, progress = batch.stage, result.progress
        if progress not in _VALID_PROGRESS.get(stage, ()):
            raise ValueError("unsupported stage/progress combination")
        update = SchedulerUpdate()
        signals = (
            {(s.request_id, s.generation): s for s in result.exit_signals}
            if stage == Stage.RECURRENT
            else {}
        )
        for row, item in enumerate(batch.items):
            request = self.get_live(item.request_id, item.generation)
            if request is None:
                continue
            if stage == Stage.PREFILL:
                request.num_prefilled_tokens += result.prefill_ranges[row][1]
                self.cache_manager.publish_prefix(
                    item.request_id,
                    request.prompt_token_ids,
                    request.num_prefilled_tokens,
                    result.completion[row],
                )
                if request.num_prefilled_tokens == len(request.prompt_token_ids):
                    request.loops_done = self.total_ut_steps
                    self.enqueue(request, Stage.CODA)
                else:
                    self.enqueue(request, Stage.PREFILL)
            elif stage == Stage.PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                request.pending_exit_depth = None
                self.enqueue(request, Stage.RECURRENT)
            elif stage == Stage.RECURRENT:
                signal = signals.get((item.request_id, item.generation))
                if progress == Progress.COMPLETED:
                    should_exit = self._recurrent_completed(request, signal)
                else:
                    should_exit = self._recurrent_submitted(request, signal, update, row)
                if should_exit:
                    update.exited.append((item.request_id, item.generation))
                    self.enqueue(request, Stage.CODA)
                else:
                    self.enqueue(request, Stage.RECURRENT)
            elif stage == Stage.CODA:
                if progress == Progress.SUBMITTED:
                    request.num_output_placeholders += 1
                    if request.num_scheduled_outputs < request.sampling_params.max_tokens:
                        update.next_prelude.append(row)
                        self.enqueue(request, Stage.PRELUDE)
                else:
                    if progress == Progress.DELIVERED:
                        if item.output_index != len(request.generated_token_ids):
                            raise RuntimeError("out-of-order coda delivery")
                        if request.num_output_placeholders != 1:
                            raise RuntimeError("invalid pending output count")
                        request.num_output_placeholders -= 1
                    reason = self._record_token(
                        request, result.sampled_token_ids[row], item.loops_done
                    )
                    if reason is not None:
                        update.finished.append((item.request_id, item.generation, reason))
                    elif progress == Progress.COMPLETED:
                        self.enqueue(
                            request,
                            Stage.SPECULATIVE if self.speculative_config else Stage.PRELUDE,
                        )
                    update.output_rows.append(item)
            elif stage == Stage.SPECULATIVE:
                spec = result.speculative[row]
                emitted = 0
                reason = None
                for token in spec.token_ids:
                    reason = self._record_token(
                        request, token, self.speculative_config.target_loops
                    )
                    emitted += 1
                    if reason is not None:
                        break
                update.speculative_committed.append((emitted, min(spec.accepted_count, emitted)))
                if reason is not None:
                    update.finished.append((item.request_id, item.generation, reason))
                else:
                    # The final emitted token has not been forwarded yet.
                    self.cache_manager.truncate_suffix(item.request_id, item.token_start + emitted)
                    request.loops_done = 0
                    self.enqueue(request, Stage.SPECULATIVE)
                update.output_rows.append(item)
        return update
