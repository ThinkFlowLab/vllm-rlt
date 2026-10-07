from dataclasses import replace

from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.output_adapter import (
    SignalHandle,
    adapt_async_delivery,
    adapt_async_submit,
    adapt_speculative_execute,
    adapt_sync_execute,
)
from vllm_rlt.engine.preemption import PreemptionManager
from vllm_rlt.kernels.flash_attention import FLASH_BACKENDS
from vllm_rlt.profiling import Profiler
from vllm_rlt.request import FinishReason, Request, RequestOutput, Stage
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.output import ExitSignal
from vllm_rlt.worker.speculative import SpeculativeRunner


class LLMEngine:
    """Single-device loop-level engine with synchronous or pipelined scheduling."""

    def __init__(
        self,
        model,
        *,
        cache_config=None,
        scheduler_config=None,
        attention_backend="torch",
        exit_config=None,
        execution_config=None,
        speculative_config=None,
    ):
        self.model = model
        self.profiling = Profiler(next(model.parameters()).device)
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()
        parameter = next(model.parameters())
        config = model.config
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = execution_config or ExecutionConfig()
        self.speculative_config = speculative_config
        if speculative_config is not None:
            if cache_config.layout != "last_exited":
                raise ValueError("speculative decoding requires last_exited KV")
            if speculative_config.target_loops != config.total_ut_steps:
                raise ValueError("speculative target_loops must equal the model full depth")
            if self.exit_config.mode != "ouro":
                raise ValueError("speculative decoding requires fixed-depth ouro exit mode")
            if self.execution_config.async_scheduling:
                raise ValueError("speculative decoding currently requires synchronous execution")
            if scheduler_config.enable_preemption or scheduler_config.mode != "refill":
                raise ValueError(
                    "speculative decoding requires refill scheduling without preemption"
                )
            # Retained q distributions and sampling scratch coexist with target
            # logits. Reserve beyond ordinary prefill/core profiling, even when
            # requests later choose sampling rather than greedy.
            scratch = scheduler_config.max_num_batched_tokens * (
                config.vocab_size * 24 + config.hidden_size * parameter.element_size() * 2
            )
            cache_config = replace(
                cache_config, memory_reserve_bytes=cache_config.memory_reserve_bytes + scratch
            )
        if self.execution_config.cuda_graphs and (
            parameter.device.type != "cuda" or attention_backend not in ("triton", *FLASH_BACKENDS)
        ):
            raise ValueError("CUDA graphs require CUDA with Triton or FlashAttention")
        if self.execution_config.async_scheduling:
            if self.exit_config.mode not in ("ouro_delayed", "random_lookahead", "trace"):
                raise ValueError(
                    "async scheduling requires ouro_delayed, random_lookahead or trace exit mode"
                )
            if parameter.device.type == "cuda" and attention_backend not in (
                "triton",
                *FLASH_BACKENDS,
            ):
                raise ValueError("CUDA async scheduling requires Triton or FlashAttention")
        num_blocks, self.memory_plan = plan_cache(
            model, cache_config, scheduler_config, self.execution_config, attention_backend
        )
        self.cache_manager = KVCacheManager(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_loops=config.total_ut_steps,
            num_blocks=num_blocks,
            layout=cache_config.layout,
            block_size=cache_config.block_size,
            device=parameter.device,
            dtype=parameter.dtype,
            backend=attention_backend,
            enable_prefix_caching=cache_config.enable_prefix_caching,
            incremental_allocation=cache_config.incremental_allocation,
            watermark_ratio=cache_config.watermark_ratio,
        )
        if self.execution_config.prefill_uva and (
            parameter.device.type != "cuda"
            or cache_config.layout != "last_exited"
            or getattr(self.cache_manager.attention, "generation", None) != 4
        ):
            raise ValueError("prefill_uva requires CUDA FA4 with last_exited KV")
        eos = config.eos_token_id
        eos_ids = eos if isinstance(eos, (tuple, list)) else (eos,)
        self.scheduler = Scheduler(
            scheduler_config,
            self.cache_manager,
            speculative_config,
            total_ut_steps=config.total_ut_steps,
            exit_mode=self.exit_config.mode,
            eos_token_ids=eos_ids,
        )
        self.speculative_runner = (
            SpeculativeRunner(model, self.cache_manager, speculative_config, self.execution_config)
            if speculative_config is not None
            else None
        )
        self.model_runner = ModelRunner(
            model,
            self.cache_manager,
            exit_config=self.exit_config,
            execution_config=self.execution_config,
            scheduler_config=scheduler_config,
        )
        self._exit_traces = {
            key: tuple(values) for key, values in (self.exit_config.depths_by_request or {}).items()
        }
        # Delayed scores are read after the next core submission.
        self._pending_exit_signals = {}
        self._pending_coda = []
        self._inflight = []
        self._overlap_boundary = False
        self.last_schedule = None
        self.preemption = PreemptionManager(self)
        if scheduler_config.enable_preemption:
            self.scheduler.preempt_callback = self.preemption.preempt
            self.scheduler.resume_callback = self.preemption.resume

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        trace_id: str | None = None,
    ):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        params = sampling_params or SamplingParams()
        config = self.model.config
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if any(type(t) is not int or not 0 <= t < config.vocab_size for t in prompt_token_ids):
            raise ValueError("prompt token IDs must be integers within the model vocabulary")
        max_loops = params.max_loops or config.total_ut_steps
        if max_loops > config.total_ut_steps or params.min_loops > max_loops:
            raise ValueError("requested loop bounds exceed the model's supported depth")
        if self.speculative_config is not None and (
            max_loops != self.speculative_config.target_loops or params.exit_threshold != 1.0
        ):
            raise ValueError("speculative requests require fixed target depth and exit_threshold=1")
        if trace_id is not None:
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError("trace_id must be a nonempty string")
            if self.exit_config.mode != "trace":
                raise ValueError("trace_id requires trace exit mode")
        trace = ()
        if self.exit_config.mode == "trace":
            key = request_id if trace_id is None else trace_id
            if key not in self._exit_traces:
                raise ValueError(f"unknown exit trace: {key!r}; specify a configured trace_id")
            trace = self._exit_traces[key]
            if (
                len(trace) < params.max_tokens
                or trace[0] != config.total_ut_steps
                or any(
                    type(d) is not int or not params.min_loops <= d <= max_loops
                    for d in trace[1 : params.max_tokens]
                )
            ):
                raise ValueError(
                    "exit trace must cover every output and obey prefill/decode bounds"
                )
        capacity = len(prompt_token_ids) + params.max_tokens - 1
        if capacity > config.max_position_embeddings:
            raise ValueError("prompt plus decode positions exceed the model context length")
        cache = self.cache_manager
        required = cache.required_blocks(capacity)
        if required > cache.num_blocks:
            raise ValueError(
                f"request requires {required} KV blocks but cache has {cache.num_blocks}; "
                "increase num_blocks or reduce prompt/max_tokens"
            )
        self.scheduler.add_request(
            Request(request_id, list(prompt_token_ids), params, exit_trace=trace)
        )

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests

    def _is_live(self, item) -> bool:
        return self.scheduler.get_live(item.request_id, item.generation) is not None

    def abort_request(self, request_id: str) -> RequestOutput:
        self.preemption.discard_snapshot(request_id)
        self.model_runner.release(request_id)
        self._pending_exit_signals.pop(request_id, None)
        return RequestOutput.from_request(self.scheduler.abort(request_id))

    def start_profile(self, config, *, scheduled=False, **identity):
        return self.profiling.start(config, scheduled=scheduled, **identity)

    def stop_profile(self):
        return self.profiling.stop()

    def profile_status(self):
        return self.profiling.status()

    def wait_for_profile_artifacts(self, timeout=None):
        return self.profiling.wait(timeout)

    def close(self):
        self.profiling.close()

    def step(self) -> list[RequestOutput]:
        if not self.profiling.recording:
            return self._step()
        try:
            outputs = self._step()
            self.profiling.step()
            return outputs
        except BaseException:
            self.profiling.stop()
            raise

    def _step(self) -> list[RequestOutput]:
        if self.execution_config.async_scheduling:
            try:
                return self._step_async()
            except Exception:
                # A submission can fail before recording an event. Drain owned
                # streams before invalidating requests or recycling any memory.
                self.model_runner.synchronize()
                for rid in list(self.scheduler.requests):
                    self.abort_request(rid)
                self._pending_exit_signals.clear()
                self._pending_coda.clear()
                self._inflight.clear()
                raise
        return self._step_sync()

    def _step_sync(self) -> list[RequestOutput]:
        batch = self.scheduler.schedule()
        self.last_schedule = batch
        if batch is None:
            # PD imports wait for external KV completion; yield to the IPC loop.
            if any(r.stage != Stage.RECEIVING for r in self.scheduler.requests.values()):
                raise RuntimeError("scheduler made no progress")
            return []
        try:
            if batch.stage == Stage.SPECULATIVE:
                raw = self.speculative_runner.execute(batch)
                result = adapt_speculative_execute(batch, raw)
            else:
                raw = self.model_runner.execute(batch)
                result = adapt_sync_execute(batch, raw)
            update = self.scheduler.update_from_output(batch, result)
            return self._apply_scheduler_update(batch, update)
        except Exception:
            # A failed execution or result application invalidates this batch.
            for item in batch.items:
                if self._is_live(item):
                    self.abort_request(item.request_id)
            raise

    def _finish(self, request: Request, reason: FinishReason) -> None:
        self.model_runner.release(request.request_id)
        self.cache_manager.poll_prefixes()
        self._pending_exit_signals.pop(request.request_id, None)
        self.scheduler.finish(request, reason)

    def _apply_scheduler_update(self, batch, update, ticket=None) -> list[RequestOutput]:
        if batch.stage == Stage.PRELUDE:
            for item in batch.items:
                if self._is_live(item):
                    self._pending_exit_signals.pop(item.request_id, None)
        if update.exited:
            requests = []
            for rid, generation in update.exited:
                request = self.scheduler.get_live(rid, generation)
                if request is not None:
                    requests.append(request)
            self.model_runner.finalize_many(requests)
        for rid, generation, reason in update.finished:
            request = self.scheduler.get_live(rid, generation)
            if request is not None:
                self._finish(request, reason)
        if update.speculative_committed:
            stats = self.speculative_runner.stats
            for committed, accepted in update.speculative_committed:
                stats.committed_tokens += committed
                stats.accepted_tokens += accepted
        if ticket is not None:
            for row in update.next_prelude:
                item = batch.items[row]
                if self._is_live(item):
                    # Prelude uses the device token before CPU delivery.
                    item.request.input_token_tensor = ticket.device_values[row]
            for row, rid, generation, position, signal_depth in update.retain_signal:
                item = batch.items[row]
                if self._is_live(item):
                    self._pending_exit_signals[rid] = SignalHandle(
                        ticket=ticket,
                        row=row,
                        generation=generation,
                        position=position,
                        signal_depth=signal_depth,
                    )
        return [RequestOutput.from_request(item.request) for item in update.output_rows]

    def _deliver_coda(self, ticket) -> list[RequestOutput]:
        result = adapt_async_delivery(ticket)
        update = self.scheduler.update_from_output(ticket.batch, result)
        return self._apply_scheduler_update(ticket.batch, update, ticket)

    def _collect_coda(self, wait=False) -> list[RequestOutput]:
        outputs = []
        pending = []
        for ticket in self._pending_coda:
            if ticket.ready() or wait:
                outputs.extend(self._deliver_coda(ticket))
                wait = False
            else:
                pending.append(ticket)
        self._pending_coda = pending
        return outputs

    def _step_async(self) -> list[RequestOutput]:
        # Hold readback buffers until their DMA completes, even for discarded scores.
        self._inflight = [t for t in self._inflight if not t.ready()]
        outputs = self._collect_coda()
        batch = self.scheduler.schedule(
            # Once a core has completed, refill first; a briefly pending coda
            # readback should not fragment the next batch.
            prefer_recurrent=self._overlap_boundary
            and any(t.batch.stage == Stage.RECURRENT and not t.ready() for t in self._inflight)
        )
        self._overlap_boundary = False
        self.last_schedule = batch
        if batch is None:
            if self._pending_coda:
                outputs.extend(self._collect_coda(wait=True))
            elif any(r.stage != Stage.RECEIVING for r in self.scheduler.requests.values()):
                raise RuntimeError("scheduler made no progress")
            return outputs
        if batch.stage == Stage.CODA:
            # One outstanding sample per request. Device-token prelude/core may
            # proceed, but a second sample waits for the first CPU delivery.
            while True:
                pending = {
                    (item.request_id, item.generation)
                    for ticket in self._pending_coda
                    for item in ticket.batch.items
                }
                if not any((i.request_id, i.generation) in pending for i in batch.items):
                    break
                outputs.extend(self._collect_coda(wait=True))
            batch = replace(batch, items=[i for i in batch.items if self._is_live(i)])
            self.last_schedule = batch
            if not batch.items:
                return outputs
        ticket = self.model_runner.submit(batch)
        self._inflight.append(ticket)
        result = adapt_async_submit(batch, self.model_runner.events)
        if batch.stage == Stage.CODA:
            self._pending_coda.append(ticket)
            runner = self.model_runner
            self._overlap_boundary = (
                runner.boundary_stream is not None
                and runner.boundary_stream is not runner.core_stream
            )
        elif batch.stage == Stage.RECURRENT:
            # Submit the next core before reading retained scores.
            for item in batch.items:
                previous = self._pending_exit_signals.pop(item.request_id, None)
                if previous is None or previous.generation != item.generation:
                    continue
                score = previous.ticket.collect()[previous.row]
                result.exit_signals.append(
                    ExitSignal(
                        item.request_id,
                        previous.generation,
                        previous.position,
                        previous.signal_depth,
                        previous.ticket.batch.seq,
                        score,
                    )
                )
        update = self.scheduler.update_from_output(batch, result)
        outputs.extend(self._apply_scheduler_update(batch, update, ticket))
        return outputs
