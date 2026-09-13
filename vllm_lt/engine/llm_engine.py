import math

from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.core.scheduler import Scheduler
from vllm_lt.request import Request, RequestOutput, Stage
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.worker.model_runner import ModelRunner


class LLMEngine:
    """Single-device synchronous engine with continuous loop-level batching."""

    def __init__(
        self, model, *, cache_config=None, scheduler_config=None, attention_backend="torch"
    ):
        self.model = model
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()
        parameter = next(model.parameters())
        config = model.config
        self.cache_manager = KVCacheManager(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_loops=config.total_ut_steps,
            num_blocks=cache_config.num_blocks,
            block_size=cache_config.block_size,
            device=parameter.device,
            dtype=parameter.dtype,
            backend=attention_backend,
        )
        self.scheduler = Scheduler(scheduler_config, self.cache_manager)
        self.model_runner = ModelRunner(model, self.cache_manager)
        self.last_schedule = None

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ):
        self.model_runner._require_execution_usable()
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
        capacity = len(prompt_token_ids) + params.max_tokens - 1
        if capacity > config.max_position_embeddings:
            raise ValueError("prompt plus decode positions exceed the model context length")
        cache = self.cache_manager
        required = math.ceil(capacity / cache.block_size) * cache.max_loops
        if required > cache.num_blocks:
            raise ValueError(
                f"request requires {required} KV blocks but cache has {cache.num_blocks}; "
                "increase num_blocks or reduce prompt/max_tokens"
            )
        self.scheduler.add_request(Request(request_id, list(prompt_token_ids), params))

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests

    def abort_request(self, request_id: str) -> RequestOutput:
        self.cache_manager._require_usable()
        return RequestOutput.from_request(self.scheduler.abort(request_id))

    def step(self) -> list[RequestOutput]:
        self.model_runner._require_execution_usable()
        batch = self.scheduler.schedule()
        self.last_schedule = batch
        if batch is None:
            return []
        try:
            result = self.model_runner.execute(batch)
            return self._update(batch, result)
        except BaseException as error:
            if not isinstance(error, Exception) and self.model_runner._persistent is None:
                # Preserve the compact engine's existing interrupt behavior.
                raise
            # Finalization in _update can submit work after the runner's gate
            # readback. Settle that work too, before returning pages to the pool.
            if self.model_runner._settle_persistent_failure(error):
                for item in batch.items:
                    if item.request.request_id in self.scheduler.requests:
                        try:
                            self.scheduler.abort(item.request.request_id)
                        except BaseException as cleanup_error:
                            self.model_runner._record_cleanup_failure(cleanup_error)
                            break
            # Cleanup must never replace the original execution exception.
            raise

    def _update(self, batch, result) -> list[RequestOutput]:
        outputs = []
        for index, item in enumerate(batch.items):
            request = item.request
            params = request.sampling_params
            if batch.stage == Stage.PREFILL:
                request.num_prefilled_tokens += item.token_count
                if request.num_prefilled_tokens == len(request.prompt_token_ids):
                    request.loops_done = self.model.config.total_ut_steps
                    self.scheduler.enqueue(request, Stage.CODA)
                else:
                    self.scheduler.enqueue(request, Stage.PREFILL)
            elif batch.stage == Stage.PRELUDE:
                request.loops_done = 0
                request.remaining_probability = 1.0
                self.scheduler.enqueue(request, Stage.RECURRENT)
            elif batch.stage == Stage.RECURRENT:
                request.loops_done += 1
                # Ouro learns a conditional hazard at each depth, not a direct exit CDF.
                request.remaining_probability *= 1.0 - result[index]
                if self._should_exit(request):
                    self.cache_manager.finalize_token(
                        request.request_id, request.position, request.loops_done - 1
                    )
                    self.scheduler.enqueue(request, Stage.CODA)
                else:
                    self.scheduler.enqueue(request, Stage.RECURRENT)
            elif batch.stage == Stage.CODA:
                token_id = result[index]
                request.generated_token_ids.append(token_id)
                request.exit_depths.append(request.loops_done)
                eos = self.model.config.eos_token_id
                eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
                if token_id in eos_ids and not params.ignore_eos:
                    self.scheduler.finish(request, "stop")
                elif len(request.generated_token_ids) >= params.max_tokens:
                    self.scheduler.finish(request, "length")
                else:
                    self.scheduler.enqueue(request, Stage.PRELUDE)
                outputs.append(RequestOutput.from_request(request))
        return outputs

    def _should_exit(self, request: Request) -> bool:
        """Evaluate the policy after recording the completed loop's actual hazard."""
        params = request.sampling_params
        max_loops = params.max_loops or self.model.config.total_ut_steps
        reached_threshold = (
            params.exit_threshold < 1.0
            and request.loops_done >= params.min_loops
            and 1.0 - request.remaining_probability >= params.exit_threshold
        )
        return request.loops_done >= max_loops or reached_threshold
