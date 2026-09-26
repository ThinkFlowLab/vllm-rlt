from dataclasses import replace

import torch

from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM
from vllm_rlt.sampling_params import SamplingParams


class LLM:
    """vLLM-style offline facade; token ID input does not require Transformers."""

    def __init__(
        self,
        model,
        *,
        tokenizer=None,
        revision=None,
        device="cpu",
        dtype=torch.bfloat16,
        cache_config=None,
        scheduler_config=None,
        attention_backend="torch",
        exit_config=None,
        execution_config=None,
        speculative_config=None,
    ):
        self._tokenizer_source = None
        if isinstance(model, str):
            model_name = model
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, device=device, dtype=dtype
            )
            if tokenizer is None:
                from vllm_rlt.models.ouro import OURO_REVISION

                self._tokenizer_source = (
                    model_name,
                    revision or (OURO_REVISION if model_name == "ByteDance/Ouro-1.4B" else None),
                )
        self.tokenizer = tokenizer
        self.engine = LLMEngine(
            model,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            attention_backend=attention_backend,
            exit_config=exit_config,
            execution_config=execution_config,
            speculative_config=speculative_config,
        )
        self._next_request_id = 0

    def generate(self, prompts, sampling_params=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()
        all_params = params if isinstance(params, list) else [params] * len(prompts)
        if len(all_params) != len(prompts):
            raise ValueError("provide one SamplingParams per prompt")
        if self.engine.has_unfinished_requests():
            raise RuntimeError("finish existing engine requests before calling generate")
        request_ids = []
        try:
            for prompt, request_params in zip(prompts, all_params):
                if isinstance(prompt, str):
                    if self.tokenizer is None and self._tokenizer_source is not None:
                        from transformers import AutoTokenizer

                        source, revision = self._tokenizer_source
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            source, revision=revision, trust_remote_code=False
                        )
                    if self.tokenizer is None:
                        raise ValueError(
                            "text prompts require a tokenizer; pass token ID lists instead"
                        )
                    prompt = self.tokenizer.encode(prompt)
                request_id = str(self._next_request_id)
                self._next_request_id += 1
                self.engine.add_request(request_id, list(prompt), request_params)
                request_ids.append(request_id)
            results = {}
            while self.engine.has_unfinished_requests():
                for output in self.engine.step():
                    if output.finished:
                        if self.tokenizer is not None:
                            output = replace(
                                output,
                                text=self.tokenizer.decode(
                                    output.token_ids, skip_special_tokens=True
                                ),
                            )
                        results[output.request_id] = output
            return [results[request_id] for request_id in request_ids]
        except Exception:
            for request_id in request_ids:
                if request_id in self.engine.scheduler.requests:
                    self.engine.abort_request(request_id)
            raise
