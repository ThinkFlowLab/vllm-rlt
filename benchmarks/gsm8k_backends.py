"""Single-request BF16 generation with identical token inputs and stop criteria."""

import torch
from lm_eval.models.utils import postprocess_generated_text, stop_sequences_criteria
from transformers import AutoModelForCausalLM, DynamicCache, LogitsProcessor

from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig


def check_logits(logits):
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Non-finite generation logits")


class FiniteLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        check_logits(scores)
        return scores


class Generator:
    def __init__(self, backend, model_path, tokenizer, max_length):
        self.backend, self.tokenizer = backend, tokenizer
        if backend == "transformers":
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .eval()
                .to("cuda")
            )
            # Ouro indexes KV by depth * physical_layers + layer. Allocate all
            # slots in the standard HF cache so the release need not replace it
            # with its cache class, whose interface predates Transformers 4.55.
            self.cache_slots = (
                self.model.config.num_hidden_layers * self.model.config.total_ut_steps
            )
        else:
            self.llm = LLM(
                model_path,
                device="cuda",
                dtype=torch.bfloat16,
                attention_backend="triton",
                cache_config=CacheConfig(num_blocks=4 * ((max_length + 15) // 16)),
                scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
            )
            # Observe the existing sampling boundary without changing arithmetic.
            self.model = self.llm.engine.model
            self._finite_hook = self.model.lm_head.register_forward_hook(
                lambda module, inputs, output: check_logits(output)
            )

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens, stops):
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        criteria = stop_sequences_criteria(self.tokenizer, stops, len(prompt_ids), 1)
        if self.backend == "transformers":
            cache = DynamicCache()
            cache.append_new_layers(self.cache_slots - 1)
            ids = self.model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=criteria,
                logits_processor=[FiniteLogits()],
                use_cache=True,
                past_key_values=cache,
                exit_at_step=3,
                use_weighted_exit=False,
                logits_to_keep=1,
            )[0, len(prompt_ids) :].tolist()
            if len(cache.layers) != self.cache_slots:
                raise RuntimeError("Official cache does not cover every loop/layer")
        else:
            engine = self.llm.engine
            engine.add_request(
                "gsm8k",
                prompt_ids,
                SamplingParams(
                    max_tokens=max_new_tokens,
                    min_loops=4,
                    max_loops=4,
                    temperature=0,
                ),
            )
            ids = []
            try:
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        ids = output.token_ids
                        if any(depth != 4 for depth in output.exit_depths):
                            raise RuntimeError("Expected fixed-four-loop generation")
                        sequence = torch.tensor([prompt_ids + ids], device="cuda")
                        if not output.finished and criteria(sequence, None).all().item():
                            engine.abort_request("gsm8k")
            finally:
                if engine.has_unfinished_requests():
                    engine.abort_request("gsm8k")
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        text = postprocess_generated_text(raw, stop=stops, think_end_token=None)
        eos = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        reason = "eos" if eos else "stop" if any(s in raw for s in stops) else "length"
        return {"token_ids": ids, "raw_text": raw, "text": text, "finish_reason": reason}
