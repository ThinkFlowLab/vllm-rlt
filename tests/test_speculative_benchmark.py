"""Checks for the matched decode-only benchmark boundary."""

import subprocess
import sys

import torch

from vllm_rlt import CacheConfig, SamplingParams, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def test_decode_trial_counts_only_tokens_after_matched_prefill():
    from benchmarks.speculative import run_decode_trial

    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    params = SamplingParams(max_tokens=4, max_loops=4, exit_threshold=1.0, ignore_eos=True)
    prompts = [[2, 3, 4], [5, 6, 7]]
    native = LLMEngine(model, cache_config=CacheConfig(64, 2))
    speculative = LLMEngine(
        model, cache_config=CacheConfig(64, 2), speculative_config=SpeculativeConfig(2)
    )

    native_result = run_decode_trial(native, prompts, params)
    speculative_result = run_decode_trial(speculative, prompts, params)

    assert native_result["decode_tokens"] == 6
    assert speculative_result["decode_tokens"] == 6
    assert native_result["token_ids"] == speculative_result["token_ids"]
    assert all(len(tokens) == 4 for tokens in native_result["token_ids"])
    assert native.cache_manager.num_used_blocks == 0
    assert speculative.cache_manager.num_used_blocks == 0


def test_case_repeats_matched_requests_and_keeps_raw_measurements():
    from benchmarks.speculative import make_prompts, run_case

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [2, 3, 4, 5] * 25

    prompts = make_prompts(Tokenizer(), prompt_length=6, concurrency=2)
    assert len(prompts) == 2
    assert all(len(prompt) == 6 for prompt in prompts)

    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    result = run_case(model, prompts, output_tokens=4, k=2, trials=2, backend="torch")
    assert len(result["runs"]) == 2
    assert all(run["same_tokens"] for run in result["runs"])
    assert all(run["native"]["decode_tokens"] == 6 for run in result["runs"])
    assert all(run["speculative"]["decode_tokens"] == 6 for run in result["runs"])
    assert result["summary"]["exact_agreement_fraction"] == 1.0


def test_benchmark_exposes_documented_command_line_options():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.speculative", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--model" in result.stdout
    assert "--prompt-lengths" in result.stdout
    assert "--concurrencies" in result.stdout
    assert "--ks" in result.stdout
    assert "--resume" in result.stdout
    assert "--memory-only" in result.stdout


def test_cache_reservation_fits_long_output_eight_request_batch():
    from benchmarks.speculative import cache_blocks_for_case

    blocks = cache_blocks_for_case(
        prompt_tokens=128, output_tokens=128, concurrency=8, depth=4, k=8
    )
    assert blocks >= 8 * 4 * 17


def test_isolated_memory_trial_releases_each_engine_before_next_mode():
    from benchmarks.speculative import measure_isolated_memory

    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    result = measure_isolated_memory(model, [[2, 3, 4]], output_tokens=3, k=2, backend="torch")
    assert set(result) == {"native", "speculative"}
    assert result["native"]["decode_tokens"] == 2
    assert result["speculative"]["decode_tokens"] == 2
    assert result["native"]["peak_allocated_bytes"] is None
