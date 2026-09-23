"""Checks for out-of-band speculative phase attribution."""

import torch

from vllm_rlt import CacheConfig, SamplingParams, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def test_profile_records_draft_target_and_commit_without_changing_tokens():
    from benchmarks.speculative import run_decode_trial
    from benchmarks.speculative_profile import profile_trial

    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(64, 2),
        speculative_config=SpeculativeConfig(2),
    )
    params = SamplingParams(max_tokens=4, max_loops=4, ignore_eos=True)
    baseline = run_decode_trial(engine, [[2, 3, 4]], params)
    result = profile_trial(engine, [[2, 3, 4]], params)
    assert result["trial"]["token_ids"] == baseline["token_ids"]
    assert result["phases"]["draft_core"]["calls"] > 0
    assert result["phases"]["target_core"]["calls"] > 0
    assert result["phases"]["commit"]["calls"] > 0
