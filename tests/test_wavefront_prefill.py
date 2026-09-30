"""Correctness checks for depth-aware wavefront prompt prefill."""

from dataclasses import replace

import pytest
import torch

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.request import Stage


def make_engine(*, wavefront, asynchronous=False, prefix=False, graphs=False):
    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    return LLMEngine(
        model,
        cache_config=CacheConfig(
            128,
            2,
            enable_prefix_caching=prefix,
            incremental_allocation=prefix,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            wavefront_prefill=wavefront,
        ),
        exit_config=ExitConfig("trace", depths_by_request={"a": [4, 4, 4]}),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous,
            static_buffers=asynchronous,
            pad_to_power_of_two=asynchronous,
            cuda_graphs=graphs,
        ),
    )


def drain(engine):
    outputs = []
    batches = []
    for _ in range(300):
        if not engine.has_unfinished_requests():
            assert engine.cache_manager.num_used_blocks == 0
            return outputs, batches
        outputs.extend(output for output in engine.step() if output.finished)
        if engine.last_schedule is not None:
            batches.append(engine.last_schedule)
    pytest.fail("wavefront engine failed to drain")


def test_wavefront_prefill_matches_legacy_and_forms_mixed_depth_batches():
    params = SamplingParams(max_tokens=2, min_loops=4, max_loops=4, ignore_eos=True)
    baseline = make_engine(wavefront=False)
    wavefront = make_engine(wavefront=True)
    for engine in (baseline, wavefront):
        engine.add_request("a", [2, 3, 4, 5, 6], params)

    expected, _ = drain(baseline)
    actual, batches = drain(wavefront)
    assert [(o.token_ids, o.exit_depths) for o in actual] == [
        (o.token_ids, o.exit_depths) for o in expected
    ]
    mixed = [
        item
        for batch in batches
        if batch.stage == Stage.PREFILL
        for item in batch.items
        if item.prefill_task is not None
    ]
    assert any(
        len(batch.items) >= 2 and {item.prefill_task.depth for item in batch.items} == {0, 1}
        for batch in batches
        if batch.stage == Stage.PREFILL
    )
    assert mixed


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wavefront_prefix_cache_and_async_match_eager(asynchronous):
    params = SamplingParams(max_tokens=2, min_loops=4, max_loops=4, ignore_eos=True)
    engine = make_engine(wavefront=True, asynchronous=asynchronous, prefix=True)
    engine.add_request("a", [2, 3, 4, 5, 6], params)
    first, _ = drain(engine)
    engine.add_request("a", [2, 3, 4, 5, 6], params)
    second, _ = drain(engine)
    assert [(o.token_ids, o.exit_depths) for o in second] == [
        (o.token_ids, o.exit_depths) for o in first
    ]
    assert engine.cache_manager.prefix_hits > 0
    assert engine.cache_manager.num_used_blocks == 0


def test_wavefront_prefill_requires_last_exited_layout():
    with pytest.raises(ValueError, match="LAST_EXITED"):
        LLMEngine(
            OuroForCausalLM(OuroConfig.tiny()),
            cache_config=CacheConfig(64, 2, "shared"),
            scheduler_config=SchedulerConfig(wavefront_prefill=True),
        )


@pytest.mark.gpu
def test_wavefront_async_prefix_and_decode_graphs_match_eager():
    if torch.version.hip:
        pytest.skip("CUDA Graph validation is NVIDIA-specific")
    config = replace(OuroConfig.tiny(), head_dim=64)
    params = SamplingParams(max_tokens=2, min_loops=4, max_loops=4, ignore_eos=True)

    def make(wavefront, graphs):
        torch.manual_seed(123)
        return LLMEngine(
            OuroForCausalLM(config).to("cuda", torch.bfloat16),
            cache_config=CacheConfig(
                128,
                16,
                enable_prefix_caching=True,
                incremental_allocation=True,
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=4,
                prefill_chunk_size=2,
                wavefront_prefill=wavefront,
            ),
            exit_config=ExitConfig("trace", depths_by_request={"a": [4, 4]}),
            execution_config=ExecutionConfig(
                async_scheduling=True,
                static_buffers=True,
                pad_to_power_of_two=True,
                cuda_graphs=graphs,
            ),
            attention_backend="triton",
        )

    baseline = make(False, False)
    expected_engine = make(True, True)
    prompt = [1 + (index % 63) for index in range(17)]
    baseline.add_request("a", prompt, params)
    expected_engine.add_request("a", prompt, params)
    expected, _ = drain(baseline)
    actual, _ = drain(expected_engine)
    assert [(o.token_ids, o.exit_depths) for o in actual] == [
        (o.token_ids, o.exit_depths) for o in expected
    ]
    assert expected_engine.model_runner.graphs.captures > 0
    assert expected_engine.model_runner.graphs.replays > 0

    expected_engine.add_request("a", prompt, params)
    second, _ = drain(expected_engine)
    assert [(o.token_ids, o.exit_depths) for o in second] == [
        (o.token_ids, o.exit_depths) for o in actual
    ]
    assert expected_engine.cache_manager.prefix_hits > 0
