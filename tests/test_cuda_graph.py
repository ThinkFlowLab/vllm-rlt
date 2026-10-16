"""Graph replay must update KV addresses, request identities and host progress."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


def test_cuda_graph_validation():
    with pytest.raises(ValueError, match="positive"):
        ExecutionConfig(cuda_graph_max_graphs=0)
    with pytest.raises(ValueError, match="boolean"):
        ExecutionConfig(cuda_graphs=1)
    with pytest.raises(ValueError, match="CUDA graphs require CUDA"):
        LLMEngine(
            OuroForCausalLM(tiny_ouro_config()), execution_config=ExecutionConfig(cuda_graphs=True)
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn"])
@pytest.mark.parametrize("mode", ["sync", "async", "multi"])
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("static", [False, True])
def test_graph_replay_matches_eager_and_reuses_requests(backend, mode, layout, static):
    torch.manual_seed(321)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    outputs = []
    for graphs in [False, True]:
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=16, layout=layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(
                async_scheduling=mode != "sync",
                multi_stream=mode == "multi",
                static_buffers=static,
                pad_to_power_of_two=static,
                cuda_graphs=graphs,
            ),
            exit_config=ExitConfig("ouro_delayed"),
            attention_backend=backend,
        )
        rounds = []
        for reuse in range(2):
            for i in range(3):
                engine.add_request(
                    str(i),
                    [2 + i, 3, 4] * (i + 1 + reuse),
                    SamplingParams(
                        max_tokens=5 + i, min_loops=2, exit_threshold=0.2, ignore_eos=True
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        finished[out.request_id] = (out.token_ids, out.exit_depths)
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
        if graphs:
            assert engine.model_runner.graphs.captures > 0
            assert engine.model_runner.graphs.replays > engine.model_runner.graphs.captures
        outputs.append(rounds)
    assert outputs[0] == outputs[1]


@pytest.mark.gpu
def test_graph_cache_limit_fallback_and_abort():
    torch.manual_seed(19)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(
            cuda_graphs=True,
            cuda_graph_max_batch_size=2,
            cuda_graph_max_graphs=1,
            async_scheduling=True,
        ),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    for count in [1, 2, 3]:
        for i in range(count):
            engine.add_request(str(i), [2, 3, 4], SamplingParams(max_tokens=8, ignore_eos=True))
        for _ in range(12):
            engine.step()
        for rid in list(engine.scheduler.requests):
            engine.abort_request(rid)
        while engine.has_unfinished_requests():
            engine.step()
        assert engine.cache_manager.num_used_blocks == 0
    assert engine.model_runner.graphs.captures == 1
    assert engine.model_runner.graphs.fallbacks > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_speculative_graph_memory_budget_scales_with_pools_and_retained_outputs(enabled):
    from dataclasses import replace

    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.core.memory import graph_memory_reserve

    model = OuroForCausalLM(tiny_ouro_config())
    scheduler = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=4096)
    execution = ExecutionConfig(cuda_graphs=enabled, cuda_graph_max_batch_size=768)
    base = graph_memory_reserve(model, scheduler, execution)
    spec = graph_memory_reserve(model, scheduler, execution, SpeculativeConfig(2))
    if not enabled:
        assert base == spec == 0
        return
    assert base == execution.cuda_graph_memory_reserve_bytes
    logits = 16 * 768 * model.config.vocab_size * next(model.parameters()).element_size()
    assert spec == 3 * base + logits
    # Reducing token/row capacity also reduces the persistent-output allowance.
    limited = graph_memory_reserve(
        model, replace(scheduler, max_num_batched_tokens=64), execution, SpeculativeConfig(4)
    )
    assert limited == 3 * base + logits // 12


def test_async_graph_memory_accounts_for_every_precaptured_bucket():
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.core.memory import graph_memory_reserve

    model = OuroForCausalLM(tiny_ouro_config())
    execution = ExecutionConfig(cuda_graphs=True, async_scheduling=True)
    # Even tiny admitted work has the configured capture capacity prepared.
    scheduler = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=1)
    reserve = graph_memory_reserve(model, scheduler, execution, SpeculativeConfig(2))
    retained = 255 * model.config.vocab_size * next(model.parameters()).element_size()
    assert reserve == 3 * execution.cuda_graph_memory_reserve_bytes + retained
