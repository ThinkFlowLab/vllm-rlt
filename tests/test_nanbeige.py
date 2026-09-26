"""CPU numerical checks for Nanbeige against an independent dense oracle."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.models.reference import dense_nanbeige_reference
from vllm_rlt.sampling_params import SamplingParams


def make_cache(config, *, blocks=64):
    return KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=blocks,
        block_size=2,
        max_loops=config.total_ut_steps,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture
def model():
    torch.manual_seed(42)
    return NanbeigeForCausalLM(NanbeigeConfig.tiny())


def test_nanbeige_config_validation():
    config = NanbeigeConfig.tiny()
    assert config.num_loops == 2
    assert config.total_ut_steps == 2
    assert config.hidden_act == "silu"

    with pytest.raises(ValueError, match="head_dim"):
        NanbeigeConfig.tiny(head_dim=15)

    with pytest.raises(ValueError, match="num_attention_heads"):
        NanbeigeConfig.tiny(num_attention_heads=5, num_key_value_heads=2)

    with pytest.raises(ValueError, match="hidden_act"):
        NanbeigeConfig.tiny(hidden_act="gelu")

    with pytest.raises(ValueError, match="model_type"):
        NanbeigeConfig.from_dict({"model_type": "llama"})


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_packed_prefill_matches_dense_at_every_depth(kv_heads):
    torch.manual_seed(123)
    config = NanbeigeConfig.tiny(num_key_value_heads=kv_heads)
    model = NanbeigeForCausalLM(config)
    tokens = torch.tensor([5, 7, 2, 11, 3])
    expected = dense_nanbeige_reference(model, tokens, config.total_ut_steps)
    cache = make_cache(config)
    assert cache.allocate("prompt", len(tokens))
    hidden = model.prelude(tokens)
    for depth, (expected_hidden, expected_logits) in enumerate(expected):
        hidden, gate = model.recurrent(
            hidden, ["prompt"] * len(tokens), [depth] * len(tokens), list(range(len(tokens))), cache
        )
        torch.testing.assert_close(hidden, expected_hidden, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(model.coda(hidden), expected_logits, atol=2e-6, rtol=3e-5)


def test_incremental_tokens_match_dense_fixed_depth(model):
    tokens = torch.tensor([4, 15, 6, 2, 9, 12, 3])
    expected = dense_nanbeige_reference(model, tokens, model.config.total_ut_steps)
    cache = make_cache(model.config)
    assert cache.allocate("sequence", len(tokens))
    for position, token in enumerate(tokens):
        hidden = model.prelude(token.unsqueeze(0))
        for depth in range(model.config.total_ut_steps):
            hidden, _ = model.recurrent(hidden, ["sequence"], [depth], [position], cache)
            expected_hidden, expected_logits = expected[depth]
            torch.testing.assert_close(hidden[0], expected_hidden[position], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(
                model.coda(hidden)[0], expected_logits[position], atol=2e-6, rtol=3e-5
            )


def test_mixed_request_depth_batch_matches_separate_dense(model):
    first = torch.tensor([5, 8, 12])
    second = torch.tensor([6, 13, 2, 4])
    expected_first = dense_nanbeige_reference(model, first, 2)[1]
    expected_second = dense_nanbeige_reference(model, second, 1)[0]
    cache = make_cache(model.config)
    assert cache.allocate("first", len(first))
    assert cache.allocate("second", len(second))
    first_hidden = model.prelude(first)
    first_hidden, _ = model.recurrent(first_hidden, ["first"] * 3, [0] * 3, [0, 1, 2], cache)
    hidden, _ = model.recurrent(
        torch.cat([first_hidden, model.prelude(second)]),
        ["first"] * 3 + ["second"] * 4,
        [1] * 3 + [0] * 4,
        [0, 1, 2, 0, 1, 2, 3],
        cache,
    )
    torch.testing.assert_close(
        hidden, torch.cat([expected_first[0], expected_second[0]]), atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_native_safetensors_roundtrip(tmp_path, model, sharded):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = model.state_dict()
    if sharded:
        names = list(weights.keys())
        half = len(names) // 2
        first = {name: weights[name] for name in names[:half]}
        second = {name: weights[name] for name in names[half:]}
        save_file(first, tmp_path / "model-00001-of-00002.safetensors")
        save_file(second, tmp_path / "model-00002-of-00002.safetensors")
        index = {
            "weight_map": {
                name: (
                    "model-00001-of-00002.safetensors"
                    if name in first
                    else "model-00002-of-00002.safetensors"
                )
                for name in names
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(weights, tmp_path / "model.safetensors")

    loaded = NanbeigeForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, weights[name])

    auto_loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(auto_loaded, NanbeigeForCausalLM)


def test_nanbeige_engine_generation(model):
    engine = LLMEngine(model, attention_backend="torch")
    prompt = [1, 5, 8]
    engine.add_request(
        "request-1",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
    )
    all_outputs = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        all_outputs.extend(outputs)

    assert len(all_outputs) > 0
    final = [o for o in all_outputs if o.request_id == "request-1"][-1]
    assert len(final.token_ids) == 5
    assert final.finish_reason == "length"
