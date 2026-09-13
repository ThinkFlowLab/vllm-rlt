"""Tiny CPU checks of the actual pinned cache, mask creation and request lifecycle."""

import importlib.metadata
import importlib.util
import socket

import pytest
import torch

from vllm_lt.models.config import OuroConfig
from vllm_lt.models.ouro import OuroForCausalLM
from vllm_lt.validation.official import OfficialOuroReference
from vllm_lt.validation.official_cached import OfficialOuroCachedReference


def _compatible():
    try:
        return (
            importlib.metadata.version("transformers") == "4.55.0"
            and importlib.util.find_spec("kernels") is None
        )
    except importlib.metadata.PackageNotFoundError:
        return False


requires_official = pytest.mark.skipif(
    not _compatible(), reason="requires prepared Transformers 4.55.0 reference env"
)


@pytest.fixture(autouse=True)
def no_device_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cached reference CPU tests must not use CUDA or network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def model(request):
    torch.manual_seed(17)
    return OuroForCausalLM(
        OuroConfig.tiny(
            hidden_size=16, intermediate_size=32, num_attention_heads=2, max_position_embeddings=256
        )
    ).to(dtype=getattr(request, "param", torch.float32))


@pytest.fixture
def reference(model):
    ref = OfficialOuroCachedReference(model.config.to_dict(), dict(model.named_parameters()))
    yield ref
    ref.close(completion_confirmed=True)


@requires_official
@pytest.mark.parametrize("model", [torch.float32, torch.bfloat16], indirect=True)
def test_cached_greedy_matches_uncached_prefixes_and_preserves_shared_weights(model, reference):
    caller = dict(model.named_parameters())
    original = {name: value.detach().clone() for name, value in caller.items()}
    flags = {name: value.requires_grad for name, value in caller.items()}
    uncached = OfficialOuroReference(model.config.to_dict(), caller)
    assert reference.config.use_cache is True and uncached.config.use_cache is False
    for name, parameter in reference.model.named_parameters():
        assert parameter.data_ptr() == caller[name].data_ptr()
        assert parameter.dtype == caller[name].dtype and not parameter.requires_grad
    prompt, generated = [1, 2, 3, 4], []
    for index in range(5):
        logits = reference.start(prompt, 5) if index == 0 else reference.advance(generated[-1])
        expected = uncached.predict(prompt + generated, [])[0]
        assert logits.shape == (model.config.vocab_size,)
        assert logits.dtype == next(model.parameters()).dtype
        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
        assert logits.argmax().item() == expected.argmax().item()
        generated.append(logits.argmax().item())
    reference.complete(completion_confirmed=True)
    state = reference.snapshot(inspect_cache=True, check_finite=True)
    assert state["status"] == "finished"
    assert state["forward_calls"] == state["output_count"] == 5
    assert state["final_summary"]["lengths"] == [8] * 8
    assert state["final_summary"]["distinct_storage"] is True
    assert state["final_summary"]["all_finite"] is True
    reference.reset()
    assert reference.cache_slots == 0
    uncached.close()
    for name, tensor in caller.items():
        assert tensor.requires_grad == flags[name]
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@requires_official
def test_reset_completion_and_close_boundaries(reference):
    with pytest.raises(RuntimeError, match="ready"):
        reference.advance(1)
    reference.start([1, 2], 2)
    old_cache = reference.cache
    before = reference.snapshot()
    with pytest.raises(ValueError):
        reference.advance(True)
    with pytest.raises(RuntimeError, match="active"):
        reference.start([3], 1)
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.reset()
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.complete(completion_confirmed=True)
    assert reference.snapshot() == before
    reference.advance(3)
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.complete(completion_confirmed=False)
    reference.complete(completion_confirmed=True)
    reference.reset()
    assert old_cache.key_cache == old_cache.value_cache == []
    reference.start([4], 1)
    assert reference.cache is not old_cache
    assert reference.snapshot(inspect_cache=True)["final_summary"]["lengths"] == [1] * 8
    reference.complete(completion_confirmed=True)
    reference.close()
    reference.close()
    assert reference.cache_slots == 0 and reference.model is None
    for call in (lambda: reference.start([1], 1), reference.reset, lambda: reference.advance(1)):
        with pytest.raises(RuntimeError, match="closed"):
            call()


@requires_official
def test_partial_forward_failure_preserves_error_and_requires_confirmed_cleanup(
    reference, monkeypatch
):
    reference.start([1, 2], 2)
    error = RuntimeError("injected after first slot append")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(reference.model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError) as observed:
        reference.advance(3)
    assert observed.value is error
    assert reference.status == "failed" and reference.forward_calls == 2
    assert reference.output_count == 1 and reference.expected_position == 2
    assert reference.cache.get_seq_length(0) == 3 and reference.cache.get_seq_length(1) == 2
    with pytest.raises(RuntimeError, match="failed"):
        reference.advance(4)
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.close()
    reference.reset(completion_confirmed=True)
    assert reference.cache_slots == 0 and reference.status == "ready"
