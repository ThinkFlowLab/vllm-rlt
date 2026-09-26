"""CPU tests for native generation stopping and cleanup."""

from types import SimpleNamespace

import pytest
import torch

ADAPTIVE_EXIT = {
    "mode": "ouro",
    "threshold": 0.5,
    "min_loops": 1,
    "max_loops": 4,
    "async_scheduling": False,
}


class Engine:
    def __init__(self, failure=None, depths=(4, 4)):
        self.failure = failure
        self.depths = list(depths)
        self.active = False
        self.steps = 0
        self.aborts = 0
        self.params = None

    def add_request(self, request_id, prompt_ids, params):
        assert request_id == "gsm8k"
        assert prompt_ids == [1, 2]
        assert params.max_tokens == 8
        self.params = params
        self.active = True

    def has_unfinished_requests(self):
        return self.active

    def step(self):
        self.steps += 1
        if self.failure == "execution":
            raise RuntimeError("execution failed")
        return [
            SimpleNamespace(
                token_ids=[10, 11][: self.steps],
                exit_depths=(
                    [3] * self.steps if self.failure == "depth" else self.depths[: self.steps]
                ),
                finished=False,
            )
        ]

    def abort_request(self, request_id):
        assert request_id == "gsm8k"
        self.aborts += 1
        self.active = False


@pytest.fixture
def generator(monkeypatch):
    pytest.importorskip("lm_eval")
    from benchmarks import gsm8k_backends as backends

    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(backends.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(
        backends,
        "stop_sequences_criteria",
        lambda tokenizer, stops, length, batch: (
            lambda ids, scores: original_tensor([ids.shape[1] >= length + 2])
        ),
    )
    generator = backends.Generator.__new__(backends.Generator)
    generator.backend = "native"
    generator.exit = backends.FIXED_EXIT
    generator.tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda ids, **kw: "answer STOP")
    generator.llm = SimpleNamespace(engine=Engine())
    return generator


def test_stop_releases_request_without_an_extra_token(generator):
    result = generator.generate([1, 2], 8, ["STOP"])
    assert result["token_ids"] == [10, 11]
    assert result["text"] == "answer "
    assert result["finish_reason"] == "stop"
    assert generator.llm.engine.steps == 2
    assert generator.llm.engine.aborts == 1
    assert not generator.llm.engine.active
    params = generator.llm.engine.params
    assert params.min_loops == params.max_loops == 4 and params.exit_threshold == 1
    assert result["exit_depths"] == [4, 4]


def test_adaptive_exit_records_depths_below_four(generator):
    generator.exit = ADAPTIVE_EXIT
    generator.llm.engine = Engine(depths=(4, 2))
    result = generator.generate([1, 2], 8, ["STOP"])
    params = generator.llm.engine.params
    assert (params.min_loops, params.max_loops, params.exit_threshold) == (1, 4, 0.5)
    # Prefill produces the first token at full depth; decode may exit earlier.
    assert result["exit_depths"] == [4, 2]
    assert generator.llm.engine.aborts == 1


@pytest.mark.parametrize("failure", ["execution", "depth"])
def test_failure_releases_request(generator, failure):
    generator.llm.engine.failure = failure
    with pytest.raises(RuntimeError):
        generator.generate([1, 2], 8, ["STOP"])
    assert generator.llm.engine.aborts == 1
    assert not generator.llm.engine.active


class ReleaseModel:
    config = SimpleNamespace(num_hidden_layers=1, total_ut_steps=4)

    def __init__(self, depths=None, recorded=(4, 2)):
        self.kwargs = None
        self.depths = depths
        self.recorded = recorded

    def generate(self, inputs, **kwargs):
        self.kwargs = kwargs
        if self.depths is not None:
            # Stands in for ReleaseExitHooks, which records one depth per forward.
            self.depths.depths.extend(self.recorded)
        return torch.cat([inputs.cpu(), torch.tensor([[10, 11]])], dim=1)


@pytest.mark.parametrize("adaptive", [False, True])
def test_transformers_exit_arguments(generator, monkeypatch, adaptive):
    from benchmarks import gsm8k_backends as backends

    class Cache:
        def __init__(self):
            self.layers = [None]

        def append_new_layers(self, index):
            self.layers.extend([None] * index)

    monkeypatch.setattr(backends, "DynamicCache", Cache)
    generator.backend = "transformers"
    generator.exit_hooks = SimpleNamespace(depths=["stale"]) if adaptive else None
    generator.model = ReleaseModel(generator.exit_hooks)
    generator.cache_slots = 4
    generator.exit = ADAPTIVE_EXIT if adaptive else backends.FIXED_EXIT
    result = generator.generate([1, 2], 8, ["STOP"])
    kwargs = generator.model.kwargs
    if adaptive:
        assert kwargs["exit_threshold"] == 0.5 and "exit_at_step" not in kwargs
    else:
        assert kwargs["exit_at_step"] == 3 and "exit_threshold" not in kwargs
    assert result["token_ids"] == [10, 11]
    if adaptive:
        # Depths from the previous question are cleared before generation.
        assert result["exit_depths"] == [4, 2]
    else:
        assert "exit_depths" not in result


def test_transformers_depth_count_must_match_tokens(generator, monkeypatch):
    from benchmarks import gsm8k_backends as backends

    class Cache:
        layers = [None] * 4

        def append_new_layers(self, index):
            pass

    monkeypatch.setattr(backends, "DynamicCache", Cache)
    generator.backend = "transformers"
    generator.exit_hooks = SimpleNamespace(depths=[])
    generator.model = ReleaseModel(generator.exit_hooks, recorded=(4,))
    generator.cache_slots = 4
    generator.exit = ADAPTIVE_EXIT
    with pytest.raises(RuntimeError, match="exit depths"):
        generator.generate([1, 2], 8, ["STOP"])


class GateModel(torch.nn.Module):
    """Minimal stand-in for the release: the inner model returns (outputs, hidden, gates)."""

    config = SimpleNamespace(total_ut_steps=4)

    def __init__(self, hazards):
        super().__init__()
        logits = torch.logit(torch.tensor(hazards, dtype=torch.float32))

        class Inner(torch.nn.Module):
            def forward(self, length):
                return None, None, [value.expand(1, length, 1) for value in logits]

        self.model = Inner()
        self.exit_at_step = []

    def forward(self, length, cache_position=None, exit_at_step=None):
        self.exit_at_step.append(exit_at_step)
        return self.model(length)


@pytest.mark.parametrize("threshold,depth", [(0.05, 1), (0.2, 2), (0.5, 3), (0.9, 4), (1.0, 4)])
def test_release_exit_hooks(threshold, depth):
    pytest.importorskip("lm_eval")
    from benchmarks.gsm8k_backends import ReleaseExitHooks

    # Hazards 0.1, 0.3, 0.5 give cumulative exit probabilities 0.1, 0.37, 0.685, then 1.
    model = GateModel([0.1, 0.3, 0.5, 0.9])
    hooks = ReleaseExitHooks(model, threshold)
    model(3, cache_position=torch.tensor([0, 1, 2]))
    model(1, cache_position=torch.tensor([3]))
    # Prefill is forced to full depth like native; decode follows the release's rule.
    assert model.exit_at_step == [3, None]
    assert hooks.depths == [4, depth]
