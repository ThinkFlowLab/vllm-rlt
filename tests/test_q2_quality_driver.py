"""Real CPU engine EOS/prefix checks and bounded mocked worker/controller failures."""

import gc
import math
import sys
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_lt.benchmarks import q2_quality as quality
from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import RequestOutput


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("quality CPU test attempted CUDA discovery or initialization")

    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    torch.set_num_threads(1)


class Head(nn.Module):
    def __init__(self, tokens, *, nonfinite_at=None):
        super().__init__()
        self.tokens = list(tokens)
        self.calls = 0
        self.nonfinite_at = nonfinite_at

    def forward(self, hidden):
        logits = hidden.new_full((hidden.shape[0], 16), -10)
        logits[:, self.tokens[self.calls]] = 10
        self.calls += 1
        if self.calls == self.nonfinite_at:
            logits[0, 3] = float("nan")
        return logits


def engine_for(tokens, *, nonfinite_at=None):
    config = OuroConfig.tiny(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=1024,
    )
    model = OuroForCausalLM(config)
    model.lm_head = Head(tokens, nonfinite_at=nonfinite_at)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(192, 16),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
    )
    return engine


def invoke(engine, *, prompt=None, phase="evaluation", run_id="F-eval-00"):
    prompt = prompt if prompt is not None else [3, 4]
    return quality.generate_example(
        engine,
        {"run_id": run_id, "phase": phase, "prompt_token_ids": prompt},
        deadline_ns=time.perf_counter_ns() + 30 * 10**9,
        max_steps=math.ceil(len(prompt) / 128) + 1531,
    )


@pytest.mark.parametrize("phase", ["evaluation", "feasibility"])
def test_real_engine_first_eos_and_reuse_remove_all_hooks(phase):
    engine = engine_for([0, 5, 0])
    first = invoke(engine, phase=phase)
    assert first["output_token_ids"] == [0]
    assert first["finish_reason"] == "stop"
    assert first["counts"] == {
        "steps": 2,
        "stage_counts": {"prefill": 1, "prelude": 0, "recurrent": 0, "coda": 1},
    }
    assert first["finite_checks"] == {
        "lm_head": 1,
        "recurrent": 4 if phase == "feasibility" else 0,
        "coda": 1 if phase == "feasibility" else 0,
    }
    second = invoke(engine, run_id="F-eval-01")
    assert second["output_token_ids"] == [5, 0]
    assert second["exit_depths"] == [4, 4]
    assert not engine.model.lm_head._forward_hooks
    assert "recurrent" not in engine.model.__dict__
    assert "coda" not in engine.model.__dict__
    assert quality.require_empty(engine) == dict.fromkeys(
        ("requests", "queued", "allocated_pages", "used_pages"), 0
    )
    assert engine.last_schedule is None


def test_chunked_prefill_and_final_prediction_is_never_forwarded():
    engine = engine_for([7, 0])
    inputs = []
    original = engine.model.prelude

    def observe(ids):
        inputs.extend(ids.tolist())
        return original(ids)

    engine.model.prelude = observe
    result = invoke(engine, prompt=[2] * 129, phase="feasibility")
    assert inputs == [2] * 129 + [7]
    assert result["counts"] == {
        "steps": 9,
        "stage_counts": {"prefill": 2, "prelude": 1, "recurrent": 4, "coda": 2},
    }
    assert result["finite_checks"] == {"lm_head": 2, "recurrent": 12, "coda": 2}


@pytest.mark.parametrize("last,reason", [(0, "stop"), (6, "length")])
def test_output_256_eos_precedes_length_and_no_token_257(last, reason):
    engine = engine_for([6] * 255 + [last])
    result = invoke(engine)
    assert len(result["output_token_ids"]) == 256
    assert result["finish_reason"] == reason
    assert result["truncated"] == (reason == "length")
    assert result["counts"]["steps"] == 1532
    assert result["finite_checks"]["lm_head"] == 256
    assert engine.model.lm_head.calls == 256


def test_nonfinite_actual_logits_fail_before_argmax_keep_prefix_and_cleanup():
    engine = engine_for([5, 0], nonfinite_at=2)
    sample_calls = []
    sample = engine.model_runner._sample

    def observed(logits, request):
        sample_calls.append(request.request_id)
        return sample(logits, request)

    engine.model_runner._sample = observed
    with pytest.raises(ValueError, match="before sampling") as caught:
        invoke(engine)
    result = caught.value.q2_partial_result
    assert result["output_token_ids"] == [5]
    assert result["finite_checks"]["lm_head"] == 2
    assert sample_calls == ["F-eval-00"]
    assert not any(result["cleanup"].values())
    assert not engine.model.lm_head._forward_hooks


@pytest.mark.parametrize("mutation", ["prefix", "depth", "finish", "repeat", "identity"])
def test_real_outputs_are_validated_instead_of_replaced(mutation):
    engine = engine_for([5, 6, 0])
    original = engine.step

    def changed():
        outputs = original()
        if outputs and len(outputs[0].token_ids) == 2:
            output = outputs[0]
            values = dict(output.__dict__)
            if mutation == "prefix":
                values["token_ids"] = [4, 6]
            elif mutation == "depth":
                values["exit_depths"] = [4, 3]
            elif mutation == "finish":
                values.update(finished=True, finish_reason="stop")
            elif mutation == "repeat":
                values.update(token_ids=[5], exit_depths=[4])
            else:
                values["request_id"] = "other"
            return [RequestOutput(**values)]
        return outputs

    engine.step = changed
    with pytest.raises(ValueError) as caught:
        invoke(engine)
    assert caught.value.q2_partial_result["output_token_ids"] == [5]
    assert not any(caught.value.q2_partial_result["cleanup"].values())


def test_deadline_during_final_cleanup_keeps_raw_outputs(monkeypatch):
    engine = engine_for([0])
    calls = 0
    original = quality.require_empty

    def expire_after_cleanup(current):
        nonlocal calls
        calls += 1
        result = original(current)
        if calls == 2:
            monkeypatch.setattr(
                quality,
                "check_deadline",
                lambda _: (_ for _ in ()).throw(TimeoutError("late cleanup")),
            )
        return result

    monkeypatch.setattr(quality, "require_empty", expire_after_cleanup)
    with pytest.raises(TimeoutError) as caught:
        invoke(engine)
    assert caught.value.q2_partial_result["output_token_ids"] == [0]
    assert not any(caught.value.q2_partial_result["cleanup"].values())


def minimal_plan(count=2):
    return {
        "plan_sha256": "a" * 64,
        "interpreter": sys.executable,
        "contract": {
            "limits": {
                "executions": count,
                "case_timeout_s": 600,
                "total_timeout_s": 7200,
                "case_bytes_max": 2 * 1024**2,
                "artifact_bytes_max": 256 * 1024**2,
            },
            "controls": {"gpu_ids": [2]},
        },
        "resource_estimates": {"native_pool_bytes": 1207959552},
        "model_path": "/unused",
        "execution_order": [
            {"run_id": f"F-feas-{i:02d}", "phase": "feasibility"} for i in range(count)
        ],
    }


def worker_standins(monkeypatch, *, count=2):
    from tokenizers import Tokenizer

    from vllm_lt.benchmarks import runner

    engine = engine_for([0])
    engine_ref = weakref.ref(engine)
    holder = [engine]
    monkeypatch.setattr(
        quality,
        "_schema",
        lambda: SimpleNamespace(
            verify_quality_plan=lambda p: None,
            validate_quality_plan=lambda p: None,
            source_probe=lambda: {"cpu_standin": True},
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(runner, "configure_process", lambda c: None)
    monkeypatch.setattr(quality, "_environment", lambda p: {"cpu_standin": True})
    monkeypatch.setattr(runner, "load_model", lambda p, m: holder[0].model)
    monkeypatch.setattr(quality, "_engine", lambda m, p: holder.pop())
    monkeypatch.setattr(quality, "_pool", lambda e: {"size_bytes": 1207959552})
    monkeypatch.setattr(Tokenizer, "from_file", lambda p: object())
    calls, releases = [], []

    def execute(plan, row, current, tokenizer, **times):
        calls.append(row["run_id"])
        return {"status": "complete", "output_token_ids": [0], "failures": [], **times}

    def release(worker):
        gc.collect()
        releases.append(engine_ref() is None)
        worker["teardown_after_workspace_release"] = {"allocated_bytes": 0, "reserved_bytes": 0}

    monkeypatch.setattr(quality, "execute_example", execute)
    monkeypatch.setattr(runner, "release_device", release)
    del engine
    return minimal_plan(count), calls, releases


def test_worker_single_engine_all_rows_ack_chains_and_zero_owners(tmp_path, monkeypatch):
    plan, calls, releases = worker_standins(monkeypatch)
    result = quality.run_worker(plan, tmp_path, time.perf_counter_ns() + 60 * 10**9)
    assert result["status"] == "complete"
    assert calls == result["completed_runs"] == ["F-feas-00", "F-feas-01"]
    assert releases == [True]
    for row in plan["execution_order"]:
        directory = tmp_path / "runs" / row["run_id"]
        completion = quality.read_json(directory / "completed.json")
        ack = quality.read_json(directory / "acknowledged.json")
        assert completion["result"] == quality.file_record(directory / "result.json")
        assert ack["completion"] == quality.file_record(directory / "completed.json")
        assert (
            completion["started_ns"]
            <= completion["completed_ns"]
            <= ack["acknowledged_ns"]
            < ack["deadline_ns"]
        )


def test_late_export_failure_preserves_completed_bytes_no_next_row(tmp_path, monkeypatch):
    plan, calls, releases = worker_standins(monkeypatch)
    original = quality.artifact_usage
    preserved = {}

    def fail_after_completion(root, limits):
        directory = root / "runs" / "F-feas-00"
        if (directory / "completed.json").exists():
            for name in ("result.json", "completed.json"):
                preserved.setdefault(name, (directory / name).read_bytes())
            raise OSError(5, "late export I/O failure")
        return original(root, limits)

    monkeypatch.setattr(quality, "artifact_usage", fail_after_completion)
    result = quality.run_worker(plan, tmp_path, time.perf_counter_ns() + 60 * 10**9)
    assert result["status"] == "failed"
    assert calls == ["F-feas-00"]
    assert releases == [True]
    directory = tmp_path / "runs" / "F-feas-00"
    assert {name: (directory / name).read_bytes() for name in preserved} == preserved
    assert quality.read_json(directory / "failure.json")["failure"]["type"] == "OSError"
    assert not (directory / "acknowledged.json").exists()


def test_rejected_cvd_or_source_never_calls_environment(tmp_path, monkeypatch):
    plan, _, _ = worker_standins(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(quality, "_environment", lambda _: pytest.fail("device entered"))
    result = quality.run_worker(plan, tmp_path, time.perf_counter_ns() + 60 * 10**9)
    assert result["status"] == "failed"
    assert result["device_initialization"] == "not_started"


def test_parent_failed_worker_prefix_is_retained_without_retry(tmp_path, monkeypatch):
    plan = minimal_plan()
    monkeypatch.setattr(
        quality,
        "_schema",
        lambda: SimpleNamespace(
            verify_quality_plan=lambda p: None, validate_quality_plan=lambda p: None
        ),
    )
    monkeypatch.setattr(quality, "_copy_inputs", lambda p, r: None)
    launched = []

    class Child:
        pid, returncode = 12345, 1

        def poll(self):
            return self.returncode

    output = tmp_path / "run"

    def launch(command, **kwargs):
        launched.append((command, kwargs["start_new_session"]))
        quality.write_json(
            output / "worker.json", {"status": "failed", "completed_runs": ["F-feas-00"]}
        )
        return Child()

    monkeypatch.setattr(quality.subprocess, "Popen", launch)
    result = quality.run_quality(plan, output)
    assert len(launched) == 1 and launched[0][1] is True
    assert result["status"] == "failed"
    assert result["completed_runs"] == ["F-feas-00"]


def test_parent_case_watchdog_stops_only_its_group(tmp_path, monkeypatch):
    plan = minimal_plan()
    monkeypatch.setattr(
        quality,
        "_schema",
        lambda: SimpleNamespace(
            verify_quality_plan=lambda p: None, validate_quality_plan=lambda p: None
        ),
    )
    monkeypatch.setattr(quality, "_copy_inputs", lambda p, r: None)
    killed = []
    output = tmp_path / "run"

    class Child:
        pid, returncode = 99123, None

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            self.returncode = -15
            return self.returncode

    def launch(*args, **kwargs):
        quality.write_json(output / "active-case.json", {"deadline_ns": 0})
        return Child()

    monkeypatch.setattr(quality.subprocess, "Popen", launch)
    monkeypatch.setattr(quality.os, "killpg", lambda *args: killed.append(args))
    result = quality.run_quality(plan, output)
    assert result["status"] == "failed"
    assert killed == [(99123, quality.signal.SIGTERM)]
    assert result["failures"][0]["type"] == "TimeoutError"


def test_execute_example_decodes_actual_eos_then_parses_exact_number():
    from vllm_lt.benchmarks.q2_quality_data import parse_answer

    class Tokenizer:
        def get_vocab_size(self, **kwargs):
            return 16

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return "".join({7: "reason\n#### 1,200.00", 0: "<|endoftext|>"}[token] for token in ids)

    engine = engine_for([7, 0])
    row = {
        "run_id": "F-eval-00",
        "phase": "evaluation",
        "example_index": 0,
        "example_id": "dataset/0",
        "max_steps": 1532,
    }
    example = {
        "source_id": "dataset/0",
        "prompt_token_ids": [3, 4],
        "prompt_sha256": "a" * 64,
        "parsed_reference": parse_answer("#### 1200"),
    }
    plan = {"plan_sha256": "b" * 64, "selection": {"evaluation": [example]}}
    now = time.perf_counter_ns()
    result = quality.execute_example(
        plan, row, engine, Tokenizer(), started_ns=now, deadline_ns=now + 30 * 10**9
    )
    assert result["status"] == "complete"
    assert result["correct"] is True
    assert result["raw_text"] == "reason\n#### 1,200.00<|endoftext|>"
    assert result["scoring_text"] == "reason\n#### 1,200.00"
    assert result["output_token_ids"] == [7, 0]
    assert result["finite_checks"] == {"lm_head": 2, "recurrent": 0, "coda": 0}


@pytest.mark.parametrize("initialized", [False, True])
def test_partial_environment_failure_has_best_effort_cleanup_without_false_zero(
    initialized, tmp_path, monkeypatch
):
    from vllm_lt.benchmarks import runner

    plan, calls, _ = worker_standins(monkeypatch)
    releases = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(
        quality,
        "_environment",
        lambda _: (_ for _ in ()).throw(
            ValueError("environment failed after possible CUDA initialization")
        ),
    )

    def release(worker):
        releases.append(True)
        worker["teardown_after_workspace_release"] = {"allocated_bytes": 0, "reserved_bytes": 0}

    monkeypatch.setattr(runner, "release_device", release)
    worker = quality.run_worker(plan, tmp_path, time.perf_counter_ns() + 30 * 10**9)
    assert worker["status"] == "failed"
    assert not calls
    assert releases == ([True] if initialized else [])
    if not initialized:
        assert worker["device_initialization"] == "not_initialized"
        assert "teardown_after_workspace_release" not in worker
    assert worker["failures"][0]["type"] == "ValueError"


def test_persistence_after_ack_deadline_fails_and_keeps_linked_bytes(tmp_path, monkeypatch):
    plan, calls, _ = worker_standins(monkeypatch)
    original = quality.check_deadline
    original_write = quality.write_json
    late = False

    def write(path, value):
        nonlocal late
        original_write(path, value)
        if Path(path).name == "acknowledged.json":
            late = True

    def check(deadline):
        if late:
            raise TimeoutError("deadline during ACK export")
        original(deadline)

    monkeypatch.setattr(quality, "write_json", write)
    monkeypatch.setattr(quality, "check_deadline", check)
    worker = quality.run_worker(plan, tmp_path, time.perf_counter_ns() + 30 * 10**9)
    assert calls == ["F-feas-00"]
    assert worker["status"] == "failed"
    assert worker["completed_runs"] == []
    directory = tmp_path / "runs" / "F-feas-00"
    ack = quality.read_json(directory / "acknowledged.json")
    assert ack["result"] == quality.file_record(directory / "result.json")
    assert ack["completion"] == quality.file_record(directory / "completed.json")
    assert quality.read_json(directory / "failure.json")["failure"]["type"] == "TimeoutError"
