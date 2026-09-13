"""Exercise runner orchestration with real tiny CPU engines and mocked CUDA boundaries."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from vllm_lt.benchmarks import runner
from vllm_lt.benchmarks.schema import read_json
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Stage

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "fixtures"


@pytest.fixture
def cuda_boundaries(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("mocked runner test attempted real device discovery/initialization")

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            calls.append("event_record")

        def elapsed_time(self, other):
            return 1.0  # Synthetic value, never interpreted as a measured GPU result.

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        torch._C,
        "_cuda_clearCublasWorkspaces",
        lambda: calls.append("workspace_release"),
        raising=False,
    )
    monkeypatch.setattr(runner, "memory", lambda: {"allocated_bytes": 0, "reserved_bytes": 0})
    return calls


def case():
    contract = read_json(FIXTURES / "ouro-m1-contract.json")
    # Private execution mechanics use a tiny CPU pool; this is not an accepted M1 plan.
    contract["engine"]["attention_backend"] = "torch"
    contract["engine"]["cache"] = {"num_blocks": 64, "block_size": 2}
    workload = {
        "workload_id": "cpu",
        "kind": "fixed_depth_generation",
        "requests": [
            {
                "request_id": "request",
                "prompt_token_ids": [1, 2],
                "max_output_tokens": 3,
                "arrival_offset_ns": 0,
            },
        ],
    }
    run = {
        "run_id": "measured-cpu-refill-1",
        "cell_id": "cpu-refill",
        "workload_id": "cpu",
        "mode": "refill",
        "phase": "measured",
        "repetition": 1,
        "pair_id": None,
        "instrumentation": "timing",
        "max_steps": 100,
        "max_events": 1000,
        "controls_sha256": "a" * 64,
        "workload_sha256": "b" * 64,
    }
    plan = {"contract": contract, "plan_sha256": "c" * 64}
    return plan, run, workload


def test_execute_preserves_hashes_and_starts_clock_after_observer_setup(
    tmp_path, monkeypatch, cuda_boundaries
):
    plan, run, workload = case()
    original_instrument = runner.instrument_engine
    ready = []

    @contextmanager
    def instrument(*args, **kwargs):
        with original_instrument(*args, **kwargs) as observed:
            ready.append(runner.time.perf_counter_ns())
            yield observed

    monkeypatch.setattr(runner, "instrument_engine", instrument)
    result = runner._execute(
        OuroForCausalLM(OuroConfig.tiny()), plan, run, workload, tmp_path, deadline=10**30
    )
    assert result["status"] == "complete", result["failures"]
    assert result["comparison_eligible"]
    assert result["controls_sha256"] == run["controls_sha256"]
    assert result["workload_sha256"] == run["workload_sha256"]
    assert result["arrival_ns"] >= ready[0]
    assert len(result["requests"][0]["token_ids"]) == 3
    assert result["requests"][0]["exit_depths"] == [4, 4, 4]
    assert result["cleanup"] == {"active_requests": 0, "used_kv_blocks": 0}
    folder = tmp_path / "runs" / run["run_id"]
    assert read_json(folder / "result.json")["status"] == "complete"
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "token_emitted"]) == 3
    assert "workspace_release" not in cuda_boundaries  # Between-run workspace flush is forbidden.


def test_execute_keeps_partial_outputs_and_releases_requests_on_failure(
    tmp_path, monkeypatch, cuda_boundaries
):
    plan, run, workload = case()

    def failing_engine(*args, **kwargs):
        instance = LLMEngine(*args, **kwargs)
        original = instance.model_runner.execute

        def execute(batch):
            if batch.stage == Stage.PRELUDE:
                raise RuntimeError("injected failure after first output")
            return original(batch)

        instance.model_runner.execute = execute
        return instance

    monkeypatch.setattr(runner, "LLMEngine", failing_engine)
    result = runner._execute(
        OuroForCausalLM(OuroConfig.tiny()), plan, run, workload, tmp_path, deadline=10**30
    )
    assert result["status"] == "failed"
    assert not result["comparison_eligible"]
    assert result["synchronized_ns"] is None
    assert len(result["requests"]) == 1
    assert len(result["requests"][0]["token_ids"]) == 1
    assert not result["requests"][0]["finished"]
    assert result["cleanup"] == {"active_requests": 0, "used_kv_blocks": 0}
    events_path = tmp_path / "runs" / run["run_id"] / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "token_emitted"]) == 1
    assert len([event for event in events if event["kind"] == "run_failed"]) == 1
    assert "workspace_release" not in cuda_boundaries


def test_unavailable_scheduler_rejected_before_device_query(monkeypatch, cuda_boundaries):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="scheduler"):
        runner.environment()


def test_invalid_plan_rejected_before_output_or_device_access(
    tmp_path, monkeypatch, cuda_boundaries
):
    def reject(plan):
        raise ValueError("frozen source changed")

    monkeypatch.setattr(runner, "verify_plan", reject)
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="source changed"):
        runner.run_plan({}, output_dir=output)
    assert not output.exists()
    assert not cuda_boundaries


def test_failed_execution_stops_plan_without_retry_or_later_run(
    tmp_path, monkeypatch, cuda_boundaries
):
    plan, first, workload = case()
    second = {**first, "run_id": "measured-cpu-refill-2", "repetition": 2}
    plan.update(
        {
            "source": {},
            "model_path": "unused",
            "execution_order": [first, second],
            "suite": {"workloads": [workload]},
        }
    )
    monkeypatch.setattr(runner, "verify_plan", lambda _: None)
    monkeypatch.setattr(runner, "environment", lambda: {"mocked": True})
    monkeypatch.setattr(runner, "capacity_check", lambda _: {"mocked": True})
    for name in ("set_num_threads", "set_num_interop_threads", "manual_seed"):
        monkeypatch.setattr(torch, name, lambda _: None)
    for owner, names in [
        (
            torch.backends.cuda.matmul,
            [
                "allow_tf32",
                "allow_bf16_reduced_precision_reduction",
                "allow_fp16_reduced_precision_reduction",
            ],
        ),
        (torch.backends.cudnn, ["allow_tf32"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))  # Restore controls after test.
    loaded = []

    def load(*args, **kwargs):
        loaded.append(kwargs)
        return object()

    monkeypatch.setattr(runner.OuroForCausalLM, "from_pretrained", load)
    executions = []

    def fail(model, plan, run, workload, output_dir, deadline):
        executions.append(run["run_id"])
        folder = output_dir / "runs" / run["run_id"]
        folder.mkdir(parents=True)
        result = {**run, "status": "failed", "failures": [{"message": "injected failure"}]}
        runner.write_json(folder / "result.json", result)
        return result

    monkeypatch.setattr(runner, "_execute", fail)
    output = tmp_path / "experiment"
    manifest = runner.run_plan(plan, output_dir=output)
    assert manifest["status"] == "failed"
    assert executions == [first["run_id"]]
    assert len(loaded) == 1
    assert manifest["completed_runs"] == []
    assert not (output / "runs" / second["run_id"]).exists()
    assert read_json(output / "manifest.json")["status"] == "failed"
    assert cuda_boundaries.count("workspace_release") == 1  # Final teardown only.


@pytest.mark.parametrize("headroom", [-1, 0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_capacity_preflight_uses_meta_parameters_and_rejects_impossible_pool(
    monkeypatch, cuda_boundaries, headroom, dtype
):
    config = OuroConfig.tiny()
    original = runner.OuroForCausalLM
    weight_bytes = sum(
        parameter.numel() * dtype.itemsize for parameter in original(config).parameters()
    )
    pool_bytes = 1024
    minimum = weight_bytes + pool_bytes
    observed = []

    def sizing(configuration):
        model = original(configuration)
        observed.append(all(parameter.is_meta for parameter in model.parameters()))
        return model

    monkeypatch.setattr(runner, "OuroForCausalLM", sizing)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (minimum + headroom, minimum * 2))
    plan = {
        "model_config": config.to_dict(),
        "workload_stats": {"cpu": {"pool_bytes": pool_bytes}},
        "contract": {"engine": {"dtype": str(dtype).removeprefix("torch.")}},
    }
    if headroom <= 0:
        with pytest.raises(ValueError, match="insufficient device memory"):
            runner.capacity_check(plan)
    else:
        result = runner.capacity_check(plan)
        assert result["minimum_resident_bytes"] == minimum
        assert result["remaining_bytes"] == headroom
    assert observed == [True]
