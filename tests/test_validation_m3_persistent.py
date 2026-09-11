"""Real tiny CPU trajectories and adverse pointer evidence, without CUDA discovery."""

import math
import time
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import m3_inactive
from vllm_lt.validation import m3_persistent as persistent
from vllm_lt.validation.schema import read_json

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (
        read_json(ROOT / "benchmarks/fixtures/ouro-q1.json"),
        read_json(ROOT / "benchmarks/fixtures/ouro-q1-contract.json"),
    )


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("persistent CPU validation attempted CUDA")

    for name in ("is_available", "device_count", "current_device", "_lazy_init", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_plan_retains_same_physical_shape_histories_gates_and_finite_bounds():
    suite, contract = inputs()
    config = OuroConfig().to_dict()
    plan = persistent.build_model_plan(suite, contract, config)
    persistent.validate_model_plan(plan)
    prior = m3_inactive.build_model_plan(suite, contract, config)
    assert len(plan["execution_order"]) == 13 and len(plan["comparison_order"]) == 28
    assert plan["contract"]["comparison_policy"] == contract["comparison_policy"]
    assert plan["resource_estimates"]["retained_tensor_bytes_upper_bound"] == 3067709328
    assert plan["resource_estimates"]["comparison_records_upper_bound"] == 133776
    assert all(not row["require_exact"] for row in plan["comparison_order"])
    for before, case in zip(prior["execution_order"], plan["execution_order"]):
        for field in (
            "fixture_ids",
            "fixture_sha256",
            "exit_policy",
            "history_mode",
            "expected_capacity",
            "max_steps",
        ):
            assert case[field] == before[field]
        if case["implementation"] == "oracle":
            assert case["storage_strategy"] == "oracle_compact" and case["padding"] == {
                "mode": "compact"
            }
        else:
            assert case["padding"] == m3_inactive.PADDING
            assert case["storage_strategy"] == (
                "allocating_padded" if case["implementation_id"] == "A" else "persistent_decode"
            )
    records = sum(
        c["max_steps"] for c in plan["execution_order"] if c["implementation"] == "native"
    )
    assert plan["resource_estimates"]["pointer_records_upper_bound"] == records
    assert plan["resource_estimates"]["pointer_evidence_bytes_upper_bound"] == records * 32768


@pytest.mark.parametrize("change", ["shape", "strategy", "policy", "extra", "pointer_cap"])
def test_rehashed_strategy_shape_or_acceptance_changes_are_rejected(change):
    plan = persistent.build_model_plan(*inputs(), OuroConfig().to_dict())
    if change == "shape":
        plan["execution_order"][-1]["padding"]["row_count"] = 4
    elif change == "strategy":
        plan["execution_order"][-1]["storage_strategy"] = "allocating_padded"
    elif change == "policy":
        plan["comparison_order"][0]["require_exact"] = True
    elif change == "extra":
        plan["execution_order"].append(deepcopy(plan["execution_order"][-1]))
    else:
        plan["storage_evidence"]["dispatch_record_bytes"] *= 2
    plan["numerical_plan_sha256"] = persistent._digest(
        {k: v for k, v in plan.items() if k != "numerical_plan_sha256"}
    )
    with pytest.raises(ValueError, match="frozen same-shape subset"):
        persistent.validate_model_plan(plan)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    from vllm_lt.core.kv_cache_manager import KVCacheManager

    with pytest.MonkeyPatch.context() as patch:
        initialize = KVCacheManager.__init__

        def cpu(self, *args, **kwargs):
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)

        def forbidden(*args, **kwargs):
            pytest.fail("tiny persistent trajectory attempted CUDA")

        patch.setattr(KVCacheManager, "__init__", cpu)
        for name in ("is_available", "device_count", "current_device", "_lazy_init", "synchronize"):
            patch.setattr(torch.cuda, name, forbidden)
        config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
        torch.manual_seed(41)
        model = OuroForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        parent = {
            "plan_sha256": "a" * 64,
            "numerical": persistent.build_model_plan(*inputs(), config.to_dict()),
        }
        output = tmp_path_factory.mktemp("m3-persistent")
        for implementation in ("A", "B"):
            result = persistent.run_model_rows(
                model, parent, implementation, output, time.perf_counter_ns() + 180 * 10**9
            )
            assert result["complete"] and result["passed"], result["errors"]
        yield output, parent


def test_same_shape_real_execution_and_full_offline_pointer_coverage(completed_run):
    output, parent = completed_run
    result = persistent.audit_model_rows(output, parent, expected_device="cpu")
    assert result["complete"] and result["passed"], result["errors"]
    assert result["counts"]["verified_cases"] == 13
    assert result["counts"]["verified_comparisons"] == 28
    assert result["counts"]["verified_qualification_comparisons"] == 27
    assert result["counts"]["verified_storage_dispatches"] > 100
    assert (
        result["counts"]["verified_persistent_dispatches"]
        == result["counts"]["verified_allocating_padded_dispatches"]
    )
    assert (
        result["counts"]["verified_storage_dispatches"]
        == 2 * result["counts"]["verified_persistent_dispatches"]
    )
    for case in parent["numerical"]["execution_order"]:
        value = read_json(output / "numerical/cases" / case["case_id"] / "result.json")
        events = value["storage_observations"]
        if case["implementation"] == "oracle":
            assert events == []
            continue
        assert len(events) == len(value["padding_observations"])
        for event in events:
            assert event["model_input"]["shape"] == [8, 32]
            assert event["metadata"]["block_tables"]["shape"] == [8, 32]
        if case["implementation_id"] == "B":
            assert len({event["model_input"]["data_ptr"] for event in events}) == 1
            assert all(
                event["before"]["status"] == event["after"]["status"] == "ready" for event in events
            )
            assert all(event["in_flight"]["status"] == "in_flight" for event in events)
            assert len(events) > 1
    # The ordinary public audit cannot qualify this CPU stand-in as CUDA evidence.
    unqualified = persistent.audit_model_rows(output, parent)
    assert not unqualified["complete"] and not unqualified["passed"]
    assert "tensor device differs" in unqualified["errors"][-1]["message"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "actual_input",
        "missing_publication",
        "logical",
        "staging",
        "moving_storage",
        "fallback",
        "phase",
        "publication_alias",
        "record_cap",
    ],
)
def test_omitted_changed_or_aliased_pointer_evidence_cannot_pass(completed_run, change):
    output, parent = completed_run
    case = next(c for c in parent["numerical"]["execution_order"] if c["implementation_id"] == "B")
    value = read_json(output / "numerical/cases" / case["case_id"] / "result.json")
    events = value["storage_observations"]
    event = events[1]
    if change == "missing":
        events.pop()
    elif change == "duplicate":
        events.append(deepcopy(events[-1]))
    elif change == "actual_input":
        event["model_input"] = deepcopy(event["in_flight"]["tensors"]["hidden_out"])
    elif change == "missing_publication":
        del event["publication"]
    elif change == "logical":
        event["logical"]["request_ids"] = ["another-request"]
    elif change == "staging":
        for snapshot in (event["before"], event["in_flight"], event["after"]):
            del snapshot["staging_tensors"]["active"]
    elif change == "moving_storage":
        moved = event["after"]["tensors"]["hidden_out"]
        moved["data_ptr"] += 2**40
        moved["storage_ptr"] += 2**40
    elif change == "fallback":
        event["after"]["fallback_counts"]["table_width"] += 1
    elif change == "phase":
        event["in_flight"]["status"] = "ready"
    elif change == "publication_alias":
        address = event["after"]["tensors"]["hidden_out"]["storage_ptr"]
        old = event["publication"]["hidden"]["storage_ptr"]
        for descriptor in [
            event["publication"]["hidden"],
            event["after"]["last_publication"]["hidden"],
            *event["request_hidden"],
        ]:
            descriptor["storage_ptr"] = address
            descriptor["data_ptr"] += address - old
    else:
        event["unexpected_payload"] = "x" * 32768
    with pytest.raises((ValueError, KeyError)):
        persistent._audit_storage_case(case, value, 32, expected_device="cpu")


def test_model_failure_retains_partial_observation_and_restores_hooks(tmp_path, monkeypatch):
    from vllm_lt.core.kv_cache_manager import KVCacheManager
    from vllm_lt.validation.diagnostics import DiagnosticDump, SpoolBudget

    initialize = KVCacheManager.__init__

    def cpu(self, *args, **kwargs):
        kwargs["backend"] = "torch"
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(KVCacheManager, "__init__", cpu)
    model = OuroForCausalLM(OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512))
    parent = {
        "plan_sha256": "b" * 64,
        "numerical": persistent.build_model_plan(*inputs(), model.config.to_dict()),
    }
    view = persistent.model_view(parent)
    case = view["execution_order"][0]
    original = model._recurrent_prepared

    def fail(hidden, batch, cache):
        if batch.active is not None:
            raise RuntimeError("injected recurrent storage failure")
        return original(hidden, batch, cache)

    monkeypatch.setattr(model, "_recurrent_prepared", fail)
    budget = SpoolBudget()
    dumps = DiagnosticDump(
        tmp_path / "dumps",
        view["contract"]["diagnostics"]["preselected_dump_fixture_ids"],
        budget=budget,
    )
    try:
        with pytest.raises(RuntimeError, match="injected recurrent storage failure"):
            persistent.execute_model_case(
                model, view, case, tmp_path, budget, dumps, time.monotonic() + 30
            )
    finally:
        dumps.close()
    result = read_json(tmp_path / "cases" / case["case_id"] / "result.json")
    assert result["status"] == "failed" and result["cleanup"] == {
        "requests_remaining": 0,
        "used_blocks": 0,
    }
    assert len(result["storage_observations"]) == 1
    assert result["storage_observations"][0]["status"] == "incomplete"
    assert "model_input" in result["storage_observations"][0]
    assert "publication" not in result["storage_observations"][0]
    assert model._recurrent_prepared is fail


def test_completed_driver_cannot_omit_dispatch_observations_and_advance(tmp_path, monkeypatch):
    from vllm_lt.validation import runner
    from vllm_lt.validation.schema import write_json

    model = OuroForCausalLM(OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512))
    plan = persistent.build_model_plan(*inputs(), model.config.to_dict())
    case = plan["execution_order"][0]

    def missing_observers(model, view, case, output, budget, dumps, deadline):
        folder = output / "cases" / case["case_id"]
        folder.mkdir(parents=True)
        value = {
            "status": "complete",
            "failures": [],
            "schedule": [
                {
                    "stage": "recurrent",
                    "request_ids": case["fixture_ids"],
                    "depths_after_step": [1],
                    "positions_after_step": [24],
                }
            ],
        }
        write_json(folder / "result.json", value)
        return value

    monkeypatch.setattr(runner, "execute_case", missing_observers)
    with pytest.raises(ValueError, match="padded dispatch coverage"):
        persistent.execute_model_case(
            model, plan, case, tmp_path, None, None, time.monotonic() + 30
        )
    value = read_json(tmp_path / "cases" / case["case_id"] / "result.json")
    assert value["status"] == "failed"
    assert value["failures"][0]["phase"] == "storage_audit"
