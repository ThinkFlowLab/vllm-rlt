"""M2's exact subset, retained A evidence and offline gates use real CPU calls."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import m2
from vllm_lt.validation.diagnostics import DiagnosticDump, SpoolBudget, TensorSpool
from vllm_lt.validation.evidence import ComparisonStream, boundary_key
from vllm_lt.validation.schema import _digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (
        read_json(ROOT / "benchmarks/fixtures/ouro-q1.json"),
        read_json(ROOT / "benchmarks/fixtures/ouro-q1-contract.json"),
    )


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("M2 CPU checks must not discover CUDA")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_real_subset_has_exact_case_comparison_and_byte_budget():
    suite, contract = inputs()
    plan = m2.build_numerical_plan(suite, contract, OuroConfig().to_dict())
    m2.validate_numerical_plan(json.loads(json.dumps(plan)))
    rows, comparisons = plan["execution_order"], plan["comparison_order"]
    assert len(rows) == 27 and len(comparisons) == 51
    assert [row["implementation_id"] for row in rows] == ["A"] * 16 + ["B"] * 11
    assert [row["implementation"] for row in rows[:5]] == ["oracle"] * 5
    assert sum(row["require_exact"] for row in comparisons) == 17
    assert all(row["dtype"] == "float32" and row["max_tokens"] == 9 for row in rows)
    assert all(row["retain_evidence"] == (row["implementation_id"] == "A") for row in rows)
    assert {row["spool_group"] for row in rows} == {row["case_id"] for row in rows}
    assert plan["resource_estimates"] == {
        "retained_tensor_bytes_upper_bound": 4860873504,
        "retained_index_records_upper_bound": 105414,
        "comparison_records_upper_bound": 243495,
        "max_retained_case_bytes": 939262464,
        "cases": 27,
        "comparisons": 51,
    }
    packed = next(row for row in rows if len(row["fixture_ids"]) == 4)
    assert (
        sum(
            4 * math.ceil(size / packed["block_size"])
            for size in packed["expected_capacity"].values()
        )
        == 132
    )
    assert plan["contract"]["comparison_policy"] == contract["comparison_policy"]
    selected = {row["fixture_id"]: row for row in plan["suite"]["fixtures"]}
    assert selected == {
        row["fixture_id"]: row for row in suite["fixtures"] if row["fixture_id"] in m2.FIXTURE_IDS
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_case",
        "relaxed_exact",
        "relaxed_oracle",
        "extra_fixture",
        "reset_group",
        "forged_q1",
    ],
)
def test_strict_subset_rejects_rehashed_control_changes(mutation):
    plan = m2.build_numerical_plan(*inputs(), OuroConfig().to_dict())
    if mutation == "missing_case":
        plan["execution_order"].pop()
    elif mutation == "relaxed_exact":
        next(row for row in plan["comparison_order"] if row["require_exact"])["require_exact"] = (
            False
        )
    elif mutation == "relaxed_oracle":
        plan["contract"]["comparison_policy"]["logits"]["float32"]["atol"] = 0.01
    elif mutation == "extra_fixture":
        plan["suite"]["fixtures"].append(copy.deepcopy(plan["suite"]["fixtures"][0]))
    elif mutation == "reset_group":
        plan["execution_order"][0]["spool_group"] = "one-unbounded-shared-group"
    else:
        plan["artifact_type"] = "validation_execution_plan"
    plan["numerical_plan_sha256"] = _digest(
        {key: value for key, value in plan.items() if key != "numerical_plan_sha256"}
    )
    with pytest.raises(ValueError):
        m2.validate_numerical_plan(plan)


def test_exact_gate_changes_required_result_without_changing_diagnostic_stats(tmp_path):
    _, contract = inputs()
    meta = {
        "fixture_id": "fixture",
        "operation": "layer_output",
        "positions": [15],
        "depth": 1,
        "layer": 0,
        "output_index": 0,
        "history_sha256": "unchanged-prefix",
    }
    spool = TensorSpool(tmp_path / "spools", "reference", "fixture", SpoolBudget(), "group")
    spool.write(boundary_key(meta), torch.tensor([1.0, 2.0]), meta)
    spool.close()
    dumps = DiagnosticDump(tmp_path / "dumps", ["pre1", "pre2"])
    row = {
        "comparison_id": "B-versus-A",
        "family": "main",
        "dtype": "float32",
        "reference_case_id": "reference",
        "candidate_case_id": "B",
        "fixture_id": "fixture",
        "comparison_kind": "implementation_exact",
        "require_exact": True,
        "expected_boundary_records": 1,
        "counts_are_upper_bounds": False,
    }
    stream = ComparisonStream(
        tmp_path,
        row,
        contract["comparison_policy"],
        dumps,
        "hash",
        diagnostics={
            "preselected_dump_dtype": "float32",
            "preselected_dump_comparison_kind": "implementation_exact",
        },
    )
    stream.observe(meta, torch.tensor([1.0, 2.00001]))
    stream.close()
    record = json.loads(stream.path.read_text())
    assert record["stats"]["passed"] is True and record["stats"]["status"] == "diagnostic_only"
    assert record["stats"]["allclose"] is None and record["stats"]["exact_equal"] is False
    assert stream.summary["required_failures"] == 1
    assert dumps.failure_fixture_ids == ["fixture"] and dumps.written_bytes == 16
    dumps.close()


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    from vllm_lt.core import kv_cache_manager

    with pytest.MonkeyPatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("actual CPU M2 integration must not discover CUDA")

        for name in ("is_available", "device_count", "current_device", "_lazy_init"):
            patch.setattr(torch.cuda, name, forbidden)
        # Exercise the frozen case structure with ordinary CPU cache execution;
        # this verifies collector/ledger integration, not a GPU kernel result.
        initialize = kv_cache_manager.KVCacheManager.__init__

        def initialize_cpu(self, *args, **kwargs):
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)

        patch.setattr(kv_cache_manager.KVCacheManager, "__init__", initialize_cpu)
        config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
        torch.manual_seed(41)
        model = OuroForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        parent = {
            "plan_sha256": "a" * 64,
            "numerical": m2.build_numerical_plan(*inputs(), config.to_dict()),
        }
        output = tmp_path_factory.mktemp("m2-numerical")
        a = m2.run_numerical_rows(
            model, parent, "A", output, time.perf_counter_ns() + 180_000_000_000
        )
        assert a["passed"], a["errors"]
        b = m2.run_numerical_rows(
            model, parent, "B", output, time.perf_counter_ns() + 180_000_000_000
        )
        assert b["passed"], b["errors"]
        yield output, parent, a, b


def test_two_loaded_workers_retain_A_and_restore_cumulative_ledger(completed_run):
    output, parent, a, b = completed_run
    assert a["complete"] and len(a["completed_cases"]) == 16
    assert b["complete"] and len(b["completed_cases"]) == 11
    assert a["ledger"]["diagnostic_dumps"]["written_bytes"] == 0
    assert b["ledger"]["tensor_written_bytes"] == (
        a["ledger"]["tensor_written_bytes"] + b["ledger"]["diagnostic_dumps"]["written_bytes"]
    )
    assert b["ledger"]["diagnostic_dumps"]["selected_fixture_ids"] == list(m2.PRESELECTED)
    result = m2.audit_numerical(output, parent)
    assert result["complete"] and result["passed"], result["errors"]
    assert result["counts"] == {
        "planned_cases": 27,
        "verified_cases": 27,
        "planned_comparisons": 51,
        "verified_comparisons": 51,
    }
    assert len(result["raw_evidence"]["retained_references"]) == 22


def test_offline_report_rejects_missing_coverage_and_invented_exactness(completed_run):
    output, parent, _, _ = completed_run
    folder = output / "numerical"
    comparison = next(
        row for row in parent["numerical"]["comparison_order"] if row["require_exact"]
    )
    path = folder / "comparisons" / (comparison["comparison_id"] + ".jsonl")
    summary_path = path.with_suffix(".summary.json")
    original, summary_original = path.read_bytes(), summary_path.read_bytes()
    try:
        records = [json.loads(line) for line in original.splitlines()]
        records[0]["require_exact"] = False
        raw = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in records)
        path.write_bytes(raw)
        summary = json.loads(summary_original)
        summary.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        write_json(summary_path, summary)
        report = m2.audit_numerical(output, parent)
        assert not report["passed"] and "exact-comparison" in report["errors"][0]["message"]
        path.write_bytes(original.split(b"\n", 1)[1])
        report = m2.audit_numerical(output, parent)
        assert not report["complete"]
    finally:
        path.write_bytes(original)
        summary_path.write_bytes(summary_original)


def test_cross_worker_ledger_rejects_reset_budget_and_missing_baseline(completed_run, tmp_path):
    output, parent, _, _ = completed_run
    view = m2._view(parent)
    case = view["execution_order"][0]
    spool = TensorSpool(
        tmp_path / "spools",
        case["case_id"],
        case["fixture_ids"][0],
        SpoolBudget(),
        case["spool_group"],
    )
    spool.write("one", torch.tensor([1.0, 2.0]), {})
    spool.close()
    ledger = {
        "completed_cases": [case["case_id"]],
        "tensor_written_bytes": 8,
        "spool_bytes_by_group": {case["spool_group"]: 8},
        "diagnostic_dumps": {
            "selected_fixture_ids": list(m2.PRESELECTED),
            "written_bytes": 0,
            "fixture_written_bytes": {},
        },
    }
    assert m2._restore_budget(tmp_path, view, ledger).written_bytes == 8
    ledger["tensor_written_bytes"] = 0
    with pytest.raises(ValueError, match="cumulative tensor ledger"):
        m2._restore_budget(tmp_path, view, ledger)
    report = m2.audit_numerical(output, {**parent, "plan_sha256": "b" * 64})
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("failure", ["execute", "elapsed", "lifetime"])
def test_case_watchdog_marker_survives_failure_and_stops_next_case(tmp_path, monkeypatch, failure):
    from vllm_lt.validation import runner

    config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
    model = OuroForCausalLM(config)
    parent = {
        "plan_sha256": "a" * 64,
        "numerical": m2.build_numerical_plan(*inputs(), config.to_dict()),
    }
    calls, now = [], [1_000_000_000]
    monkeypatch.setattr(m2.time, "perf_counter_ns", lambda: now[0])

    def execute(model, view, case, folder, budget, dumps, deadline):
        calls.append(case["case_id"])
        marker = read_json(folder / "ledger.json")["active_case"]
        assert marker == {
            "case_id": case["case_id"],
            "started_ns": 1_000_000_000,
            "deadline_ns": 601_000_000_000,
        }
        if failure == "execute":
            raise RuntimeError("intentional execution failure")
        if failure == "lifetime":
            now[0] = 601_000_000_001
        return {
            "status": "complete",
            "comparisons": [],
            "elapsed_s": 601 if failure == "elapsed" else 1,
        }

    monkeypatch.setattr(runner, "execute_case", execute)
    result = m2.run_numerical_rows(model, parent, "A", tmp_path, 999_000_000_000)
    assert not result["passed"] and not result["complete"]
    assert len(calls) == 1 and result["completed_cases"] == []
    ledger = read_json(tmp_path / "numerical/ledger.json")
    assert ledger["active_case"]["case_id"] == calls[0] and ledger["case_lifetimes"] == {}


def test_offline_report_rejects_forged_case_lifetime_and_worker_flags(completed_run):
    output, parent, _, _ = completed_run
    path = output / "numerical/ledger.json"
    original = path.read_bytes()
    try:
        ledger = json.loads(original)
        assert ledger["active_case"] is None
        assert len(ledger["case_lifetimes"]) == 27
        first = ledger["case_lifetimes"][ledger["completed_cases"][0]]
        first["finished_ns"] = first["deadline_ns"] + 1
        write_json(path, ledger)
        report = m2.audit_numerical(output, parent)
        assert not report["complete"] and "lifetime" in report["errors"][0]["message"]
        ledger = json.loads(original)
        ledger["workers"]["B"]["passed"] = False
        write_json(path, ledger)
        report = m2.audit_numerical(output, parent)
        assert not report["complete"] and "worker summary" in report["errors"][0]["message"]
        ledger["workers"]["B"]["complete"] = False
        write_json(path, ledger)
        report = m2.audit_numerical(output, parent)
        assert not report["complete"] and not report["passed"]
    finally:
        path.write_bytes(original)
