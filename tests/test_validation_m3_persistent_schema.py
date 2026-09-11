"""CPU-only provenance and exact-budget checks for the persistent protocol."""

from copy import deepcopy
from pathlib import Path

import pytest
import test_benchmark_ab_schema as fixtures

from vllm_lt.benchmarks import ab_schema
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.validation import m3_inactive_schema as common
from vllm_lt.validation import m3_persistent_schema as schema

ab_plan = fixtures.ab_plan


@pytest.fixture(scope="module")
def lifecycle_template():
    from vllm_lt.validation.m3_persistent_lifecycle import build_lifecycle_plan

    return build_lifecycle_plan()


@pytest.fixture
def persistent_plan(ab_plan, lifecycle_template, monkeypatch, tmp_path):
    from vllm_lt.validation import m3_persistent_lifecycle as lifecycle

    roots = {key: Path(value["root"]) for key, value in ab_plan["implementations"].items()}
    # The reusable byte-only fixture starts with M2's production differences.
    # Persistent A/B restores exactly the three accepted runtime modules instead.
    for path in ab_schema.DIFF_PATHS:
        if path not in schema.DIFF_PATHS:
            (roots["B"] / path).write_bytes((roots["A"] / path).read_bytes())
    for key, root in roots.items():
        for relative in schema.DIFF_PATHS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(key + " persistent-source fixture\n")
    monkeypatch.setattr(common, "probe_checkout", ab_schema.probe_checkout)
    monkeypatch.setattr(common, "affinity_snapshot", lambda: deepcopy(fixtures.AFFINITY))
    monkeypatch.setattr(lifecycle, "build_lifecycle_plan", lambda: deepcopy(lifecycle_template))
    contract = tmp_path / "persistent-contract.json"
    write_json(
        contract, read_json(fixtures.ROOT / "benchmarks/fixtures/ouro-m3-persistent-contract.json")
    )
    return schema.make_plan(
        baseline_root=roots["A"],
        candidate_root=roots["B"],
        contract_path=contract,
        model_path=ab_plan["model_path"],
        gpu_ids=[7],
        affinity=deepcopy(fixtures.AFFINITY),
    )


def rehash(plan):
    plan["plan_sha256"] = schema._digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def test_cpu_plan_freezes17_executions_exact_sources_and_one_bucket(persistent_plan):
    schema.verify_plan(persistent_plan)
    rows = persistent_plan["execution_order"]
    assert len(rows) == len({r["execution_id"] for r in rows}) == 17
    assert [len(w["execution_ids"]) for w in persistent_plan["workers"]] == [11, 6]
    assert sum(r["kind"] == "model" for r in rows) == 13
    assert sum(r["kind"] == "lifecycle" for r in rows) == 4
    assert sum(r["phase"] == "feasibility" for r in rows) == 2
    for worker in persistent_plan["workers"]:
        selected = [r for r in rows if r["worker_id"] == worker["worker_id"]]
        assert selected[0]["phase"] == "feasibility" and selected[0]["kind"] == "model"
        assert [r["kind"] for r in selected[1:3]] == ["lifecycle", "lifecycle"]
        assert all(r["kind"] == "model" for r in selected[3:])
    assert persistent_plan["production_differences"] == sorted(schema.DIFF_PATHS)
    assert len(schema.DIFF_PATHS) == 3
    limits = persistent_plan["contract"]["limits"]
    assert limits["qualification_cases"] == 11 and limits["qualification_comparison_streams"] == 27
    assert limits["persistent_buckets"] == 1
    assert limits["profile_total_bytes_max"] == limits["profile_trace_bytes_max"] == 0
    assert limits["sanitizer_evaluations"] == 0
    assert (
        schema.loading_view(persistent_plan)["workload_stats"]["model"]["pool_bytes"] == 1006632960
    )


@pytest.mark.parametrize(
    "change",
    [
        "null_gpu",
        "bool_gpu",
        "second_bucket",
        "extra_profile",
        "larger_timeout",
        "lifecycle_omitted",
        "lifecycle_reordered",
        "lifecycle_plan",
        "allowlist",
        "dirty_source",
        "outside_import",
        "affinity",
        "unknown",
    ],
)
def test_rehashed_plan_rejects_scope_source_and_control_drift(persistent_plan, change):
    plan = deepcopy(persistent_plan)
    if change == "null_gpu":
        plan["contract"]["controls"]["gpu_ids"] = None
    elif change == "bool_gpu":
        plan["contract"]["controls"]["gpu_ids"] = [True]
    elif change == "second_bucket":
        plan["contract"]["limits"]["persistent_buckets"] = 2
    elif change == "extra_profile":
        plan["contract"]["limits"]["profile_total_bytes_max"] = 1
    elif change == "larger_timeout":
        plan["contract"]["limits"]["case_timeout_s"] = 601
    elif change == "lifecycle_omitted":
        del plan["execution_order"][1]
    elif change == "lifecycle_reordered":
        plan["execution_order"][1:3] = reversed(plan["execution_order"][1:3])
    elif change == "lifecycle_plan":
        plan["lifecycle"]["execution_order"][0]["backend"] = "invented"
    elif change == "allowlist":
        plan["contract"]["production_diff_paths"].append("vllm_lt/models/ouro.py")
    elif change == "dirty_source":
        plan["implementations"]["B"]["source"]["status"] = " M unreviewed.py"
    elif change == "outside_import":
        key = next(iter(plan["implementations"]["B"]["imports"]))
        plan["implementations"]["B"]["imports"][key] = "/other/check-out.py"
    elif change == "affinity":
        plan["contract"]["controls"]["affinity"]["numactl_show"]["policy"] = "default"
    else:
        plan["unrecognized"] = True
    rehash(plan)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(plan)


@pytest.mark.parametrize("change", ["checkpoint", "source", "environment", "affinity"])
def test_verification_rejects_actual_byte_or_runtime_control_drift(
    persistent_plan, monkeypatch, change
):
    if change == "checkpoint":
        (Path(persistent_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    elif change == "source":
        root = Path(persistent_plan["implementations"]["B"]["root"])
        (root / schema.DIFF_PATHS[0]).write_text("changed source\n")
    elif change == "environment":
        monkeypatch.setenv("OMP_NUM_THREADS", "changed")
    else:
        affinity = deepcopy(fixtures.AFFINITY)
        affinity["numactl_show"]["membind"] = "0"
        monkeypatch.setattr(common, "affinity_snapshot", lambda: affinity)
    with pytest.raises(ValueError):
        schema.verify_plan(persistent_plan)


def test_rehashed_embedded_inputs_still_must_match_original_file(persistent_plan):
    plan = deepcopy(persistent_plan)
    plan["inputs"]["suite"]["contents"]["provenance"]["prompt_construction"] += " changed"
    plan["numerical"] = schema.build_model_plan(
        plan["inputs"]["suite"]["contents"],
        plan["inputs"]["contract"]["contents"],
        plan["model_config"],
    )
    rehash(plan)
    schema.validate_plan(plan)
    with pytest.raises(ValueError, match="embedded input"):
        schema.verify_plan(plan)
