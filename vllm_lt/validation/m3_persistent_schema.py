"""Frozen 17-execution protocol for eager persistent decode storage."""

from pathlib import Path

from vllm_lt.benchmarks.ab_schema import equal, require, validate_affinity
from vllm_lt.benchmarks.schema import _constants, _file_record, _integer, _keys, _text

from .m3_inactive_schema import (
    CONTROLS,
    INPUTS,
    _probe_inputs,
    _validate_identity,
    _verify_inputs,
    loading_view,
)
from .m3_persistent import build_model_plan, validate_model_plan
from .schema import _digest, read_json

DIFF_PATHS = (
    "vllm_lt/core/kv_cache_manager.py",
    "vllm_lt/engine/llm_engine.py",
    "vllm_lt/worker/model_runner.py",
)
ACCEPTANCE = {
    "purpose": "eager-persistent-storage-correctness-only",
    "passes_per_case": 1,
    "full_model": "original-Q1-FP32-logits-and-actual-token-exit-gates",
    "intermediates": "finite-required-numerical-deltas-diagnostic",
    "storage": "stable-owned-boundary-tensors-and-independent-request-publication",
    "lifecycle": "exact-held-input-pairs-guards-leases-reuse-and-counted-fallbacks",
    "bucket": {"row_count": 8, "table_width": 32, "max_live_rows": 4, "mapping": "odd_slots"},
    "sanitizer": "unavailable-no-executions-no-coverage-claim",
}
LIMITS = {
    "total_timeout_s": 3600,
    "case_timeout_s": 600,
    "workers": 2,
    "model_loads": 2,
    "model_cases": 13,
    "excluded_feasibility_cases": 2,
    "qualification_cases": 11,
    "comparison_streams": 28,
    "qualification_comparison_streams": 27,
    "lifecycle_evaluations": 4,
    "sanitizer_evaluations": 0,
    "persistent_device_payload_bytes": 256 * 1024,
    "persistent_cpu_staging_bytes": 16 * 1024,
    "persistent_buckets": 1,
    "lifecycle_evidence_bytes": 256 * 1024**2,
    "artifact_bytes_max": 12 * 1024**3,
    "profile_total_bytes_max": 0,
    "profile_trace_bytes_max": 0,
}


def validate_contract(contract, *, resolved=False):
    _keys(
        contract,
        (
            "schema_version",
            "artifact_type",
            "contract_id",
            "hypothesis",
            "isolated_variable",
            "inputs",
            "production_diff_paths",
            "controls",
            "acceptance",
            "limits",
            "stop_conditions",
        ),
        name="persistent contract",
    )
    equal(
        [contract["schema_version"], contract["artifact_type"]],
        [1, "m3_persistent_contract"],
        "persistent contract version",
    )
    for key in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[key], key)
    _constants(contract["inputs"], INPUTS, "input references")
    equal(contract["production_diff_paths"], list(DIFF_PATHS), "production source allowlist")
    _constants(contract["acceptance"], ACCEPTANCE, "persistent acceptance")
    _constants(contract["limits"], LIMITS, "execution/evidence/storage budgets")
    controls = contract["controls"]
    _keys(controls, (*CONTROLS, "gpu_ids", "affinity"), name="controls")
    _constants({key: controls[key] for key in CONTROLS}, CONTROLS, "fixed controls")
    if resolved or controls["gpu_ids"] is not None:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one explicit physical GPU required",
        )
        _integer(controls["gpu_ids"][0], "physical GPU ID")
    if resolved or controls["affinity"] is not None:
        validate_affinity(controls["affinity"])
    require(
        isinstance(contract["stop_conditions"], list) and bool(contract["stop_conditions"]),
        "explicit stop conditions required",
    )
    for condition in contract["stop_conditions"]:
        _text(condition, "stop condition")


def execution_rows(numerical, lifecycle):
    rows, workers = [], []
    for implementation in ("A", "B"):
        worker_id = "N-" + implementation
        cases = [
            r for r in numerical["execution_order"] if r["implementation_id"] == implementation
        ]
        checks = [
            r for r in lifecycle["execution_order"] if r["implementation_id"] == implementation
        ]
        entries = [
            {"execution_id": cases[0]["case_id"], "kind": "model", "phase": "feasibility"},
            *[
                {"execution_id": r["evaluation_id"], "kind": "lifecycle", "phase": "validation"}
                for r in checks
            ],
            *[
                {"execution_id": r["case_id"], "kind": "model", "phase": "validation"}
                for r in cases[1:]
            ],
        ]
        for entry in entries:
            entry.update(worker_id=worker_id, implementation_id=implementation)
        rows.extend(entries)
        workers.append(
            {
                "worker_id": worker_id,
                "implementation_id": implementation,
                "execution_ids": [row["execution_id"] for row in entries],
            }
        )
    require(
        len(rows) == len({row["execution_id"] for row in rows}) == 17, "exact17 unique executions"
    )
    return workers, rows


def make_plan(*, baseline_root, candidate_root, contract_path, model_path, gpu_ids, affinity):
    from .m3_persistent_lifecycle import build_lifecycle_plan

    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    contract["controls"].update(gpu_ids=gpu_ids, affinity=affinity)
    validate_contract(contract, resolved=True)
    common = _probe_inputs(baseline_root, candidate_root, model_path, diff_paths=DIFF_PATHS)
    numerical = build_model_plan(
        common["inputs"]["suite"]["contents"],
        common["inputs"]["contract"]["contents"],
        common["model_config"],
    )
    lifecycle = build_lifecycle_plan()
    workers, rows = execution_rows(numerical, lifecycle)
    plan = {
        "schema_version": 1,
        "artifact_type": "m3_persistent_plan",
        "contract": contract,
        "contract_file": _file_record(contract_path),
        **common,
        "numerical": numerical,
        "lifecycle": lifecycle,
        "workers": workers,
        "execution_order": rows,
    }
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    from .m3_persistent_lifecycle import build_lifecycle_plan

    _keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "contract",
            "contract_file",
            "inputs",
            "implementations",
            "harness",
            "production_differences",
            "dependencies",
            "model_path",
            "model_config",
            "model_files",
            "numerical",
            "lifecycle",
            "workers",
            "execution_order",
            "interpreter",
            "runtime_environment",
            "plan_sha256",
        ),
        name="persistent plan",
    )
    equal(
        [plan["schema_version"], plan["artifact_type"]], [1, "m3_persistent_plan"], "plan version"
    )
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan selfhash",
    )
    validate_contract(plan["contract"], resolved=True)
    _validate_identity(plan, diff_paths=DIFF_PATHS)
    validate_model_plan(plan["numerical"])
    equal(plan["numerical"]["model_config"], plan["model_config"], "model configuration")
    equal(
        plan["numerical"]["source_inputs"],
        {key: value["contents"] for key, value in plan["inputs"].items()},
        "embedded original inputs",
    )
    equal(plan["lifecycle"], build_lifecycle_plan(), "exact held-input lifecycle plan")
    workers, rows = execution_rows(plan["numerical"], plan["lifecycle"])
    equal(
        [plan["workers"], plan["execution_order"]], [workers, rows], "exact worker execution order"
    )
    resources = plan["numerical"]["resource_estimates"]
    estimate = (
        resources["retained_tensor_bytes_upper_bound"]
        + resources["retained_index_records_upper_bound"] * 1024
        + resources["comparison_records_upper_bound"] * 2048
        + resources["pointer_evidence_bytes_upper_bound"]
        + plan["numerical"]["contract"]["limits"]["persisted_dump_bytes"]
        + LIMITS["lifecycle_evidence_bytes"]
        + 1024**3
    )
    require(
        estimate <= LIMITS["artifact_bytes_max"], "estimated complete artifacts exceed disk cap"
    )


def verify_plan(plan, *, implementation_id=None):
    validate_plan(plan)
    _verify_inputs(plan, implementation_id=implementation_id)


__all__ = ["make_plan", "validate_plan", "verify_plan", "loading_view"]
