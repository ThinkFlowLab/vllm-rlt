"""CPU-only source/inputs and exact65-execution inactive-row contract."""

import os
import sys
from copy import deepcopy
from pathlib import Path

from vllm_lt.benchmarks.ab_schema import (
    IMPORT_MODULES,
    RUNTIME_VARIABLES,
    _harness,
    affinity_snapshot,
    equal,
    probe_checkout,
    require,
    source_probe,
    validate_affinity,
    validate_model_config,
)
from vllm_lt.benchmarks.schema import _constants, _file_record, _integer, _keys, _model_files, _text
from vllm_lt.models import OuroConfig

from .m3_inactive import PADDING, build_model_plan, validate_model_plan
from .schema import _digest, _file_records, _validate_dependencies, read_json

INPUTS = {
    "suite": "benchmarks/fixtures/ouro-q1.json",
    "contract": "benchmarks/fixtures/ouro-q1-contract.json",
}
DIFF_PATHS = (
    "vllm_lt/core/kv_cache_manager.py",
    "vllm_lt/kernels/paged_attention.py",
    "vllm_lt/kernels/triton_attention.py",
    "vllm_lt/models/ouro.py",
    "vllm_lt/worker/model_runner.py",
)
CONTROLS = {
    "cpu_threads": 1,
    "interop_threads": 1,
    "seed": 0,
    "dtype": "float32",
    "numa_policy": "inherit-verified-parent-binding",
    "residency_policy": "two-fresh-workers-one-model-each-one-pool-at-a-time",
    "cache_policy": "fresh-case-state-preserve-worker-allocator",
    "environment_policy": "same-prepared-Q1-environment",
}
ACCEPTANCE = {
    "purpose": "inactive-row-correctness-only",
    "passes_per_case": 1,
    "full_model": "original-Q1-FP32-logits-and-actual-token-exit-gates",
    "intermediates": "finite-required-numerical-deltas-diagnostic",
    "held_input": "exact-active-results-and-whole-cache-guards-zero-inactive-attention",
    "padding": PADDING,
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
    "kernel_evaluations": 52,
    "sanitizer_evaluations": 0,
    "kernel_evidence_bytes": 256 * 1024**2,
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
        name="inactive contract",
    )
    equal(
        [contract["schema_version"], contract["artifact_type"]],
        [1, "m3_inactive_contract"],
        "version",
    )
    for key in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[key], key)
    _constants(contract["inputs"], INPUTS, "input references")
    equal(contract["production_diff_paths"], list(DIFF_PATHS), "production source allowlist")
    _constants(contract["limits"], LIMITS, "execution/evidence budgets")
    _constants(contract["acceptance"], ACCEPTANCE, "correctness gates")
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
    for value in contract["stop_conditions"]:
        _text(value, "stop condition")


def source_controls(implementations, *, diff_paths=DIFF_PATHS):
    _keys(implementations, ("A", "B"), name="implementations")
    for item in implementations.values():
        _keys(item, ("root", "source", "imports"), name="implementation")
        source = item["source"]
        _keys(source, ("root", "commit", "status", "files"), name="source")
        require(Path(item["root"]).is_absolute(), "absolute execution source path required")
        equal(source["root"], item["root"], "source root")
        require(source["status"] == "", "frozen execution checkout must be clean")
        sha = source["commit"]
        require(
            isinstance(sha, str) and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha),
            "exact source commit required",
        )
        _file_records(source["files"], "source files")
        for record in source["files"]:
            require(
                not Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts,
                "source relative paths must stay in checkout",
            )
        _keys(item["imports"], IMPORT_MODULES, name="source imports")
        for name, path in item["imports"].items():
            equal(path, name.replace(".", "/") + ".py", "frozen module location")
    require(
        implementations["A"]["root"] != implementations["B"]["root"], "distinct execution checkouts"
    )
    a, b = (implementations[k]["source"] for k in ("A", "B"))
    harness = _harness(a)
    equal(_harness(b), harness, "common harness bytes")
    amap, bmap = ({r["path"]: r for r in source["files"]} for source in (a, b))
    differences = sorted(k for k in set(amap) | set(bmap) if amap.get(k) != bmap.get(k))
    require(
        bool(differences) and set(differences) <= set(diff_paths),
        "unexpected production differences",
    )
    return harness, differences


def execution_rows(numerical, kernels):
    rows, workers = [], []
    for implementation in ("A", "B"):
        worker_id = "N-" + implementation
        cases = [
            r for r in numerical["execution_order"] if r["implementation_id"] == implementation
        ]
        kernel_rows = [
            r for r in kernels["execution_order"] if r["implementation_id"] == implementation
        ]
        entries = [
            {"execution_id": cases[0]["case_id"], "kind": "model", "phase": "feasibility"},
            *[
                {"execution_id": r["evaluation_id"], "kind": "kernel", "phase": "validation"}
                for r in kernel_rows
            ],
            *[
                {"execution_id": r["case_id"], "kind": "model", "phase": "validation"}
                for r in cases[1:]
            ],
        ]
        for row in entries:
            row.update(worker_id=worker_id, implementation_id=implementation)
        rows.extend(entries)
        workers.append(
            {
                "worker_id": worker_id,
                "implementation_id": implementation,
                "execution_ids": [r["execution_id"] for r in entries],
            }
        )
    require(len(rows) == len({r["execution_id"] for r in rows}) == 65, "exact65 unique executions")
    return workers, rows


def _probe_inputs(baseline_root, candidate_root, model_path, *, diff_paths=DIFF_PATHS):
    """Freeze the shared source/checkpoint/environment identity without CUDA."""
    roots = {"A": Path(baseline_root).resolve(), "B": Path(candidate_root).resolve()}
    probes = {key: probe_checkout(root) for key, root in roots.items()}
    equal(probes["A"]["dependencies"], probes["B"]["dependencies"], "A/B dependencies")
    implementations = {
        key: {"root": str(roots[key]), "source": p["source"], "imports": p["imports"]}
        for key, p in probes.items()
    }
    harness, differences = source_controls(implementations, diff_paths=diff_paths)
    inputs = {}
    for key, relative in INPUTS.items():
        record, other = (_file_record(roots[k] / relative) for k in ("A", "B"))
        equal(
            [record["sha256"], record["size_bytes"]],
            [other["sha256"], other["size_bytes"]],
            "source inputs",
        )
        inputs[key] = {"file": record, "contents": read_json(roots["A"] / relative)}
    config = OuroConfig.from_dict(read_json(model_path / "config.json")).to_dict()
    validate_model_config(config)
    files = _model_files(model_path)
    tokenizer = next(row for row in files if row["path"] == "tokenizer.json")
    for key in ("size_bytes", "sha256"):
        equal(
            tokenizer[key],
            inputs["suite"]["contents"]["provenance"]["tokenizer"][key],
            "fixture tokenizer",
        )
    return {
        "inputs": inputs,
        "implementations": implementations,
        "harness": harness,
        "production_differences": differences,
        "dependencies": probes["A"]["dependencies"],
        "model_path": str(model_path),
        "model_config": config,
        "model_files": files,
        "interpreter": sys.executable,
        "runtime_environment": {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
    }


def make_plan(*, baseline_root, candidate_root, contract_path, model_path, gpu_ids, affinity):
    from .m3_inactive_kernels import build_kernel_plan

    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    contract["controls"].update(gpu_ids=gpu_ids, affinity=affinity)
    validate_contract(contract, resolved=True)
    common = _probe_inputs(baseline_root, candidate_root, model_path)
    inputs, config = common["inputs"], common["model_config"]
    numerical = build_model_plan(
        inputs["suite"]["contents"], inputs["contract"]["contents"], config
    )
    kernels = build_kernel_plan()
    workers, rows = execution_rows(numerical, kernels)
    plan = {
        "schema_version": 1,
        "artifact_type": "m3_inactive_plan",
        "contract": contract,
        "contract_file": _file_record(contract_path),
        **common,
        "numerical": numerical,
        "kernels": kernels,
        "workers": workers,
        "execution_order": rows,
    }
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan)
    return plan


def _validate_identity(plan, *, diff_paths=DIFF_PATHS):
    """Validate shared snapshot, environment and file-record structure offline."""
    _validate_dependencies(plan["dependencies"])
    equal(
        plan["dependencies"]["official"]["dependencies"]["transformers"],
        "4.55.0",
        "prepared transformers",
    )
    equal(
        plan["dependencies"]["official"]["optional_kernels_present"],
        False,
        "optional kernels absence",
    )
    harness, differences = source_controls(plan["implementations"], diff_paths=diff_paths)
    equal(
        [plan["harness"], plan["production_differences"]],
        [harness, differences],
        "frozen source controls",
    )
    _file_records(plan["model_files"], "model files")
    _file_records([plan["contract_file"]], "contract file")
    _keys(plan["inputs"], INPUTS, name="inputs")
    for value in plan["inputs"].values():
        _keys(value, ("file", "contents"), name="input")
        _file_records([value["file"]], "input files")
    validate_model_config(plan["model_config"])
    _keys(plan["runtime_environment"], RUNTIME_VARIABLES, name="runtime environment")
    require(
        all(
            value is None or isinstance(value, str)
            for value in plan["runtime_environment"].values()
        ),
        "runtime variables require strings or null",
    )
    for key in ("interpreter", "model_path"):
        require(isinstance(plan[key], str) and Path(plan[key]).is_absolute(), "absolute " + key)


def validate_plan(plan):
    from .m3_inactive_kernels import build_kernel_plan

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
            "kernels",
            "workers",
            "execution_order",
            "interpreter",
            "runtime_environment",
            "plan_sha256",
        ),
        name="inactive plan",
    )
    equal([plan["schema_version"], plan["artifact_type"]], [1, "m3_inactive_plan"], "plan version")
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan selfhash",
    )
    validate_contract(plan["contract"], resolved=True)
    _validate_identity(plan)
    validate_model_plan(plan["numerical"])
    equal(plan["numerical"]["model_config"], plan["model_config"], "model configuration")
    equal(
        plan["numerical"]["source_inputs"],
        {key: value["contents"] for key, value in plan["inputs"].items()},
        "embedded original inputs",
    )
    equal(plan["kernels"], build_kernel_plan(), "exact kernel layout/evaluation plan")
    workers, rows = execution_rows(plan["numerical"], plan["kernels"])
    equal(
        [plan["workers"], plan["execution_order"]], [workers, rows], "exact worker execution order"
    )
    resources = plan["numerical"]["resource_estimates"]
    estimate = (
        resources["retained_tensor_bytes_upper_bound"]
        + resources["retained_index_records_upper_bound"] * 1024
        + resources["comparison_records_upper_bound"] * 2048
        + plan["numerical"]["contract"]["limits"]["persisted_dump_bytes"]
        + LIMITS["kernel_evidence_bytes"]
        + 1024**3
    )
    require(
        estimate <= LIMITS["artifact_bytes_max"], "estimated complete artifacts exceed disk cap"
    )


def verify_plan(plan, *, implementation_id=None):
    validate_plan(plan)
    _verify_inputs(plan, implementation_id=implementation_id)


def _verify_inputs(plan, *, implementation_id=None):
    """Verify already-validated correctness plans against actual source and controls."""
    for key in (implementation_id,) if implementation_id else ("A", "B"):
        expected = plan["implementations"][key]
        actual = source_probe() if implementation_id else probe_checkout(expected["root"])
        equal(actual["source"], expected["source"], "actual frozen source")
        equal(actual["imports"], expected["imports"], "actual import locations")
        equal(actual["dependencies"], plan["dependencies"], "actual dependency environment")
    equal(
        _model_files(Path(plan["model_path"])), plan["model_files"], "actual checkpoint file bytes"
    )
    equal(
        OuroConfig.from_dict(read_json(Path(plan["model_path"]) / "config.json")).to_dict(),
        plan["model_config"],
        "embedded checkpoint configuration",
    )
    equal(
        _file_record(Path(plan["contract_file"]["path"])), plan["contract_file"], "contract bytes"
    )
    contract = read_json(Path(plan["contract_file"]["path"]))
    contract["controls"].update(
        gpu_ids=plan["contract"]["controls"]["gpu_ids"],
        affinity=plan["contract"]["controls"]["affinity"],
    )
    equal(contract, plan["contract"], "resolved contract")
    for name, value in plan["inputs"].items():
        equal(_file_record(Path(value["file"]["path"])), value["file"], "input bytes " + name)
        equal(read_json(Path(value["file"]["path"])), value["contents"], "embedded input " + name)
    equal(sys.executable, plan["interpreter"], "prepared interpreter")
    equal(
        {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
        plan["runtime_environment"],
        "runtime environment",
    )
    equal(affinity_snapshot(), plan["contract"]["controls"]["affinity"], "active CPU/NUMA affinity")


def loading_view(plan):
    """Private shared model-loader view; no benchmark plan identity is claimed."""
    config = plan["model_config"]
    cache = plan["numerical"]["contract"]["engine"]["cache"]
    pool_bytes = (
        2
        * config["num_hidden_layers"]
        * cache["num_blocks"]
        * cache["block_size"]
        * config["num_key_value_heads"]
        * config["head_dim"]
        * 4
    )
    contract = deepcopy(plan["inputs"]["contract"]["contents"])
    contract["controls"].update(plan["contract"]["controls"])
    return {
        "model_path": plan["model_path"],
        "model_config": config,
        "contract": contract,
        "workload_stats": {"model": {"pool_bytes": pool_bytes}},
    }
