"""The single frozen M2 experiment: eight processes, seven cells, 105 executions."""

import json
import os
import subprocess
import sys
from pathlib import Path

from vllm_lt.models.config import OuroConfig
from vllm_lt.validation.schema import _file_records, _validate_dependencies, dependency_manifest

from .schema import (
    _constants,
    _digest,
    _file_record,
    _integer,
    _keys,
    _model_files,
    _source_manifest,
    _text,
    _validate_contract,
    _validate_suite,
    _workload_stats,
    read_json,
)

CELLS = (
    "W1-refill",
    "W2-refill",
    "W3-refill",
    "W4-refill",
    "W4-no_refill",
    "W5-refill",
    "W5-no_refill",
)
WORKERS = ("N-A", "N-B", "A1", "B1", "B2", "A2", "P-A", "P-B")
DIFF_PATHS = ("vllm_lt/core/kv_cache_manager.py", "vllm_lt/models/ouro.py")
IMPORT_MODULES = (
    "vllm_lt.benchmarks.ab_schema",
    "vllm_lt.benchmarks.runner",
    "vllm_lt.core.kv_cache_manager",
    "vllm_lt.models.ouro",
    "vllm_lt.validation.runner",
)
LIMITS = {
    "total_timeout_s": 7200,
    "case_timeout_s": 600,
    "workers": 8,
    "model_loads": 8,
    "executions": 105,
    "numerical_executions": 27,
    "feasibility_executions": 14,
    "warmup_executions": 32,
    "measured_executions": 28,
    "profile_executions": 4,
    "profile_decode_outputs": 16,
    "profile_trace_bytes_max": 2 * 1024**3,
    "profile_total_bytes_max": 4 * 1024**3,
    "artifact_bytes_max": 12 * 1024**3,
}
ACCEPTANCE = {
    "target_cell": "W1-refill",
    "target_ratio_min": 1.1,
    "control_ratio_min": 0.95,
    "peak_increase_bytes_max": 64 * 1024**2,
    "setup_increase_ns_max": 100_000_000,
    "variation": "min-B-strictly-greater-than-max-A",
    "numerical": "all-required-reference-and-base-candidate-exact-gates",
}
CONTROL_CONSTANTS = {
    "cpu_threads": 1,
    "interop_threads": 1,
    "seed": 0,
    "numa_policy": "inherit-verified-parent-binding",
    "residency_policy": "one-fresh-worker-one-model-one-engine",
    "cache_policy": "fresh-engine-preserve-worker-allocator",
    "environment_policy": "same-prepared-environment-all-workers",
}
INPUTS = {
    "benchmark_suite": "benchmarks/fixtures/ouro-m1.json",
    "benchmark_contract": "benchmarks/fixtures/ouro-m1-contract.json",
    "numerical_suite": "benchmarks/fixtures/ouro-q1.json",
    "numerical_contract": "benchmarks/fixtures/ouro-q1-contract.json",
}
RUNTIME_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUDA_MODULE_LOADING",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "HF_HOME",
    "HF_MODULES_CACHE",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, name):
    require(_digest(actual) == _digest(expected), f"{name} differs from the frozen contract")


def affinity_snapshot():
    shown = subprocess.run(["numactl", "--show"], capture_output=True, text=True, check=True)
    require(not shown.stderr.strip(), "active NUMA policy query was unsuccessful")
    policy = {}
    for line in shown.stdout.splitlines():
        key, value = line.split(":", 1)
        policy[key.strip().replace(" ", "_")] = " ".join(value.split())
    return {
        "numactl_show": policy,
        "cpu_ids": sorted(os.sched_getaffinity(0)),
        "numa_status": [
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:"))
        ],
    }


def validate_affinity(value):
    _keys(value, ("cpu_ids", "numa_status", "numactl_show"), name="affinity")
    require(isinstance(value["cpu_ids"], list) and bool(value["cpu_ids"]), "CPU IDs required")
    for item in value["cpu_ids"]:
        _integer(item, "CPU ID")
    equal(value["cpu_ids"], sorted(set(value["cpu_ids"])), "sorted unique CPU IDs")
    require(
        isinstance(value["numa_status"], list) and len(value["numa_status"]) == 2,
        "CPU and memory allowed masks are required",
    )
    _keys(
        value["numactl_show"],
        ("policy", "preferred_node", "physcpubind", "cpubind", "nodebind", "membind"),
        name="active NUMA policy",
    )
    require(value["numactl_show"]["policy"] == "bind", "explicit parent memory binding required")
    for key, item in value["numactl_show"].items():
        _text(item, "NUMA " + key)
        require(item == " ".join(item.split()), "NUMA policy must have normalized whitespace")
    require(
        value["numactl_show"]["physcpubind"] == " ".join(map(str, value["cpu_ids"])),
        "active NUMA CPU mask differs from CPU affinity",
    )
    for line, prefix in zip(value["numa_status"], ("Cpus_allowed_list:", "Mems_allowed_list:")):
        require(isinstance(line, str) and line.startswith(prefix), "invalid affinity record")


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
        name="M2 contract",
    )
    equal([contract["schema_version"], contract["artifact_type"]], [1, "m2_ab_contract"], "version")
    for name in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[name], name)
    _constants(contract["inputs"], INPUTS, "M2 input references")
    equal(contract["production_diff_paths"], list(DIFF_PATHS), "allowed production differences")
    _constants(contract["limits"], LIMITS, "M2 execution/resource budgets")
    _constants(contract["acceptance"], ACCEPTANCE, "M2 acceptance gates")
    controls = contract["controls"]
    _keys(controls, (*CONTROL_CONSTANTS, "gpu_ids", "affinity"), name="M2 controls")
    _constants({key: controls[key] for key in CONTROL_CONSTANTS}, CONTROL_CONSTANTS, "controls")
    if resolved or controls["gpu_ids"] is not None:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one explicitly selected physical GPU ID required",
        )
        _integer(controls["gpu_ids"][0], "physical GPU ID")
    if resolved or controls["affinity"] is not None:
        validate_affinity(controls["affinity"])
    require(
        isinstance(contract["stop_conditions"], list) and bool(contract["stop_conditions"]),
        "stop conditions are required",
    )
    for item in contract["stop_conditions"]:
        _text(item, "stop condition")


def source_probe(*, import_modules=IMPORT_MODULES):
    """Called in the selected checkout; imports must originate there, without CUDA."""
    import importlib

    source = _source_manifest()
    root = Path(source["root"]).resolve()
    imports = {}
    for name in import_modules:
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to(root), f"import outside selected implementation: {name}")
        imports[name] = str(path.relative_to(root))
    return {"source": source, "imports": imports, "dependencies": dependency_manifest()}


def probe_checkout(root, *, module="vllm_lt.benchmarks.ab"):
    root = Path(root).resolve()
    env = dict(os.environ, PYTHONPATH=str(root))
    raw = subprocess.check_output(
        [sys.executable, "-m", module, "source-probe"],
        cwd=root,
        env=env,
        text=True,
    )
    result = read_json_string(raw)
    require(Path(result["source"]["root"]).resolve() == root, "source probe used another checkout")
    return result


def read_json_string(raw):
    from .schema import _finite_float, _object_pairs, _reject_constant

    return json.loads(
        raw,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_object_pairs,
    )


def _harness(source):
    prefixes = ("vllm_lt/benchmarks/", "vllm_lt/validation/", "benchmarks/fixtures/")
    files = [
        row
        for row in source["files"]
        if row["path"].startswith(prefixes) or row["path"] == "vllm_lt/models/serial_oracle.py"
    ]
    require(bool(files), "common harness is missing")
    return {"files": files, "sha256": _digest(files)}


def _source_controls(implementations):
    _keys(implementations, ("A", "B"), name="implementations")
    for item in implementations.values():
        _keys(item, ("root", "source", "imports"), name="implementation")
        _keys(item["source"], ("root", "commit", "status", "files"), name="source")
        require(Path(item["root"]).is_absolute(), "implementation root must be absolute")
        equal(item["root"], item["source"]["root"], "source root")
        _file_records(item["source"]["files"], "source files")
        for row in item["source"]["files"]:
            require(
                not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts,
                "source paths must remain inside the checkout",
            )
        _keys(item["imports"], IMPORT_MODULES, name="import provenance")
        for name, path in item["imports"].items():
            equal(path, name.replace(".", "/") + ".py", "module import location")
    a, b = (implementations[key]["source"] for key in ("A", "B"))
    for source in (a, b):
        require(source["status"] == "", "execution source must be clean before freezing")
        require(
            isinstance(source["commit"], str)
            and len(source["commit"]) == 40
            and all(char in "0123456789abcdef" for char in source["commit"]),
            "source needs an exact commit SHA",
        )
    harness = _harness(a)
    equal(_harness(b), harness, "common harness bytes")
    amap, bmap = ({row["path"]: row for row in source["files"]} for source in (a, b))
    differences = sorted(path for path in set(amap) | set(bmap) if amap.get(path) != bmap.get(path))
    require(
        bool(differences) and set(differences) <= set(DIFF_PATHS),
        f"source differences are not confined to selected metadata change: {differences}",
    )
    return harness, differences


def execution_rows(suite, contract, stats, numerical, *, pair_prefix="M2"):
    workloads = {row["workload_id"]: row for row in suite["workloads"]}
    rows, workers = [], []
    shared_hash = _digest(contract)
    for worker_id in WORKERS:
        implementation = "A" if "A" in worker_id else "B"
        worker = {"worker_id": worker_id, "implementation_id": implementation, "execution_ids": []}

        def add(cell, phase, repetition=1):
            workload_id, mode = cell.split("-", 1)
            run_id = f"{worker_id}-{phase}-{cell}"
            row = {
                "execution_id": run_id,
                "kind": "benchmark",
                "run_id": run_id,
                "worker_id": worker_id,
                "implementation_id": implementation,
                "cell_id": cell,
                "workload_id": workload_id,
                "mode": mode,
                "phase": phase,
                "repetition": repetition,
                "pair_id": f"{pair_prefix}-{cell}-{repetition}" if phase == "measured" else None,
                "instrumentation": {"feasibility": "validation", "profile": "profile"}.get(
                    phase, "timing"
                ),
                "case_lifetime_timeout_s": 600,
                "max_steps": stats[workload_id]["max_steps"],
                "max_events": stats[workload_id]["max_events"],
                "workload_sha256": _digest(workloads[workload_id]),
                "controls_sha256": shared_hash,
            }
            rows.append(row)
            worker["execution_ids"].append(run_id)

        if worker_id.startswith("N-"):
            for cell in CELLS:
                add(cell, "feasibility")
            for case in numerical["execution_order"]:
                if case["implementation_id"] == implementation:
                    row = {
                        "execution_id": case["case_id"],
                        "kind": "numerical",
                        "worker_id": worker_id,
                        "implementation_id": implementation,
                    }
                    rows.append(row)
                    worker["execution_ids"].append(row["execution_id"])
        elif worker_id.startswith("P-"):
            for phase in ("warmup", "profile"):
                for cell in ("W1-refill", "W4-refill"):
                    add(cell, phase)
        else:
            for phase in ("warmup", "measured"):
                for cell in CELLS:
                    add(cell, phase, int(worker_id[-1]))
        workers.append(worker)
    expected = len(numerical["execution_order"]) + 78
    require(
        len(rows) == len({row["execution_id"] for row in rows}) == expected,
        "unique numerical cases and 78 benchmark executions required",
    )
    return workers, rows


def validate_model_config(config):
    expected = OuroConfig().to_dict()
    for name in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "total_ut_steps",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
    ):
        equal(config[name], expected[name], "pinned Ouro " + name)


def make_ab_plan(*, baseline_root, candidate_root, contract_path, model_path, gpu_ids, affinity):
    from vllm_lt.validation.m2 import build_numerical_plan

    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    contract["controls"].update(gpu_ids=gpu_ids, affinity=affinity)
    validate_contract(contract, resolved=True)
    roots = {"A": Path(baseline_root).resolve(), "B": Path(candidate_root).resolve()}
    require(roots["A"] != roots["B"], "A and B need distinct immutable execution checkouts")
    probes = {key: probe_checkout(root) for key, root in roots.items()}
    equal(probes["A"]["dependencies"], probes["B"]["dependencies"], "A/B dependencies")
    implementations = {
        key: {
            "root": str(roots[key]),
            "source": probes[key]["source"],
            "imports": probes[key]["imports"],
        }
        for key in roots
    }
    harness, differences = _source_controls(implementations)
    inputs = {}
    for key, name in contract["inputs"].items():
        record = _file_record(roots["A"] / name)
        other = _file_record(roots["B"] / name)
        equal(
            [record["sha256"], record["size_bytes"]],
            [other["sha256"], other["size_bytes"]],
            f"A/B {key}",
        )
        inputs[key] = {"file": record, "contents": read_json(roots["A"] / name)}
    suite, benchmark = (
        inputs["benchmark_suite"]["contents"],
        inputs["benchmark_contract"]["contents"],
    )
    _validate_suite(suite)
    _validate_contract(benchmark)
    config = OuroConfig.from_dict(read_json(model_path / "config.json"))
    validate_model_config(config.to_dict())
    files = _model_files(model_path)
    tokenizer = next(row for row in files if row["path"] == "tokenizer.json")
    for name in ("sha256", "size_bytes"):
        equal(tokenizer[name], suite["provenance"]["tokenizer"][name], "tokenizer provenance")
    numerical = build_numerical_plan(
        inputs["numerical_suite"]["contents"],
        inputs["numerical_contract"]["contents"],
        config.to_dict(),
    )
    stats = _workload_stats(suite, config, benchmark)
    workers, rows = execution_rows(suite, contract, stats, numerical)
    plan = {
        "schema_version": 1,
        "artifact_type": "m2_ab_plan",
        "contract": contract,
        "contract_file": _file_record(contract_path),
        "implementations": implementations,
        "harness": harness,
        "production_differences": differences,
        "dependencies": probes["A"]["dependencies"],
        "inputs": inputs,
        "model_path": str(model_path),
        "model_config": config.to_dict(),
        "model_files": files,
        "suite": suite,
        "benchmark_contract": benchmark,
        "workload_stats": stats,
        "numerical": numerical,
        "workers": workers,
        "execution_order": rows,
        "interpreter": sys.executable,
        "runtime_environment": {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
    }
    plan["plan_sha256"] = _digest(plan)
    validate_ab_plan(plan)
    return plan


def validate_ab_plan(plan):
    from vllm_lt.validation.m2 import validate_numerical_plan

    _keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "contract",
            "contract_file",
            "implementations",
            "harness",
            "production_differences",
            "dependencies",
            "inputs",
            "model_path",
            "model_config",
            "model_files",
            "suite",
            "benchmark_contract",
            "workload_stats",
            "numerical",
            "workers",
            "execution_order",
            "interpreter",
            "runtime_environment",
            "plan_sha256",
        ),
        name="M2 plan",
    )
    equal([plan["schema_version"], plan["artifact_type"]], [1, "m2_ab_plan"], "M2 plan version")
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "M2 plan content hash",
    )
    validate_contract(plan["contract"], resolved=True)
    _validate_dependencies(plan["dependencies"])
    official = plan["dependencies"]["official"]
    equal(official["dependencies"]["transformers"], "4.55.0", "prepared Q1 transformers")
    equal(official["optional_kernels_present"], False, "prepared Q1 optional kernels absence")
    _file_records(plan["model_files"], "model files")
    _file_records([plan["contract_file"]], "M2 contract file")
    _keys(plan["inputs"], INPUTS, name="input references")
    for item in plan["inputs"].values():
        _keys(item, ("file", "contents"), name="input")
        _file_records([item["file"]], "input file")
    _validate_suite(plan["suite"])
    _validate_contract(plan["benchmark_contract"])
    validate_numerical_plan(plan["numerical"])
    harness, differences = _source_controls(plan["implementations"])
    equal([plan["harness"], plan["production_differences"]], [harness, differences], "source diff")
    equal(plan["suite"], plan["inputs"]["benchmark_suite"]["contents"], "suite inputs")
    equal(
        plan["benchmark_contract"],
        plan["inputs"]["benchmark_contract"]["contents"],
        "contract inputs",
    )
    validate_model_config(plan["model_config"])
    equal(
        plan["numerical"]["source_inputs"],
        {
            "suite": plan["inputs"]["numerical_suite"]["contents"],
            "contract": plan["inputs"]["numerical_contract"]["contents"],
        },
        "numerical original inputs",
    )
    equal(plan["numerical"]["model_config"], plan["model_config"], "numerical model configuration")
    config = OuroConfig.from_dict(plan["model_config"])
    stats = _workload_stats(plan["suite"], config, plan["benchmark_contract"])
    equal(plan["workload_stats"], stats, "derived workload limits")
    workers, rows = execution_rows(plan["suite"], plan["contract"], stats, plan["numerical"])
    equal([plan["workers"], plan["execution_order"]], [workers, rows], "resolved execution order")
    resources = plan["numerical"]["resource_estimates"]
    numerical_limits = plan["numerical"]["contract"]["limits"]
    estimate = (
        resources["retained_tensor_bytes_upper_bound"]
        + numerical_limits["persisted_dump_bytes"]
        + resources["retained_index_records_upper_bound"] * 1024
        + resources["comparison_records_upper_bound"] * 2048
        + LIMITS["profile_total_bytes_max"]
        + 1024**3
    )
    require(
        estimate <= LIMITS["artifact_bytes_max"], "planned numerical/profile artifacts exceed cap"
    )
    _keys(plan["runtime_environment"], RUNTIME_VARIABLES, name="runtime environment")
    for value in plan["runtime_environment"].values():
        require(
            value is None or isinstance(value, str), "runtime variables must be strings or null"
        )
    for field in ("interpreter", "model_path"):
        require(
            isinstance(plan[field], str) and Path(plan[field]).is_absolute(), f"absolute {field}"
        )


def verify_ab_plan(plan, *, implementation_id=None):
    validate_ab_plan(plan)
    selected = (implementation_id,) if implementation_id else ("A", "B")
    for key in selected:
        expected = plan["implementations"][key]
        actual = source_probe() if implementation_id else probe_checkout(expected["root"])
        equal(actual["source"], expected["source"], "frozen implementation source")
        equal(actual["imports"], expected["imports"], "frozen import paths")
        equal(actual["dependencies"], plan["dependencies"], "frozen dependencies")
    verify_inputs(plan, affinity=affinity_snapshot())


def verify_inputs(plan, *, affinity):
    """Verify checkpoint/input bytes and the prepared host environment without CUDA."""
    equal(_model_files(Path(plan["model_path"])), plan["model_files"], "frozen checkpoint files")
    equal(
        OuroConfig.from_dict(read_json(Path(plan["model_path"]) / "config.json")).to_dict(),
        plan["model_config"],
        "embedded checkpoint configuration",
    )
    equal(
        _file_record(Path(plan["contract_file"]["path"])),
        plan["contract_file"],
        "frozen contract",
    )
    original = read_json(Path(plan["contract_file"]["path"]))
    original["controls"].update(
        gpu_ids=plan["contract"]["controls"]["gpu_ids"],
        affinity=plan["contract"]["controls"]["affinity"],
    )
    equal(original, plan["contract"], "resolved contract contents")
    for name, record in plan["inputs"].items():
        equal(_file_record(Path(record["file"]["path"])), record["file"], "frozen input " + name)
        equal(read_json(Path(record["file"]["path"])), record["contents"], "embedded input " + name)
    equal(sys.executable, plan["interpreter"], "prepared interpreter")
    equal(
        {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
        plan["runtime_environment"],
        "runtime environment",
    )
    equal(affinity, plan["contract"]["controls"]["affinity"], "CPU/NUMA affinity")


def execution_view(plan):
    """Private M1 executor view; never serialized as a purported M1 plan."""
    return {
        "plan_sha256": plan["plan_sha256"],
        "contract": plan["benchmark_contract"],
        "suite": plan["suite"],
        "model_config": plan["model_config"],
        "model_path": plan["model_path"],
        "workload_stats": plan["workload_stats"],
    }
