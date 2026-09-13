"""CPU-only freeze for the single 93-execution eager/replay experiment."""

import os
import sys
from pathlib import Path

from benchmarks.capture.validation import GRAPH_LIMITS, build_model_plan
from vllm_lt.benchmarks import ab_schema as ab
from vllm_lt.benchmarks.ab_schema import (
    INPUTS,
    RUNTIME_VARIABLES,
    WORKERS,
    affinity_snapshot,
    equal,
    require,
    validate_affinity,
    validate_model_config,
)
from vllm_lt.benchmarks.ab_schema import (
    execution_view as execution_view,
)
from vllm_lt.benchmarks.schema import (
    _digest,
    _file_record,
    _integer,
    _keys,
    _model_files,
    _text,
    _validate_contract,
    _validate_suite,
    _workload_stats,
    read_json,
)
from vllm_lt.models.config import OuroConfig
from vllm_lt.validation.schema import _file_records, _validate_dependencies
from vllm_lt.worker.decode_buffers import DecodeBucketLayout
from vllm_lt.worker.graph_diagnostics import SETUP_WARMUPS

IMPORT_MODULES = (
    *ab.IMPORT_MODULES,
    "benchmarks.capture.schema",
    "benchmarks.capture.runner",
    "benchmarks.capture.report",
    "benchmarks.capture.runtime",
    "benchmarks.capture.validation",
    "vllm_lt.worker.recurrent_graph",
)
DEFAULT_CONTRACT = read_json(Path(__file__).parent / "fixtures/ouro-m3-capture-contract.json")
CONTROLS = {
    k: v for k, v in DEFAULT_CONTRACT["controls"].items() if k not in ("gpu_ids", "affinity")
}
LIMITS = DEFAULT_CONTRACT["limits"]
PAIR_PREFIX = "M3-capture"


def validate_contract(contract, *, resolved=False):
    _keys(contract, DEFAULT_CONTRACT, name="capture contract")
    equal(contract["graph_limits"], GRAPH_LIMITS, "runtime graph limit defaults")
    equal(
        [contract["schema_version"], contract["artifact_type"]],
        [2, "m3_capture_contract"],
        "version",
    )
    for name in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[name], name)
    equal(contract["inputs"], INPUTS, "input paths")
    for name in (
        "implementation_options",
        "acceptance",
        "limits",
        "graph_limits",
        "stop_conditions",
    ):
        equal(contract[name], DEFAULT_CONTRACT[name], "capture " + name)
    controls = contract["controls"]
    _keys(controls, (*CONTROLS, "gpu_ids", "affinity"), name="capture controls")
    equal({key: controls[key] for key in CONTROLS}, CONTROLS, "fixed controls")
    if resolved or controls["gpu_ids"] is not None:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one explicitly selected physical GPU required",
        )
        _integer(controls["gpu_ids"][0], "physical GPU ID")
    if resolved or controls["affinity"] is not None:
        validate_affinity(controls["affinity"])


def source_probe():
    return ab.source_probe(import_modules=IMPORT_MODULES)


def probe_checkout(root):
    return ab.probe_checkout(root, module="benchmarks.capture")


def _source_controls(implementations):
    _keys(implementations, ("A", "B"), name="implementations")
    for item in implementations.values():
        _keys(item, ("root", "source", "imports"), name="implementation")
        source = item["source"]
        _keys(source, ("root", "commit", "status", "files"), name="source")
        require(Path(item["root"]).is_absolute(), "absolute implementation root required")
        equal(item["root"], source["root"], "source root")
        require(source["status"] == "", "execution source must be clean")
        require(
            isinstance(source["commit"], str)
            and len(source["commit"]) == 40
            and all(c in "0123456789abcdef" for c in source["commit"]),
            "exact source SHA",
        )
        _file_records(source["files"], "source files")
        paths = [row["path"] for row in source["files"]]
        equal(paths, sorted(paths), "sorted source manifest")
        for path in paths:
            require(
                not Path(path).is_absolute() and ".." not in Path(path).parts,
                "source paths must remain inside checkout",
            )
        _keys(item["imports"], IMPORT_MODULES, name="source imports")
        for name, path in item["imports"].items():
            equal(path, name.replace(".", "/") + ".py", "module import origin")
            require(path in paths, "imported module absent from frozen source")
    a, b = (implementations[side]["source"] for side in ("A", "B"))
    equal(a["commit"], b["commit"], "A/B identical commit")
    equal(a["files"], b["files"], "A/B all source bytes identical")
    common_paths = {row["path"] for row in ab._harness(a)["files"]}
    files = [
        row
        for row in a["files"]
        if row["path"] in common_paths
        or row["path"].startswith("benchmarks/capture/")
        or row["path"] == "benchmarks/__init__.py"
    ]
    return {"files": files, "sha256": _digest(files)}


def execution_rows(suite, contract, stats, numerical):
    workers, rows = ab.execution_rows(suite, contract, stats, numerical, pair_prefix=PAIR_PREFIX)
    cases = {c["case_id"]: c for c in numerical["execution_order"]}
    for row in rows:
        row["use_graphs"] = row["implementation_id"] == "B"
        if row["kind"] == "numerical":
            row.update(kind="model", phase=cases[row["execution_id"]]["phase"])
    # The graph prerequisite precedes benchmark feasibility in each numerical worker.
    rows.sort(key=lambda r: (WORKERS.index(r["worker_id"]), r["kind"] == "benchmark"))
    for worker in workers:
        worker["use_graphs"] = worker["implementation_id"] == "B"
        own = [r for r in rows if r["worker_id"] == worker["worker_id"]]
        worker["execution_ids"] = [r["execution_id"] for r in own]
        if worker["worker_id"].startswith("N-"):
            equal(own[0]["phase"], "feasibility", "first excluded model feasibility")
    equal(len(rows), LIMITS["executions"], "93 frozen executions")
    return workers, rows


def resource_estimates(numerical, rows, scheduler):
    r = numerical["resource_estimates"]
    graph_models = sum(
        c["implementation_id"] == "B" and c["storage_strategy"] == "graph_replay"
        for c in numerical["execution_order"]
    )
    graph_cases = graph_models + sum(
        row["implementation_id"] == "B" and row["kind"] == "benchmark" for row in rows
    )
    layout = DecodeBucketLayout(max_num_seqs=scheduler["max_num_seqs"])
    payload, staging = layout.payload_bytes(numerical["model_config"]["hidden_size"])
    require(
        payload <= GRAPH_LIMITS["common_payload_bytes"]
        and staging <= GRAPH_LIMITS["cpu_staging_bytes"],
        "scheduler bucket storage exceeds caps",
    )
    captures = graph_models * 2 + (graph_cases - graph_models) * len(layout.row_counts)
    pointer = r["pointer_evidence_bytes_upper_bound"]
    require(pointer <= LIMITS["pointer_evidence_bytes_max"], "projected pointer evidence cap")
    total = (
        r["retained_tensor_bytes_upper_bound"]
        + r["retained_index_records_upper_bound"] * 1024
        + r["comparison_records_upper_bound"] * 2048
        + LIMITS["pointer_evidence_bytes_max"]
        + numerical["contract"]["limits"]["persisted_dump_bytes"]
        + LIMITS["profile_total_bytes_max"]
        + 1024**3
    )
    require(total <= LIMITS["artifact_bytes_max"], "estimated artifacts exceed total cap")
    return {
        "artifact_bytes_upper_bound": total,
        "graph_owning_B_executions": graph_cases,
        "capture_recordings": captures,
        "warmup_device_traversals": captures * SETUP_WARMUPS,
        "verification_device_traversals": captures,
        "total_device_scratch_traversals": captures * (SETUP_WARMUPS + 1),
        "capture_semantics": "recording-does-not-execute-captured-device-kernels",
    }


def make_plan(*, baseline_root, candidate_root, contract_path, model_path, gpu_ids, affinity):
    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    contract["controls"].update(gpu_ids=gpu_ids, affinity=affinity)
    validate_contract(contract, resolved=True)
    roots = {"A": Path(baseline_root).resolve(), "B": Path(candidate_root).resolve()}
    probes = {side: probe_checkout(root) for side, root in roots.items()}
    equal(probes["A"]["dependencies"], probes["B"]["dependencies"], "A/B dependencies")
    implementations = {
        side: {"root": str(roots[side]), "source": probe["source"], "imports": probe["imports"]}
        for side, probe in probes.items()
    }
    inputs = {
        key: {"file": _file_record(roots["A"] / path), "contents": read_json(roots["A"] / path)}
        for key, path in INPUTS.items()
    }
    for key, path in INPUTS.items():
        equal(
            _file_record(roots["B"] / path, relative_to=roots["B"]),
            _file_record(roots["A"] / path, relative_to=roots["A"]),
            "A/B input " + key,
        )
    config = OuroConfig.from_dict(read_json(model_path / "config.json"))
    plan = {
        "contract": contract,
        "contract_file": _file_record(contract_path),
        "implementations": implementations,
        "dependencies": probes["A"]["dependencies"],
        "inputs": inputs,
        "model_path": str(model_path),
        "model_config": config.to_dict(),
        "model_files": _model_files(model_path),
        "interpreter": sys.executable,
        "runtime_environment": {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
    }
    plan.update(_derived_plan(plan))
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan)
    return plan


def _derived_plan(plan):
    """Build and validate the same derived fields from frozen inputs."""
    inputs, config = plan["inputs"], OuroConfig.from_dict(plan["model_config"])
    validate_model_config(plan["model_config"])
    suite = inputs["benchmark_suite"]["contents"]
    benchmark = inputs["benchmark_contract"]["contents"]
    _validate_suite(suite)
    _validate_contract(benchmark)
    numerical = build_model_plan(
        inputs["numerical_suite"]["contents"],
        inputs["numerical_contract"]["contents"],
        config.to_dict(),
    )
    stats = _workload_stats(suite, config, benchmark)
    workers, rows = execution_rows(suite, plan["contract"], stats, numerical)
    return dict(
        schema_version=2,
        artifact_type="m3_capture_plan",
        production_differences=[],
        harness=_source_controls(plan["implementations"]),
        suite=suite,
        benchmark_contract=benchmark,
        numerical=numerical,
        graph_limits=dict(GRAPH_LIMITS),
        workload_stats=stats,
        workers=workers,
        execution_order=rows,
        resource_estimates=resource_estimates(numerical, rows, benchmark["engine"]["scheduler"]),
    )


def validate_plan(plan):
    expected = _derived_plan(plan)
    _keys(
        plan,
        (
            *expected,
            "contract",
            "contract_file",
            "implementations",
            "dependencies",
            "inputs",
            "model_path",
            "model_config",
            "model_files",
            "interpreter",
            "runtime_environment",
            "plan_sha256",
        ),
        name="capture plan",
    )
    for key, value in expected.items():
        equal(plan[key], value, "derived " + key)
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan content hash",
    )
    validate_contract(plan["contract"], resolved=True)
    _validate_dependencies(plan["dependencies"])
    official = plan["dependencies"]["official"]
    equal(official["dependencies"]["transformers"], "4.55.0", "prepared Q1 transformers")
    equal(official["optional_kernels_present"], False, "prepared Q1 optional kernels absence")
    _file_records(plan["model_files"], "model files")
    _file_records([plan["contract_file"]], "capture contract file")
    require(Path(plan["contract_file"]["path"]).is_absolute(), "absolute contract path required")
    for row in plan["model_files"]:
        require(
            Path(row["path"]).name == row["path"], "checkpoint file must remain inside model root"
        )
    _keys(plan["inputs"], INPUTS, name="input references")
    for name, item in plan["inputs"].items():
        _keys(item, ("file", "contents"), name="input")
        _file_records([item["file"]], "input file")
        equal(
            item["file"]["path"],
            str(Path(plan["implementations"]["A"]["root"]) / INPUTS[name]),
            "input belongs to frozen A source",
        )
        source_record = next(
            (
                r
                for r in plan["implementations"]["A"]["source"]["files"]
                if r["path"] == INPUTS[name]
            ),
            None,
        )
        require(source_record is not None, "input missing from source manifest")
        equal({**item["file"], "path": INPUTS[name]}, source_record, "input frozen source bytes")
    tokenizer = next((r for r in plan["model_files"] if r["path"] == "tokenizer.json"), None)
    require(tokenizer is not None, "checkpoint tokenizer required")
    for suite in (plan["suite"], plan["inputs"]["numerical_suite"]["contents"]):
        expected = suite["provenance"]["tokenizer"]
        equal(
            [tokenizer[k] for k in ("sha256", "size_bytes")],
            [expected[k] for k in ("sha256", "size_bytes")],
            "tokenizer provenance",
        )
    _keys(plan["runtime_environment"], RUNTIME_VARIABLES, name="runtime environment")
    for value in plan["runtime_environment"].values():
        require(
            value is None or isinstance(value, str), "runtime variables must be strings or null"
        )
    for field in ("interpreter", "model_path"):
        require(
            isinstance(plan[field], str) and Path(plan[field]).is_absolute(), "absolute " + field
        )


def verify_plan(plan, *, implementation_id=None):
    validate_plan(plan)
    require(implementation_id in (None, "A", "B"), "unknown implementation")
    for side in (implementation_id,) if implementation_id else ("A", "B"):
        expected = plan["implementations"][side]
        actual = source_probe() if implementation_id else probe_checkout(expected["root"])
        equal(actual["source"], expected["source"], "frozen source")
        equal(actual["imports"], expected["imports"], "frozen imports")
        equal(actual["dependencies"], plan["dependencies"], "frozen dependencies")
    ab.verify_inputs(plan, affinity=affinity_snapshot())
