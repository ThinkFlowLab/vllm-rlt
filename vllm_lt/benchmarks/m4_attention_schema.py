"""The frozen M4 tile32/tile64 experiment; all preparation is CPU-only."""

import ast
import hashlib
import importlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from vllm_lt.models.config import OuroConfig
from vllm_lt.validation.m3_inactive_schema import source_controls as _common_source_controls
from vllm_lt.validation.schema import _file_records, _validate_dependencies, dependency_manifest

from .ab_schema import (
    CELLS,
    INPUTS,
    RUNTIME_VARIABLES,
    WORKERS,
    affinity_snapshot,
    equal,
    read_json_string,
    require,
    validate_affinity,
    validate_model_config,
)
from .ab_schema import (
    IMPORT_MODULES as COMMON_IMPORT_MODULES,
)
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

DIFF_PATHS = ("vllm_lt/kernels/triton_attention.py",)
BASE_ATTENTION_SHA256 = "0861c57c25d7e5b284cfe4c259ae324a2a420a42c72fa3d3124d0fd13e63098e"
IMPORT_MODULES = (
    *COMMON_IMPORT_MODULES,
    *(
        "vllm_lt.benchmarks.m4_attention_schema",
        "vllm_lt.benchmarks.m4_attention",
        "vllm_lt.benchmarks.m4_attention_report",
        "vllm_lt.validation.m4_attention",
        "vllm_lt.validation.m4_attention_held",
        "vllm_lt.worker.model_runner",
        "vllm_lt.kernels.triton_attention",
    ),
)
TILE_POLICY = {"A": 32, "B": 64, "num_warps": 4, "baseline_sha256": BASE_ATTENTION_SHA256}
CONTROLS = {
    "cpu_threads": 1,
    "interop_threads": 1,
    "seed": 0,
    "numa_policy": "inherit-verified-parent-binding",
    "residency_policy": "one-fresh-worker-one-model-one-engine",
    "cache_policy": "fresh-engine-preserve-worker-allocator",
    "environment_policy": "same-prepared-environment-all-workers",
    "benchmark_path": "compact-synchronous-private-persistent-disabled",
    "source_policy": "accepted-attention-source-one-literal-tile-change-only",
}
ACCEPTANCE = {
    "target_cell": "W1-refill",
    "target_ratio_min": 1.01,
    "control_ratio_min": 0.95,
    "peak_increase_bytes_max": 64 * 1024**2,
    "setup_increase_ns_max": 100_000_000,
    "variation": "min-B-strictly-greater-than-max-A",
    "numerical": "Q1-FP32-final-logits-and-token-exit-history-Torch-exact-Triton-fidelity",
    "intermediates": "finite-required-hidden-gate-KV-deltas-diagnostic",
    "held_attention": {"atol": 2e-5, "rtol": 2e-5},
    "held_invariants": "same-tile-masked-compact-exact-whole-pool-guards-and-inactive-zeros",
    "profile": "matched-W1-W4-compact-prepared-real-kernel-copy-attribution",
    "verdict_precedence": "invalid-evidence-then-required-failure-then-missing-or-variation",
    "sanitizer": "unavailable-zero-executions-no-coverage-claim",
}
LIMITS = {
    "total_timeout_s": 7200,
    "case_timeout_s": 600,
    "workers": 8,
    "model_loads": 8,
    "executions": 139,
    "numerical_executions": 27,
    "comparison_streams": 51,
    "kernel_evaluations": 30,
    "lifecycle_evaluations": 4,
    "feasibility_executions": 14,
    "warmup_executions": 28,
    "measured_executions": 28,
    "diagnostic_warmup_executions": 4,
    "profile_executions": 4,
    "profile_decode_outputs": 16,
    "profile_trace_bytes_max": 2 * 1024**3,
    "profile_total_bytes_max": 4 * 1024**3,
    "held_evidence_bytes_max": 312 * 1024**2,
    "auxiliary_artifact_bytes_max": 1024**3,
    "artifact_bytes_max": 12 * 1024**3,
    "sanitizer_executions": 0,
}
STOP_CONDITIONS = [
    "Source, input, dependency, device, affinity, arithmetic, numerical, ownership or cleanup "
    "failure stops subsequent work.",
    "Seven feasibility cells, direct held/lifecycle and native/reference gates pass on A before "
    "B; both sides and their raw prerequisite audit pass before A1.",
    "Each complete case has a 600-second watchdog and the controller a 7200-second deadline, "
    "including preparation/export/cleanup; the earlier deadline wins.",
    "Any frozen tensor, dump, index, comparison, held, profile or total artifact cap ends the "
    "attempt and preserves its completed prefix.",
    "No retry, replacement execution, extra tile, profile or measured sample; failed pairs "
    "cannot be rescued by averages or additional runs.",
]


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
            "tile_policy",
            "controls",
            "acceptance",
            "limits",
            "stop_conditions",
        ),
        name="M4 contract",
    )
    equal(
        [contract["schema_version"], contract["artifact_type"]],
        [1, "m4_attention_contract"],
        "version",
    )
    for name in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[name], name)
    for key, expected in (
        ("inputs", INPUTS),
        ("tile_policy", TILE_POLICY),
        ("acceptance", ACCEPTANCE),
        ("limits", LIMITS),
    ):
        _constants(contract[key], expected, "M4 " + key)
    equal(contract["production_diff_paths"], list(DIFF_PATHS), "only attention source differs")
    equal(contract["stop_conditions"], STOP_CONDITIONS, "finite stop conditions")
    controls = contract["controls"]
    _keys(controls, (*CONTROLS, "gpu_ids", "affinity"), name="controls")
    _constants({k: controls[k] for k in CONTROLS}, CONTROLS, "common controls")
    if resolved or controls["gpu_ids"] is not None:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one resolved physical GPU required",
        )
        _integer(controls["gpu_ids"][0], "physical GPU ID")
    if resolved or controls["affinity"] is not None:
        validate_affinity(controls["affinity"])


def source_probe():
    source = _source_manifest()
    root = Path(source["root"]).resolve()
    imports = {}
    for name in IMPORT_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to(root), "import outside selected source: " + name)
        imports[name] = str(path.relative_to(root))
    return {"source": source, "imports": imports, "dependencies": dependency_manifest()}


def probe_checkout(root):
    root = Path(root).resolve()
    raw = subprocess.check_output(
        [sys.executable, "-m", "vllm_lt.benchmarks.m4_attention", "source-probe"],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        text=True,
    )
    result = read_json_string(raw)
    equal(result["source"]["root"], str(root), "selected source root")
    return result


def source_controls(implementations, texts):
    """Reuse generic manifests; prove the exact changed token from embedded bytes."""
    _keys(texts, ("A", "B"), name="attention source texts")
    _keys(implementations, ("A", "B"), name="implementations")
    common = deepcopy(implementations)
    for side, item in implementations.items():
        _keys(item["imports"], IMPORT_MODULES, name="import origins")
        for name, path in item["imports"].items():
            equal(path, name.replace(".", "/") + ".py", "actual import origin")
        common[side]["imports"] = {k: item["imports"][k] for k in COMMON_IMPORT_MODULES}
    harness, differences = _common_source_controls(common, diff_paths=DIFF_PATHS)
    equal(differences, list(DIFF_PATHS), "exact one-file difference")
    require(
        implementations["A"]["source"]["commit"] != implementations["B"]["source"]["commit"],
        "different production bytes need different immutable commits",
    )
    trees, spans = {}, {}
    for side in ("A", "B"):
        _text(texts[side], "attention source")
        raw = texts[side].encode()
        require(len(raw) <= 64 * 1024, "bounded attention source text")
        record = next(
            (r for r in implementations[side]["source"]["files"] if r["path"] == DIFF_PATHS[0]),
            None,
        )
        require(record is not None, "attention source file required on both sides")
        equal(
            [len(raw), hashlib.sha256(raw).hexdigest()],
            [record["size_bytes"], record["sha256"]],
            "attention text/file binding",
        )
        if side == "A":
            equal(record["sha256"], BASE_ATTENTION_SHA256, "accepted compact/padded baseline")
        try:
            tree = ast.parse(texts[side])
        except SyntaxError as error:
            raise ValueError("invalid attention source syntax") from error
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and any(k.arg == "BLOCK_T" for k in n.keywords)
        ]
        require(len(calls) == 1, "exactly one tile launch")
        call = calls[0]
        require(
            isinstance(call.func, ast.Subscript)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "_paged_attention_kernel",
            "tile belongs to kernel launch",
        )
        keywords = {k.arg: k.value for k in call.keywords}
        for name, expected in (("BLOCK_T", TILE_POLICY[side]), ("num_warps", 4)):
            node = keywords[name]
            require(
                isinstance(node, ast.Constant)
                and type(node.value) is int
                and node.value == expected,
                "literal launch " + name,
            )
        node = keywords["BLOCK_T"]
        lines = raw.splitlines(keepends=True)
        start = sum(map(len, lines[: node.lineno - 1])) + node.col_offset
        end = sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset
        spans[side] = (start, end)
        node.value = 0
        trees[side] = ast.dump(tree, include_attributes=False)
    start, end = spans["A"]
    equal(
        texts["B"].encode().hex(),
        (texts["A"].encode()[:start] + b"64" + texts["A"].encode()[end:]).hex(),
        "only original 32 literal changes to 64; comments/format/arithmetic stay identical",
    )
    equal(trees["A"], trees["B"], "identical AST after tile normalization")
    return harness, differences


def execution_rows(suite, contract, stats, numerical, held):
    workloads = {w["workload_id"]: w for w in suite["workloads"]}
    rows, workers = [], []
    for worker_id in WORKERS:
        side = "A" if "A" in worker_id else "B"
        worker = {"worker_id": worker_id, "implementation_id": side, "execution_ids": []}

        def add(cell, phase, repetition=1):
            wid, mode = cell.split("-", 1)
            run_id = f"M4-{worker_id}-{phase}-{cell}"
            rows.append(
                {
                    "execution_id": run_id,
                    "kind": "benchmark",
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "implementation_id": side,
                    "cell_id": cell,
                    "workload_id": wid,
                    "mode": mode,
                    "phase": phase,
                    "repetition": repetition,
                    "pair_id": f"M4-{cell}-{repetition}" if phase == "measured" else None,
                    "instrumentation": {"feasibility": "validation", "profile": "profile"}.get(
                        phase, "timing"
                    ),
                    "case_lifetime_timeout_s": 600,
                    "max_steps": stats[wid]["max_steps"],
                    "max_events": stats[wid]["max_events"],
                    "workload_sha256": _digest(workloads[wid]),
                    "controls_sha256": _digest(contract),
                }
            )

        if worker_id.startswith("N-"):
            for cell in CELLS:
                add(cell, "feasibility")
            for row in held["execution_order"]:
                if row["implementation_id"] == side:
                    rows.append(
                        {
                            "execution_id": row["execution_id"],
                            "kind": row["kind"],
                            "phase": "validation",
                            "worker_id": worker_id,
                            "implementation_id": side,
                        }
                    )
            for case in numerical["execution_order"]:
                if case["implementation_id"] == side:
                    rows.append(
                        {
                            "execution_id": case["case_id"],
                            "kind": "numerical",
                            "phase": "validation",
                            "worker_id": worker_id,
                            "implementation_id": side,
                        }
                    )
        elif worker_id.startswith("P-"):
            for phase in ("diagnostic_warmup", "profile"):
                for cell in ("W1-refill", "W4-refill"):
                    add(cell, phase)
        else:
            for phase in ("warmup", "measured"):
                for cell in CELLS:
                    add(cell, phase, int(worker_id[-1]))
        worker["execution_ids"] = [r["execution_id"] for r in rows if r["worker_id"] == worker_id]
        workers.append(worker)
    require(
        len(rows) == len({r["execution_id"] for r in rows}) == 139, "exact 139 unique executions"
    )
    equal(
        [len(w["execution_ids"]) for w in workers], [40, 35, 14, 14, 14, 14, 4, 4], "worker budgets"
    )
    return workers, rows


def resource_estimates(numerical, held):
    n, limits = numerical["resource_estimates"], numerical["contract"]["limits"]
    require(
        held["resource_estimates"]["artifact_bytes_upper_bound"]
        <= LIMITS["held_evidence_bytes_max"],
        "held plans exceed their combined artifact allowance",
    )
    require(n["max_retained_case_bytes"] <= limits["group_spool_bytes"], "per-group tensor cap")
    require(
        n["retained_tensor_bytes_upper_bound"] + limits["persisted_dump_bytes"]
        <= limits["cumulative_spool_written_bytes"],
        "cumulative tensor cap",
    )
    # A dump pair originates in one comparison observation. Bounding every
    # comparison by two dumped index entries is conservative even without payloads.
    components = {
        "retained_tensor_bytes": n["retained_tensor_bytes_upper_bound"],
        "dump_tensor_bytes": limits["persisted_dump_bytes"],
        "retained_index_bytes": n["retained_index_records_upper_bound"]
        * limits["spool_index_record_bytes"],
        "dump_index_bytes": 2
        * n["comparison_records_upper_bound"]
        * limits["spool_index_record_bytes"],
        "comparison_json_bytes": n["comparison_records_upper_bound"]
        * limits["comparison_record_bytes"],
        "held_artifact_bytes": LIMITS["held_evidence_bytes_max"],
        "profile_trace_bytes": LIMITS["profile_total_bytes_max"],
        "other_artifact_bytes": LIMITS["auxiliary_artifact_bytes_max"],
    }
    total = sum(components.values())
    require(total <= LIMITS["artifact_bytes_max"], "complete planned artifacts exceed 12 GiB")
    return {
        "components": components,
        "total_artifact_bytes_upper_bound": total,
        "artifact_headroom_bytes": LIMITS["artifact_bytes_max"] - total,
        "numerical": deepcopy(n),
        "held": deepcopy(held["resource_estimates"]),
    }


def make_plan(*, baseline_root, candidate_root, contract_path, model_path, gpu_ids, affinity):
    from vllm_lt.validation.m4_attention import build_numerical_plan
    from vllm_lt.validation.m4_attention_held import build_held_plan

    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    contract["controls"].update(gpu_ids=gpu_ids, affinity=affinity)
    validate_contract(contract, resolved=True)
    roots = {"A": Path(baseline_root).resolve(), "B": Path(candidate_root).resolve()}
    probes = {k: probe_checkout(v) for k, v in roots.items()}
    equal(probes["A"]["dependencies"], probes["B"]["dependencies"], "common dependency environment")
    implementations = {
        k: {"root": str(roots[k]), "source": probes[k]["source"], "imports": probes[k]["imports"]}
        for k in roots
    }
    texts = {k: (roots[k] / DIFF_PATHS[0]).read_text() for k in roots}
    harness, differences = source_controls(implementations, texts)
    inputs = {}
    for key, name in INPUTS.items():
        record, other = (_file_record(roots[k] / name) for k in ("A", "B"))
        equal(
            [record["size_bytes"], record["sha256"]],
            [other["size_bytes"], other["sha256"]],
            "common " + key,
        )
        inputs[key] = {"file": record, "contents": read_json(roots["A"] / name)}
    suite, benchmark = (inputs[k]["contents"] for k in ("benchmark_suite", "benchmark_contract"))
    _validate_suite(suite)
    _validate_contract(benchmark)
    config = OuroConfig.from_dict(read_json(model_path / "config.json"))
    validate_model_config(config.to_dict())
    files = _model_files(model_path)
    tokenizer = next(r for r in files if r["path"] == "tokenizer.json")
    for key in ("size_bytes", "sha256"):
        equal(tokenizer[key], suite["provenance"]["tokenizer"][key], "actual tokenizer provenance")
    numerical = build_numerical_plan(
        inputs["numerical_suite"]["contents"],
        inputs["numerical_contract"]["contents"],
        config.to_dict(),
    )
    held = build_held_plan()
    stats = _workload_stats(suite, config, benchmark)
    workers, rows = execution_rows(suite, contract, stats, numerical, held)
    plan = {
        "schema_version": 1,
        "artifact_type": "m4_attention_plan",
        "contract": contract,
        "contract_file": _file_record(contract_path),
        "implementations": implementations,
        "harness": harness,
        "production_differences": differences,
        "attention_source": texts,
        "dependencies": probes["A"]["dependencies"],
        "inputs": inputs,
        "model_path": str(model_path),
        "model_config": config.to_dict(),
        "model_files": files,
        "suite": suite,
        "benchmark_contract": benchmark,
        "workload_stats": stats,
        "numerical": numerical,
        "held": held,
        "resource_estimates": resource_estimates(numerical, held),
        "workers": workers,
        "execution_order": rows,
        "interpreter": sys.executable,
        "runtime_environment": {k: os.environ.get(k) for k in RUNTIME_VARIABLES},
    }
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    from vllm_lt.validation.m4_attention import validate_numerical_plan
    from vllm_lt.validation.m4_attention_held import validate_held_plan

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
            "attention_source",
            "dependencies",
            "inputs",
            "model_path",
            "model_config",
            "model_files",
            "suite",
            "benchmark_contract",
            "workload_stats",
            "numerical",
            "held",
            "resource_estimates",
            "workers",
            "execution_order",
            "interpreter",
            "runtime_environment",
            "plan_sha256",
        ),
        name="M4 plan",
    )
    equal([plan["schema_version"], plan["artifact_type"]], [1, "m4_attention_plan"], "plan version")
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan selfhash",
    )
    validate_contract(plan["contract"], resolved=True)
    _validate_dependencies(plan["dependencies"])
    official = plan["dependencies"]["official"]
    equal(official["dependencies"]["transformers"], "4.55.0", "prepared official dependency")
    equal(official["optional_kernels_present"], False, "optional kernels absent")
    for field in ("model_files",):
        _file_records(plan[field], field)
    for record in plan["model_files"]:
        require(
            not Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts,
            "checkpoint file names must remain inside the model directory",
        )
    _file_records([plan["contract_file"]], "contract file")
    require(Path(plan["contract_file"]["path"]).is_absolute(), "absolute contract path")
    _keys(plan["inputs"], INPUTS, name="input references")
    for value in plan["inputs"].values():
        _keys(value, ("file", "contents"), name="input")
        _file_records([value["file"]], "input file")
        require(Path(value["file"]["path"]).is_absolute(), "absolute frozen input path")
    _validate_suite(plan["suite"])
    _validate_contract(plan["benchmark_contract"])
    validate_numerical_plan(plan["numerical"])
    validate_held_plan(plan["held"])
    harness, differences = source_controls(plan["implementations"], plan["attention_source"])
    equal([plan["harness"], plan["production_differences"]], [harness, differences], "source proof")
    equal(plan["suite"], plan["inputs"]["benchmark_suite"]["contents"], "embedded benchmark suite")
    equal(
        plan["benchmark_contract"],
        plan["inputs"]["benchmark_contract"]["contents"],
        "embedded benchmark contract",
    )
    equal(
        plan["numerical"]["source_inputs"],
        {
            "suite": plan["inputs"]["numerical_suite"]["contents"],
            "contract": plan["inputs"]["numerical_contract"]["contents"],
        },
        "original numerical inputs",
    )
    validate_model_config(plan["model_config"])
    equal(plan["numerical"]["model_config"], plan["model_config"], "numerical model")
    stats = _workload_stats(
        plan["suite"], OuroConfig.from_dict(plan["model_config"]), plan["benchmark_contract"]
    )
    equal(plan["workload_stats"], stats, "workload capacity bounds")
    workers, rows = execution_rows(
        plan["suite"], plan["contract"], stats, plan["numerical"], plan["held"]
    )
    equal([plan["workers"], plan["execution_order"]], [workers, rows], "exact complete order")
    equal(
        plan["resource_estimates"],
        resource_estimates(plan["numerical"], plan["held"]),
        "complete resource bounds",
    )
    _keys(plan["runtime_environment"], RUNTIME_VARIABLES, name="runtime environment")
    require(
        all(v is None or isinstance(v, str) for v in plan["runtime_environment"].values()),
        "runtime variable types",
    )
    for name in ("interpreter", "model_path"):
        require(isinstance(plan[name], str) and Path(plan[name]).is_absolute(), "absolute " + name)


def verify_plan(plan, *, implementation_id=None):
    validate_plan(plan)
    require(implementation_id in (None, "A", "B"), "valid implementation ID")
    for key in (implementation_id,) if implementation_id else ("A", "B"):
        expected = plan["implementations"][key]
        actual = source_probe() if implementation_id else probe_checkout(expected["root"])
        equal(actual["source"], expected["source"], "actual frozen source")
        equal(actual["imports"], expected["imports"], "actual import origins")
        equal(actual["dependencies"], plan["dependencies"], "actual dependencies")
        equal(
            (Path(expected["root"]) / DIFF_PATHS[0]).read_text(),
            plan["attention_source"][key],
            "actual tile source",
        )
    equal(_model_files(Path(plan["model_path"])), plan["model_files"], "frozen checkpoint bytes")
    equal(
        OuroConfig.from_dict(read_json(Path(plan["model_path"]) / "config.json")).to_dict(),
        plan["model_config"],
        "actual model configuration",
    )
    equal(
        _file_record(Path(plan["contract_file"]["path"])),
        plan["contract_file"],
        "frozen contract bytes",
    )
    original = read_json(Path(plan["contract_file"]["path"]))
    original["controls"].update(
        gpu_ids=plan["contract"]["controls"]["gpu_ids"],
        affinity=plan["contract"]["controls"]["affinity"],
    )
    equal(original, plan["contract"], "resolved contract contents")
    for key, value in plan["inputs"].items():
        equal(_file_record(Path(value["file"]["path"])), value["file"], "input bytes " + key)
        equal(
            read_json(Path(value["file"]["path"])),
            value["contents"],
            "input parsed contents " + key,
        )
    equal(sys.executable, plan["interpreter"], "prepared interpreter")
    equal(
        {k: os.environ.get(k) for k in RUNTIME_VARIABLES},
        plan["runtime_environment"],
        "runtime environment",
    )
    equal(affinity_snapshot(), plan["contract"]["controls"]["affinity"], "CPU/NUMA controls")


def execution_view(plan):
    """Private M1 executor view; no fake M1/M2 serialized identity."""
    return {
        "plan_sha256": plan["plan_sha256"],
        "contract": plan["benchmark_contract"],
        "suite": plan["suite"],
        "model_config": plan["model_config"],
        "model_path": plan["model_path"],
        "workload_stats": plan["workload_stats"],
    }
