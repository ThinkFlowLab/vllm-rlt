"""Frozen FP32-only GSM8K plan; offline integrity checks need only the stdlib.

Heavy shared preparation helpers are imported only by make/verify/source_probe.
The scorer can load this file directly without the package's Torch imports.
"""

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

BASE_SHA = "630a8fdc0dd47b6da68a3d30db6b851fc08af5c5"
BASE_PRODUCTION_SHA256 = "58b93e0c79f8c402d7ab2910bc7aeb75634b21ccc0cafe200e44233a27d91e3f"
CONTRACT_SHA256 = "43f51e4cdc7421df7b41efe1707d55d09639d8f99a4fe8527675385d0974b369"
IMPORT_MODULES = (
    "vllm_lt.benchmarks.q2_quality_schema",
    "vllm_lt.benchmarks.q2_quality_data",
    "vllm_lt.benchmarks.q2_quality_score",
    "vllm_lt.benchmarks.q2_quality",
    "vllm_lt.engine.llm_engine",
    "vllm_lt.worker.model_runner",
    "vllm_lt.models.ouro",
)
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
LIMITS = {
    "workers": 1,
    "model_loads": 1,
    "executions": 66,
    "feasibility_executions": 2,
    "evaluation_executions": 64,
    "total_timeout_s": 7200,
    "case_timeout_s": 600,
    "case_bytes_max": 2 * 1024**2,
    "artifact_bytes_max": 256 * 1024**2,
    "profile_bytes_max": 0,
    "tensor_bytes_max": 0,
}
PRODUCTION_PREFIXES = tuple(
    "vllm_lt/" + p + "/" for p in ("core", "kernels", "engine", "models", "worker", "entrypoints")
)
PRODUCTION_FILES = ("vllm_lt/config.py", "vllm_lt/request.py", "vllm_lt/sampling_params.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def equal(actual, expected, name):
    require(digest(actual) == digest(expected), name + " differs from the frozen contract")


def keys(value, expected, name):
    require(isinstance(value, dict) and set(value) == set(expected), name + " fields")


def integer(value, name, lower=0, upper=None):
    require(
        type(value) is int and value >= lower and (upper is None or value <= upper),
        name + " must be a bounded integer",
    )
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key " + key)
        result[key] = value
    return result


def read_json(path):
    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=_pairs,
        parse_float=number,
        parse_constant=lambda x: require(False, "nonfinite JSON " + x),
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def data_module():
    """Sibling loading also supports the independent, no-Torch scoring command."""
    name = "_q2_quality_data_offline"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("q2_quality_data.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def file_record(path):
    path = Path(path)
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            checksum.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": checksum.hexdigest(),
    }


def validate_file(record, *, absolute=False):
    keys(record, ("path", "size_bytes", "sha256"), "file record")
    require(isinstance(record["path"], str) and bool(record["path"]), "file path")
    integer(record["size_bytes"], "file bytes")
    require(
        isinstance(record["sha256"], str) and re.fullmatch("[0-9a-f]{64}", record["sha256"]),
        "file SHA256",
    )
    path = Path(record["path"])
    require(
        path.is_absolute() if absolute else not path.is_absolute() and ".." not in path.parts,
        "file path scope",
    )


def validate_affinity(value):
    keys(value, ("cpu_ids", "numa_status", "numactl_show"), "affinity")
    cpus = value["cpu_ids"]
    require(isinstance(cpus, list) and bool(cpus), "CPU IDs")
    for cpu in cpus:
        integer(cpu, "CPU ID")
    equal(cpus, sorted(set(cpus)), "CPU order")
    policy = value["numactl_show"]
    keys(
        policy,
        ("policy", "preferred_node", "physcpubind", "cpubind", "nodebind", "membind"),
        "active NUMA policy",
    )
    equal(policy["policy"], "bind", "memory binding")
    equal(policy["physcpubind"], " ".join(map(str, cpus)), "CPU binding")
    require(
        all(isinstance(v, str) and v and v == " ".join(v.split()) for v in policy.values()),
        "normalized NUMA policy",
    )
    require(
        isinstance(value["numa_status"], list) and len(value["numa_status"]) == 2, "affinity status"
    )
    for line, prefix in zip(value["numa_status"], ("Cpus_allowed_list:", "Mems_allowed_list:")):
        require(isinstance(line, str) and line.startswith(prefix), "affinity status field")


def canonical_gpu_uuid(value):
    require(isinstance(value, str), "GPU UUID must be text")
    raw = value.removeprefix("GPU-")
    require(
        re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", raw),
        "malformed GPU UUID",
    )
    return "GPU-" + raw


def validate_contract(contract, *, resolved=False):
    original = deepcopy(contract)
    controls = original["controls"]
    for key in ("gpu_ids", "gpu_uuid", "affinity"):
        controls[key] = None
    equal(digest(original), CONTRACT_SHA256, "fixed quality contract")
    controls = contract["controls"]
    if resolved:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one physical GPU",
        )
        integer(controls["gpu_ids"][0], "GPU ID")
        require(
            isinstance(controls["gpu_uuid"], str)
            and re.fullmatch(
                r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                controls["gpu_uuid"],
            ),
            "physical GPU UUID",
        )
        validate_affinity(controls["affinity"])
    else:
        equal(
            [controls[k] for k in ("gpu_ids", "gpu_uuid", "affinity")],
            [None] * 3,
            "unresolved template",
        )


def execution_order(selection):
    rows = []
    for phase, short in (("feasibility", "feas"), ("evaluation", "eval")):
        for index, example in enumerate(selection[phase]):
            length = example["prompt_token_count"]
            rows.append(
                {
                    "run_id": f"F-{short}-{index:02d}",
                    "phase": phase,
                    "example_id": example["source_id"],
                    "example_index": index,
                    "max_steps": (length + 127) // 128 + 1531,
                    "prompt_tokens": length,
                    "reserved_pages": 4 * ((length + 255 + 15) // 16),
                }
            )
    return rows


def _resources(selection, inputs):
    source_bytes = sum(
        inputs["dataset"]["contents"][key]["size_bytes"] for key in ("raw", "readme", "normalized")
    )
    preparation_bytes = (
        source_bytes
        + sum(v["file"]["size_bytes"] for v in inputs.values())
        + selection["tokenizer"]["file"]["size_bytes"]
        + inputs["prerequisite"]["contents"]["qualification"]["size_bytes"]
    )
    # Fixed case caps include every token prefix, source text, result and immutable ACK.
    return {
        "native_page_bytes": 6291456,
        "native_pool_bytes": 1207959552,
        "max_reserved_pages": max(r["reserved_pages"] for r in execution_order(selection)),
        "max_steps": max(r["max_steps"] for r in execution_order(selection)),
        "case_artifacts_bytes_upper_bound": 66 * LIMITS["case_bytes_max"],
        "prepared_input_bytes": preparation_bytes,
        "auxiliary_bytes_max": LIMITS["artifact_bytes_max"] - 66 * LIMITS["case_bytes_max"],
        "artifact_bytes_upper_bound": LIMITS["artifact_bytes_max"],
        "tensor_payload_bytes": 0,
        "profile_bytes": 0,
        "max_output_tokens": 256,
        "vocabulary_token_utf8_bytes_max": 162,
        "decoded_output_utf8_bytes_upper_bound": 3 * 162 * 256,
        "max_selected_prompt_utf8_bytes": max(
            len(e["prompt_text"].encode())
            for phase in ("feasibility", "evaluation")
            for e in selection[phase]
        ),
    }


def _controls_hash(plan):
    return digest(
        {
            k: plan[k]
            for k in (
                "contract",
                "selection",
                "inputs",
                "model_config",
                "model_files",
                "source",
                "imports",
                "dependencies",
                "interpreter",
                "runtime_environment",
            )
        }
    )


def _validate_dataset(plan):
    manifest = plan["inputs"]["dataset"]["contents"]
    equal(manifest["artifact_type"], "q2_quality_dataset_manifest", "dataset manifest")
    equal(manifest["dataset"], plan["contract"]["dataset"], "original dataset identity")
    for key in ("raw", "readme", "normalized"):
        validate_file(
            {k: manifest[key][k] for k in ("path", "size_bytes", "sha256")}, absolute=True
        )
    for key in ("sha256", "size_bytes"):
        equal(manifest["raw"][key], manifest["dataset"][key], "original dataset " + key)
    equal(manifest["normalized"], plan["selection"]["dataset"]["file"], "normalized data")
    equal(manifest["row_count"], 1319, "original source count")
    equal(manifest["conversion"]["source_string_transformations"], "none", "unmodified strings")
    equal(manifest["conversion"]["device_work"], False, "CPU preparation")
    validate_file(manifest["conversion"]["producer"], absolute=True)


def _validate_prerequisite(value):
    equal(value["artifact_type"], "q2_quality_q1_prerequisite", "Q1 prerequisite type")
    equal(value["dtype"], "float32", "Q1 prerequisite dtype")
    equal(value["decision"], "passed", "Q1 prerequisite decision")
    equal(
        value["fp32"],
        {"decision": "passed", "numerical_required_failures": 0, "behavior_required_failures": 0},
        "Q1 FP32 eligibility",
    )
    equal(value["required_trajectories"], 93, "Q1 FP32 coverage")
    equal(
        value["families"],
        {"main": 64, "live_gate": 16, "official": 4, "original": 9},
        "Q1 families",
    )
    equal(value["q1_evidence_status"], "complete", "Q1 evidence coverage")
    equal(value["q1_execution_commit"], "3fe9e002bd049084df47cda47e82f1e51cb35b2b", "Q1 source")
    equal(
        value["q1_plan_sha256"],
        "43d9994bfbf10aa5fc28366f6f106bf70b6ef9222e426401b0ebdc29639f35e9",
        "Q1 plan",
    )
    equal(
        value["qualification"]["sha256"],
        "e1b77c625962aa21b9a68f89dfe29ced9c20def0844fa2f6737043b2e539586f",
        "Q1 report",
    )
    equal(value["qualification"]["size_bytes"], 669248, "Q1 report bytes")
    equal(value["bf16_adaptive_eligible"], False, "BF16 remains blocked")
    validate_file(value["qualification"], absolute=True)


def validate_quality_plan(plan):
    keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "contract",
            "inputs",
            "selection",
            "model_path",
            "model_config",
            "model_files",
            "source",
            "imports",
            "dependencies",
            "interpreter",
            "runtime_environment",
            "execution_order",
            "resource_estimates",
            "controls_sha256",
            "plan_sha256",
        ),
        "quality plan",
    )
    equal([plan["schema_version"], plan["artifact_type"]], [1, "q2_quality_plan"], "plan version")
    equal(
        plan["plan_sha256"],
        digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan hash",
    )
    validate_contract(plan["contract"], resolved=True)
    keys(plan["inputs"], ("contract", "selection", "dataset", "prerequisite"), "inputs")
    for value in plan["inputs"].values():
        keys(value, ("file", "contents"), "embedded input")
        validate_file(value["file"], absolute=True)
    original = deepcopy(plan["inputs"]["contract"]["contents"])
    validate_contract(original)
    original["controls"] = plan["contract"]["controls"]
    equal(original, plan["contract"], "resolved original contract")
    equal(plan["selection"], plan["inputs"]["selection"]["contents"], "embedded selection")
    data_module().validate_selection(plan["selection"])
    _validate_dataset(plan)
    _validate_prerequisite(plan["inputs"]["prerequisite"]["contents"])
    config = plan["model_config"]
    name = "_q2_ouro_configuration_offline"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parents[1] / "models/config.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    cls = sys.modules[name].OuroConfig
    equal(cls.from_dict(config).to_dict(), config, "complete supported model configuration")
    for key, expected in {
        "vocab_size": 49152,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "total_ut_steps": 4,
        "eos_token_id": 0,
        "max_position_embeddings": 65536,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }.items():
        equal(config[key], expected, "pinned model " + key)
    for value in plan["model_files"]:
        validate_file(value)
    model_files = {r["path"]: r for r in plan["model_files"]}
    require(
        len(model_files) == len(plan["model_files"])
        and {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}
        <= model_files.keys(),
        "checkpoint files",
    )
    for key in ("size_bytes", "sha256"):
        equal(
            model_files["tokenizer.json"][key],
            plan["selection"]["tokenizer"]["file"][key],
            "model tokenizer " + key,
        )
    source = plan["source"]
    keys(source, ("root", "commit", "status", "files"), "source")
    require(
        isinstance(source["commit"], str) and re.fullmatch("[0-9a-f]{40}", source["commit"]),
        "source commit",
    )
    equal(source["status"], "", "clean source")
    for record in source["files"]:
        validate_file(record)
    paths = [r["path"] for r in source["files"]]
    equal(paths, sorted(set(paths)), "source files")
    equal(
        digest(
            [
                r
                for r in source["files"]
                if r["path"].startswith(PRODUCTION_PREFIXES) or r["path"] in PRODUCTION_FILES
            ]
        ),
        BASE_PRODUCTION_SHA256,
        "PR14 production",
    )
    equal(
        plan["imports"], {k: k.replace(".", "/") + ".py" for k in IMPORT_MODULES}, "import origins"
    )
    require(set(plan["imports"].values()) <= set(paths), "source-bound imports")
    source_files = {r["path"]: r for r in source["files"]}
    for item in plan["inputs"].values():
        path = Path(item["file"]["path"])
        if (
            path.is_relative_to(source["root"])
            and not path.relative_to(source["root"]).parts[0] == "artifacts"
        ):
            rel = str(path.relative_to(source["root"]))
            equal(source_files.get(rel), {**item["file"], "path": rel}, "source-bound input")
    for value in (source["root"], plan["model_path"], plan["interpreter"]):
        require(isinstance(value, str) and Path(value).is_absolute(), "absolute execution path")
    deps = plan["dependencies"]
    keys(deps, ("python", "torch", "torch_cuda_build", "distributions", "official"), "dependencies")
    require(
        all(isinstance(deps[k], str) and deps[k] for k in ("python", "torch")),
        "dependency versions",
    )
    distributions = deps["distributions"]
    for record in distributions:
        keys(record, ("name", "version"), "distribution")
        require(all(isinstance(v, str) and v for v in record.values()), "distribution value")
    equal(
        distributions,
        sorted(distributions, key=lambda r: (r["name"].lower(), r["version"], r["name"])),
        "distribution order",
    )
    equal(
        [r["version"] for r in distributions if r["name"].lower() == "tokenizers"],
        ["0.21.4"],
        "prepared tokenizer version",
    )
    keys(plan["runtime_environment"], RUNTIME_VARIABLES, "runtime environment")
    require(
        all(v is None or isinstance(v, str) for v in plan["runtime_environment"].values()),
        "runtime environment values",
    )
    equal(plan["execution_order"], execution_order(plan["selection"]), "66 exact executions")
    equal(
        plan["resource_estimates"],
        _resources(plan["selection"], plan["inputs"]),
        "byte/step bounds",
    )
    require(
        plan["resource_estimates"]["prepared_input_bytes"]
        <= plan["resource_estimates"]["auxiliary_bytes_max"] // 2,
        "prepared artifacts leave no support budget",
    )
    equal(plan["controls_sha256"], _controls_hash(plan), "controls hash")


def source_probe():
    import importlib

    from vllm_lt.benchmarks.schema import _source_manifest
    from vllm_lt.validation.schema import dependency_manifest

    source = _source_manifest()
    root = Path(source["root"]).resolve()
    imports = {}
    for name in IMPORT_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to(root), "import outside selected checkout: " + name)
        imports[name] = str(path.relative_to(root))
    return {"source": source, "imports": imports, "dependencies": dependency_manifest()}


def make_quality_plan(
    contract_path,
    selection_path,
    model_path,
    *,
    dataset_manifest_path,
    prerequisite_path,
    gpu_ids,
    gpu_uuid,
    affinity=None,
):
    from vllm_lt.benchmarks.ab_schema import affinity_snapshot
    from vllm_lt.benchmarks.schema import _model_files
    from vllm_lt.models.config import OuroConfig

    paths = {
        "contract": contract_path,
        "selection": selection_path,
        "dataset": dataset_manifest_path,
        "prerequisite": prerequisite_path,
    }
    inputs = {k: {"file": file_record(p), "contents": read_json(p)} for k, p in paths.items()}
    contract = deepcopy(inputs["contract"]["contents"])
    validate_contract(contract)
    contract["controls"].update(
        gpu_ids=gpu_ids,
        gpu_uuid=gpu_uuid,
        affinity=affinity if affinity is not None else affinity_snapshot(),
    )
    selection = deepcopy(inputs["selection"]["contents"])
    equal(
        data_module().prepare_selection(
            selection["dataset"]["file"]["path"], selection["tokenizer"]["file"]["path"]
        ),
        selection,
        "actual selection rebuild",
    )
    model_path = Path(model_path).resolve()
    plan = {
        "schema_version": 1,
        "artifact_type": "q2_quality_plan",
        "contract": contract,
        "inputs": inputs,
        "selection": selection,
        "model_path": str(model_path),
        "model_config": OuroConfig.from_dict(read_json(model_path / "config.json")).to_dict(),
        "model_files": _model_files(model_path),
        **source_probe(),
        "interpreter": sys.executable,
        "runtime_environment": {k: os.environ.get(k) for k in RUNTIME_VARIABLES},
        "execution_order": execution_order(selection),
        "resource_estimates": _resources(selection, inputs),
    }
    plan["controls_sha256"] = _controls_hash(plan)
    plan["plan_sha256"] = digest(plan)
    validate_quality_plan(plan)
    return plan


def verify_quality_plan(plan):
    from vllm_lt.benchmarks.ab_schema import affinity_snapshot
    from vllm_lt.benchmarks.schema import _model_files
    from vllm_lt.models.config import OuroConfig

    validate_quality_plan(plan)
    for key, value in source_probe().items():
        equal(value, plan[key], "actual " + key)
    for key, item in plan["inputs"].items():
        path = item["file"]["path"]
        equal(file_record(path), item["file"], "actual input " + key)
        equal(read_json(path), item["contents"], "actual embedded input " + key)
    selection = plan["selection"]
    equal(
        data_module().prepare_selection(
            selection["dataset"]["file"]["path"], selection["tokenizer"]["file"]["path"]
        ),
        selection,
        "actual selection",
    )
    manifest = plan["inputs"]["dataset"]["contents"]
    for key in ("raw", "readme", "normalized"):
        equal(
            file_record(manifest[key]["path"]),
            {k: manifest[key][k] for k in ("path", "size_bytes", "sha256")},
            "dataset " + key,
        )
    prerequisite = plan["inputs"]["prerequisite"]["contents"]
    equal(
        file_record(prerequisite["qualification"]["path"]),
        prerequisite["qualification"],
        "Q1 evidence",
    )
    equal(_model_files(Path(plan["model_path"])), plan["model_files"], "actual checkpoint")
    equal(
        OuroConfig.from_dict(read_json(Path(plan["model_path"]) / "config.json")).to_dict(),
        plan["model_config"],
        "actual model config",
    )
    equal(sys.executable, plan["interpreter"], "interpreter")
    equal(
        {k: os.environ.get(k) for k in RUNTIME_VARIABLES},
        plan["runtime_environment"],
        "environment",
    )
    equal(affinity_snapshot(), plan["contract"]["controls"]["affinity"], "CPU/NUMA")
