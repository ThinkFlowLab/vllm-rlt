"""Offline evidence audit; recorded errors are audited, not recomputed from absent tensors."""

import hashlib
import json
import math
from pathlib import Path

from vllm_lt.benchmarks.schema import _finite_float, _object_pairs, _reject_constant
from vllm_lt.validation.schema import PROJECTION

from .evidence import accumulate, boundary_key, compare_behavior, empty_summary
from .schema import _digest, _file_record, read_json, validate_plan_integrity, write_json

_LAYER_OPS = ("attention_input", "query", "key", "value", "attention_output", "layer_output")
_SUMMARY_FIELDS = (
    "count",
    "compared_count",
    "incomparable_count",
    "required_failures",
    "first_failure",
    "by_operation",
    "behavior",
    "behavior_failures",
    "first_behavior_failure",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=0):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= minimum,
        f"{name} must be finite and >= {minimum}",
    )
    return value


def _equal(left, right, name):
    _require(_digest(left) == _digest(right), f"{name} differs from independently derived evidence")


def _history(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _fixtures(plan):
    result = {row["fixture_id"]: row for row in plan["suite"]["fixtures"]}
    result.update({row["fixture_id"]: row for row in plan["suite"]["feasibility_fixtures"]})
    for index, prompt in enumerate(plan["suite"]["original_reproduction"]["prompt_token_ids"]):
        result[f"Q1-original-{index}"] = {
            "fixture_id": f"Q1-original-{index}",
            "prompt_token_ids": prompt,
        }
    return result


def _audit_traces(case, fixture, traces, config):
    _require(isinstance(traces, list) and len(traces) == 9, "nine genuine predictions required")
    history = list(fixture["prompt_token_ids"])
    for index, trace in enumerate(traces):
        for name in (
            "position",
            "output_index",
            "exit_depth",
            "actual_token_id",
            "emitted_token_id",
        ):
            _integer(trace[name], name)
        _equal(
            [trace["position"], trace["output_index"], trace["history_sha256"]],
            [len(history) - 1, index, _history(history)],
            "trace position/history",
        )
        depth = trace["exit_depth"]
        _require(depth in (2, 3, 4) and (index != 0 or depth == 4), "invalid executed exit depth")
        _require(trace["actual_token_id"] < config["vocab_size"], "actual token outside vocabulary")
        if case["history_mode"] == "teacher_forced":
            _equal(depth, fixture["forced_exit_depths"][index], "fixed/forced loop work")
            emitted = (
                fixture["continuation_input_ids"][index] if index < 8 else trace["actual_token_id"]
            )
        else:
            emitted = trace["actual_token_id"]
        _equal(trace["emitted_token_id"], emitted, "supplied versus sampled next input")
        _equal(
            trace["gate_usage"],
            "full_depth_prefill_diagnostic" if index == 0 else "actual_decode",
            "gate usage",
        )
        _number(trace["top_two_margin"], "top-two margin")
        for name in ("gate_logits", "gate_probabilities", "cumulative_probabilities"):
            _require(
                isinstance(trace[name], list) and len(trace[name]) == depth,
                "gate evidence must include every executed loop",
            )
            for value in trace[name]:
                _require(
                    type(value) in (int, float) and math.isfinite(value), "nonfinite gate evidence"
                )
        remaining = 1.0
        for loop, (probability, cdf) in enumerate(
            zip(trace["gate_probabilities"], trace["cumulative_probabilities"]), 1
        ):
            _require(0 <= probability <= 1 and 0 <= cdf <= 1, "gate probability outside [0,1]")
            remaining *= 1.0 - probability
            _require(cdf == 1.0 - remaining, "CDF does not accumulate the recorded real hazards")
            if case["history_mode"] == "live_gate" and index > 0 and loop >= 2:
                if loop < depth:
                    _require(
                        cdf < case["exit_policy"]["threshold"],
                        "live exit skipped threshold crossing",
                    )
                elif depth < 4:
                    _require(cdf >= case["exit_policy"]["threshold"], "live exit below threshold")
        if index < 8:
            history.append(emitted)


def _metadata(fixture_id, operation, positions, depth, layer, output_index, history_sha256):
    return {
        "fixture_id": fixture_id,
        "operation": operation,
        "positions": positions,
        "depth": depth,
        "layer": layer,
        "output_index": output_index,
        "history_sha256": history_sha256,
    }


def _expected_boundaries(case, fixture, traces, config):
    """Exact candidate keys/shapes follow executed traces; live counts are never padded."""
    projection = case.get("boundary_projection")
    _require(
        projection in (None, PROJECTION),
        "unknown numerical boundary projection",
    )
    result = {}
    fixture_id = fixture["fixture_id"]
    hidden, layers, vocabulary = (
        config["hidden_size"],
        config["num_hidden_layers"],
        config["vocab_size"],
    )
    kvheads, headdim = config["num_key_value_heads"], config["head_dim"]

    def add(metadata, shape):
        result[boundary_key(metadata)] = (metadata, shape)

    if case["family"] == "original":
        positions = list(range(len(fixture["prompt_token_ids"])))
        for depth in range(1, 5):
            add(
                _metadata(
                    fixture_id,
                    "logits",
                    positions,
                    depth,
                    None,
                    None,
                    _history(fixture["prompt_token_ids"]),
                ),
                [len(positions), vocabulary],
            )
        return result
    history = list(fixture["prompt_token_ids"])
    if case["implementation"] == "official":
        for index in range(9):
            add(
                _metadata(
                    fixture_id, "logits", [len(history) - 1], 4, None, index, _history(history)
                ),
                [vocabulary],
            )
            if index < 8:
                history.append(fixture["continuation_input_ids"][index])
        return result
    for trace in traces:
        index, position, selected = trace["output_index"], trace["position"], trace["exit_depth"]
        for depth in range(1, selected + 1):
            for layer in range(0 if projection else layers):
                for operation in _LAYER_OPS:
                    shape = (
                        [config["num_attention_heads"], headdim]
                        if operation == "query"
                        else [kvheads, headdim]
                        if operation in ("key", "value")
                        else [hidden]
                    )
                    add(
                        _metadata(
                            fixture_id,
                            operation,
                            [position],
                            depth,
                            layer,
                            index,
                            trace["history_sha256"],
                        ),
                        shape,
                    )
            for operation, shape in (("loop_hidden", [hidden]), ("gate_logits", [])):
                add(
                    _metadata(
                        fixture_id,
                        operation,
                        [position],
                        depth,
                        None,
                        index,
                        trace["history_sha256"],
                    ),
                    shape,
                )
        add(
            _metadata(
                fixture_id, "logits", [position], selected, None, index, trace["history_sha256"]
            ),
            [vocabulary],
        )
    capacity = len(fixture["prompt_token_ids"]) + 8
    for start in range(0, capacity, 4):
        positions = list(range(start, min(start + 4, capacity)))
        for component in ("keys", "values"):
            meta = _metadata(
                fixture_id, "populated_kv", positions, None, None, 8, traces[-1]["history_sha256"]
            )
            meta.update(
                component=component,
                axes=["depth", "layer", "position", "kv_head", "head_dim"],
                depth_range=[1, 4],
                layer_range=[0, layers - 1],
            )
            add(meta, [4, layers, len(positions), kvheads, headdim])
    return result


def _coordinate(value, shape, name):
    _require(isinstance(value, list) and len(value) == len(shape), f"invalid {name}")
    for coordinate, size in zip(value, shape):
        _require(type(coordinate) is int and 0 <= coordinate < size, f"invalid {name}")


def _audit_stats(record, comparison, shape, policy):
    stats, metadata = record["stats"], record["metadata"]
    _require(isinstance(stats, dict), "compared record has no tensor statistics")
    _equal(
        [stats["schema_version"], stats["artifact_type"]],
        [1, "tensor_comparison"],
        "statistics schema",
    )
    _equal(stats["policy_sha256"], _digest(policy), "statistics policy hash")
    _equal(stats["policy_id"], policy["policy_id"], "statistics policy ID")
    operation = (
        "bf16_fp32_sensitivity"
        if comparison["comparison_kind"] == "bf16_fp32_sensitivity"
        else metadata["operation"]
    )
    _equal(stats["operation"], operation, "statistics operation")
    _equal(stats["shape"], shape, "boundary tensor shape")
    count = math.prod(shape)
    _equal(_integer(stats["numel"], "numel", 1), count, "tensor element count")
    actual_dtype = comparison["dtype"]
    reference_dtype = (
        "float32" if comparison["comparison_kind"] == "bf16_fp32_sensitivity" else actual_dtype
    )
    _equal(
        [stats["actual_dtype"], stats["reference_dtype"]],
        [actual_dtype, reference_dtype],
        "tensor dtypes",
    )
    actual_finite = _integer(stats["actual_finite_count"], "actual finite count")
    reference_finite = _integer(stats["reference_finite_count"], "reference finite count")
    _require(
        actual_finite <= count and reference_finite <= count, "finite counts exceed tensor size"
    )
    finite = actual_finite == reference_finite == count
    _equal(stats["all_finite"], finite, "finiteness flag")
    numeric = operation == "logits"
    _equal(stats["numeric_required"], numeric, "numeric-required flag")
    _equal(stats["bounds"], policy["logits"][actual_dtype] if numeric else None, "numerical bounds")
    raw_top1 = not (comparison["family"] == "original" and metadata["depth"] < 4)
    _equal(record["top1_required"], raw_top1, "effective original top1 exception")
    for side, finite_count in (("actual", actual_finite), ("reference", reference_finite)):
        coordinate = stats[f"first_nonfinite_{side}_coordinate"]
        if finite_count == count:
            _equal(coordinate, None, "finite tensor nonfinite coordinate")
        else:
            _coordinate(coordinate, shape, "first nonfinite coordinate")
    if not finite:
        for key in (
            "max_abs_error",
            "rms_error",
            "reference_rms",
            "max_error_coordinate",
            "allclose",
        ):
            _equal(stats[key], None, "nonfinite aggregate")
        _equal(stats["exact_equal"], False, "nonfinite exact equality")
        _equal(stats["passed"], False, "nonfinite pass flag")
        _equal(stats["status"], "nonfinite", "nonfinite status")
        return False
    for key in ("max_abs_error", "rms_error", "reference_rms"):
        _number(stats[key], key)
    _coordinate(stats["max_error_coordinate"], shape, "maximum-error coordinate")
    _equal(
        stats["exact_equal"],
        actual_dtype == reference_dtype and stats["max_abs_error"] == 0,
        "exact-equality flag",
    )
    intrinsic_pass = True
    if numeric:
        _require(type(stats["allclose"]) is bool, "allclose must be an explicit recorded boolean")
        rows = count // shape[-1]
        for name in (
            "actual_top1_ids",
            "reference_top1_ids",
            "actual_top2_margins",
            "reference_top2_margins",
        ):
            _require(
                isinstance(stats[name], list) and len(stats[name]) == rows,
                "top1/margin row coverage differs",
            )
            for value in stats[name]:
                if name.endswith("ids"):
                    _require(
                        type(value) is int and 0 <= value < shape[-1], "invalid genuine top1 token"
                    )
                else:
                    _number(value, "top-two margin")
        equal_top1 = stats["actual_top1_ids"] == stats["reference_top1_ids"]
        _equal(stats["top1_equal"], equal_top1, "top1 flag versus genuine IDs")
        intrinsic_pass = stats["allclose"] and equal_top1
    else:
        _equal(stats["allclose"], None, "diagnostic-only allclose")
        _equal(stats["top1_equal"], None, "diagnostic-only top1")
    _equal(stats["passed"], intrinsic_pass, "intrinsic statistics pass flag")
    _equal(
        stats["status"],
        ("passed" if intrinsic_pass else "failed") if numeric else "diagnostic_only",
        "statistics status",
    )
    return (
        finite
        and (not comparison.get("require_exact", False) or stats["exact_equal"])
        and (not numeric or (stats["allclose"] and (not raw_top1 or stats["top1_equal"])))
    )


def _first_live_divergence(actual, reference):
    for index, (left, right) in enumerate(zip(actual, reference)):
        if (left["actual_token_id"], left["exit_depth"]) != (
            right["actual_token_id"],
            right["exit_depth"],
        ):
            return index
    return None


def _must_be_incomparable(metadata, key, reference_keys, actual_traces, reference_traces):
    first = _first_live_divergence(actual_traces, reference_traces)
    index = metadata["output_index"]
    if metadata["operation"] == "populated_kv":
        return any(
            (
                a["exit_depth"] != b["exit_depth"]
                or (i < 8 and a["actual_token_id"] != b["actual_token_id"])
            )
            for i, (a, b) in enumerate(zip(actual_traces, reference_traces))
        )
    return key not in reference_keys or (first is not None and index > first)


def _read_records(path, cap):
    with path.open("rb") as stream:
        for index in range(1, 1_000_001):
            line = stream.readline(cap + 1)
            if not line:
                return
            _require(
                len(line) <= cap and line.endswith(b"\n"),
                f"record {index} exceeds cap or lacks newline",
            )
            value = json.loads(
                line.decode("utf-8"),
                parse_constant=_reject_constant,
                parse_float=_finite_float,
                object_pairs_hook=_object_pairs,
            )
            _require(isinstance(value, dict), f"record {index} must be an object")
            yield value
    raise ValueError("comparison stream exceeds the bounded record count")


def _audit_comparison(folder, plan, comparison, cases, fixtures, observation_sink=None):
    comparison_id = comparison["comparison_id"]
    path = folder / "comparisons" / (comparison_id + ".jsonl")
    stored = read_json(path.with_suffix(".summary.json"))
    for key, value in comparison.items():
        _equal(stored[key], value, f"comparison plan field {key}")
    _equal(
        [stored["schema_version"], stored["artifact_type"], stored["plan_sha256"]],
        [1, "validation_comparison_summary", plan["plan_sha256"]],
        "comparison summary identity",
    )
    file_record = _file_record(path)
    _equal(
        [stored["sha256"], stored["size_bytes"]],
        [file_record["sha256"], file_record["size_bytes"]],
        "comparison JSONL bytes/hash",
    )
    candidate = cases[comparison["candidate_case_id"]]
    reference = cases[comparison["reference_case_id"]]
    fixture = fixtures[comparison["fixture_id"]]
    actual_traces = candidate["traces"].get(fixture["fixture_id"], [])
    reference_traces = reference["traces"].get(fixture["fixture_id"], [])
    expected = _expected_boundaries(candidate["case"], fixture, actual_traces, plan["model_config"])
    reference_keys = _expected_boundaries(
        reference["case"], fixture, reference_traces, plan["model_config"]
    )
    if comparison["family"] == "official":
        expected = {key: item for key, item in expected.items() if item[0]["operation"] == "logits"}
    summary, seen = empty_summary(comparison), set()
    first_numeric_failure = None
    earliest_nonexact = None
    previous_observation = 0
    for record in _read_records(path, plan["contract"]["limits"]["comparison_record_bytes"]):
        key, metadata = record["key"], record["metadata"]
        observation = _integer(record["observation_index"], "observation index", 1)
        _require(observation > previous_observation, "comparison observations are not increasing")
        previous_observation = observation
        if observation_sink is not None:
            observation_sink(observation, metadata["fixture_id"], key)
        _require(key in expected and key not in seen, "unexpected or duplicate candidate boundary")
        seen.add(key)
        expected_meta, shape = expected[key]
        _equal(metadata, expected_meta, "boundary position/depth/layer/history metadata")
        _equal(boundary_key(metadata), key, "boundary key")
        incomparable = comparison["family"] == "live_gate" and _must_be_incomparable(
            metadata, key, reference_keys, actual_traces, reference_traces
        )
        _equal(
            record["status"],
            "incomparable" if incomparable else "compared",
            "boundary comparability",
        )
        _equal(
            record["top1_required"],
            not (comparison["family"] == "original" and metadata["depth"] < 4),
            "effective top1 gate",
        )
        _equal(
            record.get("require_exact", False),
            comparison.get("require_exact", False),
            "effective exact-comparison gate",
        )
        if incomparable:
            _require(
                isinstance(record.get("reason"), str) and bool(record["reason"]),
                "incomparable reason missing",
            )
            stats = record["stats"]
            _equal(stats["check"], "candidate_finiteness_only", "incomparable check kind")
            _equal(stats["operation"], metadata["operation"], "incomparable operation")
            _equal(stats["shape"], shape, "incomparable shape")
            _equal(stats["actual_dtype"], comparison["dtype"], "incomparable dtype")
            _equal(stats["reference_available"], False, "incomparable reference availability")
            _equal(
                [
                    stats["all_finite"],
                    stats["actual_finite_count"],
                    stats["numel"],
                    stats["first_nonfinite_flat_index"],
                ],
                [True, math.prod(shape), math.prod(shape), None],
                "incomparable finiteness evidence",
            )
        else:
            _equal(record["reason"], None, "comparable record reason")
            _require(key in reference_keys, "missing corresponding reference boundary")
            _equal(
                reference_keys[key][0]["history_sha256"],
                metadata["history_sha256"],
                "matching supplied input history",
            )
            passed = _audit_stats(record, comparison, shape, plan["contract"]["comparison_policy"])
            if not passed and first_numeric_failure is None:
                first_numeric_failure = record
            if not record["stats"]["exact_equal"]:
                canonical = (
                    metadata["output_index"] if metadata["output_index"] is not None else -1,
                    metadata["depth"] or 0,
                    -1 if metadata["layer"] is None else metadata["layer"],
                    metadata["operation"],
                    metadata["positions"],
                )
                if earliest_nonexact is None or canonical < earliest_nonexact[0]:
                    earliest_nonexact = canonical, record
            if (
                metadata["operation"] == "logits"
                and comparison["comparison_kind"] != "bf16_fp32_sensitivity"
            ):
                index = metadata["output_index"]
                if actual_traces:
                    _equal(
                        record["stats"]["actual_top1_ids"],
                        [actual_traces[index]["actual_token_id"]],
                        "native genuine top1 trace",
                    )
                if reference_traces:
                    _equal(
                        record["stats"]["reference_top1_ids"],
                        [reference_traces[index]["actual_token_id"]],
                        "oracle genuine top1 trace",
                    )
        accumulate(summary, record)
    _equal(sorted(seen), sorted(expected), "complete executed boundary coverage")
    _require(stored["complete"] is True, "comparison stream is incomplete")
    _equal(summary["count"], len(expected), "dynamic executed-boundary count")
    if not comparison["counts_are_upper_bounds"]:
        _equal(summary["count"], comparison["expected_boundary_records"], "frozen boundary count")
    else:
        _require(
            summary["count"] <= comparison["expected_boundary_records"], "live depth bound exceeded"
        )
    if (
        comparison["family"] not in ("original", "official")
        and comparison["comparison_kind"] != "bf16_fp32_sensitivity"
    ):
        exit_policy = reference["case"]["exit_policy"]
        summary["behavior"] = compare_behavior(
            actual_traces,
            reference_traces,
            live=comparison["family"] == "live_gate",
            threshold=exit_policy["threshold"],
            route_control=exit_policy["mode"],
        )
    failures = [row for row in summary["behavior"] if row["status"] == "diverged"]
    summary["behavior_failures"] = len(failures)
    summary["first_behavior_failure"] = failures[0] if failures else None
    for key in _SUMMARY_FIELDS:
        _equal(stored[key], summary[key], f"reconstructed summary {key}")
    return {
        "comparison_id": comparison_id,
        "dtype": comparison["dtype"],
        "required": comparison["required"],
        "count": summary["count"],
        "compared_count": summary["compared_count"],
        "incomparable_count": summary["incomparable_count"],
        "required_failures": summary["required_failures"],
        "behavior_failures": len(failures),
        "first_failure": first_numeric_failure,
        "first_behavior_failure": summary["first_behavior_failure"],
        "earliest_nonexact_boundary": earliest_nonexact[1] if earliest_nonexact else None,
        "jsonl_sha256": file_record["sha256"],
        "size_bytes": file_record["size_bytes"],
    }


def _audit_case(result, plan, case, fixtures):
    _equal(
        [result["schema_version"], result["artifact_type"], result["plan_sha256"]],
        [1, "validation_case_result", plan["plan_sha256"]],
        "case result identity",
    )
    _equal(result["case"], case, "resolved case controls")
    _equal(result["status"], "complete", "case completion")
    _equal(result["failures"], [], "completed case failures")
    _equal(result["cleanup"], {"requests_remaining": 0, "used_blocks": 0}, "request/KV cleanup")
    steps = _integer(result["steps"], "steps", 1)
    _require(steps <= case["max_steps"], "case step budget exceeded")
    _require(
        _number(result["elapsed_s"], "elapsed seconds")
        <= plan["contract"]["limits"]["case_timeout_s"],
        "case timeout exceeded",
    )
    expected_comparisons = [
        row["comparison_id"]
        for row in plan["comparison_order"]
        if row["candidate_case_id"] == case["case_id"]
    ]
    _equal(result["comparisons"], expected_comparisons, "case comparison order/coverage")
    expected_counts = {}
    if case["implementation"] in ("oracle", "native"):
        _equal(sorted(result["traces"]), sorted(case["fixture_ids"]), "trace fixture coverage")
        for fixture_id in case["fixture_ids"]:
            _audit_traces(
                case, fixtures[fixture_id], result["traces"][fixture_id], plan["model_config"]
            )
            expected_counts[fixture_id] = len(
                _expected_boundaries(
                    case, fixtures[fixture_id], result["traces"][fixture_id], plan["model_config"]
                )
            )
    else:
        _equal(result["traces"], {}, "official/legacy unavailable generation traces")
        expected_counts = {
            key: 9 if case["implementation"] == "official" else 4 for key in case["fixture_ids"]
        }
    _equal(result["boundary_counts"], expected_counts, "executed boundary counts")
    _equal(result["observed_boundaries"], sum(expected_counts.values()), "global observer count")
    if case["implementation"] == "native":
        _require(
            isinstance(result["schedule"], list) and len(result["schedule"]) == steps,
            "native schedule summary coverage differs from steps",
        )
    return True


def _audit_environment(manifest, plan):
    environment, controls = manifest["environment"], plan["contract"]["controls"]
    _require(isinstance(environment, dict), "runtime environment is missing")
    _equal(
        environment["cuda_visible_devices"], str(controls["gpu_ids"][0]), "reserved physical GPU"
    )
    _equal(environment["logical_device"], "cuda:0", "logical device")
    scheduler = environment["scheduler"]
    _require(
        isinstance(scheduler, list) and len(scheduler) == 1, "one scheduler RUN record required"
    )
    _equal(str(scheduler[0]["gpu_id"]), str(controls["gpu_ids"][0]), "scheduler physical GPU")
    _equal(scheduler[0]["type"], "RUN", "scheduler reservation kind")
    _equal(scheduler[0]["user"], environment["account"], "scheduler reservation account")
    _equal(
        environment["actual_torch_threads"],
        {"intraop": controls["cpu_threads"], "interop": controls["interop_threads"]},
        "CPU threads",
    )
    _equal(environment["arithmetic"], plan["contract"]["arithmetic"], "actual arithmetic flags")
    _require(
        isinstance(environment["cpu_affinity"], list) and bool(environment["cpu_affinity"]),
        "CPU affinity was not recorded",
    )
    _require(
        isinstance(environment["numa_status"], list) and len(environment["numa_status"]) >= 2,
        "NUMA CPU/memory affinity was not recorded",
    )
    _equal(environment["python"], plan["dependencies"]["python"], "Python dependency freeze")
    _equal(
        environment["torch_cuda_version"],
        plan["dependencies"]["torch_cuda_build"],
        "Torch CUDA build",
    )
    _equal(
        manifest["official_provenance"],
        plan["dependencies"]["official"],
        "official dependency/source freeze",
    )
    official = manifest["official_provenance"]
    _equal(
        official["dependencies"]["transformers"],
        plan["contract"]["official"]["transformers_version"],
        "official Transformers compatibility prerequisite",
    )
    _equal(official["optional_kernels_present"], False, "official optional kernel replacement")
    for package in ("torch", "safetensors", "huggingface-hub"):
        _equal(
            environment["software"][package],
            official["dependencies"][package],
            "runtime software version",
        )
    _equal(
        manifest["teardown_after_workspace_release"],
        {"allocated_bytes": 0, "reserved_bytes": 0},
        "final task-owned CUDA teardown",
    )
    limits = plan["contract"]["limits"]
    _require(
        _number(manifest["elapsed_s"], "total elapsed seconds") <= limits["total_timeout_s"],
        "outer execution timeout exceeded",
    )
    _require(
        _integer(manifest["tensor_written_bytes"], "tensor-written bytes")
        <= limits["cumulative_spool_written_bytes"],
        "cumulative tensor-write budget exceeded",
    )
    _require(
        _integer(manifest["artifact_disk_bytes"], "artifact disk bytes")
        <= limits["total_disk_bytes"],
        "total artifact budget exceeded",
    )
    dumps = manifest["diagnostic_dumps"]
    selected = dumps["selected_fixture_ids"]
    _require(
        isinstance(selected, list) and len(set(selected)) == len(selected) and len(selected) <= 4,
        "dump fixture selection exceeds the declared budget",
    )
    _equal(
        selected[:2],
        plan["contract"]["diagnostics"]["preselected_dump_fixture_ids"],
        "preselected dump fixtures",
    )
    _require(
        _integer(dumps["written_bytes"], "dump bytes") <= limits["persisted_dump_bytes"],
        "dump byte cap exceeded",
    )
    _equal(
        sum(dumps["fixture_written_bytes"].values()), dumps["written_bytes"], "dump byte accounting"
    )
    for fixture_id, size in dumps["fixture_written_bytes"].items():
        _require(
            fixture_id in selected
            and _integer(size, "fixture dump bytes") <= limits["dump_bytes_per_fixture"],
            "per-fixture dump budget exceeded",
        )


def build_report(output_dir):
    """Read local artifacts only; never read a checkpoint, query CUDA, or retry cases."""
    folder = Path(output_dir).resolve()
    report = {
        "schema_version": 1,
        "artifact_type": "validation_qualification_report",
        "output_dir": str(folder),
        "plan_sha256": None,
        "evidence_status": "incomplete",
        "runtime_qualification": "inconclusive",
        "runtime_by_dtype": {},
        "milestone_status": "manual_diagnosis_required",
        "errors": [],
        "missing_cases": [],
        "missing_comparisons": [],
        "cases": [],
        "comparisons": [],
        "limitations": [
            (
                "This audit reconstructs coverage, summaries and decisions from preserved "
                "statistics, "
                "traces and hashes; it does not recompute elementwise tensor errors without the "
                "missing full native tensors."
            ),
            (
                "Hidden, populated-KV and gate numerical deltas and BF16/FP32 sensitivity remain "
                "diagnostic-only; original logit bounds are unchanged."
            ),
            (
                "Runtime gates alone cannot close Q1: independent-source review, CPU invariants, "
                "original-failure diagnosis and manual acceptance evidence remain required."
            ),
            (
                "A numerical pass does not promote BF16 defaults or establish task quality; "
                "Q2 remains separate."
            ),
            (
                "Candidate tensor dumps begin at selection and end at fixed byte limits; "
                "earlier tensors are not backfilled."
            ),
        ],
    }
    try:
        plan = read_json(folder / "plan.json")
        validate_plan_integrity(plan)
        report["plan_sha256"] = plan["plan_sha256"]
        manifest = read_json(folder / "manifest.json")
        _equal(
            [manifest["schema_version"], manifest["artifact_type"], manifest["plan_sha256"]],
            [1, "validation_manifest", plan["plan_sha256"]],
            "manifest identity",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["evidence_status"] = "invalid"
        report["errors"].append({"scope": "plan/manifest", "message": str(exc)})
        return report
    report["manifest_sha256"] = _file_record(folder / "manifest.json")["sha256"]
    report["manifest_status"] = manifest.get("status")
    report["execution_failures"] = manifest.get("failures", [])
    fixtures = _fixtures(plan)
    planned_cases = {row["case_id"]: row for row in plan["execution_order"]}
    case_order = list(planned_cases)
    completed = manifest.get("completed_cases", [])
    if not isinstance(completed, list) or completed != case_order[: len(completed)]:
        report["errors"].append(
            {"scope": "manifest", "message": "completed cases are not the exact planned prefix"}
        )
        completed = []
    available = {path.parent.name: path for path in (folder / "cases").glob("*/result.json")}
    for case_id in sorted(set(available) - set(planned_cases)):
        report["errors"].append({"scope": case_id, "message": "unplanned case artifact"})
    cases = {}
    valid_cases = set()
    for case_id, case in planned_cases.items():
        path = available.get(case_id)
        if path is None:
            report["missing_cases"].append(case_id)
            continue
        try:
            result = read_json(path)
            _equal(result["case"], case, "case identity")
            _equal(result["plan_sha256"], plan["plan_sha256"], "case plan hash")
            cases[case_id] = result
            if result.get("status") != "complete":
                report["missing_cases"].append(case_id)
                _require(case_id not in completed, "incomplete case listed as completed")
                continue
            _audit_case(result, plan, case, fixtures)
            _require(case_id in completed, "completed case omitted from manifest")
            valid_cases.add(case_id)
            report["cases"].append(
                {
                    "case_id": case_id,
                    "dtype": case["dtype"],
                    "status": "complete",
                    "sha256": _file_record(path)["sha256"],
                }
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            report["errors"].append({"scope": case_id, "message": str(exc)})
    planned_comparisons = {row["comparison_id"]: row for row in plan["comparison_order"]}
    observed_comparisons = {
        path.name.removesuffix(".jsonl") for path in (folder / "comparisons").glob("*.jsonl")
    }
    for comparison_id in sorted(observed_comparisons - set(planned_comparisons)):
        report["errors"].append(
            {"scope": comparison_id, "message": "unplanned comparison artifact"}
        )
    current_candidate, observations, boundary_indices = None, {}, {}

    def observe(index, fixture_id, key):
        identity = (fixture_id, key)
        _require(
            index not in observations or observations[index] == identity,
            "one global observation index refers to different boundaries",
        )
        _require(
            identity not in boundary_indices or boundary_indices[identity] == index,
            "one candidate boundary has different global observation indices",
        )
        observations[index] = identity
        boundary_indices[identity] = index

    def finish_observations():
        if current_candidate is None or current_candidate not in valid_cases:
            return
        expected_ids = {
            row["comparison_id"]
            for row in plan["comparison_order"]
            if row["candidate_case_id"] == current_candidate
        }
        audited_ids = {row["comparison_id"] for row in report["comparisons"]}
        if not expected_ids <= audited_ids:
            return
        count = cases[current_candidate]["observed_boundaries"]
        if sorted(observations) != list(range(1, count + 1)):
            report["errors"].append(
                {
                    "scope": current_candidate,
                    "message": "global observation coverage is not exactly 1..N",
                }
            )

    for comparison_id, comparison in planned_comparisons.items():
        if comparison["candidate_case_id"] != current_candidate:
            finish_observations()
            current_candidate = comparison["candidate_case_id"]
            observations.clear()
            boundary_indices.clear()
        if comparison_id not in observed_comparisons or any(
            comparison[key] not in valid_cases for key in ("reference_case_id", "candidate_case_id")
        ):
            report["missing_comparisons"].append(comparison_id)
            continue
        try:
            report["comparisons"].append(
                _audit_comparison(folder, plan, comparison, cases, fixtures, observe)
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            report["errors"].append({"scope": comparison_id, "message": str(exc)})
    finish_observations()
    complete = (
        manifest.get("status") == "complete"
        and not manifest.get("failures")
        and completed == case_order
        and not report["missing_cases"]
        and not report["missing_comparisons"]
    )
    if complete:
        try:
            _audit_environment(manifest, plan)
            if not report["errors"]:
                report["raw_evidence"] = _audit_raw_evidence(
                    folder, plan, cases, fixtures, report["comparisons"], manifest
                )
            total = 0
            for path in folder.rglob("*"):
                _require(not path.is_symlink(), "evidence directory contains a symbolic link")
                if path.is_file():
                    total += path.stat().st_size
            _require(
                total <= plan["contract"]["limits"]["total_disk_bytes"],
                "actual artifact disk cap exceeded",
            )
            report["audited_artifact_disk_bytes"] = total
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append(
                {"scope": "environment/raw-evidence/budgets/teardown", "message": str(exc)}
            )
    else:
        report["missing_environment_or_cleanup_qualification"] = True
    report["counts"] = {
        "planned_cases": len(planned_cases),
        "verified_cases": len(valid_cases),
        "planned_comparisons": len(planned_comparisons),
        "verified_comparisons": len(report["comparisons"]),
        "compared_boundaries": sum(row["compared_count"] for row in report["comparisons"]),
        "incomparable_boundaries": sum(row["incomparable_count"] for row in report["comparisons"]),
    }
    report["evidence_status"] = (
        "invalid" if report["errors"] else "complete" if complete else "incomplete"
    )
    for dtype in plan["contract"]["dtypes"]:
        rows = [row for row in report["comparisons"] if row["dtype"] == dtype and row["required"]]
        numerical = sum(row["required_failures"] for row in rows)
        behavioral = sum(row["behavior_failures"] for row in rows)
        decision = (
            "inconclusive"
            if report["evidence_status"] != "complete"
            else "failed"
            if numerical or behavioral
            else "passed"
        )
        report["runtime_by_dtype"][dtype] = {
            "decision": decision,
            "numerical_required_failures": numerical,
            "behavior_required_failures": behavioral,
        }
    if report["evidence_status"] == "complete":
        report["runtime_qualification"] = (
            "failed"
            if any(row["decision"] == "failed" for row in report["runtime_by_dtype"].values())
            else "passed"
        )
    return report


def write_report(output_dir):
    report = build_report(output_dir)
    folder = Path(output_dir).resolve()
    write_json(folder / "qualification.json", report)
    lines = [
        "# Q1 numerical evidence audit",
        "",
        f"Runtime qualification: **{report['runtime_qualification']}**.",
        "",
        (
            f"Evidence: **{report['evidence_status']}**. "
            "Q1 still requires manual diagnosis and acceptance review."
        ),
        "",
    ]
    for dtype, result in report["runtime_by_dtype"].items():
        lines.append(
            f"- {dtype}: {result['decision']}; "
            f"{result['numerical_required_failures']} numerical and "
            f"{result['behavior_required_failures']} behavioral required failures."
        )
    if report.get("counts"):
        counts = report["counts"]
        lines.extend(
            [
                "",
                (
                    f"Verified {counts['verified_cases']}/{counts['planned_cases']} cases and "
                    f"{counts['verified_comparisons']}/{counts['planned_comparisons']} "
                    "comparison trajectories."
                ),
            ]
        )
    if report["errors"]:
        lines.extend(["", "Evidence errors:", ""])
        lines.extend(f"- {row['scope']}: {row['message']}" for row in report["errors"][:20])
    lines.extend(["", *report["limitations"], ""])
    (folder / "qualification.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _stream_spool(directory, namespace, fixture_id, validator, *, expected_count, index_cap):
    """Check one retained bin with a single file-open and bounded SHA256 reads."""
    from .diagnostics import TensorSpool

    index_path = directory / namespace / (fixture_id + ".index.json")
    _require(
        index_path.stat().st_size <= expected_count * index_cap + 4096,
        "tensor index exceeds its record-count byte bound",
    )
    spool = TensorSpool.open(directory, namespace, fixture_id)
    _require(len(spool.index) == expected_count, "retained tensor index has missing/extra records")
    digest = hashlib.sha256()
    before = spool.data_path.stat()
    total = 0
    with spool.data_path.open("rb") as stream:
        for record in spool.index.values():
            validator(record)
            _require(
                len(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                )
                + 2
                <= index_cap,
                "tensor index record exceeds frozen serialized cap",
            )
            checksum = hashlib.sha256()
            remaining = record["size_bytes"]
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024**2))
                _require(bool(chunk), "truncated retained tensor payload")
                remaining -= len(chunk)
                total += len(chunk)
                digest.update(chunk)
                checksum.update(chunk)
            _equal(checksum.hexdigest(), record["sha256"], "retained tensor payload SHA256")
        _require(not stream.read(1), "unexpected trailing tensor payload")
    after = spool.data_path.stat()
    _equal(
        [before.st_size, before.st_mtime_ns],
        [after.st_size, after.st_mtime_ns],
        "stable retained tensor file",
    )
    spool.close()
    return {
        "namespace": namespace,
        "fixture_id": fixture_id,
        "records": expected_count,
        "size_bytes": total,
        "sha256": digest.hexdigest(),
        "index_sha256": _file_record(index_path)["sha256"],
    }


def _audit_raw_evidence(folder, plan, cases, fixtures, comparison_results, manifest):
    retained, dumps = [], []
    expected_pairs = set()
    limits = plan["contract"]["limits"]
    for case in plan["execution_order"]:
        retain = (
            case.get("retain_evidence", False)
            or case["implementation"] in ("oracle", "legacy_dense")
            or (case["implementation"] == "legacy_packed" and case["backend"] == "torch")
        )
        if not retain:
            continue
        result = cases[case["case_id"]]
        for fixture_id in case["fixture_ids"]:
            expected_pairs.add((case["case_id"], fixture_id))
            expected = _expected_boundaries(
                case,
                fixtures[fixture_id],
                result["traces"].get(fixture_id, []),
                plan["model_config"],
            )

            def validate(record):
                _require(record["key"] in expected, "unexpected retained reference boundary")
                metadata, shape = expected[record["key"]]
                _equal(record["metadata"], metadata, "retained reference metadata")
                _equal(record["shape"], shape, "retained reference shape")
                _equal(record["dtype"], case["dtype"], "retained reference dtype")

            retained.append(
                _stream_spool(
                    folder / "spools",
                    case["case_id"],
                    fixture_id,
                    validate,
                    expected_count=len(expected),
                    index_cap=limits["spool_index_record_bytes"],
                )
            )
    actual_pairs = {
        (path.parent.name, path.name.removesuffix(".index.json"))
        for path in (folder / "spools").glob("*/*.index.json")
    }
    actual_bins = {
        (path.parent.name, path.name.removesuffix(".bin"))
        for path in (folder / "spools").glob("*/*.bin")
    }
    _equal(sorted(actual_pairs), sorted(expected_pairs), "retained reference spool set")
    _equal(sorted(actual_bins), sorted(expected_pairs), "retained reference bin set")
    comparisons = {row["comparison_id"]: row for row in plan["comparison_order"]}
    hashes = {hashlib.sha256(key.encode()).hexdigest(): row for key, row in comparisons.items()}
    preselected = plan["contract"]["diagnostics"]["preselected_dump_fixture_ids"]
    selection = list(preselected)
    selection_points = {}
    case_rank = {row["case_id"]: index for index, row in enumerate(plan["execution_order"])}
    failures = []
    for row in comparison_results:
        failure = row["first_failure"]
        if failure is not None and failure["stats"]["all_finite"]:
            comparison = comparisons[row["comparison_id"]]
            failures.append(
                (
                    case_rank[comparison["candidate_case_id"]],
                    failure["observation_index"],
                    comparison["fixture_id"],
                )
            )
    for case_index, observation, fixture_id in sorted(failures):
        if fixture_id not in selection and len(selection) < 4:
            selection.append(fixture_id)
            selection_points[fixture_id] = (case_index, observation)
    _equal(
        manifest["diagnostic_dumps"]["selected_fixture_ids"],
        selection,
        "first finite failure dump selection",
    )
    paths = sorted((folder / "dumps" / "diagnostic").glob("*.index.json"))
    written = {}
    for path in paths:
        fixture_id = path.name.removesuffix(".index.json")
        _require(fixture_id in selection, "unselected diagnostic dump fixture")
        # A dump can contain at most two sides for each declared candidate comparison boundary.
        bound = 2 * sum(
            row["expected_boundary_records"]
            for row in comparisons.values()
            if row["fixture_id"] == fixture_id
        )
        _require(
            path.stat().st_size <= bound * limits["spool_index_record_bytes"] + 4096,
            "dump index exceeds the declared candidate-boundary bound",
        )
        index = read_json(path)
        count = len(index["records"])

        expected_cache, observed_cache, reference_cache = {}, {}, {}
        pending_pair = None

        def validate_dump(record):
            nonlocal pending_pair
            metadata = record["metadata"]
            comparison = hashes.get(metadata["comparison_sha256"])
            _require(
                comparison is not None and comparison["fixture_id"] == fixture_id,
                "dump comparison provenance is missing or wrong",
            )
            side = metadata["side"]
            _require(side in ("actual", "reference"), "unknown dump side")
            candidate = cases[comparison["candidate_case_id"]]
            reference = cases[comparison["reference_case_id"]]
            base = {
                key: value
                for key, value in metadata.items()
                if key not in ("side", "comparison_sha256")
            }
            comparison_id = comparison["comparison_id"]
            if comparison_id not in expected_cache:
                expected_cache[comparison_id] = _expected_boundaries(
                    candidate["case"],
                    fixtures[fixture_id],
                    candidate["traces"].get(fixture_id, []),
                    plan["model_config"],
                )
                observed_cache[comparison_id] = {
                    item["key"]: (item["observation_index"], item["status"])
                    for item in _read_records(
                        folder / "comparisons" / (comparison_id + ".jsonl"),
                        limits["comparison_record_bytes"],
                    )
                }
                from .diagnostics import TensorSpool

                reference_cache[comparison_id] = TensorSpool.open(
                    folder / "spools", comparison["reference_case_id"], fixture_id
                )
            expected = expected_cache[comparison_id]
            key = boundary_key(base)
            _require(key in expected, "dump boundary is outside the declared candidate trajectory")
            _equal(base, expected[key][0], "dump candidate metadata")
            _equal(record["shape"], expected[key][1], "dump shape")
            _equal(
                record["key"],
                hashlib.sha256((comparison_id + key + side).encode()).hexdigest(),
                "dump record key",
            )
            observation, status = observed_cache[comparison_id][key]
            _equal(status, "compared", "dump must have a comparable tensor pair")
            identity = (comparison_id, key)
            if side == "actual":
                _require(pending_pair is None, "missing reference half of diagnostic dump pair")
                pending_pair = identity
            else:
                _equal(pending_pair, identity, "adjacent actual/reference diagnostic dump pair")
                pending_pair = None
                _equal(
                    record["sha256"],
                    reference_cache[comparison_id].index[key]["sha256"],
                    "reference dump versus retained reference payload hash",
                )
            if fixture_id not in preselected:
                _require(
                    (case_rank[comparison["candidate_case_id"]], observation)
                    >= selection_points[fixture_id],
                    "failure fixture dump predates its first finite failure",
                )
            _equal(
                record["dtype"],
                (candidate if side == "actual" else reference)["case"]["dtype"],
                "dump dtype",
            )
            if fixture_id in preselected:
                _equal(
                    [comparison["dtype"], comparison["comparison_kind"]],
                    [
                        plan["contract"]["diagnostics"]["preselected_dump_dtype"],
                        plan["contract"]["diagnostics"]["preselected_dump_comparison_kind"],
                    ],
                    "preselected dump frozen comparison eligibility",
                )

        artifact = _stream_spool(
            folder / "dumps",
            "diagnostic",
            fixture_id,
            validate_dump,
            expected_count=count,
            index_cap=limits["spool_index_record_bytes"],
        )
        _require(pending_pair is None, "missing final reference diagnostic dump half")
        _require(
            artifact["size_bytes"] <= limits["dump_bytes_per_fixture"],
            "dump fixture byte cap exceeded",
        )
        written[fixture_id] = artifact["size_bytes"]
        dumps.append(artifact)
    dump_bins = {
        path.name.removesuffix(".bin") for path in (folder / "dumps" / "diagnostic").glob("*.bin")
    }
    _equal(sorted(dump_bins), sorted(written), "diagnostic index/bin set")
    for fixture_id in selection:
        eligible = fixture_id in selection_points or any(
            row["fixture_id"] == fixture_id
            and row["dtype"] == plan["contract"]["diagnostics"]["preselected_dump_dtype"]
            and row["comparison_kind"]
            == plan["contract"]["diagnostics"]["preselected_dump_comparison_kind"]
            for row in comparisons.values()
        )
        if eligible:
            _require(
                written.get(fixture_id, 0) > 0,
                "selected fixture lacks its required paired diagnostic capture",
            )
    _equal(written, manifest["diagnostic_dumps"]["fixture_written_bytes"], "preserved dump bytes")
    _equal(
        sum(written.values()),
        manifest["diagnostic_dumps"]["written_bytes"],
        "total preserved dump bytes",
    )
    _equal(
        sum(row["size_bytes"] for row in retained) + sum(written.values()),
        manifest["tensor_written_bytes"],
        "cumulative retained tensor byte accounting",
    )
    return {
        "retained_references": retained,
        "diagnostic_dumps": dumps,
        "payload_verification": (
            "each typed record SHA256 checked with bounded sequential binary reads"
        ),
    }
