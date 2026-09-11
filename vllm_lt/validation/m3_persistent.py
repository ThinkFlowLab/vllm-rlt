"""Same-shape eager storage validation; no graph or performance qualification."""

import json
import math
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from vllm_lt.request import Stage

from . import m3_inactive
from .m2 import _audit_numerical_view, _run_numerical_view
from .schema import _digest, read_json, write_json

STORAGE_EVIDENCE = {
    "row_count": 8,
    "table_width": 32,
    "max_live_rows": 4,
    "dispatch_record_bytes": 32768,
    "device_payload_bytes_max": 262144,
    "cpu_staging_bytes": 1256,
    "model_outputs": "temporary; copied into persistent destinations before independent gather",
    "publication": "independent live storage; actual request rows match returned hidden rows",
}
_METADATA = (
    "position_ids",
    "write_blocks",
    "write_offsets",
    "block_tables",
    "context_lengths",
    "active",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(actual, expected, message):
    _require(_digest(actual) == _digest(expected), message)


def build_model_plan(suite, contract, model_config):
    """Adapt the accepted 13-case numerical subset without changing its histories/gates."""
    base = m3_inactive.build_model_plan(suite, contract, model_config)

    def rename(value):
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, str):
            return value.replace("m3_inactive", "m3_persistent").replace(
                "m3-inactive-", "m3-persistent-"
            )
        return value

    plan = rename(base)
    for case in plan["execution_order"]:
        native = case["implementation"] == "native"
        case["padding"] = deepcopy(m3_inactive.PADDING) if native else {"mode": "compact"}
        case["storage_strategy"] = (
            ("allocating_padded" if case["implementation_id"] == "A" else "persistent_decode")
            if native
            else "oracle_compact"
        )
    records = sum(
        case["max_steps"] for case in plan["execution_order"] if case["implementation"] == "native"
    )
    plan["storage_evidence"] = deepcopy(STORAGE_EVIDENCE)
    plan["resource_estimates"].update(
        pointer_records_upper_bound=records,
        pointer_evidence_bytes_upper_bound=records * STORAGE_EVIDENCE["dispatch_record_bytes"],
    )
    del plan["numerical_plan_sha256"]
    plan["numerical_plan_sha256"] = _digest(plan)
    return plan


def validate_model_plan(plan):
    expected = build_model_plan(
        plan["source_inputs"]["suite"], plan["source_inputs"]["contract"], plan["model_config"]
    )
    _equal(plan, expected, "persistent model plan differs from the frozen same-shape subset")


def model_view(parent):
    validate_model_plan(parent["numerical"])
    return {**parent["numerical"], "plan_sha256": parent["plan_sha256"]}


def _descriptor(tensor):
    return {
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "size_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
    }


@contextmanager
def _replace(obj, name, replacement):
    existed, previous = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


@contextmanager
def _storage_observation(engine, case, observations):
    """Record actual call arguments/publication; snapshots supply owned-buffer inventory."""
    from contextlib import ExitStack

    runner, model = engine.model_runner, engine.model
    persistent = case["storage_strategy"] == "persistent_decode"
    if persistent:
        del runner._recurrent  # Remove only the allocating validation override.
        runner._enable_persistent_decode()
    current = None
    core, execute = model._recurrent_prepared, runner.execute
    recurrence_name = "_recurrent_persistent" if persistent else "_recurrent"
    recurrence = getattr(runner, recurrence_name)

    def observed_core(hidden, batch, cache):
        if current is None:
            return core(hidden, batch, cache)
        _require("model_input" not in current, "multiple model calls in one storage dispatch")
        current["model_input"] = _descriptor(hidden)
        current["metadata"] = {name: _descriptor(getattr(batch, name)) for name in _METADATA}
        if persistent:
            current["in_flight"] = runner._persistent_snapshot()
        outputs = core(hidden, batch, cache)
        current["model_outputs"] = dict(zip(("hidden", "gates"), map(_descriptor, outputs)))
        return outputs

    def observed_recurrence(*args, **kwargs):
        outputs = recurrence(*args, **kwargs)
        _require(current is not None, "publication outside an observed recurrent dispatch")
        _require("publication" not in current, "multiple publications in one storage dispatch")
        current["publication"] = dict(zip(("hidden", "gates"), map(_descriptor, outputs)))
        return outputs

    def observed_execute(batch):
        nonlocal current
        if batch.stage != Stage.RECURRENT:
            return execute(batch)
        _require(current is None, "overlapping storage dispatch observation")
        _require(len(observations) < case["max_steps"], "storage dispatch count exceeds case cap")
        requests = [item.request for item in batch.items]
        count = len(requests)
        _require(1 <= count <= 4, "numerical storage matrix requires one to four live rows")
        current = {
            "dispatch_index": len(observations) + 1,
            "strategy": case["storage_strategy"],
            "logical": {
                "request_ids": [request.request_id for request in requests],
                "depths": [request.loops_done + 1 for request in requests],
                "positions": [request.position for request in requests],
                "live_rows": [2 * index + 1 for index in range(count)],
                "row_count": 8,
                "table_width": 32,
            },
            "before": runner._persistent_snapshot() if persistent else None,
            "in_flight": None,
            "after": None,
            "status": "incomplete",
        }
        try:
            result = execute(batch)
            _require(
                "publication" in current and "model_input" in current,
                "missing storage call evidence",
            )
            current["request_hidden"] = [_descriptor(request.hidden_state) for request in requests]
            current["returned_gate_count"] = len(result)
            current["after"] = runner._persistent_snapshot() if persistent else None
            current["status"] = "complete"
            return result
        finally:
            record, current = current, None
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            _require(
                len(encoded.encode()) <= STORAGE_EVIDENCE["dispatch_record_bytes"],
                "storage dispatch record exceeds 32KiB cap",
            )
            observations.append(record)

    with ExitStack() as stack:
        stack.enter_context(_replace(model, "_recurrent_prepared", observed_core))
        stack.enter_context(_replace(runner, recurrence_name, observed_recurrence))
        stack.enter_context(_replace(runner, "execute", observed_execute))
        yield


def execute_model_case(model, view, case, output, budget, dumps, deadline):
    from .runner import execute_case

    padding, storage = [], []
    with m3_inactive._padding_execution(
        case, padding, runner_context=lambda engine: _storage_observation(engine, case, storage)
    ):
        try:
            execute_case(model, view, case, output, budget, dumps, deadline)
        finally:
            path = Path(output) / "cases" / case["case_id"] / "result.json"
            if path.exists():
                result = read_json(path)
                result.update(padding_observations=padding, storage_observations=storage)
                if result["status"] == "complete":
                    try:
                        m3_inactive._audit_padding_observations(case, result)
                        _audit_storage_case(
                            case,
                            result,
                            model.config.hidden_size,
                            expected_device=str(next(model.parameters()).device),
                        )
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        result["status"] = "failed"
                        result["failures"].append(
                            {
                                "type": type(exc).__name__,
                                "phase": "storage_audit",
                                "message": str(exc),
                            }
                        )
                        write_json(path, result)
                        raise
                write_json(path, result)
    return result


def run_model_rows(model, parent, implementation, output_dir, deadline_ns, *, after_case=None):
    return _run_numerical_view(
        model,
        model_view(parent),
        implementation,
        output_dir,
        deadline_ns,
        execute=execute_model_case,
        after_case=after_case,
    )


def _check_descriptor(value, shape, dtype, *, device=None, owned=False):
    _equal(
        sorted(value),
        sorted(
            (
                "data_ptr",
                "storage_ptr",
                "shape",
                "stride",
                "dtype",
                "device",
                "size_bytes",
                "storage_bytes",
            )
        ),
        "tensor descriptor fields differ",
    )
    _equal(value["shape"], shape, "tensor shape differs")
    _equal(value["dtype"], "torch." + dtype, "tensor dtype differs")
    width = {"float32": 4, "int64": 8, "int32": 4, "bool": 1}[dtype]
    stride = [math.prod(shape[index + 1 :]) for index in range(len(shape))]
    _equal(value["stride"], stride, "tensor stride differs")
    size = math.prod(shape) * width
    _require(value["size_bytes"] == size, "tensor byte size differs")
    for key in ("data_ptr", "storage_ptr", "storage_bytes"):
        _require(type(value[key]) is int and value[key] > 0, "invalid tensor pointer/storage bytes")
    _require(
        value["storage_ptr"] <= value["data_ptr"]
        and value["data_ptr"] + size <= value["storage_ptr"] + value["storage_bytes"],
        "tensor exceeds storage",
    )
    if owned:
        _require(
            value["data_ptr"] == value["storage_ptr"] and value["storage_bytes"] == size,
            "owned tensor has extra or aliased storage",
        )
    _require(
        isinstance(value["device"], str) and value["device"] in ("cpu", "cuda:0"),
        "unsupported tensor device",
    )
    if device is not None:
        _equal(value["device"], device, "tensor device differs")


def _layouts(hidden_size):
    return {
        "hidden_in": ([8, hidden_size], "float32"),
        "hidden_out": ([8, hidden_size], "float32"),
        "gate_out": ([8], "float32"),
        "live_indices": ([4], "int64"),
        "position_ids": ([8], "int64"),
        "write_blocks": ([8], "int64"),
        "write_offsets": ([8], "int64"),
        "block_tables": ([8, 32], "int32"),
        "context_lengths": ([8], "int32"),
        "active": ([8], "bool"),
    }


def _audit_storage_case(case, evidence, hidden_size, *, expected_device="cuda:0"):
    observations = evidence["storage_observations"]
    expected = [row for row in evidence["schedule"] if row["stage"] == "recurrent"]
    if case["implementation"] == "oracle":
        _require(not observations, "compact oracle contains persistent pointer observations")
        return 0
    _require(
        len(observations) == len(expected) and expected,
        "missing or duplicate storage dispatch coverage",
    )
    layouts = _layouts(hidden_size)
    stable = None
    for index, (record, scheduled) in enumerate(zip(observations, expected), 1):
        _require(
            len(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            <= STORAGE_EVIDENCE["dispatch_record_bytes"],
            "storage record exceeds frozen byte cap",
        )
        _require(
            record["dispatch_index"] == index and record["status"] == "complete",
            "storage dispatch index/status differs",
        )
        _equal(record["strategy"], case["storage_strategy"], "storage strategy differs")
        count = len(scheduled["request_ids"])
        logical = {
            "request_ids": scheduled["request_ids"],
            "depths": scheduled["depths_after_step"],
            "positions": scheduled["positions_after_step"],
            "live_rows": [2 * i + 1 for i in range(count)],
            "row_count": 8,
            "table_width": 32,
        }
        _equal(record["logical"], logical, "storage logical rows differ from schedule")
        _require(
            1 <= count <= 4 and max(logical["positions"]) // case["block_size"] < 32,
            "unqualified dispatch support shape",
        )
        device = record["model_input"]["device"]
        _check_descriptor(
            record["model_input"], [8, hidden_size], "float32", device=expected_device
        )
        _equal(
            sorted(record["metadata"]), sorted(_METADATA), "model metadata pointer coverage differs"
        )
        for name in _METADATA:
            _check_descriptor(record["metadata"][name], *layouts[name], device=device)
        for name, shape in (("hidden", [8, hidden_size]), ("gates", [8])):
            _check_descriptor(record["model_outputs"][name], shape, "float32", device=device)
        for name, shape in (("hidden", [count, hidden_size]), ("gates", [count])):
            _check_descriptor(
                record["publication"][name], shape, "float32", device=device, owned=True
            )
        _require(
            record["returned_gate_count"] == count and len(record["request_hidden"]) == count,
            "publication row/gate count differs",
        )
        published = record["publication"]["hidden"]
        for physical, value in enumerate(record["request_hidden"]):
            _check_descriptor(value, [hidden_size], "float32", device=device)
            _equal(
                value,
                {
                    **published,
                    "shape": [hidden_size],
                    "stride": [1],
                    "size_bytes": hidden_size * 4,
                    "data_ptr": published["data_ptr"] + physical * hidden_size * 4,
                },
                "actual request hidden state differs from published row",
            )
        if case["storage_strategy"] == "allocating_padded":
            _equal(
                [record["before"], record["in_flight"], record["after"]],
                [None, None, None],
                "allocating baseline claims persistent snapshots",
            )
            owned_outputs = list(record["model_outputs"].values())
        else:
            snapshots = [record[key] for key in ("before", "in_flight", "after")]
            for snapshot in snapshots:
                _require(
                    snapshot["enabled"] is True and snapshot["failure"] is None,
                    "persistent snapshot is disabled/failed",
                )
                _equal(
                    snapshot["capacity"],
                    {"row_count": 8, "table_width": 32, "max_live_rows": 4},
                    "persistent capacity differs",
                )
                _equal(
                    sorted(snapshot["tensors"]),
                    sorted(layouts),
                    "persistent tensor inventory differs",
                )
                _equal(
                    sorted(snapshot["staging_tensors"]),
                    sorted(_METADATA),
                    "persistent staging inventory differs",
                )
                for name, desc in snapshot["tensors"].items():
                    _check_descriptor(desc, *layouts[name], device=device, owned=True)
                for name, desc in snapshot["staging_tensors"].items():
                    _require(name in layouts, "unexpected staging tensor")
                    _check_descriptor(desc, *layouts[name], device="cpu", owned=True)
                _require(
                    snapshot["device_payload_bytes"]
                    == sum(x["size_bytes"] for x in snapshot["tensors"].values()),
                    "persistent device bytes differ",
                )
                _require(
                    snapshot["cpu_staging_bytes"]
                    == sum(x["size_bytes"] for x in snapshot["staging_tensors"].values()),
                    "persistent staging bytes differ",
                )
                _require(
                    snapshot["device_payload_bytes"] <= STORAGE_EVIDENCE["device_payload_bytes_max"]
                    and snapshot["cpu_staging_bytes"] == STORAGE_EVIDENCE["cpu_staging_bytes"],
                    "persistent storage exceeds frozen byte bounds",
                )
                pointers = [item["storage_ptr"] for item in snapshot["tensors"].values()]
                _require(
                    len(set(pointers)) == len(pointers), "owned persistent tensors alias each other"
                )
                inventory = {
                    name: snapshot[name]
                    for name in (
                        "tensors",
                        "staging_tensors",
                        "device_payload_bytes",
                        "cpu_staging_bytes",
                    )
                }
                if stable is None:
                    stable = inventory
                _equal(inventory, stable, "persistent pointers/layout/bytes changed within case")
                _equal(
                    snapshot["fallback_counts"],
                    {"live_count": 0, "table_width": 0},
                    "ordinary 8x32 case used a fallback",
                )
            before, during, after = snapshots
            _equal(
                [before["status"], during["status"], after["status"]],
                ["ready", "in_flight", "ready"],
                "persistent dispatch lease phases differ",
            )
            _equal(
                [before["generation"], during["generation"], after["generation"]],
                [index - 1, index, index],
                "persistent generation differs",
            )
            for snapshot, calls, completed in (
                (before, index - 1, index - 1),
                (during, index, index - 1),
                (after, index, index),
            ):
                _equal(
                    snapshot["counters"],
                    {"calls": calls, "prepared": calls, "completed": completed, "empty": 0},
                    "persistent dispatch accounting differs",
                )
            _equal(
                record["model_input"],
                during["tensors"]["hidden_in"],
                "actual model input is not persistent hidden storage",
            )
            for name in _METADATA:
                _equal(
                    record["metadata"][name],
                    during["tensors"][name],
                    "actual metadata is not persistent storage",
                )
            dispatch = {
                "kind": "persistent",
                **logical,
                "depths": [value - 1 for value in logical["depths"]],
                "generation": index,
            }
            _equal(
                after["last_dispatch"],
                dispatch,
                "persistent last dispatch logical identity differs",
            )
            _equal(during["last_dispatch"], dispatch, "in-flight logical identity differs")
            _equal(
                after["last_publication"],
                record["publication"],
                "snapshot publication differs from actual returned tensors",
            )
            owned_outputs = list(after["tensors"].values())
        for value in record["publication"].values():
            _require(
                all(value["storage_ptr"] != owner["storage_ptr"] for owner in owned_outputs),
                "published output aliases reusable or physical storage",
            )
    return len(observations)


def audit_model_rows(output_dir, parent, *, expected_device="cuda:0"):
    view = model_view(parent)
    result = _audit_numerical_view(output_dir, view)
    m3_inactive._add_model_counts(view, result)
    result["counts"].update(
        verified_storage_dispatches=0,
        verified_persistent_dispatches=0,
        verified_allocating_padded_dispatches=0,
    )
    if not result["complete"]:
        return result
    try:
        for case in view["execution_order"]:
            evidence = read_json(
                Path(output_dir) / "numerical/cases" / case["case_id"] / "result.json"
            )
            m3_inactive._audit_padding_observations(case, evidence)
            count = _audit_storage_case(
                case, evidence, view["model_config"]["hidden_size"], expected_device=expected_device
            )
            result["counts"]["verified_storage_dispatches"] += count
            if case["storage_strategy"] != "oracle_compact":
                label = (
                    "persistent"
                    if case["storage_strategy"] == "persistent_decode"
                    else "allocating_padded"
                )
                result["counts"][f"verified_{label}_dispatches"] += count
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["complete"] = result["passed"] = False
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    return result
