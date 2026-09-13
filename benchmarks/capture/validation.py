"""Projected post-dispatch numerical evidence for the bounded graph prerequisite.

The oracle is independent. Native capture never installs Python layer hooks;
only returned loop/gate outputs, eager coda logits, and completed KV are observed.
"""

import math
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from vllm_lt.core.kv_cache_manager import decode_live_rows
from vllm_lt.models.config import OuroConfig
from vllm_lt.request import Stage
from vllm_lt.validation import m3_persistent
from vllm_lt.validation.m2 import _audit_numerical_view, _run_numerical_view, build_numerical_plan
from vllm_lt.validation.schema import PROJECTION, _digest, read_json, write_json
from vllm_lt.worker.decode_buffers import DecodeBucketLayout
from vllm_lt.worker.graph_diagnostics import REPLAY_ATOL, SCRATCH_PAGES, SETUP_WARMUPS, GraphLimits

OPERATIONS = ("loop_hidden", "gate_logits", "logits", "populated_kv")
NUMERICAL_LAYOUT = DecodeBucketLayout()
BUCKETS = {
    "row_counts": list(NUMERICAL_LAYOUT.row_counts),
    "table_width": NUMERICAL_LAYOUT.table_width,
    "max_live_rows": NUMERICAL_LAYOUT.max_num_seqs,
    "mapping": "odd",
}
GRAPH_LIMITS = asdict(GraphLimits())

EVIDENCE = {
    "projection": PROJECTION,
    "operations": list(OPERATIONS),
    "omitted": "six per-layer hidden/Q/K/V diagnostic hooks; no claimed observations",
    "selection": "last prompt query and eight continuation queries at every actual loop",
    "kv": "all populated positions, all four materialized depths and physical layers, K and V",
    "kv_chunk_positions": 4,
    "dispatch_record_bytes": 65536,
    "sampling": "validation only; no instrumentation in ordinary timing",
}


_require = m3_persistent._require
_descriptor = m3_persistent._descriptor


def _rename(value):
    if isinstance(value, dict):
        return {k: _rename(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v) for v in value]
    if isinstance(value, str):
        return value.replace("m3_persistent", "m3_capture").replace("m3-persistent-", "m3-capture-")
    return value


def projected_stats(fixture, config, *, live=False):
    """Closed-form full-KV and projected selected-query bound; no device access."""
    depth_sum = 36 if live else sum(fixture["forced_exit_depths"])
    capacity = len(fixture["prompt_token_ids"]) + 8
    kv_records = 2 * math.ceil(capacity / 4)
    selected_bytes = 4 * (config["hidden_size"] + 1) * depth_sum + 9 * config["vocab_size"] * 4
    kv_bytes = (
        2
        * 4
        * config["num_hidden_layers"]
        * capacity
        * config["num_key_value_heads"]
        * config["head_dim"]
        * 4
    )
    return {
        "records": 2 * depth_sum + 9 + kv_records,
        "selected_records": 2 * depth_sum + 9,
        "kv_records": kv_records,
        "selected_bytes": selected_bytes,
        "kv_bytes": kv_bytes,
        "bytes": selected_bytes + kv_bytes,
        "counts_are_upper_bounds": live,
    }


def build_model_plan(suite, contract, model_config):
    plan = _rename(m3_persistent.build_model_plan(suite, contract, model_config))
    config = OuroConfig.from_dict(model_config).to_dict()
    original = build_numerical_plan(suite, contract, model_config)
    extra = []
    for side in ("A", "B"):
        row = deepcopy(
            next(
                c
                for c in original["execution_order"]
                if c["implementation_id"] == side
                and c["implementation"] == "native"
                and c["backend"] == "torch"
                and c["fixture_ids"] == ["Q1-L16-F0"]
                and c["family"] == "main"
            )
        )
        row["case_id"] = row["case_id"].replace("m2-", "m3-capture-", 1)
        row["spool_group"] = row["case_id"]
        extra.append(row)
    plan["execution_order"] = [
        c
        for side in ("A", "B")
        for c in [
            *[r for r in plan["execution_order"] if r["implementation_id"] == side],
            extra[0 if side == "A" else 1],
        ]
    ]
    extra_ids = {c["case_id"] for c in extra}
    for comparison in original["comparison_order"]:
        if comparison["candidate_case_id"].replace("m2-", "m3-capture-", 1) not in extra_ids:
            continue
        row = deepcopy(comparison)
        for name in ("comparison_id", "reference_case_id", "candidate_case_id"):
            row[name] = (
                row[name]
                .replace("m2-", "m3-capture-", 1)
                .replace("implementation_exact", "implementation_fidelity")
            )
        row["comparison_kind"] = row["comparison_kind"].replace(
            "implementation_exact", "implementation_fidelity"
        )
        row["require_exact"] = False
        plan["comparison_order"].append(row)
    by_id = {f["fixture_id"]: f for f in plan["suite"]["fixtures"]}
    order = {c["case_id"]: i for i, c in enumerate(plan["execution_order"])}
    for case in plan["execution_order"]:
        case["boundary_projection"] = PROJECTION
        native = case["implementation"] == "native"
        case["padding"] = (
            deepcopy(BUCKETS) if native and case["backend"] == "triton" else {"mode": "compact"}
        )
        case["storage_strategy"] = (
            "oracle_compact"
            if not native
            else "backend_fallback"
            if case["backend"] == "torch"
            else "graph_replay"
            if case["implementation_id"] == "B"
            else "eager_tensor_body"
        )
    for row in plan["comparison_order"]:
        stats = projected_stats(by_id[row["fixture_id"]], config, live=row["family"] == "live_gate")
        row["expected_boundary_records"] = stats["records"]
        row["counts_are_upper_bounds"] = stats["counts_are_upper_bounds"]
    plan["comparison_order"].sort(key=lambda c: (order[c["candidate_case_id"]], c["comparison_id"]))
    plan["contract"]["limits"]["implementation_executions"] = 15
    plan["contract"]["diagnostics"]["operations"] = list(OPERATIONS[:-1])
    plan["boundary_evidence"] = deepcopy(EVIDENCE)
    plan["graph_limits"] = deepcopy(GRAPH_LIMITS)
    plan.pop("storage_evidence", None)
    retained = indexes = largest = 0
    for case in plan["execution_order"]:
        if not case["retain_evidence"]:
            continue
        stats = [
            projected_stats(by_id[fid], config, live=case["family"] == "live_gate")
            for fid in case["fixture_ids"]
        ]
        size = sum(s["bytes"] for s in stats)
        retained += size
        indexes += sum(s["records"] for s in stats)
        largest = max(largest, size)
    pointer_records = sum(
        c["max_steps"] for c in plan["execution_order"] if c["implementation"] == "native"
    )
    plan["resource_estimates"] = {
        "cases": 15,
        "comparisons": 31,
        "validation_cases": 13,
        "feasibility_cases": 2,
        "validation_comparisons": 30,
        "retained_tensor_bytes_upper_bound": retained,
        "retained_index_records_upper_bound": indexes,
        "comparison_records_upper_bound": sum(
            c["expected_boundary_records"] for c in plan["comparison_order"]
        ),
        "max_retained_case_bytes": largest,
        "pointer_records_upper_bound": pointer_records,
        "pointer_evidence_bytes_upper_bound": pointer_records * EVIDENCE["dispatch_record_bytes"],
    }
    limits = plan["contract"]["limits"]
    _require(
        len(plan["execution_order"]) == 15 and len(plan["comparison_order"]) == 31,
        "capture numerical matrix differs from 15 cases/31 streams",
    )
    _require(
        largest <= limits["group_spool_bytes"]
        and retained + limits["persisted_dump_bytes"] <= limits["cumulative_spool_written_bytes"],
        "capture projected evidence exceeds tensor caps",
    )
    plan.pop("numerical_plan_sha256")
    plan["numerical_plan_sha256"] = _digest(plan)
    return plan


def validate_model_plan(plan):
    expected = build_model_plan(
        plan["source_inputs"]["suite"], plan["source_inputs"]["contract"], plan["model_config"]
    )
    _require(
        _digest(plan) == _digest(expected),
        "capture plan differs from frozen projected 15-case subset",
    )


def model_view(parent):
    validate_model_plan(parent["numerical"])
    return {**parent["numerical"], "plan_sha256": parent["plan_sha256"]}


@contextmanager
def observe_projected(engine, sink, *, on_completed_kv=None, observations=None):
    """Observe returned tensors outside capture; install no Module forward hooks."""
    import torch

    from vllm_lt.validation.native import history_hash

    observations = [] if observations is None else observations
    runner, model = engine.model_runner, engine.model
    current = None
    recurrent, decode, execute = model.recurrent, runner._recurrent, runner.execute
    coda, finish = model.coda, engine.scheduler.finish

    def emit(operation, values, rows):
        _require(values.shape[0] == len(rows), "projected tensor rows differ from scheduled input")
        for index, (request_id, position, depth) in enumerate(rows):
            fixture = engine.fixtures[request_id]
            output_index = position - len(fixture["prompt_token_ids"]) + 1
            if not 0 <= output_index <= 8:
                continue
            request = engine.scheduler.requests[request_id]
            metadata = {
                "fixture_id": request_id,
                "operation": operation,
                "positions": [position],
                "depth": depth,
                "layer": None,
                "output_index": output_index,
                "history_sha256": history_hash(
                    request.prompt_token_ids + request.generated_token_ids
                ),
            }
            value = values[index]
            if operation == "gate_logits":
                engine.gates[request_id, output_index, depth] = {
                    "logit": float(value.float().item()),
                    "probability": float(value.float().sigmoid().item()),
                    "cdf": None,
                }
            elif operation == "logits":
                top = value.float().topk(2).values.cpu().tolist()
                engine.current_logits[request_id] = {"top_two_margin": top[0] - top[1]}
            sink(metadata, value)

    def observed_recurrent(hidden, request_ids, depths, positions, cache):
        outputs = recurrent(hidden, request_ids, depths, positions, cache)
        # Decode fallback can call the public model seam; its output is observed
        # exactly once at runner._recurrent, independently of that implementation.
        if current is None or current["stage"] != Stage.RECURRENT:
            rows = list(zip(request_ids, map(int, positions), (int(d) + 1 for d in depths)))
            emit("loop_hidden", outputs[0], rows)
            emit("gate_logits", outputs[1], rows)
        return outputs

    def observed_decode(hidden, request_ids, depths, positions):
        _require(
            current is not None and current["stage"] == Stage.RECURRENT,
            "decode outside scheduled observation",
        )
        rows = list(zip(request_ids, map(int, positions), (int(d) + 1 for d in depths)))
        _require(
            rows == current["rows"] and "publication" not in current,
            "decode ownership/order differs from scheduled input",
        )
        outputs = decode(hidden, request_ids, depths, positions)
        current["publication"] = dict(zip(("hidden", "gates"), map(_descriptor, outputs)))
        emit("loop_hidden", outputs[0], rows)
        emit("gate_logits", outputs[1], rows)
        return outputs

    def observed_coda(hidden):
        _require(
            current is not None and current["stage"] == Stage.CODA,
            "coda outside scheduled observation",
        )
        result = coda(hidden)
        emit("logits", result, current["rows"])
        return result

    def observed_execute(batch):
        nonlocal current
        _require(current is None, "overlapping projected execution")
        requests = [item.request for item in batch.items]
        current = {
            "stage": batch.stage,
            "rows": [
                (r.request_id, r.position, r.loops_done + int(batch.stage == Stage.RECURRENT))
                for r in requests
            ],
        }
        event = None
        if batch.stage == Stage.RECURRENT:
            event = {
                "dispatch_index": len(observations) + 1,
                "logical": {
                    "request_ids": [r.request_id for r in requests],
                    "positions": [r.position for r in requests],
                    "depths": [r.loops_done + 1 for r in requests],
                },
                "before": runner._graph_snapshot(),
                "status": "incomplete",
            }
        try:
            result = execute(batch)
            if event is not None:
                _require("publication" in current, "missing actual decode publication")
                event.update(
                    after=runner._graph_snapshot(),
                    publication=current["publication"],
                    request_hidden=[_descriptor(r.hidden_state) for r in requests],
                    returned_gate_count=len(result),
                    physical=None,
                )
                selected = event["after"]["last_dispatch"].get("bucket_id")
                if selected is not None:
                    bucket = runner.decode_executor.buckets[selected]
                    tensors = bucket["tensors"]
                    live = event["after"]["last_dispatch"]["live_rows"]
                    active = torch.zeros(
                        tensors["active"].shape, dtype=torch.bool, device=tensors["active"].device
                    )
                    active[live] = True
                    physical = {"active_mask_matches": bool(torch.equal(active, tensors["active"]))}
                    for name in ("hidden_out", "gate_out"):
                        physical[name + "_finite"] = bool(tensors[name].isfinite().all())
                        physical[name + "_inactive_zero"] = not bool(
                            torch.count_nonzero(tensors[name][~active])
                        )
                    _require(all(physical.values()), "physical captured output isolation failed")
                    event["physical"] = physical
                event["status"] = "complete"
            return result
        finally:
            current = None
            if event is not None:
                observations.append(event)

    def observed_finish(request, reason):
        if reason == "length" and on_completed_kv is not None:
            on_completed_kv(request.request_id, engine.cache_manager)
        return finish(request, reason)

    with ExitStack() as stack:
        for obj, name, value in (
            (model, "recurrent", observed_recurrent),
            (runner, "_recurrent", observed_decode),
            (model, "coda", observed_coda),
            (runner, "execute", observed_execute),
            (engine.scheduler, "finish", observed_finish),
        ):
            stack.enter_context(m3_persistent._replace(obj, name, value))
        yield


@contextmanager
def _execution(case, observations, lifecycle, limits):
    from vllm_lt.validation import runner

    original_engine, original_observer = runner.ValidationEngine, runner.observe_native
    engines = []

    class CaptureEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            engines.append(self)

            # This frozen numerical contract qualifies the original 4/8-row
            # shapes. New scheduler-sized experiments use their own plan.
            self.model_runner._decode_layout = NUMERICAL_LAYOUT
            self.model_runner._enable_recurrent_graph(
                use_graphs=case["implementation_id"] == "B", limits=limits
            )
            lifecycle["setup"] = self.model_runner._graph_snapshot()
            _require(
                self.model_runner.decode_executor is not None,
                "graph setup declined its budget; qualification requires the executor",
            )
            self.model_runner.decode_executor.record_dispatch_tensors = True

    runner.ValidationEngine = CaptureEngine
    runner.observe_native = lambda *args, **kwargs: observe_projected(
        *args, observations=observations, **kwargs
    )
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        runner.ValidationEngine, runner.observe_native = original_engine, original_observer
        try:
            for engine in engines:
                try:
                    lifecycle["before_close"] = engine.model_runner._graph_snapshot()
                    engine.model_runner._close_recurrent_graph()
                    lifecycle["after_close"] = engine.model_runner._graph_snapshot()
                except BaseException as error:
                    lifecycle["close_error"] = {"type": type(error).__name__, "message": str(error)}
                    if primary is None:
                        raise
                    if hasattr(primary, "add_note"):
                        primary.add_note(
                            f"graph cleanup also failed: {type(error).__name__}: {error}"
                        )
        finally:
            engines.clear()


def execute_model_case(model, view, case, output, budget, dumps, deadline):
    from vllm_lt.validation.runner import execute_case

    observations, lifecycle = [], {}
    failure = None
    try:
        with _execution(case, observations, lifecycle, view["graph_limits"]):
            execute_case(model, view, case, output, budget, dumps, deadline)
    except BaseException as error:
        failure = error
        raise
    finally:
        path = Path(output) / "cases" / case["case_id"] / "result.json"
        if path.exists():
            evidence = read_json(path)
            evidence.update(graph_observations=observations, graph_lifecycle=lifecycle)
            if failure is not None:
                evidence["status"] = "failed"
                evidence["failures"].append(
                    {
                        "type": type(failure).__name__,
                        "message": str(failure),
                        "phase": "capture_adapter",
                    }
                )
            write_json(path, evidence)
    if evidence["status"] == "complete":
        try:
            _audit_graph_case(
                case,
                evidence,
                view["model_config"],
                expected_device=str(next(model.parameters()).device),
            )
        except (ValueError, KeyError, TypeError) as error:
            evidence["status"] = "failed"
            evidence["failures"].append(
                {"type": type(error).__name__, "message": str(error), "phase": "graph_evidence"}
            )
            write_json(path, evidence)
            raise
    return evidence


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


def _graph_io(bucket):
    tensors = bucket["tensors"]
    return (
        {"hidden": tensors["hidden_in"], **{k: tensors[k] for k in m3_persistent._METADATA}},
        {"hidden": tensors["hidden_out"], "gates": tensors["gate_out"]},
    )


def _audit_setup_memory(setup, replay, limits):
    before, after = setup["memory_baseline"], setup["memory_after"]
    for memory in (before, after):
        _require(all(type(v) is int and v >= 0 for v in memory.values()), "integer setup memory")
        _require(memory["reserved_bytes"] >= memory["allocated_bytes"], "setup reserved memory")
    if not replay:
        _require(after == before, "eager baseline performed graph setup")
    deltas = {}
    for kind, cap_prefix in (("retained", "graph_retained_"), ("peak", "setup_peak_")):
        for field in ("allocated_bytes", "reserved_bytes"):
            actual = field if kind == "retained" else "peak_" + field
            delta = max(0, after[actual] - before[field]) if replay else 0
            _require(delta <= limits[cap_prefix + field], "graph memory budget exceeded")
            deltas[kind + "_" + field] = delta
    _require(setup["memory_deltas"] == deltas, "graph memory deltas differ from snapshots")


def _audit_graph_setup(setup, config, *, replay, block_size, expected_device="cuda:0", limits=None):
    """Audit setup, pool ownership and complete owned tensor inventory."""
    from vllm_lt.worker.decode_buffers import DecodeBucketLayout

    limits = GRAPH_LIMITS if limits is None else limits
    layout_record = setup["layout"]
    layout = DecodeBucketLayout(
        max_num_seqs=layout_record["max_num_seqs"],
        table_width=layout_record["table_width"],
    )
    rows_inventory = layout.row_counts
    _require(
        layout_record["row_counts"] == list(rows_inventory)
        and layout_record["pool_scope"] == "executor",
        "graph layout or pool scope differs",
    )
    hidden = config["hidden_size"]
    setup_record = setup["setup"]
    _require(setup_record["status"] == "complete", "graph setup incomplete")
    start, end, deadline = (setup_record[k] for k in ("started_ns", "finished_ns", "deadline_ns"))
    _require(
        all(type(x) is int and x > 0 for x in (start, end, deadline))
        and start <= end <= deadline
        and deadline - start == limits["setup_timeout_s"] * 10**9,
        "graph setup deadline differs",
    )
    _require(
        [setup_record[k] for k in ("warmups", "captures", "verification_replays")]
        == (
            [SETUP_WARMUPS * len(rows_inventory), len(rows_inventory), len(rows_inventory)]
            if replay
            else [0, 0, 0]
        ),
        "graph setup traversal/capture budget differs",
    )
    _require(
        setup["counters"]
        == dict.fromkeys(
            ("calls", "empty", "prepared", "eager", "replays", "committed", "completed"), 0
        ),
        "setup leaked into production counters",
    )
    if replay:
        phases = (
            [("scratch_save_and_seed", None)]
            + [
                (phase, rows)
                for rows in rows_inventory
                for phase in (*("warmup",) * SETUP_WARMUPS, "capture", "verification_replay")
            ]
            + [("scratch_restore", None)]
        )
        _require(
            [(e["phase"], e["row_count"]) for e in setup_record["events"]] == phases,
            "graph setup phase matrix differs",
        )
        previous = start
        for phase in setup_record["events"]:
            _require(
                phase["status"] == "complete"
                and previous <= phase["started_ns"] <= phase["finished_ns"] <= end,
                "setup phase chronology differs",
            )
            previous = phase["finished_ns"]
        scratch = setup_record["scratch"]
        expected_bytes = (
            SCRATCH_PAGES
            * block_size
            * config["num_hidden_layers"]
            * config["num_key_value_heads"]
            * config["head_dim"]
            * 4
            * 2
        )
        _require(
            scratch["restored"]
            and len(set(scratch["pages"])) == SCRATCH_PAGES
            and scratch["saved_cpu_bytes"] == expected_bytes,
            "scratch bytes or ownership restoration differs",
        )
        _require(
            scratch["free_list_before"] == scratch["free_list_after"]
            and scratch["before_hashes"] == scratch["after_hashes"],
            "scratch free-list or byte restoration differs",
        )
    else:
        _require(
            setup_record["events"] == [] and setup_record["scratch"] is None,
            "eager/backend fallback performed graph setup",
        )
    _audit_setup_memory(setup_record, replay, limits)
    stable = setup["buckets"]
    _require(set(stable) == set(map(str, rows_inventory)), "graph bucket inventory differs")
    owned = set()
    for name, bucket in stable.items():
        rows = int(name)
        _require(
            bucket["row_count"] == rows
            and bucket["table_width"] == layout.table_width
            and bucket["max_live_rows"] == rows // 2,
            "fixed bucket layout differs",
        )
        layouts = m3_persistent._layouts(hidden)
        layouts = {
            key: ([rows // 2] if key == "live_indices" else [rows, *shape[1:]], dtype)
            for key, (shape, dtype) in layouts.items()
        }
        layouts["block_tables"] = ([rows, layout.table_width], "int32")
        _require(
            set(bucket["tensors"]) == set(layouts)
            and set(bucket["staging_tensors"]) == set(m3_persistent._METADATA),
            "missing owned graph tensor inventory",
        )
        bucket_owners = m3_persistent._check_storage_inventory(
            bucket["tensors"], bucket["staging_tensors"], layouts, device=expected_device
        )
        _require(not (owned & bucket_owners), "graph buckets alias owned storage")
        owned.update(bucket_owners)
        _require(
            bool(bucket["graph_id"]) is replay, "required graph missing or eager baseline captured"
        )
        _require(
            bucket["generation"] == bucket["setup_generation"]
            and not bucket["in_use"]
            and not bucket["failed"],
            "setup left live bucket lease",
        )
        if replay:
            _require(
                all(
                    type(bucket[field]) is int and bucket[field] > 0
                    for field in ("graph_id", "graph_exec_id")
                ),
                "invalid graph or instantiated executable identity",
            )
            inputs, outputs = _graph_io(bucket)
            _require(
                bucket["captured_inputs"] == inputs and bucket["captured_outputs"] == outputs,
                "captured tensor addresses differ from owned bundle",
            )
            _require(bucket["pool_id"] is not None, "missing graph pool identity")
            _require(
                bucket["verification"]["finite"]
                and bucket["verification"]["inactive_positive_zero"],
                "scratch replay physical output verification absent",
            )
            differences = bucket["verification"]["max_abs_diff"]
            _require(
                len(differences) == 2
                and all(
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and 0 <= value <= REPLAY_ATOL
                    for value in differences
                ),
                "scratch replay differs from warmup",
            )
    payload = sum(d["size_bytes"] for b in stable.values() for d in b["tensors"].values())
    staging = sum(d["size_bytes"] for b in stable.values() for d in b["staging_tensors"].values())
    _require(
        setup["device_payload_bytes"] == payload <= limits["common_payload_bytes"]
        and setup["cpu_staging_bytes"] == staging <= limits["cpu_staging_bytes"]
        and (payload, staging) == layout.payload_bytes(hidden),
        "common storage byte accounting differs",
    )
    if replay:
        pool_ids = {tuple(bucket["pool_id"]) for bucket in stable.values()}
        _require(len(pool_ids) == 1, "executor buckets must share their owned pool")
        for bucket in stable.values():
            _require(
                bucket["pool_owner"]
                == {
                    "kind": "torch.cuda.MemPool",
                    "id": bucket["pool_id"],
                    "release_policy": "synchronize-reset-drop-captured-outputs-owner-last",
                },
                "graph pool ownership evidence differs",
            )
    return stable, owned


def _same_except(before, after, changing):
    _require(
        {k: v for k, v in before.items() if k not in changing}
        == {k: v for k, v in after.items() if k not in changing},
        "immutable graph state or captured addresses changed",
    )


def _audit_graph_snapshot(initial, current):
    """One invariant check for numerical dispatches and timed-run boundaries."""
    _require(
        initial["enabled"] is True and initial["failure"] is None and initial["status"] == "ready",
        "graph executor disabled or failed",
    )
    _same_except(
        initial,
        current,
        ("buckets", "counters", "fallback_counts", "last_dispatch", "last_publication"),
    )
    _require(set(initial["buckets"]) == set(current["buckets"]), "graph bucket inventory changed")
    for key, bucket in initial["buckets"].items():
        _same_except(bucket, current["buckets"][key], ("generation", "counters"))


def _audit_graph_close(final, closed):
    _require(
        closed["status"] == "closed"
        and closed["buckets"] == {}
        and closed["device_payload_bytes"] == closed["cpu_staging_bytes"] == 0,
        "closed graph executor retains owned buffers",
    )
    for field in ("counters", "fallback_counts", "setup", "failure"):
        _require(closed[field] == final[field], "graph state changed during close")


def _audit_graph_case(case, evidence, config, *, expected_device="cuda:0"):
    """Independently bind pointer claims to actual logical dispatch/publication."""
    import json

    events, lifecycle = evidence["graph_observations"], evidence["graph_lifecycle"]
    if case["implementation"] == "oracle":
        _require(events == [] and lifecycle == {}, "oracle cannot claim native replay observations")
        return {
            "dispatches": 0,
            "replay_dispatches": 0,
            "eager_dispatches": 0,
            "backend_fallbacks": 0,
            "buckets": [],
        }
    schedule = [s for s in evidence["schedule"] if s["stage"] == "recurrent"]
    _require(
        len(events) == len(schedule) and schedule,
        "missing or duplicate post-replay dispatch coverage",
    )
    _require("close_error" not in lifecycle, "graph executor cleanup failed")
    setup = lifecycle["setup"]
    replay = case["storage_strategy"] == "graph_replay"
    fallback = case["storage_strategy"] == "backend_fallback"
    hidden = config["hidden_size"]
    stable, owned = _audit_graph_setup(
        setup, config, replay=replay, block_size=case["block_size"], expected_device=expected_device
    )
    _require(
        setup["use_graphs"] == (case["implementation_id"] == "B")
        and setup["backend"] == case["backend"]
        and setup["limits"] == GRAPH_LIMITS,
        "graph request/backend/limits differ",
    )
    visits = {"4": 0, "8": 0}
    for index, (event, row) in enumerate(zip(events, schedule), 1):
        _require(
            len(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            <= EVIDENCE["dispatch_record_bytes"],
            "graph observation exceeds frozen record cap",
        )
        _require(
            event["status"] == "complete" and event["dispatch_index"] == index,
            "graph dispatch status/order differs",
        )
        logical = {
            "request_ids": row["request_ids"],
            "positions": row["positions_after_step"],
            "depths": row["depths_after_step"],
        }
        _require(event["logical"] == logical, "graph dispatch logical rows differ from scheduler")
        count = len(row["request_ids"])
        _require(
            1 <= count <= 4 and max(logical["positions"]) // case["block_size"] < 32,
            "numerical fixture left declared support bounds",
        )
        before, after = event["before"], event["after"]
        _require(
            before == (setup if index == 1 else events[index - 2]["after"]),
            "graph state changed before scheduled dispatch",
        )
        _audit_graph_snapshot(setup, after)
        expected_counts = {
            "calls": index,
            "empty": 0,
            "prepared": 0 if fallback else index,
            "eager": index if not replay and not fallback else 0,
            "replays": index if replay else 0,
            "committed": 0 if fallback else index,
            "completed": index,
        }
        _require(
            after["counters"] == expected_counts, "actual graph dispatch/completion counters differ"
        )
        _require(
            after["fallback_counts"]
            == {"backend": index if fallback else 0, "live_count": 0, "table_width": 0},
            "graph fallback counters differ",
        )
        dispatch = after["last_dispatch"]
        _require(
            dispatch["request_ids"] == logical["request_ids"]
            and dispatch["positions"] == logical["positions"]
            and dispatch["depths"] == [d - 1 for d in logical["depths"]],
            "actual executor inputs differ from schedule",
        )
        _require(dispatch["dispatch_id"] == index, "executor dispatch counter differs")
        _require(
            after["last_publication"] == event["publication"],
            "actual publication differs from executor record",
        )
        for name, shape in (("hidden", [count, hidden]), ("gates", [count])):
            desc = event["publication"][name]
            m3_persistent._check_descriptor(
                desc, shape, "float32", device=expected_device, owned=True
            )
            _require(
                desc["storage_ptr"] not in owned,
                "request publication aliases reusable graph storage",
            )
        _require(
            event["returned_gate_count"] == count and len(event["request_hidden"]) == count,
            "actual request/gate publication coverage differs",
        )
        publication = event["publication"]["hidden"]
        for position, desc in enumerate(event["request_hidden"]):
            _require(
                desc
                == {
                    **publication,
                    "shape": [hidden],
                    "stride": [1],
                    "size_bytes": hidden * 4,
                    "data_ptr": publication["data_ptr"] + position * hidden * 4,
                },
                "actual request row differs from returned publication",
            )
        if fallback:
            _require(
                dispatch["kind"] == "compact"
                and dispatch["bucket_id"] is None
                and event["physical"] is None,
                "Torch fallback falsely claims replay/bucket work",
            )
            _require(dispatch["ticket_state"] == "cancelled", "fallback ticket was not cancelled")
        else:
            selected = str(NUMERICAL_LAYOUT.select(count, 0)[0])
            visits[selected] += 1
            _require(
                dispatch["kind"] == ("replay" if replay else "eager")
                and str(dispatch["bucket_id"]) == selected,
                "missing required replay or wrong bucket selection",
            )
            _require(
                dispatch["live_rows"] == list(decode_live_rows(count))
                and dispatch["row_count"] == int(selected)
                and dispatch["table_width"] == NUMERICAL_LAYOUT.table_width,
                "actual physical row mapping differs",
            )
            _require(
                dispatch["ticket_state"] == "committed",
                "replay transaction was not committed after completion",
            )
            bucket = after["buckets"][selected]
            _require(
                bucket["generation"] == stable[selected]["setup_generation"] + visits[selected]
                and dispatch["generation"] == bucket["generation"],
                "production lease generation differs from setup baseline",
            )
            inputs, outputs = _graph_io(bucket)
            _require(
                dispatch["actual_inputs"] == inputs
                and dispatch["actual_physical_outputs"] == outputs,
                "actual dispatch pointers differ from captured/common tensors",
            )
            _require(
                event["physical"]
                == {
                    "active_mask_matches": True,
                    "hidden_out_finite": True,
                    "hidden_out_inactive_zero": True,
                    "gate_out_finite": True,
                    "gate_out_inactive_zero": True,
                },
                "missing inactive physical output evidence",
            )
        for key, bucket in after["buckets"].items():
            number = visits[key]
            _require(
                bucket["counters"]
                == {
                    "prepared": number,
                    "eager": number if not replay else 0,
                    "replays": number if replay else 0,
                    "committed": number,
                    "completed": number,
                },
                "per-bucket actual work differs",
            )
            _require(not bucket["in_use"] and not bucket["failed"], "bucket lease not released")
            _require(
                bucket["generation"] == stable[key]["setup_generation"] + number,
                "inactive bucket generation changed",
            )
    _require(
        lifecycle["before_close"] == events[-1]["after"],
        "final graph state differs from last completed dispatch",
    )
    _audit_graph_close(lifecycle["before_close"], lifecycle["after_close"])
    return {
        "dispatches": len(events),
        "replay_dispatches": len(events) if replay else 0,
        "eager_dispatches": len(events) if not replay and not fallback else 0,
        "backend_fallbacks": len(events) if fallback else 0,
        "buckets": [int(k) for k, v in visits.items() if v],
    }


def _graph_case_auditor(view, expected_device, totals, buckets):
    def audit(case, value, lifetime):
        counts = _audit_graph_case(
            case, value, view["model_config"], expected_device=expected_device
        )
        if case["implementation"] == "native":
            setup = value["graph_lifecycle"]["setup"]["setup"]
            _require(
                lifetime["started_ns"]
                <= setup["started_ns"]
                <= setup["finished_ns"]
                <= lifetime["finished_ns"],
                "graph setup lies outside case lifetime",
            )
        for key in totals:
            totals[key] += counts[key]
        if case["phase"] == "validation":
            buckets[case["implementation_id"]].update(counts["buckets"])

    return audit


def _audit_model(output_dir, parent, expected_device, *, prefix):
    from vllm_lt.validation.m2 import _audit_numerical_prefix

    view = model_view(parent)
    totals = dict.fromkeys(
        ("dispatches", "replay_dispatches", "eager_dispatches", "backend_fallbacks"), 0
    )
    buckets = {"A": set(), "B": set()}
    audit = _audit_numerical_prefix if prefix else _audit_numerical_view
    result = audit(
        output_dir, view, audit_case=_graph_case_auditor(view, expected_device, totals, buckets)
    )
    result["counts"].update({"verified_" + k: v for k, v in totals.items()})
    counts = dict(
        qualification_cases=13,
        excluded_feasibility_cases=2,
        qualification_comparisons=30,
        excluded_feasibility_comparisons=1,
    )
    result["counts"].update({"planned_" + k: v for k, v in counts.items()})
    if result["complete"]:
        if buckets != {"A": {4, 8}, "B": {4, 8}}:
            result["complete"] = result["passed"] = False
            result["errors"].append(
                {"type": "ValueError", "message": "missing qualification bucket coverage"}
            )
        else:
            result["counts"].update({"verified_" + k: v for k, v in counts.items()})
    return result


def audit_model_rows(output_dir, parent, *, expected_device="cuda:0"):
    return _audit_model(output_dir, parent, expected_device, prefix=False)


def audit_completed_prefix(output_dir, parent, *, expected_device="cuda:0"):
    return _audit_model(output_dir, parent, expected_device, prefix=True)
