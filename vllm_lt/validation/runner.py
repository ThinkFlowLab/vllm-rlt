"""Execute the frozen Q1 numerical experiment inside an exact GPU reservation."""

import gc
import json
import os
import time
from pathlib import Path

import torch

from vllm_lt.benchmarks.runner import environment, memory
from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.models import OuroForCausalLM
from vllm_lt.models.reference import dense_reference
from vllm_lt.models.serial_oracle import ExitPolicy, SerialOuroOracle
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.validation.schema import PROJECTION

from .diagnostics import DiagnosticDump, SpoolBudget, TensorSpool
from .evidence import ComparisonStream, boundary_key
from .native import ValidationEngine, history_hash, observe_native, snapshot_native
from .official import OfficialOuroReference, official_provenance
from .schema import verify_plan, write_json


def fixtures_for(plan):
    suite = plan["suite"]
    fixtures = {
        row["fixture_id"]: row for row in suite["fixtures"] + suite.get("feasibility_fixtures", [])
    }
    for index, prompt in enumerate(
        suite.get("original_reproduction", {}).get("prompt_token_ids", [])
    ):
        key = f"Q1-original-{index}"
        fixtures[key] = {"fixture_id": key, "prompt_token_ids": prompt}
    return fixtures


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("frozen validation time budget exhausted")


def _metadata(fixture_id, operation, positions, depth, layer, output_index, history):
    return {
        "fixture_id": fixture_id,
        "operation": operation,
        "positions": list(positions),
        "depth": depth,
        "layer": layer,
        "output_index": output_index,
        "history_sha256": history_hash(history),
    }


def _kv(sink, snapshot, fixture_id, history):
    if not bool(snapshot.initialized.all()):
        raise RuntimeError("final KV contains uninitialized history")
    metadata = _metadata(fixture_id, "populated_kv", snapshot.positions, None, None, 8, history)
    for component, tensor in (("keys", snapshot.keys), ("values", snapshot.values)):
        sink(
            {
                **metadata,
                "component": component,
                "depth_range": [1, snapshot.keys.shape[0]],
                "layer_range": [0, snapshot.keys.shape[1] - 1],
                "axes": ["depth", "layer", "position", "kv_head", "head_dim"],
            },
            tensor,
        )


def _trace(trace, history, emitted):
    logits = trace.logits.float()
    top = logits.topk(2).values
    return {
        "position": trace.position,
        "output_index": trace.output_index,
        "exit_depth": trace.exit_depth,
        "actual_token_id": int(logits.argmax()),
        "emitted_token_id": emitted,
        "history_sha256": history_hash(history),
        "gate_logits": list(trace.gate_logits),
        "gate_probabilities": list(trace.gate_probabilities),
        "cumulative_probabilities": list(trace.cumulative_probabilities),
        "gate_usage": "full_depth_prefill_diagnostic"
        if trace.output_index == 0
        else "actual_decode",
        "top_two_margin": float(top[0] - top[1]),
    }


def _oracle(model, case, fixture, sink, deadline, *, progress=None):
    fixture_id = fixture["fixture_id"]
    history = list(fixture["prompt_token_ids"])
    prompt_length, traces = len(history), []
    oracle = SerialOuroOracle(
        model.config, dict(model.named_parameters()), capacity=prompt_length + 8
    )
    if progress is not None:
        progress["traces"] = {fixture_id: traces}

    def observer(boundary, values):
        _deadline(deadline)
        for row, position in enumerate(boundary.positions):
            if position >= prompt_length - 1:
                sink(
                    _metadata(
                        fixture_id,
                        boundary.operation,
                        [position],
                        boundary.depth,
                        boundary.layer,
                        position - prompt_length + 1,
                        history,
                    ),
                    values[row],
                )

    try:
        prediction = oracle.prefill(history, observer=observer)
        for index in range(9):
            actual = int(prediction.logits.argmax())
            emitted = (
                fixture["continuation_input_ids"][index]
                if case["history_mode"] == "teacher_forced" and index < 8
                else actual
            )
            traces.append(_trace(prediction, history, emitted))
            if progress is not None:
                progress["steps"] = len(traces)
            if index < 8:
                history.append(emitted)
                prediction = oracle.advance(
                    emitted,
                    policy=ExitPolicy(
                        min_loops=2, max_loops=4, exit_threshold=case["exit_policy"]["threshold"]
                    ),
                    forced_depth=(
                        fixture["forced_exit_depths"][index + 1]
                        if case["history_mode"] == "teacher_forced"
                        else None
                    ),
                    observer=observer,
                )
        for start in range(0, len(history), 4):
            _deadline(deadline)
            snapshot = oracle.snapshot_kv(positions=range(start, min(start + 4, len(history))))
            _kv(sink, snapshot, fixture_id, history)
            del snapshot
        return {fixture_id: traces}, 9, []
    finally:
        _close_oracle(oracle, progress)


def _close_oracle(oracle, progress):
    """Observe ownership on both sides of close, including a failed close."""

    def state():
        return {
            "closed": oracle._closed,
            "retained_fields": [
                name
                for name in ("key_cache", "value_cache", "initialized", "_inv_freq")
                if getattr(oracle, name) is not None
            ],
            "weight_references": len(oracle.weights),
        }

    before = state()
    try:
        oracle.close()
    finally:
        after = state()
        if progress is not None:
            progress["oracle_lifecycle"] = {"before_close": before, "after_close": after}
            progress["cleanup"] = {
                "requests_remaining": int(not after["closed"]),
                "used_blocks": 0
                if oracle.key_cache is None and oracle.value_cache is None
                else None,
            }
    if not after["closed"] or after["retained_fields"] or after["weight_references"]:
        raise RuntimeError("oracle cleanup retained request state or weight references")


def _native(model, plan, case, fixtures, sink, deadline, *, progress=None):
    scheduler = plan["contract"]["engine"]["scheduler"]
    engine = ValidationEngine(
        model,
        fixtures=fixtures,
        history_mode=case["history_mode"],
        cache_config=CacheConfig(num_blocks=case["num_blocks"], block_size=case["block_size"]),
        scheduler_config=SchedulerConfig(
            **scheduler, mode=("refill" if case["schedule"] == "serial" else case["schedule"])
        ),
        attention_backend=case["backend"],
    )
    schedule, steps = [], 0
    if progress is not None:
        progress.update(traces=engine.traces, schedule=schedule)

    def completed(request_id, cache):
        # finish() receives the ninth emitted output; it has not entered KV history.
        request = engine.scheduler.requests[request_id]
        history = request.prompt_token_ids + request.generated_token_ids[:-1]
        for start in range(0, len(history), 4):
            _deadline(deadline)
            snapshot = snapshot_native(
                cache, request.request_id, range(start, min(start + 4, len(history)))
            )
            _kv(sink, snapshot, request.request_id, history)
            del snapshot

    try:
        with observe_native(engine, sink, on_completed_kv=completed):
            for fixture in fixtures:
                sampling = {
                    **plan["contract"]["engine"]["sampling"],
                    "exit_threshold": case["exit_policy"]["threshold"],
                }
                engine.add_request(
                    fixture["fixture_id"], fixture["prompt_token_ids"], SamplingParams(**sampling)
                )
            while engine.has_unfinished_requests():
                _deadline(deadline)
                if steps >= case["max_steps"]:
                    raise RuntimeError("native execution exceeded the frozen schedule step bound")
                engine.step()
                steps += 1
                if progress is not None:
                    progress["steps"] = steps
                batch = engine.last_schedule
                if batch is None:
                    raise RuntimeError("native scheduler made no progress")
                schedule.append(
                    {
                        "stage": batch.stage.value,
                        "request_ids": [item.request.request_id for item in batch.items],
                        "token_counts": [item.token_count for item in batch.items],
                        "positions_after_step": [item.request.position for item in batch.items],
                        "depths_after_step": [item.request.loops_done for item in batch.items],
                    }
                )
        if any(len(rows) != 9 for rows in engine.traces.values()):
            raise RuntimeError("native execution did not produce exactly nine predictions")
        if engine.cache_manager.num_used_blocks or engine.scheduler.requests:
            raise RuntimeError("native completion retained request/KV state")
        return engine.traces, steps, schedule
    finally:
        try:
            for request_id in list(engine.scheduler.requests):
                engine.abort_request(request_id)
        finally:
            remaining = len(engine.scheduler.requests)
            used = engine.cache_manager.num_used_blocks
            if progress is not None:
                progress["cleanup"] = {"requests_remaining": remaining, "used_blocks": used}
            if used or remaining:
                raise RuntimeError("native cleanup leaked task-owned KV state")


def _legacy(model, case, fixtures, sink, deadline, *, progress=None):
    parameter = next(model.parameters())
    if case["implementation"] == "legacy_dense":
        fixture = fixtures[0]
        tokens = torch.tensor(fixture["prompt_token_ids"], device=parameter.device)
        try:
            outputs = dense_reference(model, tokens, 4)
            for depth, (_, _, logits) in enumerate(outputs, 1):
                _deadline(deadline)
                sink(
                    _metadata(
                        fixture["fixture_id"],
                        "logits",
                        range(len(tokens)),
                        depth,
                        None,
                        None,
                        fixture["prompt_token_ids"],
                    ),
                    logits,
                )
                if progress is not None:
                    progress["steps"] = depth
        finally:
            if progress is not None:
                # Full-sequence functional execution has no request or cache owner.
                progress["cleanup"] = {"requests_remaining": 0, "used_blocks": 0}
        return
    config = model.config
    cache = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_loops=4,
        num_blocks=case["num_blocks"],
        block_size=case["block_size"],
        device=parameter.device,
        dtype=parameter.dtype,
        backend=case["backend"],
    )
    tokens, ids, positions = [], [], []
    try:
        for fixture in fixtures:
            prompt, fixture_id = fixture["prompt_token_ids"], fixture["fixture_id"]
            if not cache.allocate(fixture_id, len(prompt)):
                raise RuntimeError("legacy prefill could not allocate its fixed cache")
            tokens.extend(prompt)
            ids.extend([fixture_id] * len(prompt))
            positions.extend(range(len(prompt)))
        hidden = model.prelude(torch.tensor(tokens, device=parameter.device))
        for depth in range(4):
            _deadline(deadline)
            hidden, _ = model.recurrent(hidden, ids, [depth] * len(ids), positions, cache)
            logits, offset = model.coda(hidden), 0
            for fixture in fixtures:
                prompt = fixture["prompt_token_ids"]
                sink(
                    _metadata(
                        fixture["fixture_id"],
                        "logits",
                        range(len(prompt)),
                        depth + 1,
                        None,
                        None,
                        prompt,
                    ),
                    logits[offset : offset + len(prompt)],
                )
                offset += len(prompt)
            if progress is not None:
                progress["steps"] = depth + 1
    finally:
        for fixture in fixtures:
            cache.free(fixture["fixture_id"])
        if progress is not None:
            progress["cleanup"] = {
                "requests_remaining": len(cache._allocations),
                "used_blocks": cache.num_used_blocks,
            }
        if cache.num_used_blocks:
            raise RuntimeError("legacy packed cache leaked")


def _check_trace(case, fixture, traces):
    if len(traces) != 9:
        raise ValueError("execution must have nine genuine predictions")
    history = list(fixture["prompt_token_ids"])
    for index, trace in enumerate(traces):
        if (trace["position"], trace["output_index"], trace["history_sha256"]) != (
            len(history) - 1,
            index,
            history_hash(history),
        ):
            raise ValueError("trace positions or input history are inconsistent")
        depth = trace["exit_depth"]
        if depth not in (2, 3, 4) or index == 0 and depth != 4:
            raise ValueError("trace violates supported exit depths")
        if case["history_mode"] == "teacher_forced":
            if depth != fixture["forced_exit_depths"][index]:
                raise ValueError("execution did not perform the declared fixed/forced loop work")
            expected = (
                fixture["continuation_input_ids"][index] if index < 8 else trace["actual_token_id"]
            )
        else:
            expected = trace["actual_token_id"]
        if trace["emitted_token_id"] != expected:
            raise ValueError("emitted history differs from the declared input policy")
        for name in ("gate_logits", "gate_probabilities", "cumulative_probabilities"):
            if len(trace[name]) != depth:
                raise ValueError("gate evidence does not cover all executed loops")
        if index < 8:
            history.append(expected)


@torch.inference_mode()
def execute_case(model, plan, case, output, budget, dumps, deadline):
    """One execution; CPU tiny models exercise this same driver before GPU use."""
    projection = case.get("boundary_projection")
    if projection not in (None, PROJECTION):
        raise ValueError("unknown numerical boundary projection")
    output = Path(output)
    folder = output / "cases" / case["case_id"]
    folder.mkdir(parents=True, exist_ok=False)
    deadline = min(deadline, time.monotonic() + plan["contract"]["limits"]["case_timeout_s"])
    fixtures = [fixtures_for(plan)[key] for key in case["fixture_ids"]]
    result = {
        "schema_version": 1,
        "artifact_type": "validation_case_result",
        "case": case,
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "traces": {},
        "cleanup": None,
        "steps": 0,
        "schedule": [],
        "comparisons": [],
        "failures": [],
    }
    write_json(folder / "started.json", result)
    streams, spools, counts = [], {}, {key: 0 for key in case["fixture_ids"]}
    result["boundary_counts"] = counts
    result["observed_boundaries"] = 0
    start = time.monotonic()
    execution_error = None
    try:
        for comparison in plan["comparison_order"]:
            if comparison["candidate_case_id"] == case["case_id"]:
                streams.append(
                    ComparisonStream(
                        output,
                        comparison,
                        plan["contract"]["comparison_policy"],
                        dumps,
                        plan["plan_sha256"],
                        diagnostics=plan["contract"]["diagnostics"],
                    )
                )
        retain = (
            case.get("retain_evidence", False)
            or case["implementation"] in ("oracle", "legacy_dense")
            or (case["implementation"] == "legacy_packed" and case["backend"] == "torch")
        )
        if retain:
            for fixture_id in case["fixture_ids"]:
                spools[fixture_id] = TensorSpool(
                    output / "spools",
                    case["case_id"],
                    fixture_id,
                    budget,
                    case.get("spool_group", f"{case['family']}-{case['dtype']}-{case['group_id']}"),
                )

        def sink(metadata, tensor):
            _deadline(deadline)
            if projection and metadata["operation"] not in (
                "loop_hidden",
                "gate_logits",
                "logits",
                "populated_kv",
            ):
                if case["implementation"] == "oracle":
                    return
                raise ValueError("native capture observation emitted an omitted boundary")
            fixture_id = metadata["fixture_id"]
            counts[fixture_id] += 1
            result["observed_boundaries"] += 1
            if fixture_id in spools:
                if not bool(tensor.isfinite().all()):
                    raise ValueError(f"nonfinite reference boundary: {metadata}")
                spools[fixture_id].write(boundary_key(metadata), tensor, metadata)
            for stream in streams:
                if stream.comparison["fixture_id"] == fixture_id:
                    stream.observe(
                        metadata, tensor, observation_index=result["observed_boundaries"]
                    )

        implementation = case["implementation"]
        if implementation == "oracle":
            traces, steps, schedule = _oracle(
                model, case, fixtures[0], sink, deadline, progress=result
            )
        elif implementation == "native":
            traces, steps, schedule = _native(
                model, plan, case, fixtures, sink, deadline, progress=result
            )
        elif implementation == "official":
            fixture = fixtures[0]
            official = OfficialOuroReference(model.config.to_dict(), dict(model.named_parameters()))
            try:
                logits = official.predict(
                    fixture["prompt_token_ids"], fixture["continuation_input_ids"]
                )
                history = list(fixture["prompt_token_ids"])
                for index, row in enumerate(logits):
                    sink(
                        _metadata(
                            fixture["fixture_id"],
                            "logits",
                            [len(history) - 1],
                            4,
                            None,
                            index,
                            history,
                        ),
                        row,
                    )
                    if index < 8:
                        history.append(fixture["continuation_input_ids"][index])
            finally:
                official.close()
                result["cleanup"] = {
                    "requests_remaining": int(official.model is not None),
                    "used_blocks": 0,
                }
            traces, steps, schedule = {}, 1, []
        else:
            _legacy(model, case, fixtures, sink, deadline, progress=result)
            traces, steps, schedule = {}, 4, []
        if implementation in ("oracle", "native"):
            for fixture in fixtures:
                fixture_id = fixture["fixture_id"]
                _check_trace(case, fixture, traces[fixture_id])
                expected = (
                    ((0 if projection else 6 * model.config.num_hidden_layers) + 2)
                    * sum(row["exit_depth"] for row in traces[fixture_id])
                    + 9
                    + 2 * ((len(fixture["prompt_token_ids"]) + 8 + 3) // 4)
                )
                if counts[fixture_id] != expected:
                    raise RuntimeError(
                        f"executed boundary coverage {counts[fixture_id]} != {expected}"
                    )
        else:
            expected = 9 if implementation == "official" else 4
            if any(count != expected for count in counts.values()):
                raise RuntimeError(f"executed boundary coverage must be {expected} per fixture")
        result.update(traces=traces, steps=steps, schedule=schedule, boundary_counts=counts)
        for stream in streams:
            stream.finish(traces.get(stream.comparison["fixture_id"]))
            result["comparisons"].append(stream.comparison["comparison_id"])
        if result["cleanup"] != {"requests_remaining": 0, "used_blocks": 0}:
            raise RuntimeError("execution did not establish complete request/KV cleanup")
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        execution_error = exc
        result["status"] = "failed"
        result["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        finalization_error = None
        resources = [("comparison", stream) for stream in streams]
        resources.extend(("spool", spool) for spool in spools.values())
        for kind, resource in resources:
            try:
                resource.close()
            except (Exception, KeyboardInterrupt) as exc:
                if finalization_error is None:
                    finalization_error = exc
                result["status"] = "failed"
                result["failures"].append(
                    {
                        "type": type(exc).__name__,
                        "phase": f"{kind}_finalization",
                        "message": str(exc),
                    }
                )
        result["elapsed_s"] = time.monotonic() - start
        write_json(folder / "result.json", result)
        if finalization_error is not None and execution_error is None:
            raise finalization_error
    return result


def _disk_bytes(output):
    return sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file())


def run(plan, output):
    """One bounded pass. Numerical failures continue; execution/invariant failures stop."""
    verify_plan(plan)
    provenance = official_provenance()
    if (
        provenance["dependencies"]["transformers"] != "4.55.0"
        or provenance["optional_kernels_present"]
    ):
        raise ValueError("official reference dependencies do not match the frozen contract")
    expected_visibility = str(plan["contract"]["controls"]["gpu_ids"][0])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visibility:
        raise ValueError("scheduler-assigned physical GPU differs from the frozen Q1 device")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    contract, limits = plan["contract"], plan["contract"]["limits"]
    torch.set_num_threads(contract["controls"]["cpu_threads"])
    torch.set_num_interop_threads(contract["controls"]["interop_threads"])
    torch.manual_seed(contract["controls"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    budget = SpoolBudget(limits["cumulative_spool_written_bytes"], limits["group_spool_bytes"])
    dumps = DiagnosticDump(
        output / "dumps",
        contract["diagnostics"]["preselected_dump_fixture_ids"],
        budget=budget,
        per_fixture_bytes=limits["dump_bytes_per_fixture"],
        total_bytes=limits["persisted_dump_bytes"],
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "validation_manifest",
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "completed_cases": [],
        "failures": [],
        "official_provenance": provenance,
        "environment": None,
        "phase": "prepared",
        "active_case_id": None,
        "model_loads": [],
        "preparation": "local pinned checkpoint; no downloads or tokenization",
    }
    start, model, dtype, device_ready = time.monotonic(), None, None, False
    device_was_initialized = torch.cuda.is_initialized()
    deadline = start + limits["total_timeout_s"]
    write_json(output / "manifest.json", manifest)
    try:
        manifest["phase"] = "environment"
        write_json(output / "manifest.json", manifest)
        manifest["environment"] = environment()
        matmul = torch.backends.cuda.matmul
        manifest["environment"]["arithmetic"] = {
            "allow_tf32": matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        }
        device_ready = True
        manifest["phase"] = "environment_ready"
        write_json(output / "manifest.json", manifest)
        free, total = torch.cuda.mem_get_info()
        estimate = plan["resource_estimates"]
        minimum = estimate["parameter_bytes"]["float32"] + estimate["native_pool_bytes"]["float32"]
        manifest["capacity_preflight"] = {
            "free_bytes": free,
            "total_bytes": total,
            "minimum_resident_bytes": minimum,
        }
        if free <= minimum:
            raise ValueError("model and frozen KV pool do not fit the reserved device")
        for case in plan["execution_order"]:
            _deadline(deadline)
            manifest["active_case_id"] = case["case_id"]
            manifest["phase"] = "preparing_case"
            write_json(output / "manifest.json", manifest)
            if dtype != case["dtype"]:
                model = None
                gc.collect()
                torch.cuda.synchronize()
                loading = time.monotonic()
                manifest["phase"] = "model_load"
                write_json(output / "manifest.json", manifest)
                model = OuroForCausalLM.from_pretrained(
                    plan["model_path"], device="cuda", dtype=getattr(torch, case["dtype"])
                )
                model.requires_grad_(False)
                model.eval()
                dtype = case["dtype"]
                manifest["model_loads"].append(
                    {
                        "before_case": case["case_id"],
                        "dtype": dtype,
                        "elapsed_s": time.monotonic() - loading,
                    }
                )
                manifest["phase"] = "model_ready"
                write_json(output / "manifest.json", manifest)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            manifest["phase"] = "executing_case"
            write_json(output / "manifest.json", manifest)
            result = execute_case(model, plan, case, output, budget, dumps, deadline)
            gc.collect()
            torch.cuda.synchronize()
            result["memory"] = {
                **memory(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            write_json(output / "cases" / case["case_id"] / "result.json", result)
            manifest["completed_cases"].append(case["case_id"])
            manifest["active_case_id"] = None
            manifest["phase"] = "case_complete"
            manifest["tensor_written_bytes"] = budget.written_bytes
            manifest["artifact_disk_bytes"] = _disk_bytes(output)
            if manifest["artifact_disk_bytes"] > limits["total_disk_bytes"]:
                raise ValueError("total artifact disk budget exceeded")
            write_json(output / "manifest.json", manifest)
            print(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "status": result["status"],
                        "elapsed_s": result["elapsed_s"],
                        "completed": len(manifest["completed_cases"]),
                    }
                ),
                flush=True,
            )
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = (
            "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        model = None
        gc.collect()
        # is_initialized only reads the process flag; it neither discovers nor
        # initializes a device after an environment/reservation rejection.
        device_ready = device_ready or (not device_was_initialized and torch.cuda.is_initialized())

        def cleanup_failure(phase, exc):
            manifest["status"] = "failed"
            manifest["failures"].append(
                {
                    "type": "cleanup",
                    "phase": phase,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

        try:
            dumps.close()
        except (Exception, KeyboardInterrupt) as exc:
            cleanup_failure("diagnostic_dumps", exc)
        manifest["diagnostic_dumps"] = {
            "selected_fixture_ids": dumps.selected_fixture_ids,
            "written_bytes": dumps.written_bytes,
            "fixture_written_bytes": dict(dumps.fixture_written_bytes),
        }
        if device_ready:
            for phase, action in (
                ("synchronize", torch.cuda.synchronize),
                ("teardown_after_gc", memory),
                ("clear_cublas_workspaces", lambda: torch._C._cuda_clearCublasWorkspaces()),
                ("empty_cache", torch.cuda.empty_cache),
                ("teardown_after_workspace_release", memory),
            ):
                try:
                    value = action()
                    if phase.startswith("teardown_after_"):
                        manifest[phase] = value
                except (Exception, KeyboardInterrupt) as exc:
                    cleanup_failure(phase, exc)
            final_memory = manifest.get("teardown_after_workspace_release")
            if final_memory is not None and final_memory["allocated_bytes"]:
                cleanup_failure(
                    "live_tensors", RuntimeError("task-owned tensors remain after final teardown")
                )
        manifest["elapsed_s"] = time.monotonic() - start
        if manifest["status"] == "complete" and time.monotonic() >= deadline:
            manifest["status"] = "incomplete"
            manifest["failures"].append({"type": "budget", "message": "total deadline exhausted"})
        manifest["tensor_written_bytes"] = budget.written_bytes
        manifest["artifact_disk_bytes"] = _disk_bytes(output)
        if manifest["artifact_disk_bytes"] > limits["total_disk_bytes"]:
            manifest["status"] = "failed"
            manifest["failures"].append(
                {"type": "budget", "message": "total artifact disk budget exceeded"}
            )
        manifest["phase"] = "finished"
        write_json(output / "manifest.json", manifest)
    return manifest
