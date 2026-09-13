"""Execute a frozen, bounded plan. GPU selection belongs to the caller's scheduler."""

import gc
import getpass
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from vllm_lt.benchmarks.observe import RunCollector, instrument_engine
from vllm_lt.benchmarks.profile import Capture, finite_checks
from vllm_lt.benchmarks.replay import ReplayEngine
from vllm_lt.benchmarks.schema import read_json, verify_plan
from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.sampling_params import SamplingParams


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def memory():
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def capacity_check(plan):
    """Reject an impossible resident model/pool before allocating either tensor set."""
    with torch.device("meta"):
        sizing = OuroForCausalLM(OuroConfig.from_dict(plan["model_config"]))
    dtype = getattr(torch, plan.get("contract", {}).get("engine", {}).get("dtype", "float32"))
    weights = sum(parameter.numel() * dtype.itemsize for parameter in sizing.parameters())
    pool = max(row["pool_bytes"] for row in plan["workload_stats"].values())
    free, total = torch.cuda.mem_get_info()
    result = {
        "fp32_weight_bytes" if dtype == torch.float32 else "bf16_weight_bytes": weights,
        "pool_bytes": pool,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "minimum_resident_bytes": weights + pool,
        "remaining_bytes": free - weights - pool,
        "policy": "Model and pool must fit; feasibility measures activation/runtime peaks.",
    }
    if result["remaining_bytes"] <= 0:
        raise ValueError(f"insufficient device memory for resident model and KV pool: {result}")
    return result


def environment():
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visibility or len(visibility.split(",")) != 1:
        raise ValueError("run requires exactly one caller-assigned CUDA_VISIBLE_DEVICES entry")
    if not shutil.which("gpu"):
        raise ValueError("device execution requires the verified gpu scheduler")
    rows = json.loads(subprocess.check_output(["gpu", "status", "--json"], text=True))
    scheduler = [row for row in rows if str(row["gpu_id"]) == visibility]
    if len(scheduler) != 1 or scheduler[0].get("user") != getpass.getuser():
        raise ValueError("visible GPU does not match this account's scheduler reservation")
    if scheduler[0].get("type") != "RUN":
        raise ValueError("device execution requires a gpu run reservation")
    if torch.cuda.device_count() != 1:
        raise ValueError("the benchmark supports one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "account": getpass.getuser(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "torch_cuda_version": torch.version.cuda,
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "safetensors", "huggingface-hub")
        },
        "cuda_visible_devices": visibility,
        "logical_device": "cuda:0",
        "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid),
        "total_device_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "actual_torch_threads": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_MODULE_LOADING",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "numa_status": [
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:"))
        ],
        "scheduler": scheduler,
        "reservation_environment": {
            name: os.environ[name]
            for name in ("CANHAZGPU_TASK_ID", "CANHAZGPU_RUN_ID", "CANHAZGPU_RESERVATION_ID")
            if name in os.environ
        },
    }


def _execute(model, plan, run, workload, output_dir, deadline):
    contract = plan["contract"]
    engine_config = contract["engine"]
    run_dir = output_dir / "runs" / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    started_ns = time.perf_counter_ns()
    case_deadline = min(
        deadline,
        started_ns
        + int(run.get("case_lifetime_timeout_s", contract["limits"]["workload_timeout_s"]) * 1e9),
    )
    write_json(
        run_dir / "started.json",
        {"schema_version": 1, **run, "started_ns": started_ns, "deadline_ns": case_deadline},
    )
    setup = time.perf_counter_ns()
    arguments = {
        "cache_config": CacheConfig(**engine_config["cache"]),
        "scheduler_config": SchedulerConfig(**engine_config["scheduler"], mode=run["mode"]),
        "attention_backend": engine_config["attention_backend"],
    }
    engine = (
        ReplayEngine(model, replay=workload["replay"], **arguments)
        if workload["kind"] == "scheduler_replay"
        else LLMEngine(model, **arguments)
    )
    setup_ns = time.perf_counter_ns() - setup
    request_ids = [item["request_id"] for item in workload["requests"]]
    max_steps = run["max_steps"]
    failures = []
    capture = (
        Capture(
            engine,
            output_dir / "profiles" / run["run_id"],
            limit=contract["limits"]["profile_decode_outputs"],
            run=run,
        )
        if run["phase"] == "profile"
        else None
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    arrival_ns = time.perf_counter_ns()
    collector = RunCollector(request_ids, arrival_ns=arrival_ns, max_events=run["max_events"])
    observation = None
    finite_counts = {}
    status = "complete"
    collected = {"requests": [], "metrics": {}, "events": []}
    synchronized_ns = arrival_ns
    gpu_elapsed_ms = None
    steps = 0
    submissions = [
        (
            item,
            SamplingParams(
                **{**engine_config["sampling"], "max_tokens": item["max_output_tokens"]}
            ),
        )
        for item in workload["requests"]
    ]
    try:
        with finite_checks(model, run["phase"] == "feasibility") as finite_counts:
            with (
                instrument_engine(
                    engine, collector, profile=capture is not None, max_steps=max_steps
                ) as observation,
                capture or nullcontext(),
            ):
                arrival_ns = time.perf_counter_ns()
                collector.start(arrival_ns)
                limit = min(
                    case_deadline, arrival_ns + int(contract["limits"]["workload_timeout_s"] * 1e9)
                )
                # Stream span includes host submission gaps, not just GPU kernel time.
                begin.record()
                for item, params in submissions:
                    started = time.perf_counter_ns()
                    engine.add_request(item["request_id"], item["prompt_token_ids"], params)
                    collector.observe_submission(
                        item["request_id"], started_ns=started, ended_ns=time.perf_counter_ns()
                    )
                while engine.has_unfinished_requests():
                    if steps >= max_steps or time.perf_counter_ns() >= limit:
                        raise TimeoutError("declared workload step/time budget exhausted")
                    outputs = engine.step()
                    returned = time.perf_counter_ns()
                    collector.observe_outputs(outputs, step_id=steps, returned_ns=returned)
                    if capture:
                        capture.after_step(outputs)
                    steps += 1
                end.record()
                torch.cuda.synchronize()
                synchronized_ns = time.perf_counter_ns()
                if synchronized_ns >= limit:
                    raise TimeoutError("case deadline exhausted during final step/synchronization")
                gpu_elapsed_ms = begin.elapsed_time(end)
                observation.validate_gate_probabilities()
                collected = collector.finish(synchronized_ns=synchronized_ns)
                if capture and not capture.done:
                    raise ValueError("profiler did not complete its declared capture window")
                if capture and (
                    capture.metadata["gpu_events"]["kernel_count"] == 0
                    or capture.metadata["gpu_events"]["memcpy_count"] == 0
                ):
                    raise ValueError("profiler trace is missing GPU kernel/memcpy attribution")
    except (Exception, KeyboardInterrupt) as exc:
        status = "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        failures.append(
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )
        # Recovery records are host observations, not invented synchronized timings.
        collected = collector.snapshot(synchronized_ns=None)
        synchronized_ns = None
    finally:
        for request_id in list(engine.scheduler.requests):
            try:
                engine.abort_request(request_id)
            except Exception as exc:
                failures.append({"type": "cleanup", "message": str(exc)})
        cleanup = {
            "active_requests": len(engine.scheduler.requests),
            "used_kv_blocks": engine.cache_manager.num_used_blocks,
        }
        if any(cleanup.values()):
            failures.append(
                {"type": "cleanup", "message": "request state or KV reservations remain"}
            )
        if failures and status == "complete":
            status = "failed"
    memory_result = {
        "before": before,
        "pool_bytes": engine.cache_manager.num_blocks * engine.cache_manager.bytes_per_block,
    }
    try:
        memory_result.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "after_requests": memory(),
            }
        )
    except Exception as exc:
        status = "failed"
        failures.append({"type": "memory_accounting", "message": str(exc)})
    result = {
        "schema_version": 1,
        "artifact_type": "run_result",
        **run,
        "experiment_id": output_dir.name,
        "status": status,
        "comparison_eligible": status == "complete" and run["phase"] == "measured",
        "plan_sha256": plan["plan_sha256"],
        "arrival_ns": arrival_ns,
        "synchronized_ns": synchronized_ns,
        "setup_ns": setup_ns,
        "gpu_stream_span_ms": gpu_elapsed_ms,
        "requests": collected["requests"],
        "metrics": collected["metrics"],
        "metrics_error": collected.get("metrics_error"),
        "counts": observation.summary() if observation else {},
        "feasibility_finite_checks": finite_counts,
        "memory": memory_result,
        "cleanup": cleanup,
        "failures": failures,
    }
    events = collected["events"]
    if observation:
        for snapshot in result["counts"]["snapshots"]:
            events.append(
                {
                    "schema_version": 1,
                    "artifact_type": "event",
                    "event_seq": len(events),
                    "kind": "stage_observed",
                    "step_id": snapshot["batch"]["step_id"],
                    "host_offset_ns": snapshot["schedule_end_ns"] - arrival_ns,
                    **snapshot,
                }
            )
    if failures:
        events.append(
            {
                "schema_version": 1,
                "artifact_type": "event",
                "event_seq": len(events),
                "kind": "run_failed",
                "host_offset_ns": time.perf_counter_ns() - arrival_ns,
                "last_completed_step": steps - 1,
                "failures": failures,
                "cleanup": cleanup,
            }
        )
    events.sort(key=lambda event: event["host_offset_ns"])
    for event_seq, event in enumerate(events):
        event["event_seq"] = event_seq
    with (run_dir / "events.jsonl").open("w") as stream:
        for event in events:
            stream.write(
                json.dumps(
                    {
                        **event,
                        "run_id": run["run_id"],
                        "phase": run["phase"],
                        "instrumentation": run["instrumentation"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    write_json(run_dir / "result.json", result)
    return result


def configure_process(contract):
    controls = contract["controls"]
    torch.set_num_threads(controls["cpu_threads"])
    torch.set_num_interop_threads(controls["interop_threads"])
    torch.manual_seed(controls["seed"])
    arithmetic = contract["arithmetic"]
    torch.backends.cuda.matmul.allow_tf32 = arithmetic["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = arithmetic["allow_tf32"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = arithmetic[
        "allow_bf16_reduced_precision_reduction"
    ]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = arithmetic[
        "allow_fp16_reduced_precision_reduction"
    ]


def load_model(plan, manifest):
    """Shared one-model worker preparation; callers record preparation separately."""
    manifest["capacity_preflight"] = capacity_check(plan)
    loading = time.perf_counter_ns()
    dtype = getattr(torch, plan["contract"]["engine"]["dtype"])
    model = OuroForCausalLM.from_pretrained(plan["model_path"], device="cuda", dtype=dtype)
    manifest["loading_ns"] = time.perf_counter_ns() - loading
    manifest["post_load_memory"] = memory()
    return model


def release_device(manifest):
    """Final process-local cleanup only, after callers release their model reference."""
    errors = []
    for label, action in (
        ("gc", gc.collect),
        ("synchronize", torch.cuda.synchronize),
        ("before_release", lambda: manifest.update(teardown_after_gc=memory())),
        ("workspace_release", lambda: torch._C._cuda_clearCublasWorkspaces()),
        ("allocator_release", torch.cuda.empty_cache),
        ("after_release", lambda: manifest.update(teardown_after_workspace_release=memory())),
    ):
        try:
            action()
        except Exception as exc:
            errors.append({"stage": label, "type": type(exc).__name__, "message": str(exc)})
    if errors:
        manifest["cleanup_errors"] = errors
        raise RuntimeError(f"task-owned CUDA cleanup failed: {errors}")
    if any(manifest["teardown_after_workspace_release"].values()):
        raise RuntimeError("task-owned tensor or allocator memory remains after teardown")


def run_loaded_rows(model, plan, rows, *, output_dir, deadline, record_completed):
    """Execute rows through the sole M1 inference loop; stop without retries."""
    workloads = {row["workload_id"]: row for row in plan["suite"]["workloads"]}
    for run in rows:
        if time.perf_counter_ns() >= deadline:
            raise TimeoutError("overall experiment budget exhausted")
        try:
            result = _execute(model, plan, run, workloads[run["workload_id"]], output_dir, deadline)
        except (Exception, KeyboardInterrupt) as exc:
            # An allocation/setup failure may precede the inner recovery boundary.
            run_dir = output_dir / "runs" / run["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / "result.json"
            if not result_path.exists():
                write_json(
                    result_path,
                    {
                        "schema_version": 1,
                        "artifact_type": "run_result",
                        **run,
                        "experiment_id": output_dir.name,
                        "plan_sha256": plan["plan_sha256"],
                        "status": "failed",
                        "comparison_eligible": False,
                        "requests": [],
                        "metrics": None,
                        "cleanup": None,
                        "failures": [{"type": type(exc).__name__, "message": str(exc)}],
                    },
                )
            raise
        if result["status"] != "complete":
            raise RuntimeError(f"stopping after {run['run_id']}: {result['failures']}")
        gc.collect()
        torch.cuda.synchronize()
        result["memory"]["after_engine_release"] = memory()
        marker = read_json(output_dir / "runs" / run["run_id"] / "started.json")
        result["case_started_ns"] = marker["started_ns"]
        result["case_completed_ns"] = time.perf_counter_ns()
        if result["case_completed_ns"] >= marker["deadline_ns"]:
            result["status"], result["comparison_eligible"] = "incomplete", False
            result["failures"].append(
                {"type": "deadline", "message": "case lifetime including cleanup exceeded"}
            )
            write_json(output_dir / "runs" / run["run_id"] / "result.json", result)
            raise TimeoutError("case lifetime including cleanup exceeded")
        write_json(output_dir / "runs" / run["run_id"] / "result.json", result)
        record_completed(run["run_id"], result)
        print(
            json.dumps(
                {
                    "run_id": run["run_id"],
                    "status": result["status"],
                    "metrics": result["metrics"].get("generated_tokens_per_second"),
                }
            ),
            flush=True,
        )
        if time.perf_counter_ns() >= deadline:
            raise TimeoutError("overall experiment budget exhausted during artifact recording")


def run_plan(plan: dict, *, output_dir: Path) -> dict:
    """Run the exact declared sequence once; never retry or expand a failed experiment."""
    start = time.perf_counter_ns()
    verify_plan(plan)
    verified_ns = time.perf_counter_ns()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "inputs").mkdir()
    write_json(output_dir / "inputs" / "plan.json", plan)
    contract = plan["contract"]
    configure_process(contract)
    manifest = {
        "schema_version": 1,
        "artifact_type": "experiment_manifest",
        "experiment_id": output_dir.name,
        "plan_sha256": plan["plan_sha256"],
        "contract": contract,
        "source": plan["source"],
        "environment": None,
        "status": "running",
        "completed_runs": [],
        "failures": [],
        "preparation": {
            "plan_verification_ns": verified_ns - start,
            "downloads": "none; local checkpoint prepared before experiment",
            "tokenization": "none; committed token-ID fixture",
            "compilation_ns": None,
            "compilation_note": (
                "Lazy compilation is included in excluded feasibility/warmup runs; "
                "not separately isolated."
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    deadline = start + int(contract["limits"]["total_timeout_s"] * 1e9)
    model = None
    device_ready = False
    try:
        manifest["environment"] = environment()
        manifest["preparation"]["environment_setup_ns"] = time.perf_counter_ns() - verified_ns
        device_ready = True
        model = load_model(plan, manifest)
        write_json(output_dir / "manifest.json", manifest)

        def completed(run_id, result):
            manifest["completed_runs"].append(run_id)
            write_json(output_dir / "manifest.json", manifest)

        run_loaded_rows(
            model,
            plan,
            plan["execution_order"],
            output_dir=output_dir,
            deadline=deadline,
            record_completed=completed,
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
        try:
            if device_ready or torch.cuda.is_initialized():
                release_device(manifest)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failures"].append({"type": "cleanup", "message": str(exc)})
        manifest["total_ns"] = time.perf_counter_ns() - start
        if manifest["status"] == "complete" and time.perf_counter_ns() >= deadline:
            manifest["status"] = "incomplete"
            manifest["failures"].append({"type": "budget", "message": "total deadline exhausted"})
        write_json(output_dir / "manifest.json", manifest)
    return manifest
