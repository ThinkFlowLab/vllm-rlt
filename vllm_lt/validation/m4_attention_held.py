"""Finite M4 tile qualification: 30 guarded attention rows and four storage rows."""

import math
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from . import m3_inactive_kernels as kernels
from . import m3_persistent_lifecycle as lifecycle
from .diagnostics import MiB, SpoolBudget, TensorSpool
from .schema import _digest, read_json, write_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fixture(plan, layout):
    """Physical inputs are identical across sources; L14 is one shared prefix."""
    import torch

    if layout["layout_id"] != "L14":
        return kernels._fixture(plan, layout, "B")[0]
    count, width = layout["row_count"], layout["table_width"]
    q = kernels._values(0, count * 2048, plan["config"]["row_formula"]["query"])
    return {
        "query": q.reshape(count, 16, 128),
        "context_lengths": torch.arange(1, count + 1, dtype=torch.int32),
        "block_tables": torch.arange(1, 2 * width, 2, dtype=torch.int32).repeat(count, 1),
        "active": torch.ones(count, dtype=torch.bool),
    }


def _chunk(plan, layout, component, start):
    return kernels._cache_chunk(
        plan, layout, component, start, start + 8, written=layout["layout_id"] != "L14"
    )


def _history(plan, layout, tensors, logical, component):
    """Build just one causal prefix, directly from physical element coordinates."""
    import torch

    row = layout["live_rows"][logical]
    length = layout["lengths"][logical]
    positions = torch.arange(length, dtype=torch.int64)
    pages = tensors["block_tables"][row, positions // 16].to(torch.int64)
    flat = (((pages * 2 + 1) * 16 + positions % 16) * 2048)[:, None]
    flat = flat + torch.arange(2048, dtype=torch.int64)[None, :]
    mult, offset, modulus, center, divisor = plan["config"]["cache_formula"][component]
    history = (((flat * mult + offset) % modulus - center).double() / divisor).reshape(
        length, 16, 128
    )
    if layout["layout_id"] != "L14":
        history[-1] = tensors[component][row].double()
    return history


def _dense(plan, layout, tensors):
    import torch

    output = torch.zeros(tensors["query"].shape, dtype=torch.float64)
    for logical, row in enumerate(layout["live_rows"]):
        key = _history(plan, layout, tensors, logical, "key")
        value = _history(plan, layout, tensors, logical, "value")
        query = tensors["query"][row].double()
        scores = torch.einsum("hd,thd->ht", query, key) / math.sqrt(128)
        output[row] = torch.einsum("ht,thd->hd", scores.softmax(-1), value)
    return output


def _stats(actual, reference, *, bounded):
    import torch

    _require(actual.shape == reference.shape, "attention output shape differs")
    a, b = actual.double(), reference.double()
    finite = bool(a.isfinite().all() and b.isfinite().all())
    delta = a - b
    return {
        "numel": a.numel(),
        "finite": finite,
        "max_abs": float(delta.abs().max()) if finite and a.numel() else (0.0 if finite else None),
        "rms": float(delta.square().mean().sqrt())
        if finite and a.numel()
        else (0.0 if finite else None),
        "reference_rms": float(b.square().mean().sqrt())
        if finite and b.numel()
        else (0.0 if finite else None),
        "exact": bool(torch.equal(actual, reference)),
        "allclose": bool(torch.allclose(a, b, atol=2e-5, rtol=2e-5)) if bounded else None,
        "policy": {"atol": 2e-5, "rtol": 2e-5} if bounded else "finite-diagnostic-only",
    }


@lru_cache(maxsize=1)
def _resolved_plan():
    base = kernels.build_kernel_plan()
    config = deepcopy(base["config"])
    config["comparison"] = "FP64 dense atol=rtol2e-5; same-tile compact exact; whole-cache exact"
    layouts = deepcopy(base["layouts"])
    layouts += [
        {
            "layout_id": "L13",
            "row_count": 8,
            "table_width": 9,
            "live_rows": [1, 3, 5, 7],
            "lengths": [63, 64, 65, 129],
        },
        {
            "layout_id": "L14",
            "row_count": 128,
            "table_width": 8,
            "live_rows": list(range(128)),
            "lengths": list(range(1, 129)),
        },
    ]
    rows = []
    for source, tile in (("A", 32), ("B", 64)):
        rows.extend(
            {
                "execution_id": f"K-{source}-{layout['layout_id']}",
                "implementation_id": source,
                "kind": "kernel",
                "tile_tokens": tile,
                "layout_id": layout["layout_id"],
            }
            for layout in layouts
        )
        rows.extend(
            {
                "execution_id": f"LIFE-{source}-{strategy}",
                "implementation_id": source,
                "kind": "lifecycle",
                "tile_tokens": tile,
                "storage_strategy": strategy,
                "inner_evaluation_id": f"L-{inner}-triton",
            }
            for inner, strategy in (("A", "allocating"), ("B", "persistent"))
        )
    plan = {
        "schema_version": 1,
        "artifact_type": "m4_attention_held_plan",
        "config": config,
        "layouts": layouts,
        "execution_order": rows,
        "lifecycle_plan": lifecycle.build_lifecycle_plan(),
        "lifecycle_subset": ["L-A-triton", "L-B-triton"],
        "checker": deepcopy(base["checker"]),
        "input_identities": {},
        "cache_identities": {},
        "dense_identities": {},
    }
    plan["checker"]["intended_layout_ids"] = [row["layout_id"] for row in layouts]
    for layout in layouts:
        identity = layout["layout_id"]
        tensors = _fixture(plan, layout)
        plan["input_identities"][identity] = {
            name: kernels._tensor_hash(value) for name, value in tensors.items()
        }
        plan["dense_identities"][identity] = kernels._tensor_hash(_dense(plan, layout, tensors))
        plan["cache_identities"][identity] = {
            f"{component}:{start}": kernels._tensor_hash(_chunk(plan, layout, component, start))
            for component in ("key", "value")
            for start in range(0, 160, 8)
        }
    output_rows = 2 * sum(row["row_count"] + len(row["live_rows"]) for row in layouts)
    plan["resource_estimates"] = {
        "kernel_evaluations": 30,
        "lifecycle_evaluations": 4,
        "attention_wrapper_calls": 60,
        "attention_device_calls": 52,
        "kernel_guard_chunks": 2400,
        "lifecycle_guard_chunks": 1920,
        "kernel_tensor_bytes": output_rows * 2048 * 4,
        "kernel_index_records": 60,
        "tensor_bytes_upper_bound": output_rows * 2048 * 4 + 32 * MiB,
        "index_records_upper_bound": 752,
        "json_bytes_upper_bound": 52 * MiB,
        "lifecycle_tensor_records": 692,
        "cache_bytes_per_evaluation": 80 * MiB,
        "matching_bytes": 64 * MiB,
        "kernel_evidence_bytes_upper_bound": 256 * MiB,
        "lifecycle_evidence_bytes_upper_bound": 48 * MiB,
        "auxiliary_evidence_bytes_upper_bound": 8 * MiB,
        "artifact_bytes_upper_bound": 312 * MiB,
        "json_record_bytes": MiB,
        "index_record_bytes": 1024,
    }
    plan["held_plan_sha256"] = _digest(plan)
    return plan


def build_held_plan():
    """CPU only: freeze every physical input, dense result and whole-pool chunk."""
    return deepcopy(_resolved_plan())


def validate_held_plan(plan):
    _require(_digest(plan) == _digest(_resolved_plan()), "M4 held plan differs from frozen matrix")


def _write(path, value):
    import json

    _require(
        len((json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()) <= MiB,
        "held JSON record exceeds1MiB",
    )
    write_json(path, value)


def _kernel_evaluation(plan, row, root, *, device, attention=None):
    """One production device evaluation; callable injection is for CPU unit tests."""
    import torch

    from vllm_lt.kernels.paged_attention import triton_paged_attention

    attention = attention or triton_paged_attention
    layout = next(item for item in plan["layouts"] if item["layout_id"] == row["layout_id"])
    identity = row["execution_id"]
    tensors = _fixture(plan, layout)
    hashes = {name: kernels._tensor_hash(tensor) for name, tensor in tensors.items()}
    _require(hashes == plan["input_identities"][layout["layout_id"]], "held input identity differs")
    result = {
        "input_hashes": hashes,
        "cache_before": {},
        "cache_after": {},
        "errors": [],
        "cleanup": {"completion_confirmed": False, "cache_references_released": False},
    }
    caches, inputs, outputs, arguments, spool = {}, {}, {}, None, None
    try:
        spool = TensorSpool(
            root / "outputs", identity, "attention", SpoolBudget(256 * MiB, 16 * MiB), identity
        )
        for component in ("key", "value"):
            caches[component] = torch.empty(
                (160, 2, 16, 16, 128), dtype=torch.float32, device=device
            )
            for start in range(0, 160, 8):
                caches[component][start : start + 8].copy_(_chunk(plan, layout, component, start))

        def guards():
            return {
                f"{component}:{start}": kernels._tensor_hash(cache[start : start + 8].cpu())
                for component, cache in caches.items()
                for start in range(0, 160, 8)
            }

        result["cache_before"] = guards()
        inputs = {name: tensor.to(device) for name, tensor in tensors.items()}
        live = layout["live_rows"]
        for mode in ("physical", "compact"):
            query = inputs["query"] if mode == "physical" else inputs["query"][live]
            table = inputs["block_tables"] if mode == "physical" else inputs["block_tables"][live]
            lengths = (
                inputs["context_lengths"] if mode == "physical" else inputs["context_lengths"][live]
            )
            mask = inputs["active"] if mode == "physical" else None
            arguments = (query, caches["key"][:, 1], caches["value"][:, 1], table, lengths, mask)
            outputs[mode] = attention(*arguments).cpu()
            arguments = query = table = lengths = mask = None
            spool.write(
                mode,
                outputs[mode],
                {"execution_id": identity, "layout_id": row["layout_id"], "operation": mode},
            )
        result["cache_after"] = guards()
        result.update(_kernel_checks(plan, layout, tensors, outputs, result))
    except (Exception, KeyboardInterrupt) as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
    finally:
        if spool is not None:
            try:
                spool.close()
            except (Exception, KeyboardInterrupt) as exc:
                result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
        arguments = query = table = lengths = mask = None
        caches.clear()
        inputs.clear()
        result["cleanup"]["cache_references_released"] = True
        try:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            result["cleanup"]["completion_confirmed"] = True
        except (Exception, KeyboardInterrupt) as exc:
            result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
    result["passed"] = not result["errors"]
    return result


def _kernel_checks(plan, layout, tensors, outputs, record):
    import torch

    physical, compact = outputs["physical"], outputs["compact"]
    _require(physical.dtype == compact.dtype == torch.float32, "held output dtype differs")
    dense = _dense(plan, layout, tensors)
    _require(
        kernels._tensor_hash(dense) == plan["dense_identities"][layout["layout_id"]],
        "dense identity differs",
    )
    live = layout["live_rows"]
    stats = {
        "physical": _stats(physical, dense, bounded=True),
        "compact": _stats(compact, dense[live], bounded=True),
    }
    padding = kernels._output_stats(physical, live)
    exact = bool(torch.equal(physical[live], compact))
    failures = []
    if any(not value["finite"] or not value["allclose"] for value in stats.values()):
        failures.append("held attention differs from frozen FP64 dense bound")
    if not exact:
        failures.append("same-tile compact and physical active outputs differ")
    if (
        padding["inactive_zero_count"] != padding["inactive_numel"]
        or padding["inactive_negative_zero_count"]
    ):
        failures.append("inactive output is not positive zero")
    expected = plan["cache_identities"][layout["layout_id"]]
    if record["cache_before"] != expected or record["cache_after"] != expected:
        failures.append("whole-pool cache guard differs")
    return {
        "dense": stats,
        "padding": padding,
        "compact_exact": exact,
        "errors": [{"type": "QualificationFailure", "message": message} for message in failures],
    }


def run_held_row(model, parent_plan, row, held_dir, deadline_ns):
    """One declared outer row; marker covers validation through final export/cleanup."""
    started = time.perf_counter_ns()
    deadline = min(deadline_ns, started + 600 * 10**9)
    root = Path(held_dir)
    identity, source = row["execution_id"], row["implementation_id"]
    declared = next(
        item for item in parent_plan["held"]["execution_order"] if item["execution_id"] == identity
    )
    _require(
        all(row.get(key) == value for key, value in declared.items()), "outer held row differs"
    )
    row = declared
    folder = root / "evaluations" / identity
    folder.mkdir(parents=True, exist_ok=False)
    marker = {"execution_id": identity, "started_ns": started, "deadline_ns": deadline}
    _write(folder / "started.json", marker)
    _write(root / f"active-{source}.json", marker)
    result = {
        "schema_version": 1,
        "artifact_type": "m4_attention_held_result",
        **marker,
        "held_plan_sha256": parent_plan["held"]["held_plan_sha256"],
        "plan_sha256": parent_plan["plan_sha256"],
        "row": row,
        "status": "failed",
        "passed": False,
        "errors": [],
    }
    try:
        plan = parent_plan["held"]
        validate_held_plan(plan)
        _require(row in plan["execution_order"], "unknown M4 held row")
        _require(time.perf_counter_ns() < deadline, "held deadline exhausted before allocation")
        device = next(model.parameters()).device
        _require(device.type == "cuda", "production held rows require the reserved CUDA device")
        if row["kind"] == "kernel":
            result["evidence"] = _kernel_evaluation(plan, row, root, device=device)
        else:
            result["evidence"] = lifecycle.run_lifecycle_evaluation(
                plan["lifecycle_plan"],
                row["inner_evaluation_id"],
                root / "lifecycle" / source,
                device=device,
                deadline_ns=deadline,
            )
        result["passed"] = result["evidence"]["passed"]
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
    result["ended_ns"] = time.perf_counter_ns()
    _write(folder / "result.json", result)
    try:
        kernels._usage(
            root, parent_plan["held"]["resource_estimates"]["artifact_bytes_upper_bound"]
        )
        _require(time.perf_counter_ns() < deadline, "held full-case export exceeded deadline")
        marker["case_completed_ns"] = time.perf_counter_ns()
        _write(folder / "completed.json", marker)
        _write(root / f"active-{source}.json", marker)
        _require(
            time.perf_counter_ns() < deadline, "held completion-marker export exceeded deadline"
        )
    except (Exception, KeyboardInterrupt) as exc:
        result["passed"], result["status"] = False, "failed"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
        result["ended_ns"] = time.perf_counter_ns()
        _write(folder / "result.json", result)
    return result


def active_case_deadline(held_dir, implementation_id):
    path = Path(held_dir) / f"active-{implementation_id}.json"
    if path.exists():
        marker = read_json(path)
        if "case_completed_ns" not in marker:
            return marker["deadline_ns"]
    return None


def _audit_kernel(root, plan, row, evidence):
    layout = next(value for value in plan["layouts"] if value["layout_id"] == row["layout_id"])
    tensors = _fixture(plan, layout)
    _require(
        evidence["input_hashes"] == plan["input_identities"][row["layout_id"]],
        "kernel inputs differ",
    )
    _require(
        evidence["cleanup"] == {"completion_confirmed": True, "cache_references_released": True},
        "kernel cleanup differs",
    )
    output = root / "outputs" / row["execution_id"]
    _require(
        {path.name for path in output.iterdir()} == {"attention.bin", "attention.index.json"},
        "kernel output inventory differs",
    )
    spool = TensorSpool.open(root / "outputs", row["execution_id"], "attention")
    _require(set(spool.index) == {"physical", "compact"}, "kernel tensor coverage differs")
    values = {}
    for key in spool.index:
        _require(
            spool.read_metadata(key)
            == {
                "execution_id": row["execution_id"],
                "layout_id": row["layout_id"],
                "operation": key,
            },
            "kernel tensor metadata differs",
        )
        values[key] = spool.read(key)
    checked = _kernel_checks(plan, layout, tensors, values, evidence)
    _require(
        all(evidence[key] == checked[key] for key in checked),
        "kernel reported numerical statistics differ",
    )
    _require(evidence["passed"] == (not checked["errors"]), "kernel verdict differs")
    return {
        "passed": not checked["errors"],
        "guard_chunks": 80,
        "tensor_records": 2,
        "dense": checked["dense"],
        "errors": checked["errors"],
    }


def _cross_tile_lifecycle(root, strategy):
    """Only attention has a bound; other computed outputs remain diagnostics."""
    inner = "L-A-triton" if strategy == "allocating" else "L-B-triton"
    a = TensorSpool.open(root / "lifecycle" / "A" / "outputs", inner, "evidence")
    b = TensorSpool.open(root / "lifecycle" / "B" / "outputs", inner, "evidence")
    _require(set(a.index) == set(b.index), "cross-tile lifecycle tensor coverage differs")
    stats = {}
    for key in a.index:
        computed = any(
            name in key
            for name in (
                "/attention",
                "physical_hidden",
                "physical_gates",
                "publication_",
                "held_hidden",
            )
        )
        actual, reference = b.read(key), a.read(key)
        if computed:
            value = _stats(actual, reference, bounded="/attention" in key)
            stats[key] = value
        else:
            _require(
                kernels._tensor_hash(actual) == kernels._tensor_hash(reference),
                "cross-tile held input/metadata/copy differs",
            )
    return stats


def audit_held(held_dir, parent_plan, *, expected_device="cuda:0"):
    """Offline audit of settled rows; missing suffix never becomes passing evidence."""
    root, plan = Path(held_dir), parent_plan["held"]
    validate_held_plan(plan)
    report = {
        "complete": False,
        "passed": False,
        "errors": [],
        "missing": [],
        "known_required_failure": False,
        "evaluations": [],
        "cross_tile_lifecycle": {},
    }
    life_rows = {}
    expected_ids = [row["execution_id"] for row in plan["execution_order"]]
    report["missing"] = list(expected_ids)
    expected_files = set()
    try:
        actual = {path.name for path in (root / "evaluations").iterdir() if path.is_dir()}
        _require(actual <= set(expected_ids), "unexpected held evaluation directory")
        _require(
            not any(path.is_symlink() for path in root.rglob("*")), "held artifacts contain links"
        )
        for row in plan["execution_order"]:
            identity = row["execution_id"]
            folder = root / "evaluations" / identity
            if not (folder / "result.json").exists():
                continue
            value = read_json(folder / "result.json")
            marker = read_json(folder / "started.json")
            complete = read_json(folder / "completed.json")
            _require(
                value["row"] == row
                and value["plan_sha256"] == parent_plan["plan_sha256"]
                and value["held_plan_sha256"] == plan["held_plan_sha256"]
                and value["artifact_type"] == "m4_attention_held_result"
                and value["schema_version"] == 1,
                "held result identity differs",
            )
            start, end, deadline = (
                value[name] for name in ("started_ns", "ended_ns", "deadline_ns")
            )
            _require(
                all(
                    type(n) is int and n > 0
                    for n in (start, end, deadline, complete["case_completed_ns"])
                )
                and start <= end <= complete["case_completed_ns"] < deadline <= start + 600 * 10**9
                and marker
                == {"execution_id": identity, "started_ns": start, "deadline_ns": deadline}
                and complete == {**marker, "case_completed_ns": complete["case_completed_ns"]},
                "held full-case lifetime differs",
            )
            _require(
                value["status"] == "complete" and not value["errors"],
                "held evaluation execution failed",
            )
            if row["kind"] == "kernel":
                audited = _audit_kernel(root, plan, row, value["evidence"])
                expected_files.update(
                    (root / "outputs" / identity / name)
                    for name in ("attention.bin", "attention.index.json")
                )
            else:
                inner = next(
                    item
                    for item in plan["lifecycle_plan"]["execution_order"]
                    if item["evaluation_id"] == row["inner_evaluation_id"]
                )
                inner_root = root / "lifecycle" / row["implementation_id"]
                raw = read_json(inner_root / "evaluations" / inner["evaluation_id"] / "result.json")
                _require(
                    raw == value["evidence"], "outer lifecycle evidence differs from inner result"
                )
                audited = lifecycle._audit_one(
                    inner_root, plan["lifecycle_plan"], inner, expected_device=expected_device
                )
                _require(
                    start <= audited["started_ns"] < audited["finished_ns"] <= end,
                    "inner lifecycle lifetime escapes outer row",
                )
                life_rows[identity] = audited
                expected_files.update(inner_root / name for name in audited["artifacts"])
            _require(value["passed"] == audited["passed"], "outer held verdict differs")
            report["known_required_failure"] |= not audited["passed"]
            report["missing"].remove(identity)
            expected_files.update(
                folder / name for name in ("started.json", "result.json", "completed.json")
            )
            report["evaluations"].append(
                {
                    "execution_id": identity,
                    "started_ns": start,
                    "ended_ns": end,
                    "deadline_ns": deadline,
                    "case_completed_ns": complete["case_completed_ns"],
                    **{
                        key: item
                        for key, item in audited.items()
                        if key
                        not in (
                            "tensor_hashes",
                            "gate_probabilities",
                            "started_ns",
                            "finished_ns",
                            "deadline_ns",
                        )
                    },
                }
            )
        for source in ("A", "B"):
            a, b = f"LIFE-{source}-allocating", f"LIFE-{source}-persistent"
            if a in life_rows and b in life_rows:
                _require(
                    life_rows[a]["tensor_hashes"] == life_rows[b]["tensor_hashes"]
                    and life_rows[a]["gate_probabilities"] == life_rows[b]["gate_probabilities"],
                    "same-tile allocating/persistent outputs differ",
                )
        for strategy in ("allocating", "persistent"):
            if all(f"LIFE-{source}-{strategy}" in life_rows for source in ("A", "B")):
                stats = _cross_tile_lifecycle(root, strategy)
                report["cross_tile_lifecycle"][strategy] = stats
                report["known_required_failure"] |= any(
                    not row["finite"] or row["allclose"] is False for row in stats.values()
                )
        report["artifact_bytes"] = kernels._usage(
            root, plan["resource_estimates"]["artifact_bytes_upper_bound"]
        )
        report["complete"] = not report["missing"] and len(report["evaluations"]) == len(
            expected_ids
        )
        if report["complete"]:
            for source in ("A", "B"):
                active = root / f"active-{source}.json"
                last = next(
                    row
                    for row in reversed(plan["execution_order"])
                    if row["implementation_id"] == source
                )
                _require(
                    read_json(active)
                    == read_json(root / "evaluations" / last["execution_id"] / "completed.json"),
                    "held terminal active marker differs",
                )
                expected_files.add(active)
            _require(
                {path for path in root.rglob("*") if path.is_file()} == expected_files,
                "held artifact inventory differs",
            )
        report["artifacts"] = {
            str(path.relative_to(root)): {
                "size_bytes": path.stat().st_size,
                "sha256": lifecycle._file_hash(path),
            }
            for path in sorted(expected_files)
        }
        report["passed"] = report["complete"] and not report["known_required_failure"]
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
    return report
