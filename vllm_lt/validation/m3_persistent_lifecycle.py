"""Four finite held-input storage/lifetime checks; no pretrained model or timing claim."""

import hashlib
import json
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

MiB = 1024**2
_METADATA = (
    "position_ids",
    "write_blocks",
    "write_offsets",
    "block_tables",
    "context_lengths",
    "active",
)
_STEPS = (
    (1, "four_at_510", ("long", "r1", "r2", "r3"), (0, 1, 2, 3), (510, 0, 0, 0)),
    (2, "four_at_511_hold_r2", ("long", "r1", "r2", "r3"), (0, 1, 2, 3), (511, 1, 1, 1)),
    (3, "shrink_one", ("r1",), (1,), (2,)),
    (4, "empty", (), (), ()),
    (5, "reorder_two", ("r3", "r1"), (3, 1), (2, 3)),
    (6, "reallocate_r1_and_reject_stale", (), (), ()),
    (7, "recycled_two", ("r1", "r3"), (1, 3), (0, 3)),
    (8, "width_33_fallback", ("long",), (0,), (512,)),
    (9, "retire_long_then_four", ("r1", "r3", "r4", "r5"), (1, 3, 0, 1), (1, 4, 0, 0)),
    (10, "five_live_fallback", ("r1", "r3", "r4", "r5", "r6"), (1, 3, 0, 1, 2), (2, 5, 1, 1, 0)),
    (11, "return_one", ("r6",), (2,), (1,)),
)
_REQUEST_TAGS = {"long": 1, **{f"r{i}": i + 1 for i in range(1, 7)}}
_SETUP = (
    ("allocate", "guard0", 16),
    ("allocate", "temporary", 16),
    ("allocate", "guard1", 16),
    ("free", "temporary", None),
    ("allocate", "long", 513),
    *(("allocate", f"r{i}", 16) for i in range(1, 5)),
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path):
    from .schema import read_json

    return read_json(path)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor_hash(tensor):
    import torch

    value = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * MiB):
            h.update(chunk)
    return h.hexdigest()


def _hidden(step, request_ids):
    import torch

    channels = torch.arange(2048, dtype=torch.int32)
    rows = [
        ((channels * 3 + step * 19 + _REQUEST_TAGS[key] * 23) % 127 - 63).float() / 64
        for key in request_ids
    ]
    return torch.stack(rows) if rows else torch.empty((0, 2048), dtype=torch.float32)


def _seed(layer, component, positions):
    import torch

    positions = torch.as_tensor(positions, dtype=torch.int32).reshape(-1, 1)
    channels = torch.arange(2048, dtype=torch.int32).reshape(1, -1)
    return (
        (positions * 17 + channels * 3 + layer * 11 + component * 7 + 5) % 127 - 63
    ).float() / 64


def _kv(hidden, layer):
    return hidden / (2 ** (layer + 1)), hidden + (layer + 1) / 32


def _base_chunk(component, start):
    import torch

    elements_per_page = 2 * 16 * 16 * 128
    values = torch.arange(
        start * elements_per_page, (start + 8) * elements_per_page, dtype=torch.int32
    )
    values.add_(component * 53).remainder_(127).sub_(63)
    return values.float().div_(64).reshape(8, 2, 16, 16, 128)


def _expected_chunk(component, start, writes):
    result = _base_chunk(component, start)
    for (block, layer, offset), spec in writes.items():
        if not start <= block < start + 8:
            continue
        if spec[0] == "seed":
            row = _seed(layer, component, [spec[1]])[0]
        else:
            row = _kv(_hidden(spec[1], [spec[2]]), layer)[component][0]
        result[block - start, layer, offset] = row.reshape(16, 128)
    return result


def _metadata(step):
    import torch

    count = len(step["request_ids"])
    padded = step["kind"] == "supported"
    rows, width = (8, 32) if padded else (count, step["required_width"])
    values = {
        "position_ids": torch.zeros(rows, dtype=torch.int64),
        "write_blocks": torch.full((rows,), -1, dtype=torch.int64),
        "write_offsets": torch.full((rows,), -1, dtype=torch.int64),
        "block_tables": torch.full((rows, width), -1, dtype=torch.int32),
        "context_lengths": torch.zeros(rows, dtype=torch.int32),
    }
    if padded:
        values["active"] = torch.zeros(rows, dtype=torch.bool)
    for index, (key, depth, position) in enumerate(
        zip(step["request_ids"], step["depths"], step["positions"])
    ):
        row = 2 * index + 1 if padded else index
        table = step["allocations"][key]["block_tables"][depth]
        values["position_ids"][row] = position
        values["write_blocks"][row] = table[position // 16]
        values["write_offsets"][row] = position % 16
        values["context_lengths"][row] = position + 1
        selected = table[:width]
        values["block_tables"][row, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        if padded:
            values["active"][row] = True
    return values


def _physical_hidden(step):
    import torch

    hidden = _hidden(step["step_id"], step["request_ids"])
    if step["kind"] != "supported":
        return hidden
    physical = torch.zeros((8, 2048), dtype=torch.float32)
    physical[step["live_rows"]] = hidden
    return physical


@lru_cache(maxsize=1)
def _resolved_plan():
    free, allocations, serial = list(reversed(range(160))), {}, 0

    def event(action, key, maximum):
        nonlocal serial
        if action == "free":
            for table in allocations.pop(key)["block_tables"]:
                free.extend(reversed(table))
        else:
            serial += 1
            pages = (maximum + 15) // 16
            allocations[key] = {
                "allocation_id": serial,
                "max_tokens": maximum,
                "block_tables": [[free.pop() for _ in range(pages)] for _ in range(4)],
            }
        return {"action": action, "request_id": key, "max_tokens": maximum}

    setup = [event(*row) for row in _SETUP]
    initial = deepcopy(allocations)
    writes = {}
    for position in range(510):
        table = initial["long"]["block_tables"][0]
        for layer in range(2):
            writes[table[position // 16], layer, position % 16] = ("seed", position)
    guards = {}

    def snapshot(dirty=None):
        for component in range(2):
            for start in range(0, 160, 8):
                if dirty is None or start in dirty:
                    guards[component, start] = _tensor_hash(
                        _expected_chunk(component, start, writes)
                    )
        return [
            {
                "component": component,
                "start_block": start,
                "blocks": 8,
                "size_bytes": 2 * MiB,
                "sha256": guards[component, start],
            }
            for component in range(2)
            for start in range(0, 160, 8)
        ]

    initial_guards = snapshot()
    steps = []
    for number, name, keys, depths, positions in _STEPS:
        actions = []
        if number == 6:
            actions = [event("free", "r1", None), event("allocate", "r1", 16)]
        elif number == 9:
            actions = [
                event("free", "long", None),
                event("allocate", "r5", 16),
                event("allocate", "r6", 16),
            ]
        width = max((position // 16 + 1 for position in positions), default=0)
        reason = "live_count" if len(keys) > 4 else "table_width" if width > 32 else None
        kind = (
            "host_only"
            if number == 6
            else "empty"
            if not keys
            else "fallback"
            if reason
            else "supported"
        )
        step = {
            "step_id": number,
            "name": name,
            "request_ids": list(keys),
            "depths": list(depths),
            "positions": list(positions),
            "actions": actions,
            "allocations": deepcopy(allocations),
            "used_blocks": 160 - len(free),
            "required_width": width,
            "kind": kind,
            "fallback_reason": reason,
            "row_count": len(keys) if reason else 0 if not keys else 8,
            "live_rows": list(range(len(keys)))
            if reason
            else [2 * i + 1 for i in range(len(keys))],
        }
        dirty = set()
        if keys:
            physical = _physical_hidden(step)
            step["input_hashes"] = {
                "logical_hidden": _tensor_hash(_hidden(number, keys)),
                "physical_hidden": _tensor_hash(physical),
                **{key: _tensor_hash(value) for key, value in _metadata(step).items()},
            }
            for key, depth, position in zip(keys, depths, positions):
                block = allocations[key]["block_tables"][depth][position // 16]
                dirty.add(block // 8 * 8)
                for layer in range(2):
                    writes[block, layer, position % 16] = ("step", number, key)
        else:
            step["input_hashes"] = {"logical_hidden": _tensor_hash(_hidden(number, keys))}
        step["guard_chunks"] = snapshot(dirty)
        steps.append(step)
    plan = {
        "schema_version": 1,
        "artifact_type": "m3_persistent_lifecycle_plan",
        "config": {
            "dtype": "float32",
            "hidden_size": 2048,
            "heads": 16,
            "head_dim": 128,
            "num_blocks": 160,
            "layers": 2,
            "block_size": 16,
            "max_loops": 4,
            "row_count": 8,
            "table_width": 32,
            "max_live_rows": 4,
            "chunk_blocks": 8,
            "case_timeout_s": 600,
            "tensor_bytes_per_evaluation": 8 * MiB,
            "json_bytes_per_evaluation": 4 * MiB,
            "suite_evidence_bytes": 48 * MiB,
            "matching_bytes": 64 * MiB,
        },
        "empty_policy": (
            "one explicit helper invocation per side returns owned empty outputs "
            "without a core or prepare; B last_dispatch retains reserved capacity 8x32"
        ),
        "scope": (
            "deterministic held-input two-layer boundary, without pretrained weights; "
            "13 separate Ouro cases qualify full-model behavior"
        ),
        "formulas": {
            "hidden": "((channel*3 + step*19 + request_tag*23)%127-63)/64",
            "seed": "((position*17 + channel*3 + layer*11 + component*7 + 5)%127-63)/64",
            "cache": "((global_component_flat_index + component*53)%127-63)/64",
            "query": "hidden",
            "key": "hidden/(2**(layer+1))",
            "value": "hidden+(layer+1)/32",
            "output_hidden": "layer0_attention + layer1_attention",
            "gate": "output_hidden[:,0]",
            "inactive": (
                "zero hidden before held arithmetic; finite positive-zero "
                "attention/hidden/gate after masking"
            ),
            "request_tags": dict(_REQUEST_TAGS),
        },
        "setup_actions": setup,
        "initial_allocations": initial,
        "initial_used_blocks": 156,
        "seed": {
            "request_id": "long",
            "depth": 0,
            "positions": [0, 509],
            "input_hashes": {
                f"{layer}:{component}": _tensor_hash(_seed(layer, component, list(range(510))))
                for layer in range(2)
                for component in range(2)
            },
        },
        "initial_guard_chunks": initial_guards,
        "steps": steps,
        "expected_persistent": {
            "generation": 7,
            "counters": {"calls": 10, "prepared": 7, "completed": 7, "empty": 1},
            "fallback_counts": {"live_count": 1, "table_width": 1},
            "device_payload_bytes": 132392,
            "cpu_staging_bytes": 1256,
        },
        "execution_order": [
            {
                "evaluation_id": f"L-{side}-{backend}",
                "implementation_id": side,
                "backend": backend,
                "storage_strategy": "allocating_padded" if side == "A" else "persistent_decode",
            }
            for side in ("A", "B")
            for backend in ("torch", "triton")
        ],
        "resource_estimates": {
            "evaluations": 4,
            "cache_bytes_per_evaluation": 80 * MiB,
            "guard_snapshots_per_evaluation": 12,
            "guard_chunks_per_evaluation": 480,
            "tensor_records_per_evaluation": 173,
            "tensor_bytes_upper_bound": 32 * MiB,
            "json_bytes_upper_bound": 16 * MiB,
            "artifact_bytes_upper_bound": 48 * MiB,
        },
        "limits": [
            "no retries or device fault injection",
            "only valid capacity overflow falls back",
            "missing evidence cannot pass",
        ],
    }
    plan["lifecycle_plan_sha256"] = _digest(plan)
    return plan


def build_lifecycle_plan():
    """Freeze four evaluations and CPU-computed inputs/whole-pool guard hashes."""
    return deepcopy(_resolved_plan())


def validate_lifecycle_plan(plan):
    _require(
        _digest(plan) == _digest(_resolved_plan()),
        "lifecycle plan differs from the frozen exact sequence",
    )


def _description(tensor):
    return {
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "size_bytes": tensor.numel() * tensor.element_size(),
    }


def _buffer_hashes(runner):
    return {key: _tensor_hash(value) for key, value in runner._persistent["tensors"].items()}


def _allocations(cache, declared):
    _require(set(cache._allocations) == set(declared), "request allocation set changed")
    for key, expected in declared.items():
        allocation = cache._allocations[key]
        _require(
            allocation.max_tokens == expected["max_tokens"]
            and [list(table) for table in allocation.block_tables] == expected["block_tables"],
            f"physical allocation differs for {key}",
        )
    return {
        key: {
            "allocation_id": row["allocation_id"],
            "object_id": id(cache._allocations[key]),
            "block_tables": row["block_tables"],
        }
        for key, row in declared.items()
    }


def _held_state(request):
    return {
        "request_id": request.request_id,
        "stage": request.stage.value,
        "prompt_token_ids": list(request.prompt_token_ids),
        "generated_token_ids": list(request.generated_token_ids),
        "exit_depths": list(request.exit_depths),
        "loops_done": request.loops_done,
        "remaining_probability": request.remaining_probability,
        "hidden_sha256": _tensor_hash(request.hidden_state),
        "hidden_storage_ptr": request.hidden_state.untyped_storage().data_ptr(),
        "generator_sha256": _tensor_hash(request.generator.get_state()),
    }


def _probe_model(device, observer, emit):
    """A weight-free held-input boundary; it is deliberately not an Ouro reference."""
    from types import SimpleNamespace

    import torch

    class Boundary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
            self.config = SimpleNamespace(hidden_size=2048)
            self.observer, self.emit = observer, emit

        def recurrent(self, hidden, request_ids, depths, positions, cache):
            batch = cache._prepare_batch(request_ids, depths, positions)
            return self._recurrent_prepared(hidden, batch, cache)

        def _recurrent_prepared(self, hidden, batch, cache):
            cache._require_live_batch(batch)
            self.observer(hidden, batch)
            self.emit("input", hidden)
            for name in _METADATA:
                value = getattr(batch, name)
                if value is not None:
                    self.emit("metadata/" + name, value)
            active = batch.active
            clean = hidden if active is None else hidden.masked_fill(~active[:, None], 0)
            outputs = []
            for layer in range(2):
                query = clean.reshape(-1, 16, 128)
                key, value = _kv(clean, layer)
                key, value = key.reshape(-1, 16, 128), value.reshape(-1, 16, 128)
                self.emit(f"layer{layer}/query", query)
                self.emit(f"layer{layer}/key", key)
                self.emit(f"layer{layer}/value", value)
                cache._write_prepared(layer, batch, key, value)
                output = cache._attend_prepared(layer, batch, query)
                self.emit(f"layer{layer}/attention", output)
                _require(bool(output.isfinite().all()), "nonfinite held attention")
                if active is not None:
                    inactive = output[~active]
                    _require(
                        not bool(inactive.count_nonzero()) and not bool(inactive.signbit().any()),
                        "inactive attention is not positive zero",
                    )
                outputs.append(output.reshape(-1, 2048))
            hidden_out = outputs[0] + outputs[1]
            gates = hidden_out[:, 0].clone(memory_format=torch.contiguous_format)
            self.emit("physical_hidden", hidden_out)
            self.emit("physical_gates", gates)
            _require(
                bool(hidden_out.isfinite().all()) and bool(gates.isfinite().all()),
                "nonfinite held output",
            )
            if active is not None:
                _require(
                    not bool(hidden_out[~active].count_nonzero())
                    and not bool(gates[~active].count_nonzero()),
                    "inactive held output changed",
                )
            return hidden_out, gates

        def coda(self, hidden):
            raise AssertionError("held coda request must not be sampled by lifecycle checks")

    return Boundary()


def run_lifecycle_evaluation(plan, evaluation_id, output_dir, device="cuda", deadline_ns=None):
    """Execute one fixed sequence and preserve partial evidence/observed cleanup on failure."""
    started = time.perf_counter_ns()
    import gc
    import weakref

    import torch

    from vllm_lt.core.kv_cache_manager import KVCacheManager
    from vllm_lt.core.scheduler import ScheduledItem, SchedulerOutput
    from vllm_lt.request import Request, Stage
    from vllm_lt.sampling_params import SamplingParams
    from vllm_lt.worker.model_runner import ModelRunner

    from .diagnostics import SpoolBudget, TensorSpool

    _require(
        evaluation_id
        in {f"L-{side}-{backend}" for side in ("A", "B") for backend in ("torch", "triton")},
        "unknown lifecycle evaluation",
    )
    matches = [row for row in plan["execution_order"] if row["evaluation_id"] == evaluation_id]
    _require(len(matches) == 1, "unknown lifecycle evaluation")
    evaluation = matches[0]
    root, target = Path(output_dir), torch.device(device)
    folder = root / "evaluations" / evaluation_id
    _require(not folder.exists(), "lifecycle evaluation cannot be retried")
    folder.mkdir(parents=True)
    deadline = min(
        started + 600 * 10**9, deadline_ns if deadline_ns is not None else started + 600 * 10**9
    )
    _write(
        folder / "started.json",
        {"evaluation": evaluation, "started_ns": started, "deadline_ns": deadline},
    )
    result = {
        "schema_version": 1,
        "artifact_type": "m3_persistent_lifecycle_result",
        "evaluation": evaluation,
        "lifecycle_plan_sha256": plan["lifecycle_plan_sha256"],
        "started_ns": started,
        "deadline_ns": deadline,
        "status": "failed",
        "passed": False,
        "errors": [],
        "steps": [],
        "initial_guard_chunks": [],
        "initial_allocations": None,
        "persistent_initial": None,
        "persistent_final": None,
        "held_checks": [],
        "cleanup": {
            "completion_confirmed": False,
            "used_blocks": None,
            "active_requests": None,
            "cache_references_released": False,
            "buffer_references_released": False,
        },
    }
    persistent = evaluation["implementation_id"] == "B"
    cache = runner = model = spool = reference = None
    requests, state, weak_cache, weak_buffers = {}, {}, [], []
    old_host = old_descriptor = old_allocation = held = hidden = probabilities = tensor = None
    allocating = original = publication = None
    device_work_started = False

    def check_time():
        if time.perf_counter_ns() >= deadline:
            raise TimeoutError("lifecycle evaluation deadline exceeded")

    def emit(operation, tensor):
        check_time()
        step = state["step"]
        key = f"step{step['step_id']:02d}/{operation}"
        record = spool.write(
            key,
            tensor,
            {"evaluation_id": evaluation_id, "step_id": step["step_id"], "operation": operation},
        )
        if reference is not None:
            prior = reference._verified_record(key)
            reference.read(key)  # Verify retained A payload bytes, not only its index.
            _require(
                (record["dtype"], record["shape"], record["sha256"])
                == (prior["dtype"], prior["shape"], prior["sha256"]),
                f"held B/A tensor differs at {key}",
            )
        return record

    def observe(hidden, batch):
        step, record = state["step"], state["record"]
        _require("core" not in record, "multiple held cores in one dispatch")
        _require(
            batch.row_count == step["row_count"] and list(batch.live_rows) == step["live_rows"],
            "physical row map differs",
        )
        record["core"] = {
            "input": _description(hidden),
            "metadata": {
                name: _description(getattr(batch, name))
                for name in _METADATA
                if getattr(batch, name) is not None
            },
        }
        actual = {
            "physical_hidden": _tensor_hash(hidden),
            **{
                name: _tensor_hash(getattr(batch, name))
                for name in _METADATA
                if getattr(batch, name) is not None
            },
        }
        _require(
            actual == {k: v for k, v in step["input_hashes"].items() if k != "logical_hidden"},
            "held physical inputs differ from plan",
        )
        record["input_hashes"] = actual
        if persistent and step["kind"] == "supported":
            snapshot = runner._persistent_snapshot()
            _require(snapshot["status"] == "in_flight", "lease released before core completion")
            record["in_flight"] = snapshot
            owned = snapshot["tensors"]
            for name, description in {
                "hidden_in": record["core"]["input"],
                **record["core"]["metadata"],
            }.items():
                _require(
                    description == {key: owned[name][key] for key in description},
                    "core did not borrow persistent storage",
                )
            host = cache._prepare_host_batch(step["request_ids"], step["depths"], step["positions"])
            try:
                cache._prepare_into(runner._persistent["metadata"], host)
            except RuntimeError:
                record["busy_lease_rejected"] = True
            else:
                raise ValueError("in-flight metadata accepted another lease")
        state["descriptor"] = batch

    def guards(expected, rows):
        for item in expected:
            check_time()
            tensor = (cache.key_cache, cache.value_cache)[item["component"]]
            actual = _tensor_hash(tensor[item["start_block"] : item["start_block"] + 8])
            row = {**item, "actual_sha256": actual}
            rows.append(row)
            _require(actual == item["sha256"], "whole-pool write/guard hash differs")
        return rows

    def apply(actions):
        for item in actions:
            if item["action"] == "free":
                cache.free(item["request_id"])
                requests.pop(item["request_id"], None)
            else:
                _require(
                    cache.allocate(item["request_id"], item["max_tokens"]),
                    "frozen lifecycle admission failed",
                )
                if item["request_id"].startswith("r") or item["request_id"] == "long":
                    size = 511 if item["request_id"] == "long" else 1
                    requests[item["request_id"]] = Request(
                        item["request_id"], [1] * size, SamplingParams(max_tokens=600)
                    )

    try:
        validate_lifecycle_plan(plan)
        check_time()
        device_work_started = True
        cache = KVCacheManager(
            2,
            16,
            128,
            160,
            16,
            4,
            device=target,
            dtype=torch.float32,
            backend=evaluation["backend"],
        )
        weak_cache = [weakref.ref(cache.key_cache), weakref.ref(cache.value_cache)]
        for component, tensor in enumerate((cache.key_cache, cache.value_cache)):
            for start in range(0, 160, 8):
                check_time()
                tensor[start : start + 8].copy_(_base_chunk(component, start))
        tensor = None
        apply(plan["setup_actions"])
        result["initial_allocations"] = _allocations(cache, plan["initial_allocations"])
        _require(cache.num_used_blocks == 156, "initial lifecycle page count differs")
        for layer in range(2):
            key, value = (_seed(layer, component, list(range(510))) for component in range(2))
            _require(
                _tensor_hash(key) == plan["seed"]["input_hashes"][f"{layer}:0"]
                and _tensor_hash(value) == plan["seed"]["input_hashes"][f"{layer}:1"],
                "seed input differs",
            )
            cache.write(
                layer,
                ["long"] * 510,
                [0] * 510,
                list(range(510)),
                key.to(target).reshape(510, 16, 128),
                value.to(target).reshape(510, 16, 128),
            )
        key = value = None
        guards(plan["initial_guard_chunks"], result["initial_guard_chunks"])
        spool = TensorSpool(
            root / "outputs",
            evaluation_id,
            "evidence",
            SpoolBudget(8 * MiB, 8 * MiB),
            evaluation_id,
            max_tensor_bytes=MiB,
            matching_bytes=64 * MiB,
        )
        if persistent:
            prior_id = evaluation_id.replace("L-B-", "L-A-", 1)
            prior = _read(root / "evaluations" / prior_id / "result.json")
            _require(
                prior["passed"] and prior["status"] == "complete",
                "missing successful held A reference",
            )
            reference = TensorSpool.open(root / "outputs", prior_id, "evidence")
        model = _probe_model(target, observe, emit)
        runner = ModelRunner(model, cache)
        if persistent:
            from .m3_persistent import _enable_frozen_persistent_decode

            _enable_frozen_persistent_decode(runner)
            for name, tensor in runner._persistent["tensors"].items():
                if name != "live_indices":
                    tensor.zero_()
            for tensor in runner._persistent["metadata"].staging.values():
                tensor.zero_()
            tensor = None
            result["persistent_initial"] = runner._persistent_snapshot()
            weak_buffers = [
                weakref.ref(tensor) for tensor in runner._persistent["tensors"].values()
            ]
        else:

            def allocating(hidden, ids, depths, positions):
                reason = len(ids) > 4 or max((p // 16 + 1 for p in positions), default=0) > 32
                if not ids:
                    return hidden.clone(), hidden.new_empty(0)
                if reason:
                    return model.recurrent(hidden, ids, depths, positions, cache)
                return runner._recurrent_padded(
                    hidden,
                    ids,
                    depths,
                    positions,
                    row_indices=tuple(2 * i + 1 for i in range(len(ids))),
                    row_count=8,
                    table_width=32,
                )

            runner._recurrent = allocating
        for step in plan["steps"]:
            check_time()
            record = {
                "step_id": step["step_id"],
                "name": step["name"],
                "kind": step["kind"],
                "guard_chunks": [],
                "status": "incomplete",
                "allocations": None,
            }
            result["steps"].append(record)
            state["step"], state["record"] = step, record
            before_hashes = _buffer_hashes(runner) if persistent else None
            if step["step_id"] == 6:
                old_allocation = cache._allocations["r1"]
                old_host = cache._prepare_host_batch(["r1"], [1], [3]) if persistent else None
                old_descriptor = state["descriptor"]
                record["previous_r1_object_id"] = id(old_allocation)
                record["previous_r1_pages"] = sorted(
                    page for table in old_allocation.block_tables for page in table
                )
            apply(step["actions"])
            record["allocations"] = _allocations(cache, step["allocations"])
            _require(cache.num_used_blocks == step["used_blocks"], "lifecycle used pages differ")
            if step["step_id"] == 6:
                new = cache._allocations["r1"]
                _require(new is not old_allocation, "recycled ID retained the old allocation")
                _require(
                    sorted(page for table in new.block_tables for page in table)
                    == record["previous_r1_pages"],
                    "recycle did not reuse the same physical pages",
                )
                try:
                    if persistent:
                        cache._prepare_into(runner._persistent["metadata"], old_host)
                    else:
                        cache._require_live_batch(old_descriptor)
                except RuntimeError:
                    record["stale_allocation_rejected"] = True
                else:
                    raise ValueError("stale allocation was accepted after recycled request ID")
                if persistent:
                    try:
                        cache._require_live_batch(old_descriptor)
                    except RuntimeError:
                        record["stale_generation_rejected"] = True
                    else:
                        raise ValueError("expired descriptor became valid after reuse")
                new = old_allocation = old_descriptor = old_host = None
            elif step["kind"] == "empty":
                hidden = _hidden(step["step_id"], []).to(target)
                outputs = runner._recurrent(hidden, [], [], [])
                emit("publication_hidden", outputs[0])
                emit("publication_gates", outputs[1])
                _require(
                    outputs[0].shape == (0, 2048) and outputs[1].shape == (0,),
                    "empty publication differs",
                )
                outputs = None
            else:
                hidden = _hidden(step["step_id"], step["request_ids"]).to(target)
                _require(
                    _tensor_hash(hidden) == step["input_hashes"]["logical_hidden"],
                    "logical hidden differs",
                )
                selected = []
                for index, (key, depth, position) in enumerate(
                    zip(step["request_ids"], step["depths"], step["positions"])
                ):
                    request = requests[key]
                    request.generated_token_ids = [1] * (
                        position - len(request.prompt_token_ids) + 1
                    )
                    request.loops_done, request.hidden_state, request.stage = (
                        depth,
                        hidden[index],
                        Stage.RECURRENT,
                    )
                    selected.append(ScheduledItem(request))
                original = runner._recurrent_persistent if persistent else runner._recurrent

                def publication(*args, **kwargs):
                    out = original(*args, **kwargs)
                    record["publication"] = {
                        "hidden": _description(out[0]),
                        "gates": _description(out[1]),
                    }
                    emit("publication_hidden", out[0])
                    emit("publication_gates", out[1])
                    return out

                if persistent:
                    runner._recurrent_persistent = publication
                else:
                    runner._recurrent = publication
                try:
                    probabilities = runner.execute(SchedulerOutput(Stage.RECURRENT, selected))
                finally:
                    if persistent:
                        del runner._recurrent_persistent
                    else:
                        runner._recurrent = original
                _require(
                    len(probabilities) == len(selected), "gate publication includes padded rows"
                )
                record["gate_probabilities"] = probabilities
                record["request_hidden"] = [
                    _description(item.request.hidden_state) for item in selected
                ]
                if persistent:
                    owned = {
                        item["storage_ptr"]
                        for item in runner._persistent_snapshot()["tensors"].values()
                    }
                    _require(
                        all(item["storage_ptr"] not in owned for item in record["request_hidden"]),
                        "published request state aliases persistent storage",
                    )
                if step["step_id"] == 2:
                    held = requests["r2"]
                    held.stage = Stage.CODA
                    held.generator = torch.Generator(device="cpu").manual_seed(17)
                    result["held_initial"] = _held_state(held)
                    emit("held_hidden", held.hidden_state)
                    emit("held_generator", held.generator.get_state())
                selected = request = original = publication = None
            if persistent:
                snapshot = runner._persistent_snapshot()
                _require(snapshot["status"] == "ready", "completed action retained an active lease")
                record["persistent"] = snapshot
                if step["kind"] == "supported":
                    try:
                        cache._require_live_batch(state["descriptor"])
                    except RuntimeError:
                        record["expired_generation_rejected"] = True
                    else:
                        raise ValueError("completed borrowed descriptor retained validity")
                record["buffer_hashes"] = _buffer_hashes(runner)
                if step["kind"] != "supported":
                    _require(
                        record["buffer_hashes"] == before_hashes,
                        "empty/fallback/stale action mutated persistent buffers",
                    )
                record["unchanged_outside_supported"] = (
                    step["kind"] == "supported" or record["buffer_hashes"] == before_hashes
                )
                _require(
                    snapshot["tensors"] == result["persistent_initial"]["tensors"]
                    and snapshot["staging_tensors"]
                    == result["persistent_initial"]["staging_tensors"],
                    "persistent tensor pointer/shape/stride changed",
                )
            if held is not None:
                actual_held = _held_state(held)
                _require(
                    actual_held == result["held_initial"], "paused coda/RNG/publication changed"
                )
                result["held_checks"].append({"step_id": step["step_id"], "state": actual_held})
            guards(step["guard_chunks"], record["guard_chunks"])
            record["status"] = "complete"
            hidden = probabilities = None
        if persistent:
            result["persistent_final"] = runner._persistent_snapshot()
            expected = plan["expected_persistent"]
            for field, value in expected.items():
                _require(result["persistent_final"][field] == value, f"persistent {field} differs")
        result["status"], result["passed"] = "complete", True
    except (Exception, KeyboardInterrupt) as exc:
        result["status"], result["passed"] = (
            ("incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"),
            False,
        )
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1024]})
    finally:
        for writer in (spool, reference):
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    result["status"], result["passed"] = "failed", False
                    result["errors"].append(
                        {"type": "evidence_cleanup", "message": str(exc)[:1024]}
                    )
        # Establish completion before freeing pages, including execution failures.
        confirmed = target.type != "cuda" or not device_work_started
        if target.type == "cuda" and device_work_started:
            try:
                torch.cuda.synchronize(target)
                confirmed = True
            except Exception as exc:
                result["status"], result["passed"] = "failed", False
                result["errors"].append({"type": "completion_cleanup", "message": str(exc)[:1024]})
        result["cleanup"]["completion_confirmed"] = confirmed
        if cache is not None and confirmed:
            try:
                for key in list(cache._allocations):
                    cache.free(key)
                result["cleanup"]["used_blocks"] = cache.num_used_blocks
                requests.clear()
                result["cleanup"]["active_requests"] = 0
            except Exception as exc:
                result["status"], result["passed"] = "failed", False
                result["errors"].append({"type": "request_cleanup", "message": str(exc)[:1024]})
        if model is not None:
            model.observer = model.emit = None
        if runner is not None:
            runner.__dict__.pop("_recurrent", None)
            runner.__dict__.pop("_recurrent_persistent", None)
        requests.clear()
        state.clear()
        cache = runner = model = held = hidden = probabilities = tensor = None
        old_host = old_descriptor = old_allocation = None
        # These names may hold device views when a diagnostic raises mid-dispatch.
        key = value = outputs = selected = request = original = publication = new = allocating = (
            None
        )
        gc.collect()
        result["cleanup"]["cache_references_released"] = bool(weak_cache) and all(
            ref() is None for ref in weak_cache
        )
        result["cleanup"]["buffer_references_released"] = all(ref() is None for ref in weak_buffers)
        expected_cleanup = {
            "completion_confirmed": True,
            "used_blocks": 0,
            "active_requests": 0,
            "cache_references_released": True,
            "buffer_references_released": True,
        }
        if result["passed"] and result["cleanup"] != expected_cleanup:
            result["status"], result["passed"] = "failed", False
            result["errors"].append(
                {"type": "lifecycle_cleanup", "message": "lifecycle cleanup evidence is incomplete"}
            )
        _persist_result(folder, root / "outputs" / evaluation_id, result, deadline)
    return result


def _evidence_bytes(folder, output):
    files = [
        path for directory in (folder, output) for path in directory.rglob("*") if path.is_file()
    ]
    return (
        sum(path.stat().st_size for path in files if path.suffix == ".bin"),
        sum(path.stat().st_size for path in files if path.suffix != ".bin"),
    )


def _persist_result(folder, output, result, deadline):
    """Publish the terminal marker after export/cap work; late persistence cannot pass."""
    staging, terminal = folder / "result.pending.json", folder / "result.json"

    def late(now):
        if now >= deadline:
            result["status"], result["passed"] = "incomplete", False
            if not any(error["type"] == "deadline" for error in result["errors"]):
                result["errors"].append(
                    {"type": "deadline", "message": "lifecycle lifetime exceeded"}
                )

    def error(exc):
        result["status"], result["passed"] = "failed", False
        result["errors"].append({"type": "result_persistence", "message": str(exc)[:1024]})

    result["finished_ns"] = time.perf_counter_ns()
    # A nonterminal staging artifact keeps the parent watchdog active throughout
    # export and the complete cap scan. No result.json exists yet.
    other_json_bytes = None
    try:
        _write(staging, result)
        tensor_bytes, json_bytes = _evidence_bytes(folder, output)
        other_json_bytes = json_bytes - staging.stat().st_size
        _require(
            tensor_bytes <= 8 * MiB and json_bytes <= 4 * MiB, "lifecycle evidence cap exceeded"
        )
    except Exception as exc:
        error(exc)
    result["finished_ns"] = time.perf_counter_ns()
    late(result["finished_ns"])
    # This final serialization is small and still occurs before the terminal
    # marker. The following check also catches a slow filesystem write.
    try:
        _write(staging, result)
        if result["passed"]:
            _require(
                other_json_bytes is not None
                and other_json_bytes + staging.stat().st_size <= 4 * MiB,
                "final lifecycle JSON exceeds evidence cap",
            )
    except Exception as exc:
        error(exc)
        result["finished_ns"] = time.perf_counter_ns()
        late(result["finished_ns"])
        _write(staging, result)
    written = time.perf_counter_ns()
    if written >= deadline:
        result["finished_ns"] = written
        late(written)
        _write(staging, result)
    staging.replace(terminal)
    persisted = time.perf_counter_ns()
    if persisted >= deadline:
        result["finished_ns"] = persisted
        late(persisted)
        _write(terminal, result)


def _guard_evidence(actual, expected):
    _require(
        actual == [{**row, "actual_sha256": row["sha256"]} for row in expected],
        "whole-pool guard evidence is missing or differs",
    )


def _allocation_evidence(actual, expected):
    _require(
        isinstance(actual, dict) and set(actual) == set(expected), "allocation evidence set differs"
    )
    for key, row in expected.items():
        observed = actual[key]
        _require(
            set(observed) == {"allocation_id", "object_id", "block_tables"}
            and observed["allocation_id"] == row["allocation_id"]
            and observed["block_tables"] == row["block_tables"]
            and type(observed["object_id"]) is int
            and observed["object_id"] > 0,
            "allocation identity or pages differ",
        )
    _require(
        len({row["object_id"] for row in actual.values()}) == len(actual),
        "simultaneous allocations alias",
    )


def _pointer(description, shape, dtype, *, device=None, snapshot=False):
    sizes = {"torch.float32": 4, "torch.int64": 8, "torch.int32": 4, "torch.bool": 1}
    fields = {"data_ptr", "storage_ptr", "shape", "stride", "dtype", "device", "size_bytes"}
    if snapshot:
        fields.add("storage_bytes")
    stride, elements = [], 1
    for dimension in reversed(shape):
        stride.insert(0, elements)
        elements *= dimension
    _require(
        isinstance(description, dict) and set(description) == fields,
        "pointer evidence fields differ",
    )
    _require(
        description["shape"] == shape
        and description["stride"] == stride
        and description["dtype"] == dtype
        and description["size_bytes"] == elements * sizes[dtype],
        "pointer geometry differs",
    )
    for name in ("data_ptr", "storage_ptr"):
        _require(
            type(description[name]) is int and description[name] > 0, "pointer evidence missing"
        )
    _require(description["data_ptr"] >= description["storage_ptr"], "pointer precedes storage")
    _require(
        isinstance(description["device"], str)
        and (device is None or description["device"] == device),
        "pointer device differs",
    )
    if snapshot:
        _require(
            description["storage_bytes"] == description["size_bytes"]
            and description["data_ptr"] == description["storage_ptr"],
            "persistent tensor is not an owned complete storage",
        )


def _owned_layout(initial, expected_device):
    layout = {
        "position_ids": ([8], "torch.int64"),
        "write_blocks": ([8], "torch.int64"),
        "write_offsets": ([8], "torch.int64"),
        "block_tables": ([8, 32], "torch.int32"),
        "context_lengths": ([8], "torch.int32"),
        "active": ([8], "torch.bool"),
        "hidden_in": ([8, 2048], "torch.float32"),
        "hidden_out": ([8, 2048], "torch.float32"),
        "gate_out": ([8], "torch.float32"),
        "live_indices": ([4], "torch.int64"),
    }
    _require(
        set(initial["tensors"]) == set(layout)
        and set(initial["staging_tensors"]) == set(_METADATA),
        "persistent tensor set differs",
    )
    device = initial["tensors"]["hidden_in"]["device"]
    _require(device == expected_device, "persistent evidence is from the wrong execution device")
    from .m3_persistent import _check_storage_inventory

    _check_storage_inventory(
        initial["tensors"],
        initial["staging_tensors"],
        {name: (shape, dtype.removeprefix("torch.")) for name, (shape, dtype) in layout.items()},
        device=device,
    )
    return device


def _snapshot_evidence(snapshot, initial, *, generation, calls, empty, fallbacks, in_flight=False):
    expected_fields = {
        "enabled",
        "status",
        "generation",
        "capacity",
        "device_payload_bytes",
        "cpu_staging_bytes",
        "tensors",
        "staging_tensors",
        "counters",
        "fallback_counts",
        "last_dispatch",
        "last_publication",
        "failure",
    }
    if "buckets" in snapshot:
        expected_fields.add("buckets")
        _require(
            snapshot["buckets"]
            == {"8": {key: snapshot[key] for key in ("generation", "tensors", "staging_tensors")}},
            "frozen persistent bucket inventory differs",
        )
    _require(
        set(snapshot) == expected_fields
        and snapshot["enabled"] is True
        and snapshot["failure"] is None,
        "persistent snapshot missing or failed",
    )
    _require(
        snapshot["status"] == ("in_flight" if in_flight else "ready")
        and snapshot["generation"] == generation,
        "lease status or generation differs",
    )
    _require(
        snapshot["capacity"] == {"row_count": 8, "table_width": 32, "max_live_rows": 4}
        and snapshot["device_payload_bytes"] == 132392
        and snapshot["cpu_staging_bytes"] == 1256,
        "persistent capacity or payload budget differs",
    )
    _require(
        snapshot["tensors"] == initial["tensors"]
        and snapshot["staging_tensors"] == initial["staging_tensors"],
        "persistent pointer/shape/stride changed",
    )
    _require(
        snapshot["counters"]
        == {
            "calls": calls,
            "prepared": generation,
            "completed": generation - int(in_flight),
            "empty": empty,
        }
        and snapshot["fallback_counts"] == fallbacks,
        "persistent call/lease/fallback counts differ",
    )


def _audit_one(root, plan, evaluation, *, expected_device="cuda:0"):
    """Read and recompute one evaluation; no model, cache construction or device calls."""
    import math

    import torch

    from .diagnostics import TensorSpool

    identity = evaluation["evaluation_id"]
    folder, output = root / "evaluations" / identity, root / "outputs" / identity
    expected_files = {
        folder / "started.json",
        folder / "result.json",
        output / "evidence.bin",
        output / "evidence.index.json",
    }
    actual_files = {
        path for directory in (folder, output) for path in directory.rglob("*") if path.is_file()
    }
    _require(
        actual_files == expected_files
        and not any(
            path.is_symlink()
            for directory in (folder, output)
            for path in (directory, *directory.rglob("*"))
        ),
        "lifecycle artifact file set differs or contains links",
    )
    result, marker = _read(folder / "result.json"), _read(folder / "started.json")
    _require(
        marker
        == {
            "evaluation": evaluation,
            "started_ns": result["started_ns"],
            "deadline_ns": result["deadline_ns"],
        },
        "lifecycle start marker differs",
    )
    _require(
        result["evaluation"] == evaluation
        and result["lifecycle_plan_sha256"] == plan["lifecycle_plan_sha256"]
        and result["schema_version"] == 1
        and result["artifact_type"] == "m3_persistent_lifecycle_result",
        "lifecycle result identity differs",
    )
    _require(
        result["status"] == "complete" and result["passed"] is True and result["errors"] == [],
        "lifecycle execution did not pass",
    )
    start, finish, deadline = (result[key] for key in ("started_ns", "finished_ns", "deadline_ns"))
    _require(
        all(type(value) is int and value > 0 for value in (start, finish, deadline))
        and start < finish < deadline <= start + 600 * 10**9,
        "lifecycle timing/deadline evidence differs",
    )
    _require(
        result["cleanup"]
        == {
            "completion_confirmed": True,
            "used_blocks": 0,
            "active_requests": 0,
            "cache_references_released": True,
            "buffer_references_released": True,
        },
        "lifecycle cleanup is incomplete",
    )
    _allocation_evidence(result["initial_allocations"], plan["initial_allocations"])
    _guard_evidence(result["initial_guard_chunks"], plan["initial_guard_chunks"])
    _require(len(result["steps"]) == 11, "lifecycle action coverage differs")
    persistent = evaluation["implementation_id"] == "B"
    initial = result["persistent_initial"]
    if persistent:
        device = _owned_layout(initial, expected_device)
        _snapshot_evidence(
            initial,
            initial,
            generation=0,
            calls=0,
            empty=0,
            fallbacks={"live_count": 0, "table_width": 0},
        )
        _require(
            initial["last_dispatch"] is None and initial["last_publication"] is None,
            "initial persistent state has prior work",
        )
    else:
        device = expected_device
        _require(
            initial is None and result["persistent_final"] is None,
            "allocating A claims persistent state",
        )
    spool = TensorSpool.open(root / "outputs", identity, "evidence")
    observed_keys, tensor_hashes = set(), {}

    def tensor(step, operation, shape, dtype=torch.float32):
        key = f"step{step['step_id']:02d}/{operation}"
        value = spool.read(key)
        _require(
            list(value.shape) == list(shape) and value.dtype == dtype,
            "raw tensor shape/dtype differs",
        )
        _require(
            spool.read_metadata(key)
            == {"evaluation_id": identity, "step_id": step["step_id"], "operation": operation},
            "raw tensor metadata differs",
        )
        observed_keys.add(key)
        tensor_hashes[key] = spool.index[key]["sha256"]
        if value.is_floating_point():
            _require(bool(value.isfinite().all()), "nonfinite lifecycle raw evidence")
        return value

    generation = calls = empty = 0
    fallbacks = {"live_count": 0, "table_width": 0}
    previous_allocations = result["initial_allocations"]
    previous_snapshot, previous_hashes = initial, None
    held_hidden_hash = held_generator_hash = None
    for step, record in zip(plan["steps"], result["steps"]):
        _require(
            {key: record[key] for key in ("step_id", "name", "kind", "status")}
            == {**{key: step[key] for key in ("step_id", "name", "kind")}, "status": "complete"},
            "lifecycle step coverage/status differs",
        )
        _allocation_evidence(record["allocations"], step["allocations"])
        changed = {row["request_id"] for row in step["actions"]}
        for key in set(previous_allocations) & set(record["allocations"]) - changed:
            _require(
                previous_allocations[key] == record["allocations"][key],
                "unchanged allocation identity drifted",
            )
        _guard_evidence(record["guard_chunks"], step["guard_chunks"])
        count, rows = len(step["request_ids"]), step["row_count"]
        if count:
            _require(
                set(record["core"]) == {"input", "metadata"}, "actual core pointer evidence missing"
            )
            physical = tensor(step, "input", (rows, 2048))
            _require(torch.equal(physical, _physical_hidden(step)), "physical held input differs")
            metadata = _metadata(step)
            for name, expected in metadata.items():
                value = tensor(step, "metadata/" + name, expected.shape, expected.dtype)
                _require(torch.equal(value, expected), "physical metadata differs")
            _require(
                set(record["core"]["metadata"]) == set(metadata),
                "actual metadata pointer set differs",
            )
            _pointer(record["core"]["input"], [rows, 2048], "torch.float32", device=device)
            actual_device = record["core"]["input"]["device"]
            for name, value in metadata.items():
                _pointer(
                    record["core"]["metadata"][name],
                    list(value.shape),
                    str(value.dtype),
                    device=actual_device,
                )
            _require(
                record["input_hashes"]
                == {
                    key: value
                    for key, value in step["input_hashes"].items()
                    if key != "logical_hidden"
                },
                "frozen input hash evidence differs",
            )
            outputs = []
            active = metadata.get("active")
            for layer in range(2):
                query = tensor(step, f"layer{layer}/query", (rows, 16, 128))
                key = tensor(step, f"layer{layer}/key", query.shape)
                value = tensor(step, f"layer{layer}/value", query.shape)
                expected_key, expected_value = _kv(physical, layer)
                _require(
                    torch.equal(query.reshape(rows, 2048), physical)
                    and torch.equal(key.reshape(rows, 2048), expected_key)
                    and torch.equal(value.reshape(rows, 2048), expected_value),
                    "held Q/K/V formula differs",
                )
                attention = tensor(step, f"layer{layer}/attention", query.shape)
                if active is not None:
                    inactive = attention[~active]
                    _require(
                        not bool(inactive.count_nonzero()) and not bool(inactive.signbit().any()),
                        "inactive attention is not positive zero",
                    )
                outputs.append(attention.reshape(rows, 2048))
            hidden = tensor(step, "physical_hidden", (rows, 2048))
            gates = tensor(step, "physical_gates", (rows,))
            _require(
                torch.equal(hidden, outputs[0] + outputs[1]) and torch.equal(gates, hidden[:, 0]),
                "held output formula differs",
            )
            if active is not None:
                for value in (hidden[~active], gates[~active]):
                    _require(
                        not bool(value.count_nonzero()) and not bool(value.signbit().any()),
                        "inactive final output is not positive zero",
                    )
            published_hidden = tensor(step, "publication_hidden", (count, 2048))
            published_gates = tensor(step, "publication_gates", (count,))
            _require(
                torch.equal(published_hidden, hidden[step["live_rows"]])
                and torch.equal(published_gates, gates[step["live_rows"]]),
                "publication did not gather live rows in scheduler order",
            )
            _require(
                set(record["publication"]) == {"hidden", "gates"},
                "publication pointer evidence missing",
            )
            _pointer(
                record["publication"]["hidden"],
                [count, 2048],
                "torch.float32",
                device=actual_device,
            )
            _pointer(record["publication"]["gates"], [count], "torch.float32", device=actual_device)
            _require(len(record["request_hidden"]) == count, "request publication count differs")
            for index, description in enumerate(record["request_hidden"]):
                _pointer(description, [2048], "torch.float32", device=actual_device)
                _require(
                    description["storage_ptr"] == record["publication"]["hidden"]["storage_ptr"]
                    and description["data_ptr"]
                    == record["publication"]["hidden"]["data_ptr"] + index * 2048 * 4,
                    "request hidden does not match returned scheduler row",
                )
            _require(
                len(record["gate_probabilities"]) == count
                and all(
                    type(value) is float
                    and math.isclose(value, expected, rel_tol=0, abs_tol=1.2e-7)
                    for value, expected in zip(
                        record["gate_probabilities"], published_gates.sigmoid().tolist()
                    )
                ),
                "actual gate readback differs or includes padding",
            )
            if step["step_id"] == 2:
                held = tensor(step, "held_hidden", (2048,))
                generator = torch.Generator(device="cpu").manual_seed(17).get_state()
                raw_generator = tensor(step, "held_generator", generator.shape, generator.dtype)
                _require(
                    torch.equal(held, published_hidden[2])
                    and torch.equal(generator, raw_generator),
                    "held coda input/RNG differs",
                )
                held_hidden_hash, held_generator_hash = _tensor_hash(held), _tensor_hash(generator)
        else:
            _require(
                "core" not in record
                and "publication" not in record
                and "input_hashes" not in record,
                "empty/host-only action performed a core",
            )
            if step["kind"] == "empty":
                tensor(step, "publication_hidden", (0, 2048))
                tensor(step, "publication_gates", (0,))
        if step["step_id"] == 6:
            old, new = previous_allocations["r1"], record["allocations"]["r1"]
            _require(
                record["previous_r1_object_id"] == old["object_id"] != new["object_id"]
                and record["previous_r1_pages"]
                == sorted(page for table in old["block_tables"] for page in table)
                == sorted(page for table in new["block_tables"] for page in table)
                and record["stale_allocation_rejected"] is True,
                "recycled allocation/physical reuse/stale ownership evidence differs",
            )
            if persistent:
                _require(
                    record["stale_generation_rejected"] is True, "stale lease evidence missing"
                )
        if persistent:
            if step["kind"] != "host_only":
                calls += 1
            if step["kind"] == "supported":
                generation += 1
                _require(
                    record["busy_lease_rejected"] is True
                    and record["expired_generation_rejected"] is True,
                    "busy or expired lease accepted",
                )
                _snapshot_evidence(
                    record["in_flight"],
                    initial,
                    generation=generation,
                    calls=calls,
                    empty=empty,
                    fallbacks=fallbacks,
                    in_flight=True,
                )
                for name, description in {
                    "hidden_in": record["core"]["input"],
                    **record["core"]["metadata"],
                }.items():
                    owned = record["in_flight"]["tensors"][name]
                    _require(
                        description == {key: owned[key] for key in description},
                        "actual core did not borrow persistent storage",
                    )
            elif step["kind"] == "empty":
                empty += 1
            elif step["kind"] == "fallback":
                fallbacks[step["fallback_reason"]] += 1
            snapshot = record["persistent"]
            _snapshot_evidence(
                snapshot,
                initial,
                generation=generation,
                calls=calls,
                empty=empty,
                fallbacks=fallbacks,
            )
            if step["kind"] == "host_only":
                _require(
                    snapshot == previous_snapshot, "host-only rejection mutated persistent state"
                )
            else:
                compact = step["kind"] == "fallback"
                expected_dispatch = {
                    "kind": "compact" if compact else "empty" if not count else "persistent",
                    "request_ids": step["request_ids"],
                    "depths": step["depths"],
                    "positions": step["positions"],
                    "live_rows": step["live_rows"],
                    "row_count": count if compact else 8,
                    "table_width": step["required_width"] if compact else 32,
                    "generation": generation,
                }
                _require(
                    snapshot["last_dispatch"] == expected_dispatch,
                    "actual dispatch/fallback differs",
                )
                if count:
                    _require(
                        set(snapshot["last_publication"]) == {"hidden", "gates"},
                        "persistent publication missing",
                    )
                    owners = {value["storage_ptr"] for value in initial["tensors"].values()}
                    for name, description in record["publication"].items():
                        expected = snapshot["last_publication"][name]
                        _require(
                            description == {key: expected[key] for key in description}
                            and description["storage_ptr"] not in owners,
                            "returned tensors alias persistent storage or differ from publication",
                        )
                else:
                    _require(
                        snapshot["last_publication"] is None, "empty dispatch retained publication"
                    )
            _require(
                set(record["buffer_hashes"]) == set(initial["tensors"])
                and all(
                    isinstance(value, str) and len(value) == 64
                    for value in record["buffer_hashes"].values()
                ),
                "buffer hash coverage differs",
            )
            _require(
                record["unchanged_outside_supported"] is True, "persistent buffer mutation flagged"
            )
            if step["kind"] == "supported":
                prefix = f"step{step['step_id']:02d}/"
                expected_buffer_hashes = {
                    name: tensor_hashes[prefix + "metadata/" + name] for name in _METADATA
                }
                expected_buffer_hashes.update(
                    {
                        "hidden_in": tensor_hashes[prefix + "input"],
                        "hidden_out": tensor_hashes[prefix + "physical_hidden"],
                        "gate_out": tensor_hashes[prefix + "physical_gates"],
                        "live_indices": _tensor_hash(torch.tensor([1, 3, 5, 7], dtype=torch.int64)),
                    }
                )
                _require(
                    record["buffer_hashes"] == expected_buffer_hashes,
                    "persistent buffer bytes do not match actual raw inputs/outputs",
                )
                _require(
                    record["in_flight"]["last_dispatch"] == snapshot["last_dispatch"]
                    and record["in_flight"]["last_publication"] is None,
                    "in-flight dispatch or publication ordering differs",
                )
            if step["kind"] != "supported":
                _require(
                    record["buffer_hashes"] == previous_hashes,
                    "unsupported action changed persistent bytes",
                )
            previous_snapshot, previous_hashes = snapshot, record["buffer_hashes"]
        previous_allocations = record["allocations"]
    _require(
        set(spool.index) == observed_keys and len(observed_keys) == 173,
        "raw lifecycle boundary coverage differs",
    )
    _require(
        [row["step_id"] for row in result["held_checks"]] == list(range(2, 12)),
        "paused coda lifetime coverage differs",
    )
    held = result["held_initial"]
    _require(
        set(held)
        == {
            "request_id",
            "stage",
            "prompt_token_ids",
            "generated_token_ids",
            "exit_depths",
            "loops_done",
            "remaining_probability",
            "hidden_sha256",
            "hidden_storage_ptr",
            "generator_sha256",
        }
        and held["request_id"] == "r2"
        and held["stage"] == "coda"
        and held["prompt_token_ids"] == [1]
        and held["generated_token_ids"] == [1]
        and held["exit_depths"] == []
        and held["loops_done"] == 2
        and held["remaining_probability"] == 1.0
        and held["hidden_sha256"] == held_hidden_hash
        and held["generator_sha256"] == held_generator_hash
        and held["hidden_storage_ptr"] == result["steps"][1]["request_hidden"][2]["storage_ptr"],
        "held request identity/history differs",
    )
    _require(
        all(row["state"] == held for row in result["held_checks"]), "paused coda/RNG state changed"
    )
    if persistent:
        _require(result["persistent_final"] == previous_snapshot, "final persistent state differs")
        for name, expected in plan["expected_persistent"].items():
            _require(
                result["persistent_final"][name] == expected, "final persistent counters differ"
            )
    bin_bytes = (output / "evidence.bin").stat().st_size
    json_bytes = sum(path.stat().st_size for path in expected_files if path.suffix != ".bin")
    _require(bin_bytes <= 8 * MiB and json_bytes <= 4 * MiB, "lifecycle evidence cap exceeded")
    return {
        "evaluation_id": identity,
        "started_ns": start,
        "finished_ns": finish,
        "deadline_ns": deadline,
        "passed": True,
        "tensor_records": len(observed_keys),
        "guard_chunks": 480,
        "artifact_bytes": bin_bytes + json_bytes,
        "artifacts": {
            str(path.relative_to(root)): {
                "sha256": _file_hash(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(expected_files)
        },
        "tensor_hashes": tensor_hashes,
        "gate_probabilities": [row.get("gate_probabilities") for row in result["steps"]],
    }


def audit_lifecycle_outputs(output_dir, plan, *, expected_device="cuda:0"):
    """Strict, bounded offline audit. Missing/partial/corrupted evidence cannot pass."""
    root = Path(output_dir)
    report = {
        "complete": False,
        "passed": False,
        "errors": [],
        "completed_evaluations": [],
        "evaluations": [],
        "artifact_bytes": 0,
    }
    try:
        validate_lifecycle_plan(plan)
        expected_ids = {row["evaluation_id"] for row in plan["execution_order"]}
        for name in ("evaluations", "outputs"):
            directory = root / name
            _require(
                directory.is_dir()
                and not directory.is_symlink()
                and {path.name for path in directory.iterdir()} == expected_ids,
                "lifecycle evaluation directory coverage differs",
            )
        for evaluation in plan["execution_order"]:
            try:
                report["evaluations"].append(
                    _audit_one(root, plan, evaluation, expected_device=expected_device)
                )
            except Exception as exc:
                report["errors"].append(
                    f"{evaluation['evaluation_id']}: {type(exc).__name__}: {str(exc)[:1024]}"
                )
        report["completed_evaluations"] = [row["evaluation_id"] for row in report["evaluations"]]
        report["artifact_bytes"] = sum(row["artifact_bytes"] for row in report["evaluations"])
        _require(report["artifact_bytes"] <= 48 * MiB, "lifecycle suite evidence cap exceeded")
        if len(report["completed_evaluations"]) == 4:
            results = {row["evaluation_id"]: row for row in report["evaluations"]}
            for backend in ("torch", "triton"):
                a, b = results[f"L-A-{backend}"], results[f"L-B-{backend}"]
                _require(
                    a["tensor_hashes"] == b["tensor_hashes"]
                    and a["gate_probabilities"] == b["gate_probabilities"],
                    "held same-backend B/A raw tensors or actual gates differ",
                )
            for before, after in zip(report["evaluations"], report["evaluations"][1:]):
                _require(
                    before["finished_ns"] < after["started_ns"],
                    "lifecycle executions overlap or violate fixed order",
                )
            report["complete"] = report["passed"] = not report["errors"]
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {str(exc)[:1024]}")
    for row in report["evaluations"]:
        row.pop("tensor_hashes", None)
        row.pop("gate_probabilities", None)
    return report
