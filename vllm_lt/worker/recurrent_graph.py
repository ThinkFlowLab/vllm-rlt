"""Private bucketed eager/replay executor with synchronous host transactions."""

import time
from dataclasses import asdict

import torch

from vllm_lt.core.kv_cache_manager import decode_live_rows

from .capture_resources import _CudaRuntime as _CudaRuntime
from .capture_resources import _make_runtime
from .decode_buffers import DecodeBucketLayout, allocate_bucket
from .graph_diagnostics import (
    _MEMORY_LIMITS,
    REPLAY_ATOL,
    SCRATCH_PAGES,
    SCRATCH_TOKENS,
    SETUP_WARMUPS,
    CaptureBudgetExceeded,
    GraphLimits,
    _description,
    _error,
    _hash,
    _memory_deltas,
    graph_snapshot,
)


def _signature(tensor):
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class RecurrentGraphExecutor:
    """Own one ordered execution stream, a fixed bucket ladder and one graph pool.

    Construction validates host configuration only. The runner retains this
    object before calling setup(), including when setup fails partway through.
    Weights and the original cache pool must remain immutable for this lifetime.
    """

    def __init__(self, model, cache, *, use_graphs, limits=None, layout=None):
        if type(use_graphs) is not bool:
            raise ValueError("use_graphs must be bool")
        if limits is None:
            limits = GraphLimits()
        elif isinstance(limits, dict):
            if set(limits) != set(asdict(GraphLimits())):
                raise ValueError("graph limits must contain the exact declared fields")
            limits = GraphLimits(**limits)
        if not isinstance(limits, GraphLimits):
            raise ValueError("limits must be GraphLimits or its exact dictionary")
        cache._require_usable()
        if cache._allocations:
            raise RuntimeError("recurrent graph setup must precede request admission")
        parameter = next(model.parameters())
        if parameter.dtype != torch.float32 or cache.dtype != torch.float32:
            raise ValueError("recurrent graph execution supports float32 only")
        if parameter.device != cache.device:
            raise ValueError("model and cache must use the same device")
        if (
            use_graphs
            and cache.backend == "triton"
            and (
                cache.max_loops != SCRATCH_PAGES
                or cache.num_free_blocks < SCRATCH_PAGES
                or cache.block_size < SCRATCH_TOKENS
            )
        ):
            raise ValueError("graph setup requires four scratch depth pages and two token slots")
        self.layout = layout if layout is not None else DecodeBucketLayout()
        payload, staging = self.layout.payload_bytes(model.config.hidden_size)
        for name, observed in (("common_payload_bytes", payload), ("cpu_staging_bytes", staging)):
            if observed > getattr(limits, name):
                raise CaptureBudgetExceeded(limits, name, observed)
        self.model, self.cache, self.device = model, cache, cache.device
        self.use_graphs, self.limits = use_graphs, limits
        # Rich descriptors are for excluded qualification/profile runs only.
        self.record_dispatch_tensors = False
        self.runtime = _make_runtime(self.device)
        if cache.backend == "triton" and self.runtime is None:
            raise ValueError("Triton recurrent buckets require a CUDA runtime")
        self.status, self.failure = "new", None
        self.buckets, self.stream, self.setup_stream = {}, None, None
        self.pool_owner = None
        self.ticket = self.batch = self.active_bucket = None
        self.counters = dict.fromkeys(
            ("calls", "empty", "prepared", "eager", "replays", "committed", "completed"), 0
        )
        self.fallback_counts = dict.fromkeys(("backend", "live_count", "table_width"), 0)
        self.last_dispatch = self.last_publication = None
        self.setup_record = {
            "status": "pending",
            "started_ns": None,
            "finished_ns": None,
            "events": [],
            "warmups": 0,
            "captures": 0,
            "verification_replays": 0,
            "scratch": None,
            "memory_baseline": None,
            "memory_after": None,
            "memory_deltas": None,
        }
        self._scratch = None
        self._pool_tensors = (cache.key_cache, cache.value_cache)
        self._pool_signature = (_signature(cache.key_cache), _signature(cache.value_cache))
        self._weights = tuple(model.parameters())

    def _check_setup_time(self):
        now = time.perf_counter_ns()
        if now >= self.setup_record["deadline_ns"]:
            raise CaptureBudgetExceeded(
                self.limits, "setup_timeout_s", (now - self.setup_record["started_ns"]) / 1e9
            )

    def _phase(self, name, row_count, operation):
        self._check_setup_time()
        event = {
            "phase": name,
            "row_count": row_count,
            "started_ns": time.perf_counter_ns(),
            "status": "incomplete",
        }
        self.setup_record["events"].append(event)
        try:
            result = operation()
        except BaseException as error:
            event["error"] = _error(error)
            event["finished_ns"] = time.perf_counter_ns()
            raise
        event["finished_ns"] = time.perf_counter_ns()
        self._check_setup_time()
        event["status"] = "complete"
        return result

    @torch.inference_mode()
    def setup(self):
        if self.status != "new":
            raise RuntimeError("recurrent executor setup cannot be repeated")
        self.status = "setup"
        started = time.perf_counter_ns()
        self.setup_record.update(
            started_ns=started, deadline_ns=started + self.limits.setup_timeout_s * 10**9
        )
        try:
            self.stream = self.runtime.current_stream() if self.runtime is not None else None
            for rows in self.layout.row_counts:
                bucket = allocate_bucket(
                    self.cache, self.layout, rows, self.model.config.hidden_size
                )
                metadata, tensors = bucket["metadata"], bucket["tensors"]
                bucket.update(
                    {
                        "view": None,
                        "graph": None,
                        "pool_id": None,
                        "graph_exec_id": None,
                        "capture_outputs": None,
                        "setup_generation": 0,
                        "captured_inputs": None,
                        "captured_outputs": None,
                        "counters": dict.fromkeys(
                            ("prepared", "eager", "replays", "committed", "completed"), 0
                        ),
                    }
                )
                self.buckets[rows] = bucket
                self._clear(bucket)
                if self.cache.backend == "triton":
                    bucket["view"] = self.cache._make_tensor_decode_view(metadata)
                bucket["owned_tensors"] = tuple(tensors.items())
                bucket["signature"] = {name: _signature(value) for name, value in tensors.items()}
            if self.runtime is not None:
                self.stream.synchronize()
                self.setup_record["memory_baseline"] = self.runtime.memory()
            self._check_setup_time()
            if self.use_graphs and self.cache.backend == "triton":
                self._setup_graphs()
            else:
                self.setup_record["skip_reason"] = (
                    "backend" if self.cache.backend != "triton" else "eager_control"
                )
                self.setup_record["memory_after"] = self.setup_record["memory_baseline"]
                self.setup_record["memory_deltas"] = (
                    dict.fromkeys(_MEMORY_LIMITS, 0) if self.runtime is not None else None
                )
            for bucket in self.buckets.values():
                if bucket["view"] is not None:
                    self._check_bucket(bucket, full=True)
                bucket["setup_generation"] = bucket["metadata"].generation
            self._check_setup_time()
            self.setup_record["status"], self.status = "complete", "ready"
        except BaseException as error:
            self.settle_failure(error)
            self.setup_record["status"] = "failed"
            raise
        finally:
            finished = time.perf_counter_ns()
            self.setup_record["finished_ns"] = finished
            if (
                self.setup_record["status"] == "complete"
                and finished >= self.setup_record["deadline_ns"]
            ):
                error = CaptureBudgetExceeded(
                    self.limits, "setup_timeout_s", (finished - started) / 1e9
                )
                self.setup_record["status"] = "failed"
                self.settle_failure(error)
                self.setup_record["finished_ns"] = time.perf_counter_ns()
                raise error

    def _clear(self, bucket):
        for name, tensor in bucket["tensors"].items():
            if name != "live_indices":
                tensor.fill_(-1 if name in {"write_blocks", "write_offsets", "block_tables"} else 0)
        for name, tensor in bucket["metadata"].staging.items():
            tensor.fill_(-1 if name in {"write_blocks", "write_offsets", "block_tables"} else 0)

    def _save_scratch(self):
        cache = self.cache
        if cache._allocations:
            raise RuntimeError("graph scratch requires an empty request allocator")
        free = tuple(cache._free_blocks)
        request_id = "__vllm_lt_graph_setup__"
        if not cache.allocate(request_id, SCRATCH_TOKENS):
            raise RuntimeError("graph scratch reservation failed")
        allocation = cache._allocations[request_id]
        pages = [page for table in allocation.block_tables for page in table]
        self._scratch = {
            "request_id": request_id,
            "allocation": allocation,
            "free": free,
            "saved": [],
        }
        record = {
            "request_id": request_id,
            "pages": pages,
            "saved_cpu_bytes": 0,
            "free_list_before": list(free),
            "free_list_after": None,
            "restored": False,
            "before_hashes": [],
            "after_hashes": [],
        }
        self.setup_record["scratch"] = record
        for component, pool in enumerate((cache.key_cache, cache.value_cache)):
            for page in pages:
                saved = pool[page].detach().to("cpu", copy=True)
                self._scratch["saved"].append((component, page, saved))
                record["saved_cpu_bytes"] += saved.numel() * saved.element_size()
                record["before_hashes"].append(
                    {"component": component, "page": page, "sha256": _hash(saved)}
                )
        if (
            len(pages) != SCRATCH_PAGES
            or record["saved_cpu_bytes"] != cache.bytes_per_block * SCRATCH_PAGES
        ):
            raise RuntimeError("scratch page or byte accounting differs")
        seed = torch.arange(
            cache.num_kv_heads * cache.head_dim, device=self.device, dtype=torch.float32
        ).reshape(1, cache.num_kv_heads, cache.head_dim)
        seed = (seed.remainder(17) - 8) / 16
        for layer in range(cache.num_layers):
            cache.write(layer, [request_id], [0], [0], seed, seed / (layer + 1))

    @torch.inference_mode()
    def _restore_scratch(self):
        if self._scratch is None:
            return
        scratch, cache = self._scratch, self.cache
        if (
            set(cache._allocations) != {scratch["request_id"]}
            or cache._allocations[scratch["request_id"]] is not scratch["allocation"]
        ):
            raise RuntimeError("scratch ownership changed during graph setup")
        record = self.setup_record["scratch"]
        for component, page, saved in scratch["saved"]:
            (cache.key_cache, cache.value_cache)[component][page].copy_(saved)
        if self.stream is not None:
            self.stream.synchronize()
        record["after_hashes"] = [
            {
                "component": component,
                "page": page,
                "sha256": _hash((cache.key_cache, cache.value_cache)[component][page]),
            }
            for component, page, _ in scratch["saved"]
        ]
        if record["after_hashes"] != record["before_hashes"]:
            raise RuntimeError("scratch bytes were not restored")
        # This is setup-only ownership, before any admission. Preserve quarantine
        # on failed setup while releasing only confirmed-complete scratch pages.
        # Refactor tripwire: changes to free() side effects must update this exact
        # rollback and its scratch-byte/allocator-order ownership tests together.
        del cache._allocations[scratch["request_id"]]
        cache._allocation_generation += 1
        cache._free_blocks[:] = scratch["free"]
        record["free_list_after"] = list(cache._free_blocks)
        record["restored"] = record["free_list_after"] == record["free_list_before"]
        self._scratch = None

    def _inputs(self, bucket):
        return {
            "hidden": _description(bucket["tensors"]["hidden_in"]),
            **{name: _description(value) for name, value in bucket["metadata"].tensors.items()},
        }

    def _outputs(self, bucket):
        return {
            "hidden": _description(bucket["tensors"]["hidden_out"]),
            "gates": _description(bucket["tensors"]["gate_out"]),
        }

    def _tensor_body(self, bucket):
        hidden, gates = self.model._recurrent_tensor(bucket["tensors"]["hidden_in"], bucket["view"])
        bucket["tensors"]["hidden_out"].copy_(hidden)
        bucket["tensors"]["gate_out"].copy_(gates)
        return hidden, gates

    def _capture(self, bucket):
        if self.pool_owner is None:
            self.pool_owner = self.runtime.new_pool()
        bucket["pool_id"] = list(self.pool_owner.id)
        graph = self.runtime.new_graph()
        bucket["graph"] = graph
        graph.capture_begin(pool=self.pool_owner.id, capture_error_mode="global")
        try:
            bucket["capture_outputs"] = self._tensor_body(bucket)
        except BaseException:
            try:
                graph.capture_end()
            except BaseException as secondary:
                self.setup_record.setdefault("capture_cleanup_errors", []).append(_error(secondary))
            raise
        else:
            graph.capture_end()
        if list(graph.pool()) != bucket["pool_id"]:
            raise RuntimeError("captured graph must use its executor's shared pool")
        bucket["graph_exec_id"] = graph.raw_cuda_graph_exec()
        bucket["captured_inputs"], bucket["captured_outputs"] = (
            self._inputs(bucket),
            self._outputs(bucket),
        )

    def _setup_graphs(self):
        self.runtime.reset_peaks()
        self.setup_stream = self.runtime.new_stream()
        self.setup_stream.wait_stream(self.stream)
        # An outer stream context restores the original even when capture_end fails.
        with self.runtime.stream_context(self.setup_stream):
            self._phase("scratch_save_and_seed", None, self._save_scratch)
            for rows, bucket in self.buckets.items():
                host = self.cache._prepare_host_batch([self._scratch["request_id"]], [0], [1])
                self.ticket = self.cache._begin_decode_traversal(host)
                self.active_bucket = bucket
                self.batch = self.cache._prepare_into(bucket["metadata"], host)
                self.cache._bind_decode_traversal(self.ticket, self.batch)
                bucket["tensors"]["hidden_in"].zero_()
                values = torch.arange(
                    self.model.config.hidden_size, device=self.device, dtype=torch.float32
                )
                bucket["tensors"]["hidden_in"][1].copy_((values.remainder(17) - 8) / 16)
                for _ in range(SETUP_WARMUPS):
                    self._phase("warmup", rows, lambda: self._tensor_body(bucket))
                    self.setup_record["warmups"] += 1
                self.setup_stream.synchronize()
                warm = tuple(
                    bucket["tensors"][name].to("cpu", copy=True)
                    for name in ("hidden_out", "gate_out")
                )
                self._phase("capture", rows, lambda: self._capture(bucket))
                self.setup_record["captures"] += 1
                self._phase("verification_replay", rows, bucket["graph"].replay)
                self.setup_record["verification_replays"] += 1
                self.setup_stream.synchronize()
                actual = tuple(
                    bucket["tensors"][name].to("cpu", copy=True)
                    for name in ("hidden_out", "gate_out")
                )
                inactive = [index for index in range(rows) if index != 1]
                for value in actual:
                    if (
                        not bool(value.isfinite().all())
                        or bool(value[inactive].count_nonzero())
                        or bool(value[inactive].signbit().any())
                    ):
                        raise RuntimeError(
                            "graph setup verification produced invalid physical output"
                        )
                differences = [float((a - b).abs().max()) for a, b in zip(warm, actual)]
                if not all(diff <= REPLAY_ATOL for diff in differences):
                    raise RuntimeError("graph setup replay differs from warmup")
                bucket["verification"] = {
                    "finite": True,
                    "inactive_positive_zero": True,
                    "warmup_sha256": [_hash(value) for value in warm],
                    "replay_sha256": [_hash(value) for value in actual],
                    "max_abs_diff": differences,
                }
                self.cache._commit_decode_traversal(self.ticket, completion_confirmed=True)
                self.cache._release_prepared(self.batch)
                self.ticket = self.batch = self.active_bucket = None
        if any(bucket["pool_id"] != list(self.pool_owner.id) for bucket in self.buckets.values()):
            raise RuntimeError("graph buckets must retain their executor's shared pool")
        self.stream.wait_stream(self.setup_stream)
        self.stream.synchronize()
        self._phase("scratch_restore", None, self._restore_scratch)
        for bucket in self.buckets.values():
            self._clear(bucket)
        self.stream.synchronize()
        self.setup_record["memory_after"] = after = self.runtime.memory()
        before = self.setup_record["memory_baseline"]
        deltas = _memory_deltas(before, after)
        self.setup_record["memory_deltas"] = deltas
        for field, name in _MEMORY_LIMITS.items():
            if deltas[field] > getattr(self.limits, name):
                raise CaptureBudgetExceeded(self.limits, name, deltas[field])

    def require_usable(self):
        if self.status not in ("ready", "in_flight"):
            raise RuntimeError(f"recurrent executor is {self.status}; retry is forbidden")
        self.cache._require_usable()

    def select_stream(self):
        if self.runtime is not None and self.runtime.current_stream() != self.stream:
            raise RuntimeError("recurrent execution requires its original ordered stream")

    def synchronize(self):
        if self.setup_stream is not None:
            self.setup_stream.synchronize()
        if self.stream is not None:
            self.stream.synchronize()

    def _check_bucket(self, bucket, *, full=False):
        if (
            self.cache.key_cache is not self._pool_tensors[0]
            or self.cache.value_cache is not self._pool_tensors[1]
        ):
            raise RuntimeError("captured cache pool identity changed")
        if any(bucket["tensors"].get(name) is not value for name, value in bucket["owned_tensors"]):
            raise RuntimeError("decode bucket storage identity changed")
        if (
            full
            and (
                _signature(self.cache.key_cache),
                _signature(self.cache.value_cache),
            )
            != self._pool_signature
        ):
            raise RuntimeError("captured cache pool identity changed")
        if (
            full
            and {name: _signature(value) for name, value in bucket["tensors"].items()}
            != bucket["signature"]
        ):
            raise RuntimeError("decode bucket storage signature changed")
        for name, value in bucket["metadata"].tensors.items():
            if value is not bucket["tensors"][name] or getattr(bucket["view"], name) is not value:
                raise RuntimeError("tensor view metadata no longer matches the borrowed storage")

    def dispatch_route(self, count, width):
        """Read-only routing shared by execution and excluded profiling."""
        rows, reason = self.layout.select(count, width)
        if self.cache.backend != "triton":
            reason = "backend"
        kind = (
            "empty"
            if not count
            else "compact"
            if reason
            else "replay"
            if self.use_graphs
            else "eager"
        )
        return (None if reason or not count else rows), reason, kind

    def preview_dispatch(self, request_ids, positions):
        width = max((position // self.cache.block_size + 1 for position in positions), default=0)
        rows, _, kind = self.dispatch_route(len(request_ids), width)
        return rows, kind

    @torch.inference_mode()
    def recurrent(self, hidden, request_ids, depths, positions, *, defer_completion=False):
        self.require_usable()
        self.select_stream()
        if self.status != "ready" or self.ticket is not None:
            raise RuntimeError("recurrent executor already has an in-flight traversal")
        if (
            hidden.shape != (len(request_ids), self.model.config.hidden_size)
            or hidden.dtype != torch.float32
            or hidden.device != self.device
        ):
            raise ValueError("recurrent inputs must be compact float32 model-device rows")
        host = self.cache._prepare_host_batch(request_ids, depths, positions)
        count = len(host.rows)
        # Reject holes before selecting a fallback or copying a single input byte.
        ticket = self.cache._begin_decode_traversal(host) if count else None
        rows, reason, kind = self.dispatch_route(count, host.width)
        self.counters["calls"] += 1
        self.last_publication = None
        self.last_dispatch = {
            "dispatch_id": self.counters["calls"],
            "kind": kind,
            "request_ids": list(request_ids),
            "depths": list(depths),
            "positions": list(positions),
            "live_rows": list(range(count)) if reason else list(decode_live_rows(count)),
            "row_count": 0 if not count else count if reason else rows,
            "table_width": 0 if not count else host.width if reason else self.layout.table_width,
            "bucket_id": None if reason or not count else rows,
            "generation": None,
            "ticket_state": None,
            "actual_inputs": None,
            "actual_physical_outputs": None,
        }
        if not count:
            self.counters["empty"] += 1
            return hidden.clone(), hidden.new_empty(0)
        self.ticket = ticket
        try:
            if reason:
                # The general eager path owns its ordinary per-layer prefix updates.
                self.cache._cancel_decode_traversal(ticket)
                self.ticket = None
                self.last_dispatch["ticket_state"] = ticket.state
                self.fallback_counts[reason] += 1
                self.status = "in_flight"
                outputs = self.model.recurrent(hidden, request_ids, depths, positions, self.cache)
            else:
                bucket = self.buckets[rows]
                self.active_bucket = bucket
                self._check_bucket(bucket)
                self.batch = self.cache._prepare_into(bucket["metadata"], host)
                self.cache._bind_decode_traversal(ticket, self.batch)
                self.counters["prepared"] += 1
                bucket["counters"]["prepared"] += 1
                self.last_dispatch.update(
                    generation=self.batch.generation, ticket_state=ticket.state
                )
                live = bucket["tensors"]["live_indices"][:count]
                bucket["tensors"]["hidden_in"].zero_()
                bucket["tensors"]["hidden_in"].index_copy_(0, live, hidden)
                if self.record_dispatch_tensors:
                    self.last_dispatch["actual_inputs"] = self._inputs(bucket)
                self.status = "in_flight"
                if self.use_graphs:
                    bucket["graph"].replay()
                    self.counters["replays"] += 1
                    bucket["counters"]["replays"] += 1
                else:
                    self._tensor_body(bucket)
                    self.counters["eager"] += 1
                    bucket["counters"]["eager"] += 1
                if self.record_dispatch_tensors:
                    self.last_dispatch["actual_physical_outputs"] = self._outputs(bucket)
                outputs = (
                    bucket["tensors"]["hidden_out"].index_select(0, live),
                    bucket["tensors"]["gate_out"].index_select(0, live),
                )
            if self.record_dispatch_tensors:
                self.last_publication = {
                    "hidden": _description(outputs[0]),
                    "gates": _description(outputs[1]),
                }
            if not defer_completion:
                self.synchronize()
                self.complete_after_gate()
            return outputs
        except BaseException as error:
            self.settle_failure(error)
            raise

    def complete_after_gate(self):
        self.require_usable()
        if self.status != "in_flight":
            raise RuntimeError("no recurrent traversal to complete")
        try:
            if self.ticket is not None:
                self.cache._commit_decode_traversal(self.ticket, completion_confirmed=True)
                self.last_dispatch["ticket_state"] = self.ticket.state
                self.counters["committed"] += 1
                self.active_bucket["counters"]["committed"] += 1
                self.cache._release_prepared(self.batch)
                self.active_bucket["counters"]["completed"] += 1
            self.counters["completed"] += 1
            self.ticket = self.batch = self.active_bucket = None
            self.status = "ready"
        except BaseException as error:
            self.settle_failure(error)
            raise

    def settle_failure(self, error):
        if self.failure is not None:
            # Preserve later settlement attempts without retaining exception
            # tracebacks and their tensor references. Completion is not retried.
            self.failure["secondary"].append(_error(error))
            return self.failure["completion_confirmed"]
        self.status = "failed"
        self.failure = {"primary": _error(error), "secondary": [], "completion_confirmed": False}
        try:
            self.synchronize()
            self.failure["completion_confirmed"] = True
        except BaseException as secondary:
            self.failure["secondary"].append(_error(secondary))
            self.cache._quarantine("recurrent stream completion failed")
        confirmed = self.failure["completion_confirmed"]
        if self.ticket is not None:
            try:
                self.cache._abort_decode_traversal(self.ticket, completion_confirmed=confirmed)
            except BaseException as secondary:
                self.failure["secondary"].append(_error(secondary))
                self.cache._quarantine("recurrent transaction settlement failed")
        for bucket in self.buckets.values():
            bucket["metadata"].failed = True
            if confirmed:
                bucket["metadata"].in_use = False
        if confirmed:
            self.ticket = self.batch = self.active_bucket = None
            try:
                self._restore_scratch()
            except BaseException as secondary:
                self.failure["secondary"].append(_error(secondary))
                self.cache._quarantine("graph scratch restoration failed")
        return confirmed

    def record_cleanup_failure(self, error):
        if self.failure is not None:
            self.failure["secondary"].append(_error(error))
        self.cache._quarantine("recurrent request cleanup failed")

    def close(self):
        if self.status == "closed":
            return
        if self.status == "in_flight":
            raise RuntimeError("close requires completed or failed request publication")
        try:
            self.synchronize()
            # A later confirmed boundary can release resources retained after an
            # earlier failed synchronization. It never makes the executor usable.
            if self.failure is not None:
                self.failure["completion_confirmed"] = True
                if self.ticket is not None:
                    self.cache._abort_decode_traversal(self.ticket, completion_confirmed=True)
                for bucket in self.buckets.values():
                    bucket["metadata"].in_use = False
                self.ticket = self.batch = self.active_bucket = None
                self._restore_scratch()
            for bucket in self.buckets.values():
                if bucket["graph"] is not None:
                    bucket["graph"].reset()
        except BaseException as error:
            if self.failure is None:
                self.settle_failure(error)
            else:
                self.failure["secondary"].append(_error(error))
            # In particular, do not drop graph pools or borrowed inputs if a
            # synchronization or graph reset failed partway through close.
            raise
        # Complete all resets before releasing the shared owner. Captured model
        # outputs belong to the private pool, unlike the common output buffers.
        # Drop every executor-owned output reference before MemPool destruction
        # performs its pool-specific cache release. Explicit assignments also
        # work when an exception traceback retains a local bucket dictionary.
        for bucket in self.buckets.values():
            bucket["capture_outputs"] = None
            bucket["graph"] = None
        self.pool_owner = None
        self.buckets.clear()
        if self.setup_stream is not None:
            self.runtime.release_stream(self.setup_stream)
            self.setup_stream = None
        self._weights = ()
        self._pool_tensors = ()
        self.ticket = self.batch = self.active_bucket = None
        self.model = self.cache = None
        self.status = "closed"

    def snapshot(self):
        return graph_snapshot(self)
