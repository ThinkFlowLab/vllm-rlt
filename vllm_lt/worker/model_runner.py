"""Stage execution; the scheduler never invokes a whole-token model forward."""

from copy import deepcopy

import torch

from vllm_lt.core.scheduler import SchedulerOutput
from vllm_lt.request import Request, Stage
from vllm_lt.worker.decode_buffers import DecodeBucketLayout, allocate_bucket
from vllm_lt.worker.graph_diagnostics import (
    GraphLimits,
    _bucket_snapshot,
    _description,
    _error,
    _storage_snapshot,
)


class ModelRunner:
    def __init__(self, model, cache_manager, *, max_num_seqs=4):
        self.model = model.eval()
        self.cache_manager = cache_manager
        self.device = next(model.parameters()).device
        self._decode_layout = DecodeBucketLayout(max_num_seqs=max_num_seqs)
        self._persistent = None
        self._decode_executor = None
        self._graph_declined = None
        self._persistent_stream = None
        self._inside_execute = False
        self._persistent_lease = None

    @property
    def decode_executor(self):
        """Current private executor, exposed for lifecycle and diagnostics."""
        return self._decode_executor

    def _require_decode_unconfigured(self):
        self.cache_manager._require_usable()
        if (
            self._persistent is not None
            or self._decode_executor is not None
            or self._graph_declined is not None
        ):
            raise RuntimeError(
                "a private decode executor is already enabled; replacement is forbidden"
            )

    def _enable_persistent_decode(self):
        """Prepare the scheduler-sized eager bucket ladder before request admission."""
        self._require_decode_unconfigured()
        parameter = next(self.model.parameters())
        if parameter.dtype != torch.float32 or self.cache_manager.dtype != torch.float32:
            raise ValueError("persistent decode initially supports float32 only")
        if parameter.device != self.cache_manager.device:
            raise ValueError("persistent model and KV cache must use the same device")
        width = self.model.config.hidden_size
        payload_bytes, staging_bytes = self._decode_layout.payload_bytes(width)
        limits = GraphLimits()
        if payload_bytes > limits.common_payload_bytes or staging_bytes > limits.cpu_staging_bytes:
            raise ValueError("persistent decode exceeds its 1 MiB tensor / 16 KiB staging cap")
        setup_stream = (
            torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        )
        buckets = {
            rows: allocate_bucket(self.cache_manager, self._decode_layout, rows, width)
            for rows in self._decode_layout.row_counts
        }
        # Publish the bundle only after every allocation succeeds. No partially
        # constructed executor is reachable after a constructor exception.
        self._persistent_stream = setup_stream
        self._persistent = {
            **buckets[self._decode_layout.row_counts[-1]],
            "buckets": buckets,
            "counters": {"calls": 0, "prepared": 0, "completed": 0, "empty": 0},
            "fallback_counts": {"live_count": 0, "table_width": 0},
            "last_dispatch": None,
            "last_publication": None,
            "failure": None,
        }

    def _enable_recurrent_graph(self, *, use_graphs: bool, limits=None):
        """Install the private eager/replay buckets before any request admission."""
        from .recurrent_graph import CaptureBudgetExceeded, RecurrentGraphExecutor

        self._require_decode_unconfigured()
        executor = None
        try:
            executor = RecurrentGraphExecutor(
                self.model,
                self.cache_manager,
                use_graphs=use_graphs,
                limits=limits,
                layout=self._decode_layout,
            )
            # Retain partial setup storage on failure until completion permits close.
            self._decode_executor = executor
            executor.setup()
        except CaptureBudgetExceeded as error:
            attempt_setup = attempt_failure = None
            if executor is not None:
                # setup() already settles its failure. Any secondary device or
                # restoration error prevents decline, even if a later close might
                # establish completion; that remains explicit recovery work.
                failure = executor.failure
                if failure is None or not failure["completion_confirmed"] or failure["secondary"]:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            "capture budget decline refused: setup did not settle cleanly"
                        )
                    raise
                attempt_setup = deepcopy(executor.setup_record)
                attempt_failure = deepcopy(failure)
                try:
                    executor.close()
                    self.cache_manager._require_usable()
                    if self.cache_manager._allocations:
                        raise RuntimeError("capture budget decline retained scratch allocations")
                except BaseException as secondary:
                    self.cache_manager._quarantine("capture budget decline cleanup failed")
                    if hasattr(error, "add_note"):
                        error.add_note(f"capture budget decline cleanup failed: {secondary}")
                    raise error from secondary
            # Constructor declines happened before allocating/submitting anything.
            # Otherwise the entire attempted executor was closed after confirmed
            # completion, leaving the original pool available to ordinary eager.
            self._graph_declined = {
                "reason": "capture_budget",
                "attempted_use_graphs": use_graphs,
                "limits": dict(error.limits),
                "error": error.record(),
                "attempt_setup": attempt_setup,
                "attempt_failure": attempt_failure,
                "completion_confirmed": True,
                "calls": 0,
                "closed": False,
            }
            self._decode_executor = None

    def _graph_snapshot(self):
        if self._graph_declined is not None:
            return {
                "enabled": False,
                "status": "closed" if self._graph_declined["closed"] else "budget_fallback",
                "budget_decline": deepcopy(self._graph_declined),
            }
        return (
            {"enabled": False}
            if self._decode_executor is None
            else self._decode_executor.snapshot()
        )

    def _close_recurrent_graph(self):
        if self._decode_executor is not None:
            self._decode_executor.close()
        elif self._graph_declined is not None:
            self._graph_declined["closed"] = True

    def _settle_execution_failure(self, error):
        if self._decode_executor is not None:
            return self._decode_executor.settle_failure(error)
        return self._settle_persistent_failure(error)

    def _persistent_snapshot(self):
        """Detached host metadata only; never read device tensor contents."""
        bundle = self._persistent
        if bundle is None:
            return {"enabled": False}
        metadata = bundle["metadata"]
        return {
            "enabled": True,
            "status": "failed" if metadata.failed else "in_flight" if metadata.in_use else "ready",
            "capacity": {
                "row_count": self._decode_layout.row_counts[-1],
                "table_width": self._decode_layout.table_width,
                "max_live_rows": self._decode_layout.max_num_seqs,
            },
            **_storage_snapshot(bundle["buckets"]),
            **_bucket_snapshot(bundle),
            **deepcopy(
                {
                    k: bundle[k]
                    for k in (
                        "counters",
                        "fallback_counts",
                        "last_dispatch",
                        "last_publication",
                        "failure",
                    )
                }
            ),
        }

    def _require_execution_usable(self):
        self.cache_manager._require_usable()
        if self._graph_declined is not None and self._graph_declined["closed"]:
            raise RuntimeError("declined recurrent executor is closed")
        if self._decode_executor is not None:
            self._decode_executor.require_usable()
        if self._persistent is not None and any(
            b["metadata"].failed for b in self._persistent["buckets"].values()
        ):
            raise RuntimeError("persistent executor failed; retry is forbidden")

    def _select_persistent_stream(self):
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            if self._persistent_stream is not None and stream != self._persistent_stream:
                raise RuntimeError("persistent execution requires its original ordered stream")
            self._persistent_stream = stream

    def _synchronize_persistent(self):
        if self._persistent_stream is not None:
            self._persistent_stream.synchronize()

    def _settle_persistent_failure(self, error, *, completion_error=None):
        """Invalidate before cleanup; return whether pages are safe to release."""
        bundle = self._persistent
        if bundle is None:
            return True
        if bundle["failure"] is not None:
            return bundle["failure"]["completion_confirmed"]
        for bucket in bundle["buckets"].values():
            bucket["metadata"].failed = True
        failure = {
            "primary": _error(error),
            "secondary": [],
            "completion_confirmed": False,
        }
        bundle["failure"] = failure
        if completion_error is not None:
            failure["secondary"].append(_error(completion_error))
            self.cache_manager._quarantine("persistent stream completion failed")
            return False
        try:
            self._synchronize_persistent()
        except BaseException as completion_error:
            failure["secondary"].append(_error(completion_error))
            self.cache_manager._quarantine("persistent stream completion failed")
            return False
        failure["completion_confirmed"] = True
        for bucket in bundle["buckets"].values():
            bucket["metadata"].in_use = False
        self._persistent_lease = None
        return True

    def _record_cleanup_failure(self, error):
        if self._decode_executor is not None:
            self._decode_executor.record_cleanup_failure(error)
            return
        if self._persistent is not None and self._persistent["failure"] is not None:
            self._persistent["failure"]["secondary"].append(_error(error))
        self.cache_manager._quarantine("request cleanup failed after execution failure")

    def _finish_persistent_lease(self):
        if self._persistent_lease is not None:
            self.cache_manager._release_prepared(self._persistent_lease)
            self._persistent["counters"]["completed"] += 1
            self._persistent_lease = None

    @torch.inference_mode()
    def execute(self, batch: SchedulerOutput):
        self._require_execution_usable()
        if self._persistent is None and self._decode_executor is None:
            return self._execute_batch(batch)
        if self._inside_execute:
            raise RuntimeError("persistent execution cannot be reentered")
        if self._decode_executor is not None:
            self._decode_executor.select_stream()
        else:
            self._select_persistent_stream()
        self._inside_execute = True
        try:
            return self._execute_batch(batch)
        except BaseException as error:
            self._settle_execution_failure(error)
            raise
        finally:
            self._inside_execute = False

    def _execute_batch(self, batch: SchedulerOutput):
        requests = [item.request for item in batch.items]
        if not requests:
            return [] if batch.stage in (Stage.RECURRENT, Stage.CODA) else None
        if batch.stage == Stage.PREFILL:
            ids, positions, tokens = [], [], []
            for item in batch.items:
                start, count, request = item.token_start, item.token_count, item.request
                ids.extend([request.request_id] * count)
                positions.extend(range(start, start + count))
                tokens.extend(request.prompt_token_ids[start : start + count])
            hidden = self.model.prelude(torch.tensor(tokens, device=self.device, dtype=torch.long))
            # Every prompt token reaches the model's full depth, independently of decode policy.
            for depth in range(self.model.config.total_ut_steps):
                hidden, _ = self.model.recurrent(
                    hidden, ids, [depth] * len(ids), positions, self.cache_manager
                )
            offset = 0
            for item in batch.items:
                offset += item.token_count
                item.request.hidden_state = hidden[offset - 1].clone()
            return None
        if batch.stage == Stage.PRELUDE:
            token_ids = torch.tensor(
                [r.input_token_id for r in requests], device=self.device, dtype=torch.long
            )
            hidden = self.model.prelude(token_ids)
            for request, state in zip(requests, hidden):
                request.hidden_state = state
            return None
        hidden = torch.stack([r.hidden_state for r in requests])
        if batch.stage == Stage.RECURRENT:
            hidden, gate_logits = self._recurrent(
                hidden,
                [r.request_id for r in requests],
                [r.loops_done for r in requests],
                [r.position for r in requests],
            )
            if self._decode_executor is not None:
                probabilities = gate_logits.float().sigmoid().cpu().tolist()
                self._decode_executor.complete_after_gate()
                for request, state in zip(requests, hidden):
                    request.hidden_state = state
                return probabilities
            for request, state in zip(requests, hidden):
                request.hidden_state = state
            # This is explicitly synchronous. A stock gate cannot act as the paper's lookahead gate.
            probabilities = gate_logits.float().sigmoid().cpu().tolist()
            # The real gate readback completes preceding copies and publication
            # on the same stream; the private helper alone cannot release its lease.
            self._finish_persistent_lease()
            return probabilities
        if batch.stage == Stage.CODA:
            logits = self.model.coda(hidden)
            return [self._sample(row, request) for row, request in zip(logits, requests)]
        raise ValueError(f"unsupported execution stage {batch.stage}")

    def _recurrent(self, hidden, request_ids, depths, positions):
        """Private decode seam; persistent storage requires explicit setup."""
        if self._decode_executor is not None:
            return self._decode_executor.recurrent(
                hidden, request_ids, depths, positions, defer_completion=self._inside_execute
            )
        if self._persistent is not None:
            return self._recurrent_persistent(hidden, request_ids, depths, positions)
        if self._graph_declined is not None:
            self._require_execution_usable()
            self._graph_declined["calls"] += 1
        return self.model.recurrent(hidden, request_ids, depths, positions, self.cache_manager)

    @torch.inference_mode()
    def _recurrent_persistent(self, hidden, request_ids, depths, positions):
        self._require_execution_usable()
        bundle = self._persistent
        if bundle is None:
            raise RuntimeError("persistent decode is not enabled")
        if any(b["metadata"].in_use for b in bundle["buckets"].values()):
            raise RuntimeError("persistent decode still has an in-flight lease")
        if hidden.shape != (len(request_ids), self.model.config.hidden_size):
            raise ValueError("persistent decode requires compact live hidden inputs")
        if hidden.device != self.device or hidden.dtype != torch.float32:
            raise ValueError("persistent hidden inputs must match the float32 model device")
        host = self.cache_manager._prepare_host_batch(request_ids, depths, positions)
        count = len(host.rows)
        rows, reason = self._decode_layout.select(count, host.width)
        if not reason and count:
            # The selected view supports existing diagnostic callers; all
            # bucket storage remains owned by the executor between traversals.
            bundle.update(bundle["buckets"][rows])
        bundle["counters"]["calls"] += 1
        bundle["last_publication"] = None
        bundle["last_dispatch"] = {
            "kind": "empty" if not count else "compact" if reason else "persistent",
            "request_ids": list(request_ids),
            "depths": [depth for _, depth, _ in host.rows],
            "positions": [position for _, _, position in host.rows],
            "live_rows": list(range(count)) if reason else [2 * i + 1 for i in range(count)],
            "row_count": count if reason else rows,
            "table_width": host.width if reason else self._decode_layout.table_width,
            "generation": bundle["metadata"].generation,
        }
        if not count:
            bundle["counters"]["empty"] += 1
            return hidden.clone(), hidden.new_empty((0,))
        self._select_persistent_stream()
        try:
            if reason:
                bundle["fallback_counts"][reason] += 1
                outputs = self.model.recurrent(
                    hidden, request_ids, depths, positions, self.cache_manager
                )
            else:
                batch = self.cache_manager._prepare_into(bundle["metadata"], host)
                self._persistent_lease = batch
                bundle["counters"]["prepared"] += 1
                bundle["last_dispatch"]["generation"] = batch.generation
                tensors = bundle["tensors"]
                live = tensors["live_indices"][:count]
                tensors["hidden_in"].zero_()
                tensors["hidden_in"].index_copy_(0, live, hidden)
                physical, gates = self.model._recurrent_prepared(
                    tensors["hidden_in"], batch, self.cache_manager
                )
                tensors["hidden_out"].copy_(physical)
                tensors["gate_out"].copy_(gates)
                outputs = (
                    tensors["hidden_out"].index_select(0, live),
                    tensors["gate_out"].index_select(0, live),
                )
            bundle["last_publication"] = {
                "hidden": _description(outputs[0]),
                "gates": _description(outputs[1]),
            }
            if not self._inside_execute:
                try:
                    self._synchronize_persistent()
                except BaseException as error:
                    self._settle_persistent_failure(error, completion_error=error)
                    raise
                self._finish_persistent_lease()
            return outputs
        except BaseException as error:
            self._settle_persistent_failure(error)
            raise

    def _recurrent_padded(
        self,
        hidden,
        request_ids,
        depths,
        positions,
        *,
        row_indices,
        row_count,
        table_width,
    ):
        """Exercise padding eagerly, returning only live rows in scheduler order.

        No scheduler item, coda input or request generator is manufactured for
        padding. Gathered output owns storage independently of the physical
        traversal, which may outlive publication to a paused request.
        """
        if hidden.ndim != 2 or hidden.shape != (len(request_ids), self.model.config.hidden_size):
            raise ValueError("padded recurrent requires compact live hidden inputs")
        batch = self.cache_manager._prepare_batch(request_ids, depths, positions)
        batch = self.cache_manager._pad_prepared(
            batch, row_indices=row_indices, row_count=row_count, table_width=table_width
        )
        live = torch.tensor(batch.live_rows, device=self.device, dtype=torch.long)
        physical = hidden.new_zeros((row_count, hidden.shape[1]))
        physical.index_copy_(0, live, hidden)
        physical, gates = self.model._recurrent_prepared(physical, batch, self.cache_manager)
        return physical.index_select(0, live), gates.index_select(0, live)

    def _sample(self, logits: torch.Tensor, request: Request) -> int:
        params = request.sampling_params
        if params.temperature == 0:
            return int(logits.argmax().item())
        logits = logits.float() / params.temperature
        if params.top_k > 0:
            threshold = logits.topk(min(params.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if params.top_p < 1:
            sorted_logits, indices = logits.sort(descending=True)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
        if request.generator is None:
            request.generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return int(torch.multinomial(logits.softmax(-1), 1, generator=request.generator).item())
