"""Stage execution; the scheduler never invokes a whole-token model forward."""

from copy import deepcopy

import torch

from vllm_lt.core.kv_cache_manager import _metadata_payload_bytes
from vllm_lt.core.scheduler import SchedulerOutput
from vllm_lt.request import Request, Stage


class ModelRunner:
    def __init__(self, model, cache_manager):
        self.model = model.eval()
        self.cache_manager = cache_manager
        self.device = next(model.parameters()).device
        self._persistent = None
        self._persistent_stream = None
        self._inside_execute = False
        self._persistent_lease = None

    @staticmethod
    def _tensor_description(tensor):
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

    def _enable_persistent_decode(self):
        """Opt in to one fixed eager capacity; construction never changes request KV."""
        self.cache_manager._require_usable()
        if self._persistent is not None:
            raise RuntimeError("persistent decode is already enabled; replacement is forbidden")
        parameter = next(self.model.parameters())
        if parameter.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("persistent decode supports bfloat16 and float32")
        if parameter.dtype != self.cache_manager.dtype:
            raise ValueError("persistent model and KV cache must use the same dtype")
        if parameter.device != self.cache_manager.device:
            raise ValueError("persistent model and KV cache must use the same device")
        width = self.model.config.hidden_size
        payload_bytes = (
            (2 * 8 * width + 8) * parameter.element_size() + _metadata_payload_bytes() + 4 * 8
        )
        if payload_bytes > 256 * 1024:
            raise ValueError("persistent decode exceeds the 256 KiB tensor payload cap")
        setup_stream = (
            torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        )
        metadata = self.cache_manager._allocate_metadata_storage()
        tensors = {
            **metadata.tensors,
            "hidden_in": torch.empty((8, width), dtype=parameter.dtype, device=self.device),
            "hidden_out": torch.empty((8, width), dtype=parameter.dtype, device=self.device),
            "gate_out": torch.empty(8, dtype=parameter.dtype, device=self.device),
            "live_indices": torch.tensor([1, 3, 5, 7], dtype=torch.long, device=self.device),
        }
        # Publish the bundle only after every allocation succeeds. No partially
        # constructed executor is reachable after a constructor exception.
        self._persistent_stream = setup_stream
        self._persistent = {
            "metadata": metadata,
            "tensors": tensors,
            "counters": {"calls": 0, "prepared": 0, "completed": 0, "empty": 0},
            "fallback_counts": {"live_count": 0, "table_width": 0},
            "last_dispatch": None,
            "last_publication": None,
            "failure": None,
        }

    def _persistent_snapshot(self):
        """Detached host metadata only; never read device tensor contents."""
        bundle = self._persistent
        if bundle is None:
            return {"enabled": False}
        metadata = bundle["metadata"]
        return {
            "enabled": True,
            "status": "failed" if metadata.failed else "in_flight" if metadata.in_use else "ready",
            "generation": metadata.generation,
            "capacity": {"row_count": 8, "table_width": 32, "max_live_rows": 4},
            "device_payload_bytes": sum(
                t.numel() * t.element_size() for t in bundle["tensors"].values()
            ),
            "cpu_staging_bytes": sum(
                t.numel() * t.element_size() for t in metadata.staging.values()
            ),
            "tensors": {k: self._tensor_description(v) for k, v in bundle["tensors"].items()},
            "staging_tensors": {
                k: self._tensor_description(v) for k, v in metadata.staging.items()
            },
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
        if self._persistent is not None and self._persistent["metadata"].failed:
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

    @staticmethod
    def _error_description(error):
        return {"type": type(error).__name__, "message": str(error)[:512]}

    def _settle_persistent_failure(self, error, *, completion_error=None):
        """Invalidate before cleanup; return whether pages are safe to release."""
        bundle = self._persistent
        if bundle is None:
            return True
        if bundle["failure"] is not None:
            return bundle["failure"]["completion_confirmed"]
        metadata = bundle["metadata"]
        metadata.failed = True
        failure = {
            "primary": self._error_description(error),
            "secondary": [],
            "completion_confirmed": False,
        }
        bundle["failure"] = failure
        if completion_error is not None:
            failure["secondary"].append(self._error_description(completion_error))
            self.cache_manager._quarantine("persistent stream completion failed")
            return False
        try:
            self._synchronize_persistent()
        except BaseException as completion_error:
            failure["secondary"].append(self._error_description(completion_error))
            self.cache_manager._quarantine("persistent stream completion failed")
            return False
        failure["completion_confirmed"] = True
        metadata.in_use = False
        self._persistent_lease = None
        return True

    def _record_cleanup_failure(self, error):
        if self._persistent is not None and self._persistent["failure"] is not None:
            self._persistent["failure"]["secondary"].append(self._error_description(error))
        self.cache_manager._quarantine("request cleanup failed after execution failure")

    def _finish_persistent_lease(self):
        if self._persistent_lease is not None:
            self.cache_manager._release_prepared(self._persistent_lease)
            self._persistent["counters"]["completed"] += 1
            self._persistent_lease = None

    @torch.inference_mode()
    def execute(self, batch: SchedulerOutput):
        self._require_execution_usable()
        if self._persistent is None:
            return self._execute_batch(batch)
        if self._inside_execute:
            raise RuntimeError("persistent execution cannot be reentered")
        self._select_persistent_stream()
        self._inside_execute = True
        try:
            return self._execute_batch(batch)
        except BaseException as error:
            self._settle_persistent_failure(error)
            raise
        finally:
            self._inside_execute = False

    def _execute_batch(self, batch: SchedulerOutput):
        requests = [item.request for item in batch.items]
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
        if self._persistent is not None:
            return self._recurrent_persistent(hidden, request_ids, depths, positions)
        return self.model.recurrent(hidden, request_ids, depths, positions, self.cache_manager)

    @torch.inference_mode()
    def _recurrent_persistent(self, hidden, request_ids, depths, positions):
        self._require_execution_usable()
        bundle = self._persistent
        if bundle is None:
            raise RuntimeError("persistent decode is not enabled")
        if bundle["metadata"].in_use:
            raise RuntimeError("persistent decode still has an in-flight lease")
        if hidden.shape != (len(request_ids), self.model.config.hidden_size):
            raise ValueError("persistent decode requires compact live hidden inputs")
        if hidden.device != self.device or hidden.dtype != bundle["tensors"]["hidden_in"].dtype:
            raise ValueError("persistent hidden inputs must match the model dtype and device")
        host = self.cache_manager._prepare_host_batch(request_ids, depths, positions)
        count = len(host.rows)
        reason = "live_count" if count > 4 else "table_width" if host.width > 32 else None
        bundle["counters"]["calls"] += 1
        bundle["last_publication"] = None
        bundle["last_dispatch"] = {
            "kind": "empty" if not count else "compact" if reason else "persistent",
            "request_ids": list(request_ids),
            "depths": [depth for _, depth, _ in host.rows],
            "positions": [position for _, _, position in host.rows],
            "live_rows": list(range(count)) if reason else [2 * i + 1 for i in range(count)],
            "row_count": count if reason else 8,
            "table_width": host.width if reason else 32,
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
                "hidden": self._tensor_description(outputs[0]),
                "gates": self._tensor_description(outputs[1]),
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
