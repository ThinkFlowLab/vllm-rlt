"""Evidence serialization and explicit budgets, separate from graph execution."""

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch

SETUP_WARMUPS = 3
SCRATCH_PAGES = 4
SCRATCH_TOKENS = 2
REPLAY_ATOL = 1e-5

_MEMORY_LIMITS = {
    "retained_allocated_bytes": "graph_retained_allocated_bytes",
    "retained_reserved_bytes": "graph_retained_reserved_bytes",
    "peak_allocated_bytes": "setup_peak_allocated_bytes",
    "peak_reserved_bytes": "setup_peak_reserved_bytes",
}


def _memory_deltas(before, after):
    return {
        f"{kind}_{memory}_bytes": max(
            0, after[f"{prefix}{memory}_bytes"] - before[f"{memory}_bytes"]
        )
        for kind, prefix in (("retained", ""), ("peak", "peak_"))
        for memory in ("allocated", "reserved")
    }


@dataclass(frozen=True)
class GraphLimits:
    common_payload_bytes: int = 1024 * 1024
    cpu_staging_bytes: int = 16 * 1024
    graph_retained_allocated_bytes: int = 256 * 1024**2
    graph_retained_reserved_bytes: int = 256 * 1024**2
    setup_peak_allocated_bytes: int = 512 * 1024**2
    setup_peak_reserved_bytes: int = 512 * 1024**2
    setup_timeout_s: int = 60

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class CaptureBudgetExceeded(RuntimeError):
    """Only a declared executor limit, never a CUDA/allocator/device exception."""

    def __init__(self, limits, limit_name, observed):
        self.limits = asdict(limits)
        self.limit_name = limit_name
        self.limit = self.limits[limit_name]
        self.observed = observed
        super().__init__(
            f"configured capture budget {limit_name} exceeded: "
            f"observed={observed}, limit={self.limit}"
        )

    def record(self):
        return {
            "type": type(self).__name__,
            "message": str(self),
            "limit_name": self.limit_name,
            "limit": self.limit,
            "observed": self.observed,
        }


def _description(tensor):
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


def _hash(tensor):
    cpu = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(cpu.numpy()).cast("B")).hexdigest()


def _error(error):
    return {"type": type(error).__name__, "message": str(error)[:512]}


def _bucket_snapshot(bucket):
    metadata = bucket["metadata"]
    return {
        "generation": metadata.generation,
        "tensors": {name: _description(value) for name, value in bucket["tensors"].items()},
        "staging_tensors": {name: _description(value) for name, value in metadata.staging.items()},
    }


def _storage_snapshot(buckets):
    snapshots = {str(rows): _bucket_snapshot(bucket) for rows, bucket in buckets.items()}
    return {
        "buckets": snapshots,
        **{
            field: sum(t["size_bytes"] for b in snapshots.values() for t in b[key].values())
            for field, key in (
                ("device_payload_bytes", "tensors"),
                ("cpu_staging_bytes", "staging_tensors"),
            )
        },
    }


def graph_snapshot(executor):
    storage = _storage_snapshot(executor.buckets)
    for rows, bucket in executor.buckets.items():
        metadata = bucket["metadata"]
        storage["buckets"][str(rows)].update(
            {
                "row_count": rows,
                "table_width": executor.layout.table_width,
                "max_live_rows": rows // 2,
                "setup_generation": bucket["setup_generation"],
                "in_use": metadata.in_use,
                "failed": metadata.failed,
                "graph_id": id(bucket["graph"]) if bucket["graph"] is not None else None,
                "graph_exec_id": bucket["graph_exec_id"],
                "pool_id": bucket["pool_id"],
                "pool_owner": {
                    "kind": "torch.cuda.MemPool",
                    "id": list(executor.pool_owner.id),
                    "release_policy": "synchronize-reset-drop-captured-outputs-owner-last",
                }
                if bucket["pool_id"] is not None
                else None,
                "captured_inputs": bucket["captured_inputs"],
                "captured_outputs": bucket["captured_outputs"],
                "counters": bucket["counters"],
                "verification": bucket.get("verification"),
            }
        )
    return deepcopy(
        {
            "enabled": True,
            "status": executor.status,
            "use_graphs": executor.use_graphs,
            "backend": executor.cache.backend if executor.cache is not None else None,
            "limits": asdict(executor.limits),
            "layout": {
                "max_num_seqs": executor.layout.max_num_seqs,
                "row_counts": list(executor.layout.row_counts),
                "table_width": executor.layout.table_width,
                "pool_scope": "executor",
            },
            **storage,
            "setup": executor.setup_record,
            "counters": executor.counters,
            "fallback_counts": executor.fallback_counts,
            "last_dispatch": executor.last_dispatch,
            "last_publication": executor.last_publication,
            "failure": executor.failure,
        }
    )
