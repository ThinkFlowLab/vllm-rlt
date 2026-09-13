"""Shared bucket selection and tensor specifications for persistent decode."""

from dataclasses import dataclass
from math import prod

import torch

from vllm_lt.core.kv_cache_manager import _metadata_payload_bytes, decode_live_rows


@dataclass(frozen=True)
class DecodeBucketLayout:
    max_num_seqs: int = 4
    table_width: int = 32

    def __post_init__(self):
        for name in ("max_num_seqs", "table_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def row_counts(self):
        # Retain interleaved inactive rows and their existing kernel contract.
        # A physical bucket of 2*N rows supports N live requests.
        rows = [4]
        while rows[-1] // 2 < self.max_num_seqs:
            rows.append(rows[-1] * 2)
        return tuple(rows)

    def select(self, live_count, table_width):
        if live_count > self.max_num_seqs:
            return None, "live_count"
        if table_width > self.table_width:
            return None, "table_width"
        return next(rows for rows in self.row_counts if rows // 2 >= live_count), None

    def payload_bytes(self, hidden_size):
        staging = sum(_metadata_payload_bytes(r, self.table_width) for r in self.row_counts)
        device = staging + sum(
            prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
            for r in self.row_counts
            for shape, dtype in tensor_specifications(r, hidden_size).values()
        )
        return device, staging


def tensor_specifications(rows, hidden_size):
    return {
        "hidden_in": ((rows, hidden_size), torch.float32),
        "hidden_out": ((rows, hidden_size), torch.float32),
        "gate_out": ((rows,), torch.float32),
        "live_indices": ((rows // 2,), torch.long),
    }


def allocate_bucket(cache, layout, rows, hidden_size):
    metadata = cache._allocate_metadata_storage(row_count=rows, table_width=layout.table_width)
    tensors = {
        **metadata.tensors,
        **{
            name: torch.empty(shape, dtype=dtype, device=cache.device)
            for name, (shape, dtype) in tensor_specifications(rows, hidden_size).items()
        },
    }
    tensors["live_indices"].copy_(
        torch.tensor(decode_live_rows(rows // 2), device=cache.device, dtype=torch.long)
    )
    return {"metadata": metadata, "tensors": tensors}
