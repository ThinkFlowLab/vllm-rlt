"""Event-owned pinned UVA banks for packed, ragged Prefill metadata.

One host description covers all recurrence depths. No dependency on vLLM.
The consumer event protects both pinned source and device outputs on reuse.
"""

import torch

from vllm_rlt.core.kv_cache_manager import _PreparedKVBatch


class PrefillMetadataBank:
    def __init__(self, cache):
        self.cache = cache
        self.done = None
        self.capacity = 0
        self.row_capacity = 0

    def prepare(self, ids, positions, tokens):
        from vllm_rlt.kernels.prefill_metadata import expand_prefill, stage_prefill

        if self.done is not None:
            self.done.synchronize()
        self.ids, self.positions = ids, positions
        allocations = self.cache._allocations
        self.allocations = tuple(dict((rid, allocations[rid]) for rid in ids).items())
        sequences, ends, cumulative, unique = [], [], [0], []
        for i, rid in enumerate(ids):
            if not i or rid != ids[i - 1]:
                unique.append(rid)
                ends.append(positions[i] + 1)
                cumulative.append(i + 1)
            else:
                ends[-1] = positions[i] + 1
                cumulative[-1] = i + 1
            sequences.append(len(unique) - 1)
        self.n, self.q = len(ids), len(unique)
        self.width = max(positions) // self.cache.block_size + 1
        tables = []
        for depth in range(self.cache.storage_depths):
            for rid in unique:
                table = list(allocations[rid].block_tables[depth][: self.width])
                tables.extend(table + [-1] * (self.width - len(table)))
        # A single mapped buffer is consumed by a Triton copy, then all metadata
        # expansion reads device memory. CPU data is never modified in flight.
        values = tokens + positions + sequences + ends + cumulative + tables
        if len(values) > self.capacity:
            self.capacity = 1 << (len(values) - 1).bit_length()
            self.host = torch.empty(self.capacity, dtype=torch.int64, pin_memory=True)
            self.device = torch.empty(self.capacity, dtype=torch.int64, device=self.cache.device)
            self.device32 = torch.empty(self.capacity, dtype=torch.int32, device=self.cache.device)
        self.host.numpy()[: len(values)] = values
        stage_prefill[(triton_cdiv(len(values), 256),)](
            self.host, self.device, self.device32, len(values), 256
        )
        n, q = self.n, self.q
        self.tokens = self.device[:n]
        self.position_ids = self.device[n : 2 * n]
        seq = self.device[2 * n : 3 * n]
        self.lengths = self.device32[3 * n : 3 * n + q]
        self.cu = self.device32[3 * n + q : 3 * n + 2 * q + 1]
        self.tables = self.device32[3 * n + 2 * q + 1 : len(values)].view(
            self.cache.storage_depths, q, self.width
        )
        if n > self.row_capacity:
            self.row_capacity = 1 << (n - 1).bit_length()
            self.block_storage = torch.empty(
                (self.cache.storage_depths, self.row_capacity),
                dtype=torch.int64,
                device=self.cache.device,
            )
            self.offset_storage = torch.empty_like(self.block_storage)
        self.blocks = self.block_storage[:, :n]
        self.offsets = self.offset_storage[:, :n]
        for depth in range(self.cache.storage_depths):
            expand_prefill[(triton_cdiv(n, 256),)](
                self.position_ids,
                seq,
                self.tables[depth],
                self.blocks[depth],
                self.offsets[depth],
                n,
                self.width,
                self.cache.block_size,
                256,
            )
        self.max_query = max(b - a for a, b in zip(cumulative, cumulative[1:]))
        return self.tokens

    def metadata(self, depth):
        rows = tuple(
            (self.cache._allocations[rid], depth, pos) for rid, pos in zip(self.ids, self.positions)
        )
        return _PreparedKVBatch(
            self.cache,
            rows,
            self.allocations,
            self.position_ids,
            self.blocks[depth],
            self.offsets[depth],
            self.tables[depth],
            self.lengths,
            True,
            self.cu,
            self.max_query,
        )

    def release(self):
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(self.cache.device))


def triton_cdiv(n, block):
    return (n + block - 1) // block
