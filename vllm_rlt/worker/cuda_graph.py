"""Bounded recurrent-core graphs; request bookkeeping and sampling stay eager."""

import math
from dataclasses import replace
from types import SimpleNamespace

import torch


class _DeviceCache:
    """Capture only GPU cache operations, never Python allocation/written state."""

    def __init__(self, cache):
        self.cache = cache

    def _write_prepared(self, layer, batch, k, v):
        cache = self.cache
        cache.key_cache[batch.write_blocks, layer, batch.write_offsets] = k
        cache.value_cache[batch.write_blocks, layer, batch.write_offsets] = v

    def _attend_prepared(self, layer, batch, q):
        cache = self.cache
        return cache.attention(
            q,
            cache.key_cache[:, layer],
            cache.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
        )


class RecurrentGraphs:
    def __init__(self, model, cache, execution, compute_gate):
        self.model, self.cache = model, cache
        self.execution, self.compute_gate = execution, compute_gate
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=cache.device)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0

    @torch.inference_mode()
    def run(self, hidden, batch):
        count = len(batch.rows)
        if len(batch.position_ids) != count:
            batch = replace(
                batch,
                position_ids=batch.position_ids[:count],
                context_lengths=batch.context_lengths[:count],
                block_tables=batch.block_tables[:count],
            )
        if (
            count > self.execution.cuda_graph_max_batch_size
            or not count
            or batch.cu_seqlens_q is not None
            or (
                count not in self.entries
                and len(self.entries) >= self.execution.cuda_graph_max_graphs
            )
        ):
            self.fallbacks += 1
            return self.model.recurrent_prepared(
                hidden, batch, self.cache, compute_gate=self.compute_gate
            )
        cache = self.cache
        cache._require_live_batch(batch)
        # Decode writes exactly one position per request/depth. Validate history
        # before launch; capture/replay must not freeze or repeat host mutations.
        for allocation, depth, pos in batch.rows:
            for layer in range(cache.num_layers):
                cache._require_prefix(allocation, layer, depth, pos)
        stream = torch.cuda.current_stream(cache.device)
        if self.last_event is not None:
            stream.wait_event(self.last_event)
        entry = self.entries.get(count)
        if entry is None:
            width = math.ceil(self.model.config.max_position_embeddings / cache.block_size)
            entry = SimpleNamespace(
                hidden=torch.empty_like(hidden[:count]),
                metadata=SimpleNamespace(
                    position_ids=torch.empty(count, device=cache.device, dtype=torch.long),
                    write_blocks=torch.empty(count, device=cache.device, dtype=torch.long),
                    write_offsets=torch.empty(count, device=cache.device, dtype=torch.long),
                    block_tables=torch.zeros(
                        (count, width), device=cache.device, dtype=torch.int32
                    ),
                    context_lengths=torch.empty(count, device=cache.device, dtype=torch.int32),
                ),
            )
        entry.hidden.copy_(hidden[:count])
        for name in ("position_ids", "write_blocks", "write_offsets", "context_lengths"):
            getattr(entry.metadata, name).copy_(getattr(batch, name)[:count])
        width = batch.block_tables.shape[1]
        entry.metadata.block_tables[:, :width].copy_(batch.block_tables[:count])
        if count not in self.entries:
            # First-use capture is intentionally outside steady-state timing.
            # Drain outstanding streams: CUDA capture cannot race other launches.
            torch.cuda.synchronize(cache.device)
            proxy = _DeviceCache(cache)
            with torch.cuda.stream(self.capture_stream):
                for _ in range(2):
                    self.model.recurrent_prepared(
                        entry.hidden, entry.metadata, proxy, compute_gate=self.compute_gate
                    )
            self.capture_stream.synchronize()
            entry.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(entry.graph, pool=self.pool, stream=self.capture_stream):
                entry.output = self.model.recurrent_prepared(
                    entry.hidden, entry.metadata, proxy, compute_gate=self.compute_gate
                )
            self.entries[count] = entry
            self.captures += 1
        entry.graph.replay()
        # Outputs must outlive reuse of graph-private buffers and the shared pool.
        output = tuple(t.clone() if t is not None else None for t in entry.output)
        self.last_event = torch.cuda.Event()
        self.last_event.record(stream)
        for allocation, depth, pos in batch.rows:
            for layer in range(cache.num_layers):
                allocation.written[cache._plane(depth)][layer].add(pos)
        self.replays += 1
        return output
