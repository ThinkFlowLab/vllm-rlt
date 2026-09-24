"""Bounded recurrent-core graphs; request bookkeeping and sampling stay eager."""

import math
from dataclasses import replace
from types import SimpleNamespace

import torch


def _capture(hidden, stream, pool, run):
    """Warm and capture one stable-input graph on its own stream."""
    torch.cuda.synchronize(hidden.device)
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool, stream=stream):
        output = run()
    return graph, output


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
        if batch.cu_seqlens_q is not None:
            return cache.attention.prefill(
                q,
                cache.key_cache[:, layer],
                cache.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
                batch.cu_seqlens_q,
                batch.max_seqlen_q,
            )
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
        packed = batch.cu_seqlens_q is not None
        key = (count, len(batch.context_lengths), batch.max_seqlen_q if packed else 1, packed)
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
            or (
                key not in self.entries
                and len(self.entries) >= self.execution.cuda_graph_max_graphs
            )
        ):
            self.fallbacks += 1
            return self.model.recurrent_prepared(
                hidden, batch, self.cache, compute_gate=self.compute_gate
            )
        cache = self.cache
        cache._require_live_batch(batch)
        # A speculative verification batch may write consecutive positions at
        # one depth. Its first position needs committed history; later rows are
        # written together before attention reads them.
        frontier = {}
        for allocation, depth, pos in batch.rows:
            row_key = (id(allocation), depth)
            previous = frontier.get(row_key)
            if previous is None:
                for layer in range(cache.num_layers):
                    cache._require_prefix(allocation, layer, depth, pos)
            elif pos != previous + 1:
                self.fallbacks += 1
                return self.model.recurrent_prepared(
                    hidden, batch, cache, compute_gate=self.compute_gate
                )
            frontier[row_key] = pos
        stream = torch.cuda.current_stream(cache.device)
        if self.last_event is not None:
            stream.wait_event(self.last_event)
        entry = self.entries.get(key)
        if entry is None:
            width = math.ceil(self.model.config.max_position_embeddings / cache.block_size)
            tables = len(batch.context_lengths)
            entry = SimpleNamespace(
                hidden=torch.empty_like(hidden[:count]),
                metadata=SimpleNamespace(
                    position_ids=torch.empty(count, device=cache.device, dtype=torch.long),
                    write_blocks=torch.empty(count, device=cache.device, dtype=torch.long),
                    write_offsets=torch.empty(count, device=cache.device, dtype=torch.long),
                    block_tables=torch.zeros(
                        (tables, width), device=cache.device, dtype=torch.int32
                    ),
                    context_lengths=torch.empty(tables, device=cache.device, dtype=torch.int32),
                    cu_seqlens_q=(
                        torch.empty(tables + 1, device=cache.device, dtype=torch.int32)
                        if packed
                        else None
                    ),
                    max_seqlen_q=batch.max_seqlen_q,
                ),
            )
        entry.hidden.copy_(hidden[:count])
        entry.metadata.position_ids.copy_(batch.position_ids[:count])
        entry.metadata.write_blocks.copy_(batch.write_blocks[:count])
        entry.metadata.write_offsets.copy_(batch.write_offsets[:count])
        entry.metadata.context_lengths.copy_(batch.context_lengths)
        if packed:
            entry.metadata.cu_seqlens_q.copy_(batch.cu_seqlens_q)
        width = batch.block_tables.shape[1]
        entry.metadata.block_tables[:, :width].copy_(batch.block_tables)
        if key not in self.entries:
            proxy = _DeviceCache(cache)
            entry.graph, entry.output = _capture(
                entry.hidden,
                self.capture_stream,
                self.pool,
                lambda: self.model.recurrent_prepared(
                    entry.hidden, entry.metadata, proxy, compute_gate=self.compute_gate
                ),
            )
            self.entries[key] = entry
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


class CodaGraphs:
    """Capture fixed-row LM heads while keeping sampling and stop decisions eager."""

    def __init__(self, model, execution):
        self.model, self.execution = model, execution
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=next(model.parameters()).device)
        self.captures = self.replays = self.fallbacks = 0

    @torch.inference_mode()
    def run(self, hidden):
        count = len(hidden)
        if count > self.execution.cuda_graph_max_batch_size or (
            count not in self.entries and len(self.entries) >= self.execution.cuda_graph_max_graphs
        ):
            self.fallbacks += 1
            return self.model.coda(hidden)
        entry = self.entries.get(count)
        if entry is None:
            entry = SimpleNamespace(hidden=torch.empty_like(hidden))
        entry.hidden.copy_(hidden)
        if count not in self.entries:
            entry.graph, entry.output = _capture(
                entry.hidden, self.stream, self.pool, lambda: self.model.coda(entry.hidden)
            )
            self.entries[count] = entry
            self.captures += 1
        entry.graph.replay()
        self.replays += 1
        return entry.output.clone()
