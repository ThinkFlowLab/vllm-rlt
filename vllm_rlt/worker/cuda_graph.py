"""Bounded recurrent-core graphs; request bookkeeping and sampling stay eager."""

import logging
import math
from dataclasses import replace
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)


def graph_capture_sizes(execution):
    """Fixed geometric row buckets, bounded by both user capacity settings."""
    sizes = [execution.cuda_graph_max_batch_size]
    while sizes[-1] > 1 and len(sizes) < execution.cuda_graph_max_graphs:
        sizes.append((sizes[-1] + 1) // 2)
    return tuple(reversed(sizes))


def _bucket_size(sizes, count):
    return next((size for size in sizes if count <= size), count)


def _cache_full(owner, key):
    full = key not in owner.entries and len(owner.entries) >= owner.execution.cuda_graph_max_graphs
    if full and not owner.warned_cache_full:
        logger.warning(
            "%s cache is full (%d shapes); unseen shapes will run eager. "
            "Existing shapes still replay; lazy capture synchronizes the device.",
            type(owner).__name__,
            len(owner.entries),
        )
        owner.warned_cache_full = True
    return full


def _capture(hidden, stream, pool, run):
    """Warm and capture one stable-input graph on its own stream."""
    torch.cuda.synchronize(hidden.device)
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            output = run()
        return graph, output
    finally:
        # A failed warmup can leave KV writes on the capture stream without a
        # compute-stream event. Drain it before tensor/allocation owners unwind.
        stream.synchronize()


class _DeviceCache:
    """Capture only GPU cache operations, never Python allocation/written state."""

    def __init__(self, cache):
        self.cache = cache

    def _write_prepared(self, layer, batch, k, v):
        cache = self.cache
        if batch.active_rows is not None:
            from vllm_rlt.kernels.graph_kv import write_graph_kv

            write_graph_kv(cache.key_cache[:, layer], cache.value_cache[:, layer], batch, k, v)
        else:
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


def _graph_entry(
    hidden, batch, cache, max_position_embeddings, *, count, packed=False, bucketed=False
):
    tables = count if bucketed else len(batch.context_lengths)
    width = math.ceil(max_position_embeddings / cache.block_size)
    return SimpleNamespace(
        hidden=hidden.new_empty((count, *hidden.shape[1:])),
        metadata=SimpleNamespace(
            active_rows=torch.zeros((), device=cache.device, dtype=torch.int32)
            if bucketed
            else None,
            position_ids=torch.empty(count, device=cache.device, dtype=torch.long),
            write_blocks=torch.empty(count, device=cache.device, dtype=torch.long),
            write_offsets=torch.empty(count, device=cache.device, dtype=torch.long),
            block_tables=torch.zeros((tables, width), device=cache.device, dtype=torch.int32),
            context_lengths=torch.empty(tables, device=cache.device, dtype=torch.int32),
            cu_seqlens_q=(
                torch.empty(tables + 1, device=cache.device, dtype=torch.int32) if packed else None
            ),
            max_seqlen_q=batch.max_seqlen_q if packed else 1,
        ),
    )


def _copy_graph_inputs(entry, hidden, batch, *, count):
    metadata = entry.metadata
    if metadata.active_rows is not None:
        metadata.active_rows.fill_(count)
        entry.hidden[count:].zero_()
        metadata.position_ids[count:].zero_()
        metadata.write_blocks[count:].zero_()
        metadata.write_offsets[count:].zero_()
        metadata.context_lengths[count:].zero_()
        metadata.block_tables.zero_()
    entry.hidden[:count].copy_(hidden[:count])
    metadata.position_ids[:count].copy_(batch.position_ids[:count])
    metadata.write_blocks[:count].copy_(batch.write_blocks[:count])
    metadata.write_offsets[:count].copy_(batch.write_offsets[:count])
    tables = len(batch.context_lengths)
    metadata.context_lengths[:tables].copy_(batch.context_lengths)
    if metadata.cu_seqlens_q is not None:
        metadata.cu_seqlens_q.copy_(batch.cu_seqlens_q)
    metadata.block_tables[:tables, : batch.block_tables.shape[1]].copy_(batch.block_tables)


class RecurrentGraphs:
    def __init__(self, model, cache, execution, compute_gate):
        self.model, self.cache = model, cache
        self.execution, self.compute_gate = execution, compute_gate
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=cache.device)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0
        self.warned_cache_full = False

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
        if count > self.execution.cuda_graph_max_batch_size or not count or _cache_full(self, key):
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
                for layer in cache.recurrent_layers:
                    cache._require_prefix(allocation, layer, depth, pos)
            elif pos != previous + 1:
                self.fallbacks += 1
                return self.model.recurrent_prepared(
                    hidden, batch, cache, compute_gate=self.compute_gate
                )
            frontier[row_key] = pos
        output = self._replay(hidden, batch, key, count=count, packed=packed)
        for allocation, depth, pos in batch.rows:
            for layer in cache.recurrent_layers:
                allocation.written[cache._plane(depth)][layer].add(pos)
        return output

    def _replay(self, hidden, batch, key, *, count, packed=False, capacity=None):
        cache = self.cache
        stream = torch.cuda.current_stream(cache.device)
        if self.last_event is not None:
            stream.wait_event(self.last_event)
        entry = self.entries.get(key)
        if entry is None:
            entry = _graph_entry(
                hidden,
                batch,
                cache,
                self.model.config.max_position_embeddings,
                count=capacity if capacity is not None else count,
                packed=packed,
                bucketed=capacity is not None,
            )
        _copy_graph_inputs(entry, hidden, batch, count=count)
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
        output = tuple(t[:count].clone() if t is not None else None for t in entry.output)
        self.last_event = torch.cuda.Event()
        self.last_event.record(stream)
        self.replays += 1
        return output


class DeviceRecurrentGraphs(RecurrentGraphs):
    """Core graphs whose actual positions and KV frontier remain on device.

    Allocation identity is checked on every call, outside capture. Entries retain
    only tensor metadata, never request allocations or CPU written-position state.
    """

    def __init__(self, model, cache, execution, compute_gate):
        super().__init__(model, cache, execution, compute_gate)
        self.capture_sizes = graph_capture_sizes(execution)

    @torch.inference_mode()
    def prepare(self):
        """Capture the finite bucket set before any serving requests exist."""
        parameter = next(self.model.parameters())
        hidden = parameter.new_empty((0, self.model.config.hidden_size))
        batch = SimpleNamespace(
            position_ids=torch.empty(0, device=self.cache.device, dtype=torch.long),
            write_blocks=torch.empty(0, device=self.cache.device, dtype=torch.long),
            write_offsets=torch.empty(0, device=self.cache.device, dtype=torch.long),
            context_lengths=torch.empty(0, device=self.cache.device, dtype=torch.int32),
            block_tables=torch.empty((0, 1), device=self.cache.device, dtype=torch.int32),
            cu_seqlens_q=None,
        )
        for size in self.capture_sizes:
            self._replay(hidden, batch, size, count=0, capacity=size)
        self.replays = 0

    @torch.inference_mode()
    def run(self, hidden, batch):
        self.cache._require_live_batch(batch)
        count = len(batch.position_ids)
        if not count or count > self.execution.cuda_graph_max_batch_size:
            self.fallbacks += 1
            return self.model.recurrent_prepared(
                hidden, batch, self.cache, compute_gate=self.compute_gate
            )
        capacity = _bucket_size(self.capture_sizes, count)
        return self._replay(hidden, batch, capacity, count=count, capacity=capacity)


class CodaGraphs:
    """Capture decode coda and its head; commit boundary KV outside capture."""

    def __init__(self, model, execution, cache=None, *, bucketed=False):
        self.model, self.execution = model, execution
        self.cache = cache
        if bucketed and cache is not None:
            raise ValueError("bucketed coda graphs require a row-independent Ouro head")
        self.capture_sizes = graph_capture_sizes(execution) if bucketed else None
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=next(model.parameters()).device)
        self.last_event = None
        self.captures = self.replays = self.fallbacks = 0
        self.warned_cache_full = False

    @torch.inference_mode()
    def prepare(self):
        if self.capture_sizes is None:
            return
        parameter = next(self.model.parameters())
        for size in self.capture_sizes:
            self.run(parameter.new_zeros((size, self.model.config.hidden_size)))
        self.replays = 0

    @torch.inference_mode()
    def run(self, hidden, batch=None):
        count = len(hidden)
        if self.cache is not None:
            if (
                batch is None
                or not batch.writable
                or len(batch.rows) != count
                or batch.cu_seqlens_q is not None
            ):
                raise ValueError("coda graphs require matching decode rows and KV metadata")
            self.cache._require_live_batch(batch)
            for allocation, depth, position in batch.rows:
                for layer in self.model.coda_kv_layers:
                    self.cache._require_prefix(allocation, layer, depth, position)
        capacity = _bucket_size(self.capture_sizes, count) if self.capture_sizes else count
        if (
            not count
            or count > self.execution.cuda_graph_max_batch_size
            or _cache_full(self, capacity)
        ):
            self.fallbacks += 1
            return (
                self.model.coda_prepared(hidden, batch, self.cache)
                if self.cache is not None
                else self.model.coda(hidden)
            )
        stream = torch.cuda.current_stream(hidden.device)
        if self.last_event is not None:
            stream.wait_event(self.last_event)
        entry = self.entries.get(capacity)
        if entry is None:
            entry = (
                _graph_entry(
                    hidden,
                    batch,
                    self.cache,
                    self.model.config.max_position_embeddings,
                    count=count,
                )
                if self.cache is not None
                else SimpleNamespace(hidden=hidden.new_empty((capacity, *hidden.shape[1:])))
            )
        if self.cache is not None:
            _copy_graph_inputs(entry, hidden, batch, count=count)
        else:
            entry.hidden[:count].copy_(hidden)
            entry.hidden[count:].zero_()
        if capacity not in self.entries:
            run = (
                (
                    lambda: self.model.coda_prepared(
                        entry.hidden, entry.metadata, _DeviceCache(self.cache)
                    )
                )
                if self.cache is not None
                else lambda: self.model.coda(entry.hidden)
            )
            entry.graph, entry.output = _capture(entry.hidden, self.stream, self.pool, run)
            self.entries[capacity] = entry
            self.captures += 1
        entry.graph.replay()
        output = entry.output[:count].clone()
        self.last_event = torch.cuda.Event()
        self.last_event.record(stream)
        if self.cache is not None:
            for allocation, depth, position in batch.rows:
                for layer in self.model.coda_kv_layers:
                    allocation.written[self.cache._plane(depth)][layer].add(position)
        self.replays += 1
        return output
