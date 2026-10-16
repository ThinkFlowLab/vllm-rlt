"""Stage execution, random lookahead signals and event-owned CUDA submissions."""

import math
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import torch

from vllm_rlt.config import ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.core.execution import ExecutionBatch, Operation
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Request
from vllm_rlt.worker.buffers import Workspace, prefill_stage_rows
from vllm_rlt.worker.cuda_graph import CodaGraphs, RecurrentGraphs
from vllm_rlt.worker.execution import ExecutionRuntime, ForwardBatch
from vllm_rlt.worker.sampler import RngRegistry, Sampler


@dataclass
class ReadbackSlot:
    storage: torch.Tensor
    leased: bool = False
    event: object = None


@dataclass
class PrefillMetadataBuffers:
    """One in-flight batch's reusable descriptors and packed token output.

    Sources stores borrowed tensor references, never owned copies. CUDA pinned
    descriptors are read directly on NVIDIA; HIP gets a device descriptor copy.
    token_sources rows: (pointer, token_start, token_count, batch_row_start).
    hidden_sources rows: (pointer, row_stride, col_stride, token_count,
                          batch_row_start, chunk_output_pointer, last_output_pointer).
    """

    host_segments: torch.Tensor
    device_segments: torch.Tensor
    host_segments_array: object
    host_token_ids: torch.Tensor
    device_token_ids: torch.Tensor
    host_token_ids_array: object
    host_token_sources: torch.Tensor
    host_token_sources_array: object
    host_hidden_sources: torch.Tensor
    host_hidden_sources_array: object
    device_hidden_sources: torch.Tensor
    # Pointer descriptors borrow tensor storage until the final consumer event.
    sources: list[torch.Tensor] = field(default_factory=list)
    event: object = None
    in_use: bool = False


class PrefillSegment(NamedTuple):
    state_slot: int
    chunk_offset: int
    token_count: int
    batch_row_start: int
    keep_intermediate: bool


class PrefillMetadataPool:
    def __init__(self, max_num_seqs, max_num_batched_tokens, device, num_slots=3):
        self.slots = []
        self.next_slot = 0
        self.use_uva = device.type == "cuda" and not torch.version.hip
        for _ in range(num_slots):
            host = torch.empty(
                (max_num_seqs, 5),
                dtype=torch.int64,
                pin_memory=device.type == "cuda",
            )
            device_segments = (
                torch.empty_like(host, device=device)
                if device.type == "cuda" and not self.use_uva
                else host
            )
            host_tokens = torch.empty(
                max_num_batched_tokens,
                dtype=torch.long,
                pin_memory=device.type == "cuda",
            )
            device_tokens = (
                torch.empty_like(host_tokens, device=device)
                if device.type == "cuda"
                else host_tokens
            )
            token_sources = torch.empty(
                (max_num_seqs, 4), dtype=torch.int64, pin_memory=device.type == "cuda"
            )
            hidden_sources = torch.empty(
                (max_num_seqs, 7), dtype=torch.int64, pin_memory=device.type == "cuda"
            )
            device_hidden_sources = (
                torch.empty_like(hidden_sources, device=device)
                if device.type == "cuda" and not self.use_uva
                else hidden_sources
            )
            self.slots.append(
                PrefillMetadataBuffers(
                    host,
                    device_segments,
                    host.numpy(),
                    host_tokens,
                    device_tokens,
                    host_tokens.numpy(),
                    token_sources,
                    token_sources.numpy(),
                    hidden_sources,
                    hidden_sources.numpy(),
                    device_hidden_sources,
                )
            )

    def acquire(self):
        pending = None
        for offset in range(len(self.slots)):
            index = (self.next_slot + offset) % len(self.slots)
            buffers = self.slots[index]
            if buffers.in_use:
                continue
            if buffers.event is None or buffers.event.query():
                buffers.event = None
                buffers.sources.clear()
                buffers.in_use = True
                self.next_slot = (index + 1) % len(self.slots)
                return buffers
            if pending is None:
                pending = index, buffers
        if pending is not None:
            index, buffers = pending
            buffers.event.synchronize()
            buffers.event = None
            buffers.sources.clear()
            buffers.in_use = True
            self.next_slot = (index + 1) % len(self.slots)
            return buffers
        raise RuntimeError("all prefill metadata buffers are in use")

    def release(self, buffers):
        buffers.sources.clear()
        buffers.event = None
        buffers.in_use = False

    def release_after(self, buffers, event):
        buffers.event = event
        buffers.in_use = False


@dataclass(frozen=True)
class PrefillMetadata:
    buffers: PrefillMetadataBuffers
    rows: int
    max_rows: int
    token_ids: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    mixed: bool = False


@dataclass(frozen=True)
class PreparedExecution:
    batch: ExecutionBatch
    depths: tuple[int, ...]
    positions: tuple[int, ...]
    output_indices: tuple[int, ...]
    token_tensors: tuple[torch.Tensor, ...] = ()
    kv: object = None
    routing: object = None
    prefill_metadata: PrefillMetadata | None = None


@dataclass
class Submission:
    batch: SchedulerOutput | None
    values: torch.Tensor | None
    event: object = None
    cached: list | None = None
    slot: ReadbackSlot | None = None
    device_values: torch.Tensor | None = None
    depths: tuple[int, ...] = ()
    output_indices: tuple[int, ...] = ()

    def __del__(self):
        if self.slot is not None:
            self.slot.leased = False

    @property
    def has_result(self):
        return self.values is not None or self.cached is not None

    def ready(self):
        return self.event is None or self.event.query()

    def collect(self):
        if self.cached is not None:
            return self.cached
        if self.event is not None:
            self.event.synchronize()
        if self.cached is None and self.values is not None:
            self.cached = self.values.tolist()
            self.values = None
            if self.slot is not None:
                self.slot.leased = False
                self.slot = None
        return self.cached


class ModelRunner:
    def __init__(
        self,
        model,
        cache_manager,
        *,
        exit_config=None,
        execution_config=None,
        scheduler_config=None,
        rng: RngRegistry | None = None,
    ):
        self.model = model.eval()
        self.cache_manager = cache_manager
        self.runtime = ExecutionRuntime(self.model, cache_manager)
        parameter = next(model.parameters())
        self.device = parameter.device
        self.sampler = Sampler()
        # Sampling RNG state lives with the engine, which shares this registry
        # between the CODA runner and the speculative runner.
        self.rng = rng if rng is not None else RngRegistry()
        self.exit_config = exit_config or ExitConfig()
        self.execution_config = execution_config or ExecutionConfig()
        scheduler = scheduler_config or SchedulerConfig()
        self.graphs = (
            RecurrentGraphs(
                model,
                cache_manager,
                self.execution_config,
                self.exit_config.mode in ("ouro", "ouro_delayed"),
            )
            if self.execution_config.cuda_graphs
            else None
        )
        self.coda_graphs = (
            CodaGraphs(model, self.execution_config, cache_manager)
            if self.graphs is not None and getattr(model, "requires_boundary_kv", False)
            else None
        )
        self._prefill_banks = None
        self._prefill_bank_index = 0
        self.lookahead_head = None
        if self.exit_config.mode == "random_lookahead":
            # Keep auxiliary random weights separate from the strict base checkpoint.
            # CPU RNG isolation also makes initialization independent of device count.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(self.exit_config.seed)
                head = torch.nn.Linear(model.config.hidden_size, 1, device="cpu")
                torch.nn.init.normal_(
                    head.weight, std=model.config.initializer_range, generator=generator
                )
                torch.nn.init.zeros_(head.bias)
            self.lookahead_head = head.to(device=self.device, dtype=parameter.dtype).eval()
            self.lookahead_head.requires_grad_(False)
        self.split_prefill = not self.execution_config.atomic_prefill
        self.core_stream = self.boundary_stream = self.copy_stream = self.readback_stream = None
        if self.device.type == "cuda":
            self.core_stream = torch.cuda.Stream(device=self.device)
            self.boundary_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            self.copy_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            self.readback_stream = (
                torch.cuda.Stream(device=self.device)
                if self.execution_config.multi_stream
                else self.core_stream
            )
            for stream in (
                self.core_stream,
                self.boundary_stream,
                self.copy_stream,
                self.readback_stream,
            ):
                stream.wait_stream(torch.cuda.current_stream(self.device))
        self.readback_slots = []
        if self.device.type == "cuda":
            # cudaHostAlloc during submission can serialize streams. Allocate all
            # readback storage before work starts, retaining leases until consumed.
            for dtype in (torch.float32, torch.int64):
                for _ in range(scheduler.max_num_seqs + 4):
                    self.readback_slots.append(
                        ReadbackSlot(
                            torch.empty(
                                scheduler.max_num_batched_tokens, dtype=dtype, pin_memory=True
                            )
                        )
                    )
        # Two core rounds plus one interleaved boundary stage can be in flight.
        # Prepare the next batch BEFORE retiring the oldest submission.
        self.submission_events = deque()
        self.split_prefill_submissions = 0
        self.split_prefill_submission_waits = 0
        self.events = {}
        self.requestspaces = {}
        self.requestspace_index = {"core": 0, "boundary": 0}
        self.state_slots = {}
        self.free_state_slots = list(reversed(range(scheduler.max_num_seqs)))
        self.states = None
        # Prompt hidden states staged between split-prefill scheduling boundaries.
        # Sized by the same helper the memory plan reserves with.
        stage_rows = prefill_stage_rows(model.config, scheduler, self.execution_config)
        self.prefill_states = (
            torch.empty(
                (scheduler.max_num_seqs, stage_rows, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            if self.split_prefill
            else None
        )
        prefill_rows = scheduler.max_num_batched_tokens
        if self.execution_config.pad_to_power_of_two:
            prefill_rows = 1 << (prefill_rows - 1).bit_length()
        self.prefill_hidden = (
            torch.empty(
                (prefill_rows, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            if self.split_prefill
            else None
        )
        self.prefill_metadata_pool = (
            PrefillMetadataPool(
                scheduler.max_num_seqs,
                max(scheduler.max_num_batched_tokens, scheduler.prefill_chunk_size),
                self.device,
            )
            if self.split_prefill
            else None
        )
        # Immutable prompt rows are filled once per request/runner-slot lifetime.
        # Preallocation avoids cudaHostAlloc and prompt packing on every chunk.
        self.prompt_token_buffers = (
            torch.empty(
                (scheduler.max_num_seqs, model.config.max_position_embeddings),
                dtype=torch.long,
                pin_memory=self.device.type == "cuda",
            )
            if self.split_prefill
            else None
        )
        self.prompt_token_owners = {}
        self.last_effective_size = self.last_submitted_size = 0
        if self.execution_config.static_buffers:
            rows = scheduler.max_num_batched_tokens
            if self.execution_config.pad_to_power_of_two:
                rows = 1 << (rows - 1).bit_length()
            width = math.ceil(model.config.max_position_embeddings / cache_manager.block_size)
            for group in ("core", "boundary"):
                self.requestspaces[group] = [
                    Workspace(rows, width, model.config.hidden_size, self.device, parameter.dtype)
                    for _ in range(2)
                ]
            self.states = torch.empty(
                (scheduler.max_num_seqs, model.config.hidden_size),
                device=self.device,
                dtype=parameter.dtype,
            )
            # Buffers allocated after stream creation must be visible there too.
            for stream in (self.core_stream, self.boundary_stream):
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream(self.device))

        self.async_state = None
        if self.execution_config.async_scheduling and self.device.type == "cuda":
            from vllm_rlt.worker.async_state import AsyncState

            rows = scheduler.max_num_batched_tokens
            if self.execution_config.pad_to_power_of_two:
                rows = 1 << (rows - 1).bit_length()
            self.async_state = AsyncState(
                cache_manager, model.config, scheduler, rows, use_uva=not torch.version.hip
            )
            self.states = self.async_state.hidden
            self.state_slots = self.async_state.slots
            self.free_state_slots = self.async_state.free
            for stream in (self.core_stream, self.boundary_stream, self.copy_stream):
                stream.wait_stream(torch.cuda.current_stream(self.device))

    def _size(self, count):
        size = count
        if self.execution_config.pad_to_power_of_two:
            size = 1 << (count - 1).bit_length()
        self.last_effective_size, self.last_submitted_size = count, size
        return size

    def _workspace(self, group):
        if not self.requestspaces:
            return None
        index = self.requestspace_index[group]
        self.requestspace_index[group] = 1 - index
        workspace = self.requestspaces[group][index]
        workspace.acquire()
        return workspace

    def _save(self, request, state):
        if self.states is None or (
            self.async_state is not None and request.request_id not in self.async_state.owners
        ):
            request.hidden_state = state.clone()
        else:
            slot = self._ensure_request_slot(request)
            request.hidden_state = self.states[slot]
            request.hidden_state.copy_(state)

    def _ensure_request_slot(self, request):
        if self.async_state is not None:
            return self.async_state.reserve_slot(request)
        rid = request.request_id
        if rid not in self.state_slots:
            self.state_slots[rid] = self.free_state_slots.pop()
        return self.state_slots[rid]

    def _prefill_segments(self, batch, buffers):
        row_start = 0
        max_rows = 0
        for index, span in enumerate(batch.spans):
            state_slot = self._ensure_request_slot(span.request)
            chunk_offset = span.token_start - span.chunk_start
            keep_intermediate = span.retain_chunk
            segment = PrefillSegment(
                state_slot,
                chunk_offset,
                span.token_count,
                row_start,
                keep_intermediate,
            )
            buffers.host_segments_array[index] = (
                segment.state_slot,
                segment.chunk_offset,
                segment.token_count,
                segment.batch_row_start,
                int(segment.keep_intermediate),
            )
            row_start += span.token_count
            max_rows = max(max_rows, span.token_count)
        count = len(batch.spans)
        if self.device.type == "cuda" and not self.prefill_metadata_pool.use_uva:
            buffers.device_segments[:count].copy_(buffers.host_segments[:count], non_blocking=True)
        return row_start, max_rows

    def _prepare_prefill_tokens(self, batch, buffers, rows, max_rows):
        row = 0
        for index, span in enumerate(batch.spans):
            request = span.request
            end = row + span.token_count
            if span.input_tokens is not None:
                buffers.host_token_ids_array[row:end] = span.input_tokens
                source, start = buffers.host_token_ids, row
            elif span.chunk_start is not None:
                slot = self._ensure_request_slot(request)
                if self.prompt_token_owners.get(request.request_id) is not request:
                    count = len(request.prompt_token_ids)
                    self.prompt_token_buffers[slot, :count].numpy()[:] = request.prompt_token_ids
                    self.prompt_token_owners[request.request_id] = request
                source, start = self.prompt_token_buffers[slot], span.token_start
            elif request.input_token_tensor is not None:
                source, start = request.input_token_tensor.reshape(-1), 0
                if source.is_cuda:
                    source.record_stream(torch.cuda.current_stream(self.device))
                request.input_token_tensor = None
            else:
                buffers.host_token_ids_array[row] = request.input_token_id
                source, start = buffers.host_token_ids, row
            buffers.sources.append(source)
            buffers.host_token_sources_array[index] = (
                source.data_ptr(),
                start,
                span.token_count,
                row,
            )
            if not self.prefill_metadata_pool.use_uva:
                buffers.device_token_ids[row:end].copy_(
                    source[start : start + span.token_count], non_blocking=True
                )
            row = end
        if self.prefill_metadata_pool.use_uva:
            from vllm_rlt.kernels.routing import prepare_prefill_tokens_kernel

            prepare_prefill_tokens_kernel[(len(batch.spans), math.ceil(max_rows / 256))](
                buffers.host_token_sources,
                buffers.device_token_ids,
                256,
            )
        return buffers.device_token_ids[:rows]

    def _mixed_hidden_sources(self, batch, buffers):
        # Borrow inputs before updating request.hidden_state to its output storage.
        # All pointer targets stay alive through buffers.sources until event reuse.
        row = 0
        for index, span in enumerate(batch.spans):
            request = span.request
            slot = self._ensure_request_slot(request)
            chunk = None
            if span.chunk_start is not None:
                start = span.token_start - span.chunk_start
                chunk = self.prefill_states[slot, start : start + span.token_count]
                source = chunk
            else:
                source = request.hidden_state
                source = None if source is None else source.reshape(1, -1)
            last = None
            if span.retain_last:
                if self.states is None:
                    last = torch.empty(
                        self.model.config.hidden_size,
                        device=self.device,
                        dtype=self.prefill_states.dtype,
                    )
                else:
                    last = self.states[slot]
                request.hidden_state = last
            for tensor in (source, chunk, last):
                if tensor is not None:
                    buffers.sources.append(tensor)
                    if tensor.is_cuda:
                        tensor.record_stream(torch.cuda.current_stream(self.device))
            buffers.host_hidden_sources_array[index] = (
                0 if source is None else source.data_ptr(),
                0 if source is None else source.stride(0),
                0 if source is None else source.stride(1),
                span.token_count,
                row,
                chunk.data_ptr() if chunk is not None and span.retain_chunk else 0,
                0 if last is None else last.data_ptr(),
            )
            row += span.token_count
        if self.device.type == "cuda" and not self.prefill_metadata_pool.use_uva:
            buffers.device_hidden_sources.copy_(buffers.host_hidden_sources, non_blocking=True)

    def _mixed_hidden(self, hidden, prepared, *, scatter):
        metadata = prepared.prefill_metadata
        buffers = metadata.buffers
        if self.device.type == "cuda":
            from vllm_rlt.kernels.routing import mixed_hidden_kernel

            mixed_hidden_kernel[
                (
                    len(prepared.batch.spans),
                    math.ceil(metadata.max_rows / 4),
                    math.ceil(hidden.shape[1] / 256),
                )
            ](
                buffers.device_hidden_sources,
                hidden,
                hidden.stride(0),
                hidden.shape[1],
                scatter,
                4,
                256,
            )
            return
        tensors = {tensor.data_ptr(): tensor for tensor in buffers.sources}
        for (
            ptr,
            stride,
            col_stride,
            count,
            row,
            chunk_ptr,
            last_ptr,
        ) in buffers.host_hidden_sources_array[: len(prepared.batch.spans)]:
            if not scatter:
                hidden[row : row + count].copy_(tensors[ptr])
            else:
                if chunk_ptr:
                    tensors[chunk_ptr].copy_(hidden[row : row + count])
                if last_ptr:
                    tensors[last_ptr].copy_(hidden[row + count - 1])

    def _gather_prefill(self, prepared):
        metadata = prepared.prefill_metadata
        count = metadata.rows
        hidden = self.prefill_hidden[:count]
        if metadata.mixed:
            self._mixed_hidden(hidden, prepared, scatter=False)
            return hidden
        if self.device.type == "cuda":
            from vllm_rlt.kernels.routing import gather_segments_kernel

            pool = self.prefill_states.view(-1, self.model.config.hidden_size)
            gather_segments_kernel[
                (
                    len(prepared.batch.spans),
                    math.ceil(metadata.max_rows / 4),
                    math.ceil(pool.shape[1] / 256),
                )
            ](
                pool,
                metadata.buffers.device_segments[: len(prepared.batch.spans)],
                hidden,
                self.prefill_states.shape[1],
                pool.shape[1],
                pool.stride(0),
                hidden.stride(0),
                4,
                256,
            )
            return hidden
        for values in metadata.buffers.host_segments_array[: len(prepared.batch.spans)]:
            segment = PrefillSegment(*values)
            start = segment.batch_row_start
            end = start + segment.token_count
            hidden[start:end].copy_(
                self.prefill_states[
                    segment.state_slot,
                    segment.chunk_offset : segment.chunk_offset + segment.token_count,
                ]
            )
        return hidden

    def _scatter_prefill(self, hidden, prepared):
        metadata = prepared.prefill_metadata
        if metadata.mixed:
            self._mixed_hidden(hidden, prepared, scatter=True)
            return
        if self.device.type == "cuda":
            from vllm_rlt.kernels.routing import scatter_segments_kernel

            pool = self.prefill_states.view(-1, self.model.config.hidden_size)
            scatter_segments_kernel[
                (
                    len(prepared.batch.spans),
                    math.ceil(metadata.max_rows / 4),
                    math.ceil(pool.shape[1] / 256),
                )
            ](
                hidden,
                metadata.buffers.device_segments[: len(prepared.batch.spans)],
                pool,
                self.prefill_states.shape[1],
                pool.shape[1],
                hidden.stride(0),
                pool.stride(0),
                4,
                256,
            )
        else:
            for values in metadata.buffers.host_segments_array[: len(prepared.batch.spans)]:
                segment = PrefillSegment(*values)
                if not segment.keep_intermediate:
                    continue
                self.prefill_states[
                    segment.state_slot,
                    segment.chunk_offset : segment.chunk_offset + segment.token_count,
                ].copy_(
                    hidden[segment.batch_row_start : segment.batch_row_start + segment.token_count]
                )

    def _store_split_prefill_outputs(self, hidden, prepared):
        row = 0
        batch = prepared.batch
        for span in batch.spans:
            if span.retain_last:
                self._save(span.request, hidden[row + span.token_count - 1])
            row += span.token_count

    def _save_batch(self, requests, hidden, routing):
        if routing is None:
            for request, state in zip(requests, hidden):
                self._save(request, state)
            return
        routing.scatter(hidden)
        for request in requests:
            request.hidden_state = self.states[self.state_slots[request.request_id]]

    def _gather(self, requests, workspace, size):
        if workspace is None:
            return torch.stack([r.hidden_state for r in requests])
        hidden = workspace.hidden[:size]
        hidden.zero_()
        for row, request in enumerate(requests):
            hidden[row].copy_(request.hidden_state)
        return hidden

    def _core(self, hidden, ids, depths, positions, workspace, size):
        batch = (
            self.cache_manager._prepare_batch(ids, depths, positions)
            if workspace is None
            else workspace.prepare(self.cache_manager, ids, depths, positions, size)
        )
        return self.runtime.advance(
            hidden, (batch,), compute_gate=self.exit_config.mode in ("ouro", "ouro_delayed")
        )

    def _execute_resident(self, batch):
        if self.cache_manager.layout == "shared":
            # Define chunk-invariant shared semantics: complete all loops of each
            # position before advancing that request. Parallelize across requests.
            for offset in range(max(i.token_count for i in batch.spans)):
                active = [i for i in batch.spans if offset < i.token_count]
                ids = [i.request.request_id for i in active]
                positions = [i.token_start + offset for i in active]
                tokens = [i.input_tokens[offset] for i in active]
                hidden = self._advance_tokens(
                    ids, positions, tokens, [i.request for i in active], batch.loop_count
                )
                for row, span in enumerate(active):
                    self._save(span.request, hidden[row])
            return
        ids, positions, tokens, requests = [], [], [], []
        for span in batch.spans:
            start, count, request = span.token_start, span.token_count, span.request
            ids.extend([request.request_id] * count)
            positions.extend(range(start, start + count))
            tokens.extend(span.input_tokens)
            requests.extend([request] * count)
        hidden = self._advance_tokens(ids, positions, tokens, requests, batch.loop_count)
        offset = 0
        for span in batch.spans:
            offset += span.token_count
            self._save(span.request, hidden[offset - 1])

    def _populate_coda(self, hidden, ids, positions, requests):
        # The last prompt position runs coda once, when producing the first token.
        rows = (
            [
                row
                for row, (position, request) in enumerate(zip(positions, requests))
                if position < len(request.prompt_token_ids) - 1
            ]
            if requests
            else list(range(len(ids)))
        )
        if not rows:
            return
        boundary = self.cache_manager._prepare_batch(
            [ids[row] for row in rows],
            [0] * len(rows),
            [positions[row] for row in rows],
            packed_prefill=self.cache_manager.attention_capabilities.packed_prefill,
        )
        self.runtime.forward(
            ForwardBatch(
                Operation.CODA,
                hidden=hidden[rows],
                boundary_kv=boundary,
                compute_logits=False,
            )
        )

    def _advance_tokens(self, ids, positions, tokens, requests, loop_count):
        cache = self.cache_manager
        generators = (
            [
                self.rng.acquire(request.request_id, request.sampling_params, self.device)
                for request in requests
            ]
            if getattr(self.model, "requires_boundary_kv", False) and requests
            else None
        )
        if cache.layout == "last_exited" and cache.attention_capabilities.packed_prefill:
            # Prefill has genuinely ragged query sequences. Do not pad token rows
            # or reuse decode's per-query, model-max-width static page tables.
            bank = None
            if self.execution_config.prefill_uva:
                from vllm_rlt.worker.prefill_metadata import PrefillMetadataBank

                if self._prefill_banks is None:
                    self._prefill_banks = [PrefillMetadataBank(cache) for _ in range(3)]
                bank = self._prefill_banks[self._prefill_bank_index]
                self._prefill_bank_index = (self._prefill_bank_index + 1) % len(self._prefill_banks)
            try:
                tensor = (
                    bank.prepare(ids, positions, tokens)
                    if bank
                    else torch.tensor(tokens, device=self.device, dtype=torch.long)
                )
                boundary = (
                    cache._prepare_batch(ids, [0] * len(ids), positions, packed_prefill=True)
                    if getattr(self.model, "requires_boundary_kv", False)
                    else None
                )
                metadata = (
                    bank.metadata(depth)
                    if bank
                    else cache._prepare_batch(
                        ids, [depth] * len(ids), positions, packed_prefill=True
                    )
                    for depth in range(loop_count)
                )
                hidden, _ = self.runtime.advance(
                    metadata=metadata,
                    input_tokens=tensor,
                    input_kv=boundary,
                    generators=generators,
                )
                if boundary is not None:
                    self._populate_coda(hidden, ids, positions, requests)
                return hidden
            finally:
                if bank:
                    bank.release()
        size = self._size(len(tokens))
        workspace = self._workspace("core")
        tensor = (
            workspace.tokens(tokens, size)
            if workspace
            else torch.tensor(tokens, device=self.device, dtype=torch.long)
        )
        boundary = (
            cache._prepare_batch(ids, [0] * len(ids), positions)
            if getattr(self.model, "requires_boundary_kv", False)
            else None
        )
        hidden, _ = self.runtime.advance(
            metadata=self._loop_metadata(ids, positions, range(loop_count), workspace, size),
            input_tokens=tensor,
            input_kv=boundary,
            generators=generators,
            hidden_buffer=workspace.hidden[:size] if workspace else None,
            compute_gate=self.exit_config.mode in ("ouro", "ouro_delayed"),
        )
        if boundary is not None:
            self._populate_coda(hidden, ids, positions, requests)
        return hidden

    def _loop_metadata(self, ids, positions, depths, workspace, size):
        for depth in depths:
            if workspace:
                workspace.acquire()
            try:
                yield (
                    workspace.prepare(self.cache_manager, ids, [depth] * len(ids), positions, size)
                    if workspace
                    else self.cache_manager._prepare_batch(ids, [depth] * len(ids), positions)
                )
            finally:
                if workspace:
                    workspace.release()

    def prepare_batch(self, batch: ExecutionBatch):
        batch.validate_depths(self.model.config.total_ut_steps)
        requests = [span.request for span in batch.spans]
        depths = tuple(
            span.request.loops_done if batch.operation == Operation.CODA else span.depth
            for span in batch.spans
        )
        positions = tuple(span.token_start for span in batch.spans)
        indices = tuple(r.num_scheduled_outputs for r in requests)
        prepared = PreparedExecution(batch, depths, positions, indices)
        if batch.operation == Operation.PRELUDE and batch.loop_count:
            return prepared
        if batch.has_chunks:
            buffers = self.prefill_metadata_pool.acquire()
            try:
                if batch.all_chunks:
                    rows, max_rows = self._prefill_segments(batch, buffers)
                else:
                    rows = batch.num_rows
                    max_rows = max(span.token_count for span in batch.spans)
                metadata = PrefillMetadata(buffers, rows, max_rows, mixed=not batch.all_chunks)
                prepared = replace(prepared, prefill_metadata=metadata)
                if metadata.mixed:
                    self._mixed_hidden_sources(batch, buffers)
                if batch.operation == Operation.PRELUDE:
                    tokens = self._prepare_prefill_tokens(batch, buffers, rows, max_rows)
                    return replace(prepared, prefill_metadata=replace(metadata, token_ids=tokens))
                kv = self.cache_manager.prepare_recurrent(batch, packed=True)
                hidden = self._gather_prefill(prepared)
                return replace(prepared, kv=kv, prefill_metadata=replace(metadata, hidden=hidden))
            except Exception:
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                self.prefill_metadata_pool.release(buffers)
                raise
        tokens = ()
        if batch.operation == Operation.PRELUDE:
            if any(span.input_tokens is not None for span in batch.spans):
                return prepared
            tokens = tuple(r.input_token_tensor for r in requests)
            if any(t is None for t in tokens):
                if self.async_state is not None:
                    raise RuntimeError("async prelude requires the preceding device sample")
                # Host-owned inputs use the same submit path and are staged by
                # _load_tokens; no device sample exists before CPU result commit.
                tokens = ()
        if self.async_state is not None:
            routing, kv = self.async_state.prepare(
                requests,
                depths,
                positions,
                self._size(len(requests)),
                recurrent=batch.operation == Operation.RECURRENT,
            )
            return replace(prepared, token_tensors=tokens, kv=kv, routing=routing)
        kv = None
        if batch.operation == Operation.RECURRENT and (
            not self.requestspaces or self.graphs is not None
        ):
            kv = self.cache_manager.prepare_recurrent(batch)
        return replace(prepared, token_tensors=tokens, kv=kv)

    def _release_prepared_metadata(self, prepared, event=None):
        if prepared is None or prepared.prefill_metadata is None:
            return
        buffers = prepared.prefill_metadata.buffers
        if event is None:
            self.prefill_metadata_pool.release(buffers)
        else:
            self.prefill_metadata_pool.release_after(buffers, event)

    def _load_hidden(self, batch, prepared, workspace, size):
        if prepared is not None and prepared.prefill_metadata is not None:
            return prepared.prefill_metadata.hidden
        routing = prepared.routing if prepared is not None else None
        return (
            routing.gather()
            if routing is not None
            else self._gather([span.request for span in batch.spans], workspace, size)
        )

    def _store_hidden(self, batch, hidden, prepared):
        if prepared is not None and prepared.prefill_metadata is not None:
            self._scatter_prefill(hidden, prepared)
            if not prepared.prefill_metadata.mixed:
                self._store_split_prefill_outputs(hidden, prepared)
            return
        routing = prepared.routing if prepared is not None else None
        self._save_batch([span.request for span in batch.spans], hidden, routing)

    def _load_tokens(self, batch, prepared, workspace, size):
        if prepared is not None and prepared.prefill_metadata is not None:
            return prepared.prefill_metadata.token_ids
        requests = [s.request for s in batch.spans]
        if batch.has_chunks or any(s.input_tokens is not None for s in batch.spans):
            rows = []
            for span in batch.spans:
                request = span.request
                if span.input_tokens is not None:
                    rows.append(
                        torch.tensor(span.input_tokens, device=self.device, dtype=torch.long)
                    )
                elif request.input_token_tensor is not None:
                    token = request.input_token_tensor
                    if token.is_cuda:
                        token.record_stream(torch.cuda.current_stream(self.device))
                    rows.append(token.reshape(1))
                    request.input_token_tensor = None
                else:
                    rows.append(
                        torch.tensor([request.input_token_id], device=self.device, dtype=torch.long)
                    )
            return torch.cat(rows)
        if prepared is not None and (prepared.routing is not None or prepared.token_tensors):
            routing = prepared.routing
            tokens = (
                routing.gather(tokens=True)
                if routing is not None
                else torch.stack(prepared.token_tensors)
            )
            if tokens.is_cuda:
                for tensor in prepared.token_tensors:
                    tensor.record_stream(torch.cuda.current_stream(self.device))
            if routing is None and size > len(requests):
                tokens = torch.cat((tokens, tokens.new_zeros(size - len(requests))))
            for request in requests:
                request.input_token_tensor = None
            return tokens
        ids = [r.input_token_id for r in requests]
        return (
            workspace.tokens(ids, size)
            if workspace
            else torch.tensor(ids, device=self.device, dtype=torch.long)
        )

    @torch.inference_mode()
    def _execute(self, scheduled_batch, prepared):
        return self._execute_batch(prepared.batch, prepared)

    def _execute_batch(self, batch: ExecutionBatch, prepared=None):
        batch.validate_depths(self.model.config.total_ut_steps)
        if batch.operation == Operation.PRELUDE and batch.loop_count:
            self._execute_resident(batch)
            return None
        if batch.has_chunks:
            prepared = prepared or self.prepare_batch(batch)
        requests = [span.request for span in batch.spans]
        routing = prepared.routing if prepared is not None else None
        size = batch.num_rows if batch.has_chunks else self._size(batch.num_rows)
        group = "core" if batch.operation == Operation.RECURRENT else "boundary"
        workspace = None if batch.has_chunks or routing is not None else self._workspace(group)
        try:
            if batch.operation == Operation.PRELUDE:
                tokens = self._load_tokens(batch, prepared, workspace, size)
                boundary = (
                    self.cache_manager.prepare_recurrent(batch)
                    if getattr(self.model, "requires_boundary_kv", False)
                    else None
                )
                hidden, _ = self.runtime.advance(
                    input_tokens=tokens,
                    input_kv=boundary,
                    generators=[
                        self.rng.acquire(r.request_id, r.sampling_params, self.device)
                        for r in requests
                    ]
                    if boundary is not None
                    else None,
                )
                self._store_hidden(batch, hidden, prepared)
                return None

            hidden = self._load_hidden(batch, prepared, workspace, size)
            if batch.operation == Operation.RECURRENT:
                kv = prepared.kv if prepared is not None else None
                graphs = self.graphs if batch.emit_gate else None
                if batch.loop_count > 1:
                    # Keep activations resident. Only the first view may have been
                    # prepared ahead of submission; later depths bind lazily.
                    def loop_views():
                        for offset in range(batch.loop_count):
                            yield (
                                kv
                                if offset == 0 and kv is not None
                                else self.cache_manager.prepare_recurrent(
                                    batch.at_loop(offset),
                                    packed=batch.has_chunks,
                                )
                            )

                    hidden, logits = self.runtime.advance(
                        hidden[: batch.num_rows],
                        loop_views(),
                        compute_gate=batch.emit_gate
                        and self.exit_config.mode in ("ouro", "ouro_delayed"),
                        graphs=graphs,
                    )
                elif kv is not None or graphs is not None:
                    if kv is None:
                        kv = self.cache_manager.prepare_recurrent(batch)
                    hidden, logits = self.runtime.advance(
                        hidden[: batch.num_rows] if graphs is not None else hidden,
                        (kv,),
                        compute_gate=batch.emit_gate
                        and self.exit_config.mode in ("ouro", "ouro_delayed"),
                        graphs=graphs,
                    )
                else:
                    ids, depths, positions = batch.addresses()
                    if prepared is not None:
                        depths, positions = prepared.depths, prepared.positions
                    hidden, logits = self._core(hidden, ids, depths, positions, workspace, size)
                self._store_hidden(batch, hidden, prepared)
                if not batch.emit_gate:
                    return None
                if self.lookahead_head is not None:
                    logits = self.lookahead_head(hidden).squeeze(-1)
                if logits is None:
                    return None
                # One result slot per scheduled request, independent of its token count.
                # Only emit_gate consumers interpret the corresponding value.
                if batch.num_rows == len(batch.spans):
                    return logits[: len(batch.spans)].float().sigmoid()
                rows, end = [], 0
                for span in batch.spans:
                    end += span.token_count
                    rows.append(end - 1)
                return logits[rows].float().sigmoid()

            boundary = (
                self.cache_manager.prepare_recurrent(batch)
                if getattr(self.model, "requires_boundary_kv", False)
                else None
            )
            logits = self.runtime.forward(
                ForwardBatch(
                    Operation.CODA,
                    hidden=hidden[: batch.num_rows] if boundary is not None else hidden,
                    boundary_kv=boundary,
                    graphs=self.coda_graphs if boundary is not None else None,
                )
            ).logits
            result = torch.stack([self._sample_row(row, r) for row, r in zip(logits, requests)])
            if routing is not None:
                routing.scatter(result, tokens=True)
            return result
        finally:
            if workspace:
                workspace.release()

    def _readback_slot(self, result):
        pending = None
        for slot in self.readback_slots:
            if slot.storage.dtype != result.dtype or slot.leased:
                continue
            if slot.event is None or slot.event.query():
                slot.leased = True
                return slot
            if pending is None:
                pending = slot
        if pending is not None:
            pending.event.synchronize()
            pending.leased = True
            return pending
        raise RuntimeError("readback buffers exhausted; retire completed submissions")

    def _synchronize_submission(self, stream, prepared):
        if self.device.type != "cuda":
            return
        (stream or torch.cuda.current_stream(self.device)).synchronize()
        if self.readback_stream is not None and self.readback_stream is not stream:
            self.readback_stream.synchronize()
        if prepared is not None and prepared.routing is not None:
            if self.copy_stream is not None and self.copy_stream is not stream:
                self.copy_stream.synchronize()

    def _wait_for_requests(self, batch, stream):
        # A previous batch often gives many requests the same completion event.
        dependencies = set()
        for span in batch.spans:
            event = self.events.get(span.request.request_id)
            if event is not None and event not in dependencies:
                stream.wait_event(event)
                dependencies.add(event)
            if span.request.hidden_state is not None:
                span.request.hidden_state.record_stream(stream)

    @torch.inference_mode()
    def submit(self, scheduled_batch: SchedulerOutput | None = None, *, batch: ExecutionBatch):
        """Submit model computation and return its result/completion ownership.

        A scheduled batch supplies lifecycle snapshots to the engine. Direct
        execution needs only ExecutionBatch; neither path waits for readback here
        unless bounded buffers must be recycled.
        """
        split_prefilling = batch.split_prefilling
        # A fused prelude + loops stays resident on core; chunking does not
        # change the stream of an otherwise standalone prelude or coda.
        stream = self.core_stream if batch.loop_count else self.boundary_stream
        prepared = None
        routing_finish_attempted = False
        slot = None
        try:
            prepare_on_stream = stream is not None and (
                split_prefilling or self.async_state is None
            )
            if prepare_on_stream:
                source_stream = torch.cuda.current_stream(self.device)
                with torch.cuda.stream(stream):
                    if self.async_state is None and source_stream != stream:
                        stream.wait_stream(source_stream)
                    # Chunk gather already reads hidden state, before forward.
                    self._wait_for_requests(batch, stream)
                    prepared = self.prepare_batch(batch)
            else:
                prepared = self.prepare_batch(batch)
            while self.submission_events and self.submission_events[0].query():
                self.submission_events.popleft()
            if len(self.submission_events) >= 3:
                if split_prefilling:
                    self.split_prefill_submission_waits += 1
                self.submission_events.popleft().synchronize()
            if prepared.routing is not None:
                if prepared.routing.imports:
                    self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.copy_stream):
                    kv = prepared.routing.transfer(prepared.kv)
                stream.wait_event(prepared.routing.ready_event)
                prepared = replace(prepared, kv=kv)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if stream is not None and not prepare_on_stream:
                    self._wait_for_requests(batch, stream)
                try:
                    result = self._execute(scheduled_batch, prepared)
                finally:
                    if prepared.routing is not None:
                        routing_finish_attempted = True
                        prepared.routing.record_done()
                # Fixed-depth/trace execution never consumes a gate. In the
                # delayed policy the final loop's signal also has no consumer.
                # Keep graph selection and its captured compute unchanged.
                if (
                    scheduled_batch is not None
                    and self.execution_config.async_scheduling
                    and batch.operation == Operation.RECURRENT
                    and not any(
                        span.emit_gate
                        and self.exit_config.mode in ("ouro_delayed", "random_lookahead")
                        and span.request.sampling_params.exit_threshold < 1
                        and span.depth + batch.loop_count
                        < (
                            span.request.sampling_params.max_loops
                            or self.model.config.total_ut_steps
                        )
                        for span in batch.spans
                    )
                ):
                    result = None
                device_values = result
                event = compute_done = None
                if self.device.type == "cuda":
                    compute_stream = torch.cuda.current_stream(self.device)
                    compute_done = torch.cuda.Event()
                    compute_done.record(compute_stream)
                    event = compute_done
                    if result is not None:
                        slot = self._readback_slot(result)
                        host = slot.storage[: result.numel()].view(result.shape)
                        readback_stream = self.readback_stream or compute_stream
                        with torch.cuda.stream(readback_stream):
                            if readback_stream != compute_stream:
                                readback_stream.wait_event(compute_done)
                                # Results are independent sigmoid/stack outputs,
                                # never borrowed graph or request-state buffers.
                                result.record_stream(readback_stream)
                            host.copy_(result, non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(readback_stream)
                        slot.event = event
                        result = host
                    for span in batch.spans:
                        self.events[span.request.request_id] = compute_done
                    self.submission_events.append(compute_done)
            if split_prefilling:
                self.split_prefill_submissions += 1
            self._release_prepared_metadata(prepared, compute_done)
            return Submission(
                scheduled_batch,
                result,
                event,
                slot=slot,
                device_values=device_values,
                depths=prepared.depths,
                output_indices=prepared.output_indices,
            )
        except Exception:
            self._synchronize_submission(stream, prepared)
            try:
                if (
                    prepared is not None
                    and prepared.routing is not None
                    and not routing_finish_attempted
                ):
                    with torch.cuda.stream(stream) if stream is not None else nullcontext():
                        routing_finish_attempted = True
                        prepared.routing.record_done()
            finally:
                if slot is not None:
                    slot.leased = False
                self._release_prepared_metadata(prepared)
            raise

    def finalize_many(self, requests):
        if not requests:
            return
        if self.async_state is None or self.cache_manager.layout == "shared":
            for request in requests:
                self.finalize(request)
            return
        from vllm_rlt.kernels.routing import finalize_kernel

        cache = self.cache_manager
        copies = []
        for request in requests:
            depth = request.loops_done - 1
            if not cache.token_written(request.request_id, request.position, depth):
                raise RuntimeError("cannot finalize before every layer has written KV")
            if depth + 1 < cache.max_loops:
                copies.append(request)
        if not copies:
            return
        bank, _ = self.async_state.prepare(
            copies,
            [r.loops_done - 1 for r in copies],
            [r.position for r in copies],
            len(copies),
            finalize=True,
        )
        with torch.cuda.stream(self.boundary_stream):
            for request in copies:
                event = self.events.get(request.request_id)
                if event is not None:
                    self.boundary_stream.wait_event(event)
            bank.descriptor = (
                bank.host
                if self.async_state.use_uva
                else bank.host[: bank.count].to(self.device, non_blocking=True)
            )
            channels = cache.num_kv_heads * cache.head_dim
            finalize_kernel[
                (len(copies), cache.max_loops, math.ceil(cache.num_layers * channels / 256))
            ](
                bank.descriptor,
                self.async_state.tables,
                cache.key_cache,
                cache.value_cache,
                self.async_state.width,
                cache.max_loops,
                cache.block_size,
                cache.num_layers,
                channels,
                *cache.key_cache.stride()[:3],
                256,
            )
            bank.record_done()
            for request in copies:
                self.events[request.request_id] = bank.done
                cache.mark_finalized(request.request_id, request.position, request.loops_done - 1)

    def finalize(self, request):
        # The final core event must precede copies and coda on the boundary stream.
        stream = self.boundary_stream
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            event = self.events.get(request.request_id)
            if stream is not None and event is not None:
                stream.wait_event(event)
            self.cache_manager.finalize_token(
                request.request_id, request.position, request.loops_done - 1
            )
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
                self.events[request.request_id] = event

    def _free_runner_slot(self, request_id):
        # The latest submission or KV-finalization event follows earlier work
        # through stream dependencies. Wait before recycling this request's slot.
        event = self.events.pop(request_id, None)
        if event is not None:
            event.synchronize()
        self.prompt_token_owners.pop(request_id, None)
        if self.async_state is not None:
            self.async_state.release(request_id)
            return
        slot = self.state_slots.pop(request_id, None)
        if slot is not None:
            self.free_state_slots.append(slot)

    def suspend(self, request_id):
        """Free the request's runner resources but keep its sampling RNG.

        Preemption suspends a request without terminating it, so its generator
        must survive for the resumed token sequence to match an uninterrupted run.
        """
        self._free_runner_slot(request_id)

    def release(self, request_id):
        """Terminate the request: free its runner resources and drop its RNG.

        A new request reusing the same ID must start from a fresh generator, so
        termination drops the RNG slot while suspension keeps it.
        """
        self._free_runner_slot(request_id)
        self.rng.drop(request_id)

    def synchronize(self):
        for stream in (
            self.core_stream,
            self.boundary_stream,
            self.copy_stream,
            self.readback_stream,
        ):
            if stream is not None:
                stream.synchronize()

    def _sample_row(self, logits: torch.Tensor, request: Request) -> torch.Tensor:
        """Sample one token for one logits row from the engine's RNG registry.

        Greedy rows never acquire a generator. Otherwise the registry creates it on
        the distribution's device, so the generator follows the logits row rather
        than the device the model was loaded on.
        """
        params = request.sampling_params
        generator = (
            None
            if params.temperature == 0
            else self.rng.acquire(request.request_id, params, logits.device)
        )
        return self.sampler.sample(logits, params, generator)
