"""Depth-indexed physical KV pages with LAST-EXITED token propagation.

Each request owns a separate block table for every recurrence depth. When a
token exits, its final K/V states are copied into all deeper planes. Subsequent
tokens may therefore loop further without encountering holes or reading the
wrong depth's history. These are physical copies, not cross-depth aliases.

The initial admission policy reserves a request's complete token budget at all
depths. It is conservative, but admitted requests cannot deadlock on KV growth.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import prod
from numbers import Integral

import torch

from vllm_lt.kernels.paged_attention import torch_paged_attention, triton_paged_attention


def _metadata_specifications(row_count=8, table_width=32):
    """One source for fixed-storage allocation and its preallocation byte cap."""
    return {
        "position_ids": ((row_count,), torch.long),
        "write_blocks": ((row_count,), torch.long),
        "write_offsets": ((row_count,), torch.long),
        "block_tables": ((row_count, table_width), torch.int32),
        "context_lengths": ((row_count,), torch.int32),
        "active": ((row_count,), torch.bool),
    }


def _metadata_payload_bytes(row_count=8, table_width=32):
    return sum(
        prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
        for shape, dtype in _metadata_specifications(row_count, table_width).values()
    )


def decode_live_rows(count):
    """Interleave live rows with inactive sentinels in persistent storage."""
    return tuple(range(1, 2 * count, 2))


@dataclass
class _WrittenPositions:
    # Common in-order writes only advance a scalar. Keep sparse writes until
    # preceding positions arrive, so packed batches may still be out of order.
    prefix: int = 0
    pending: set[int] = field(default_factory=set)

    def add(self, position: int) -> None:
        if position < self.prefix:
            return
        if position > self.prefix:
            self.pending.add(position)
            return
        self.prefix += 1
        while self.prefix in self.pending:
            self.pending.remove(self.prefix)
            self.prefix += 1

    def __contains__(self, position: int) -> bool:
        return position < self.prefix or position in self.pending


@dataclass
class _Allocation:
    max_tokens: int
    block_tables: tuple[tuple[int, ...], ...]
    written: list[list[_WrittenPositions]]


@dataclass(frozen=True, eq=False)
class _HostKVBatch:
    owner: "KVCacheManager"
    rows: tuple[tuple[_Allocation, int, int], ...]
    allocations: tuple[tuple[str, _Allocation], ...]
    addresses: tuple[tuple[int, int], ...]
    width: int
    writable: bool


@dataclass(eq=False)
class _MetadataStorage:
    owner: "KVCacheManager"
    tensors: dict[str, torch.Tensor]
    staging: dict[str, torch.Tensor]
    packed: torch.Tensor
    packed_staging: torch.Tensor
    capacity: tuple[int, int] = (8, 32)
    generation: int = 0
    in_use: bool = False
    failed: bool = False
    transaction: "_DecodeTraversal | None" = None


@dataclass(frozen=True, eq=False)
class _PreparedKVBatch:
    """Borrowed metadata for one synchronous traversal, never a cross-step cache.

    Host rows and allocation identities are captured independently of caller
    lists. Device tensors are private, read-only inputs to the cache operations.
    """

    owner: "KVCacheManager"
    rows: tuple[tuple[_Allocation, int, int], ...]
    allocations: tuple[tuple[str, _Allocation], ...]
    row_count: int
    live_rows: tuple[int, ...]
    # None preserves the compact path without another transfer or mask launch.
    active: torch.Tensor | None
    position_ids: torch.Tensor
    write_blocks: torch.Tensor
    write_offsets: torch.Tensor
    block_tables: torch.Tensor
    context_lengths: torch.Tensor
    writable: bool
    storage: _MetadataStorage | None = None
    generation: int | None = None
    allocation_generation: int | None = None


@dataclass(eq=False)
class _DecodeTraversal:
    owner: "KVCacheManager"
    host: _HostKVBatch
    allocation_generation: int
    batch: _PreparedKVBatch | None = None
    state: str = "begun"


@dataclass(frozen=True, eq=False)
class _TensorKVView:
    """Fixed tensor arguments and kernels, with no host allocation or lease state.

    Construct once outside capture. The caller validates ownership and preceding
    history before submission, and commits written prefixes after completion.
    Inactive rows are selected only by the device mask, including during replay.
    """

    row_count: int
    position_ids: torch.Tensor
    write_blocks: torch.Tensor
    write_offsets: torch.Tensor
    block_tables: torch.Tensor
    context_lengths: torch.Tensor
    active: torch.Tensor
    key_layers: tuple[torch.Tensor, ...]
    value_layers: tuple[torch.Tensor, ...]
    write_kernel: Callable
    attention_kernel: Callable

    def _write_prepared(self, layer, batch, k, v):
        self.write_kernel(
            self.key_layers[layer],
            self.value_layers[layer],
            batch.write_blocks,
            batch.write_offsets,
            k,
            v,
            batch.active,
        )

    def _attend_prepared(self, layer, batch, q):
        return self.attention_kernel(
            q,
            self.key_layers[layer],
            self.value_layers[layer],
            batch.block_tables,
            batch.context_lengths,
            active=batch.active,
        )


class KVCacheManager:
    """Own a fixed physical page pool shared by request/depth allocations.

    ``num_blocks`` counts total physical blocks across all recurrence depths.
    Every physical block contains K and V for all decoder layers. Depth indices
    are zero-based throughout the interface.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        max_loops: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: str = "torch",
    ):
        for name, value in (
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("max_loops", max_loops),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if backend not in {"torch", "triton"}:
            raise ValueError("backend must be 'torch' or 'triton'")
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("KV dtype must be float32, float16, or bfloat16")
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = backend
        if backend == "triton" and self.device.type != "cuda":
            raise ValueError("the Triton attention backend requires a CUDA or ROCm device")
        if backend == "triton" and self.head_dim > 256:
            raise ValueError("the Triton attention backend supports head_dim <= 256")
        shape = (num_blocks, num_layers, block_size, num_kv_heads, head_dim)
        self.key_cache = torch.empty(shape, device=self.device, dtype=dtype)
        self.value_cache = torch.empty_like(self.key_cache)
        # Resolve an implicit CUDA index to the actual storage device once.
        self.device = self.key_cache.device
        self._free_blocks = list(reversed(range(num_blocks)))
        self._allocations: dict[str, _Allocation] = {}
        self._allocation_generation = 0
        self._quarantine_reason: str | None = None

    def _require_usable(self) -> None:
        if self._quarantine_reason is not None:
            raise RuntimeError(f"KV cache is quarantined: {self._quarantine_reason}")

    def _quarantine(self, reason: str) -> None:
        # Preserve allocation identities/counters. These pages cannot be recycled.
        if self._quarantine_reason is None:
            self._quarantine_reason = str(reason)[:512]

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def bytes_per_block(self) -> int:
        return (
            2
            * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.key_cache.element_size()
        )

    def required_blocks(self, max_tokens: int) -> int:
        if not isinstance(max_tokens, Integral) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return ((max_tokens + self.block_size - 1) // self.block_size) * self.max_loops

    def allocate(self, request_id: str, max_tokens: int) -> bool:
        """Reserve all depths atomically; return False only for temporary pressure."""
        self._require_usable()
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._allocations:
            raise ValueError(f"request {request_id!r} already owns KV blocks")
        required = self.required_blocks(max_tokens)
        if required > self.num_blocks:
            raise ValueError(
                f"request needs {required} physical KV blocks, but the pool has {self.num_blocks}"
            )
        if required > self.num_free_blocks:
            return False
        pages_per_depth = required // self.max_loops
        tables = tuple(
            tuple(self._free_blocks.pop() for _ in range(pages_per_depth))
            for _ in range(self.max_loops)
        )
        self._allocations[request_id] = _Allocation(
            int(max_tokens),
            tables,
            [[_WrittenPositions() for _ in range(self.num_layers)] for _ in range(self.max_loops)],
        )
        self._allocation_generation += 1
        return True

    def free(self, request_id: str) -> None:
        """Release only this request's pages; repeated cleanup is harmless."""
        self._require_usable()
        allocation = self._allocations.pop(request_id, None)
        if allocation is not None:
            self._allocation_generation += 1
            for table in allocation.block_tables:
                self._free_blocks.extend(reversed(table))

    def get_block_table(self, request_id: str, depth: int) -> tuple[int, ...]:
        self._validate_depth(depth)
        return self._get_allocation(request_id).block_tables[depth]

    def _get_allocation(self, request_id: str) -> _Allocation:
        self._require_usable()
        try:
            return self._allocations[request_id]
        except KeyError:
            raise KeyError(f"request {request_id!r} has no KV allocation") from None

    def _validate_layer(self, layer: int) -> None:
        if (
            not isinstance(layer, Integral)
            or isinstance(layer, bool)
            or not 0 <= layer < self.num_layers
        ):
            raise ValueError(f"layer must be in [0, {self.num_layers})")

    def _validate_depth(self, depth: int) -> None:
        if (
            not isinstance(depth, Integral)
            or isinstance(depth, bool)
            or not 0 <= depth < self.max_loops
        ):
            raise ValueError(f"depth must be in [0, {self.max_loops})")

    def _validate_position(self, allocation: _Allocation, position: int) -> None:
        if (
            not isinstance(position, Integral)
            or isinstance(position, bool)
            or not 0 <= position < allocation.max_tokens
        ):
            raise ValueError(f"position must be in [0, {allocation.max_tokens})")

    def _validate_rows(self, request_ids, depths, positions):
        self._require_usable()
        if isinstance(positions, torch.Tensor):
            if positions.ndim != 1 or positions.dtype not in {torch.int32, torch.int64}:
                raise ValueError("positions must be a one-dimensional integer tensor")
            positions = positions.tolist()
        if len(request_ids) != len(depths) or len(request_ids) != len(positions):
            raise ValueError("request_ids, depths, and positions must have equal lengths")
        rows = []
        for request_id, depth, position in zip(request_ids, depths, positions):
            allocation = self._get_allocation(request_id)
            self._validate_depth(depth)
            self._validate_position(allocation, position)
            rows.append((allocation, int(depth), int(position)))
        return rows

    def _prepare_host_batch(
        self,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        *,
        for_write: bool = True,
    ) -> _HostKVBatch:
        """Validate all logical ownership/addresses before allocating or copying tensors."""
        rows = tuple(self._validate_rows(request_ids, depths, positions))
        addresses = [
            (
                allocation.block_tables[depth][position // self.block_size],
                position % self.block_size,
            )
            for allocation, depth, position in rows
        ]
        if for_write and len(set(addresses)) != len(addresses):
            raise ValueError(
                "a write batch cannot contain duplicate request/depth/position addresses"
            )
        width = max((position // self.block_size + 1 for _, _, position in rows), default=0)
        allocations = dict(zip(request_ids, (allocation for allocation, _, _ in rows)))
        return _HostKVBatch(
            self, rows, tuple(allocations.items()), tuple(addresses), width, for_write
        )

    def _prepare_batch(
        self,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        *,
        for_write: bool = True,
    ) -> _PreparedKVBatch:
        """Build layer-independent addresses once; do not initialize any KV slot."""
        host = self._prepare_host_batch(request_ids, depths, positions, for_write=for_write)
        rows, addresses, width = host.rows, host.addresses, host.width
        tables = []
        for allocation, depth, _ in rows:
            table = allocation.block_tables[depth][:width]
            tables.append(list(table) + [-1] * (width - len(table)))
        return _PreparedKVBatch(
            owner=self,
            rows=rows,
            allocations=host.allocations,
            row_count=len(rows),
            live_rows=tuple(range(len(rows))),
            active=None,
            position_ids=torch.tensor(
                [position for _, _, position in rows], device=self.device, dtype=torch.long
            ),
            write_blocks=torch.tensor(
                [block for block, _ in addresses], device=self.device, dtype=torch.long
            ),
            write_offsets=torch.tensor(
                [offset for _, offset in addresses], device=self.device, dtype=torch.long
            ),
            block_tables=torch.tensor(tables, device=self.device, dtype=torch.int32).reshape(
                len(rows), width
            ),
            context_lengths=torch.tensor(
                [position + 1 for _, _, position in rows], device=self.device, dtype=torch.int32
            ),
            writable=for_write,
        )

    def _pad_prepared(
        self,
        batch: _PreparedKVBatch,
        *,
        row_indices: Sequence[int],
        row_count: int,
        table_width: int,
    ) -> _PreparedKVBatch:
        """Borrow live rows into physical slots; inactive addresses are never valid.

        The host map is the sole source of the device mask. The resulting
        tensors are private, read-only metadata, just like a compact batch's.
        This eager helper creates no allocation or initialized KV position.
        """
        self._require_live_batch(batch)
        for name, value in (("row_count", row_count), ("table_width", table_width)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        live_rows = tuple(row_indices)
        if len(live_rows) != len(batch.rows):
            raise ValueError("row_indices must map every live row exactly once")
        if any(
            not isinstance(row, Integral) or isinstance(row, bool) or not 0 <= row < row_count
            for row in live_rows
        ) or len(set(live_rows)) != len(live_rows):
            raise ValueError("row_indices must be unique integers within row_count")
        width = max((position // self.block_size + 1 for _, _, position in batch.rows), default=0)
        if table_width < width:
            raise ValueError("table_width cannot truncate a live row's causal history")
        live_rows = tuple(int(row) for row in live_rows)
        positions, lengths = [0] * row_count, [0] * row_count
        blocks, offsets = [-1] * row_count, [-1] * row_count
        tables = [[-1] * table_width for _ in range(row_count)]
        active = [False] * row_count
        for row, (allocation, depth, position) in zip(live_rows, batch.rows):
            active[row] = True
            positions[row], lengths[row] = position, position + 1
            blocks[row] = allocation.block_tables[depth][position // self.block_size]
            offsets[row] = position % self.block_size
            table = allocation.block_tables[depth][:table_width]
            tables[row][: len(table)] = table
        return _PreparedKVBatch(
            owner=self,
            rows=batch.rows,
            allocations=batch.allocations,
            row_count=int(row_count),
            live_rows=live_rows,
            active=torch.tensor(active, device=self.device, dtype=torch.bool),
            position_ids=torch.tensor(positions, device=self.device, dtype=torch.long),
            write_blocks=torch.tensor(blocks, device=self.device, dtype=torch.long),
            write_offsets=torch.tensor(offsets, device=self.device, dtype=torch.long),
            block_tables=torch.tensor(tables, device=self.device, dtype=torch.int32).reshape(
                row_count, table_width
            ),
            context_lengths=torch.tensor(lengths, device=self.device, dtype=torch.int32),
            writable=batch.writable,
        )

    def _allocate_metadata_storage(
        self, row_count: int = 8, table_width: int = 32
    ) -> _MetadataStorage:
        """Allocate fixed storage for a caller-budgeted decode bucket."""
        self._require_usable()
        if (
            not isinstance(row_count, Integral)
            or isinstance(row_count, bool)
            or row_count < 4
            or row_count & (row_count - 1)
        ):
            raise ValueError("persistent row_count must be a power of two of at least 4")
        if type(table_width) is not int or table_width <= 0:
            raise ValueError("persistent table_width must be a positive integer")
        row_count = int(row_count)
        specifications = _metadata_specifications(row_count, table_width)
        size = _metadata_payload_bytes(row_count, table_width)
        packed_staging = torch.empty(
            size, dtype=torch.uint8, device="cpu", pin_memory=self.device.type == "cuda"
        )
        packed = torch.empty(size, dtype=torch.uint8, device=self.device)

        def views(buffer):
            offset, result = 0, {}
            for name, (shape, dtype) in specifications.items():
                width = prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
                result[name] = buffer[offset : offset + width].view(dtype).reshape(shape)
                offset += width
            return result

        return _MetadataStorage(
            self,
            views(packed),
            views(packed_staging),
            packed,
            packed_staging,
            capacity=(row_count, table_width),
        )

    def _prepare_into(self, storage: _MetadataStorage, host: _HostKVBatch) -> _PreparedKVBatch:
        """Borrow fixed tensors for exactly one generation, after all host checks."""
        self._require_usable()
        if storage.owner is not self or host.owner is not self:
            raise ValueError("persistent metadata belongs to a different cache manager")
        if storage.failed or storage.in_use or storage.transaction is not None:
            raise RuntimeError("persistent metadata is failed or already in use")
        row_count, table_width = storage.tensors["block_tables"].shape
        if (row_count, table_width) != storage.capacity:
            raise ValueError("persistent metadata must retain its fixed capacity")
        if len(host.rows) > row_count // 2 or host.width > table_width:
            raise ValueError("persistent metadata capacity exceeded")
        for request_id, allocation in host.allocations:
            if self._allocations.get(request_id) is not allocation:
                raise RuntimeError(f"stale host KV batch for request {request_id!r}")
        live_rows = decode_live_rows(len(host.rows))
        # All metadata tails are initialized, including addresses never consumed
        # by inactive kernels. The lease prevents staging reuse before completion.
        staging = {name: tensor.numpy() for name, tensor in storage.staging.items()}
        for name, tensor in staging.items():
            tensor.fill(-1 if name in {"write_blocks", "write_offsets", "block_tables"} else 0)
        for row, (allocation, depth, position) in zip(live_rows, host.rows):
            staging["active"][row] = True
            staging["position_ids"][row] = position
            staging["context_lengths"][row] = position + 1
            staging["write_blocks"][row] = allocation.block_tables[depth][
                position // self.block_size
            ]
            staging["write_offsets"][row] = position % self.block_size
            table = allocation.block_tables[depth][:table_width]
            staging["block_tables"][row, : len(table)] = table
        storage.generation += 1
        storage.in_use = True
        try:
            storage.packed.copy_(storage.packed_staging, non_blocking=self.device.type == "cuda")
        except BaseException:
            storage.failed = True
            raise
        return _PreparedKVBatch(
            owner=self,
            rows=host.rows,
            allocations=host.allocations,
            row_count=row_count,
            live_rows=live_rows,
            writable=host.writable,
            storage=storage,
            generation=storage.generation,
            allocation_generation=self._allocation_generation,
            **storage.tensors,
        )

    def _release_prepared(self, batch: _PreparedKVBatch) -> None:
        """Caller establishes ordered-stream completion before releasing this lease."""
        self._require_live_batch(batch)
        if batch.storage is None:
            raise ValueError("only persistent descriptors have a releasable lease")
        if batch.storage.transaction is not None:
            raise RuntimeError("decode transaction must finish before releasing its lease")
        batch.storage.in_use = False

    def _validate_decode_host(self, host: _HostKVBatch) -> None:
        self._require_usable()
        if host.owner is not self or not host.writable or not host.rows:
            raise ValueError("decode traversal requires this cache's nonempty writable batch")
        for request_id, allocation in host.allocations:
            if self._allocations.get(request_id) is not allocation:
                raise RuntimeError(f"stale host KV batch for request {request_id!r}")
        owners = {id(allocation) for _, allocation in host.allocations}
        addresses = []
        for allocation, depth, position in host.rows:
            if id(allocation) not in owners:
                raise RuntimeError("decode row does not belong to its captured allocation")
            self._validate_depth(depth)
            self._validate_position(allocation, position)
            addresses.append(
                (
                    allocation.block_tables[depth][position // self.block_size],
                    position % self.block_size,
                )
            )
            # LAST-EXITED may already have initialized this position. Rewrites
            # are valid, but every layer must have all preceding positions.
            for layer in range(self.num_layers):
                self._require_prefix(allocation, layer, depth, position)
        if tuple(addresses) != host.addresses or len(set(addresses)) != len(addresses):
            raise ValueError("decode traversal contains changed or duplicate destinations")

    def _begin_decode_traversal(self, host: _HostKVBatch) -> _DecodeTraversal:
        """Validate every preceding layer before any metadata copy or KV write."""
        self._validate_decode_host(host)
        return _DecodeTraversal(self, host, self._allocation_generation)

    def _check_decode_ticket(self, ticket):
        self._require_usable()
        if ticket.allocation_generation != self._allocation_generation:
            raise RuntimeError("stale decode allocation generation")

    def _bind_decode_traversal(self, ticket: _DecodeTraversal, batch: _PreparedKVBatch) -> None:
        """Bind the actual newly prepared lease, not its previous generation."""
        if ticket.owner is not self or ticket.state != "begun":
            raise RuntimeError("decode transaction is foreign or already bound")
        self._check_decode_ticket(ticket)
        self._require_live_batch(batch)
        if (
            batch.storage is None
            or not batch.writable
            or batch.rows is not ticket.host.rows
            or batch.allocations is not ticket.host.allocations
            or batch.storage.transaction is not None
        ):
            raise ValueError("decode transaction must bind its own newly prepared batch")
        ticket.batch = batch
        ticket.state = "bound"
        batch.storage.transaction = ticket

    def _commit_decode_traversal(
        self, ticket: _DecodeTraversal, *, completion_confirmed: bool
    ) -> None:
        """Publish host prefixes once, after the caller confirms device completion.

        All rows are checked before the first mutation. A mutation-time failure
        cannot be rolled back honestly: quarantine the manager and storage, so
        a partially committed traversal can never be reused or retried.
        """
        if ticket.owner is not self or ticket.state != "bound":
            raise RuntimeError("decode transaction is foreign, unbound or already finished")
        if completion_confirmed is not True:
            raise RuntimeError("decode commit requires confirmed device completion")
        batch = ticket.batch
        try:
            self._require_usable()
            if (
                batch.storage.failed
                or not batch.storage.in_use
                or batch.storage.generation != batch.generation
            ):
                raise RuntimeError("stale or failed persistent KV generation")
            if batch.storage.transaction is not ticket:
                raise RuntimeError("decode transaction does not own this metadata lease")
            self._check_decode_ticket(ticket)
            for allocation, depth, position in ticket.host.rows:
                for layer in range(self.num_layers):
                    allocation.written[depth][layer].add(position)
        except BaseException:
            ticket.state = "failed"
            batch.storage.failed = True
            self._quarantine("decode prefix commit failed")
            raise
        ticket.state = "committed"
        batch.storage.transaction = None

    def _cancel_decode_traversal(self, ticket: _DecodeTraversal) -> None:
        """Cancel before submission only; leave every written prefix unchanged."""
        if ticket.owner is not self or ticket.state not in {"begun", "bound"}:
            raise RuntimeError("decode transaction cannot be cancelled")
        if ticket.batch is not None:
            self._require_live_batch(ticket.batch)
            if ticket.batch.storage.transaction is not ticket:
                raise RuntimeError("decode transaction does not own this metadata lease")
            ticket.batch.storage.transaction = None
        ticket.state = "cancelled"

    def _abort_decode_traversal(
        self, ticket: _DecodeTraversal, *, completion_confirmed: bool
    ) -> None:
        """Fail closed after submission; preserve any already committed history."""
        if ticket.owner is not self:
            raise ValueError("decode transaction belongs to a different cache manager")
        if not isinstance(completion_confirmed, bool):
            raise ValueError("completion_confirmed must be boolean")
        batch = ticket.batch
        ticket.state = "failed"
        if batch is not None:
            storage = batch.storage
            storage.failed = True
            if storage.generation != batch.generation or (
                storage.transaction is not None and storage.transaction is not ticket
            ):
                self._quarantine("stale decode transaction cannot settle another lease")
                raise RuntimeError("stale decode transaction cannot settle another lease")
            if completion_confirmed:
                storage.transaction = None
                storage.in_use = False
        if not completion_confirmed:
            self._quarantine("decode traversal completion failed")

    def _make_tensor_decode_view(self, storage: _MetadataStorage) -> _TensorKVView:
        """Resolve fixed tensor views and Triton imports before capture begins."""
        self._require_usable()
        if storage.owner is not self or storage.failed:
            raise ValueError("tensor decode view requires this cache's usable storage")
        if self.backend != "triton":
            raise ValueError("tensor-only decode requires the Triton backend")
        from vllm_lt.kernels.triton_kv_write import masked_kv_write

        return _TensorKVView(
            row_count=storage.tensors["block_tables"].shape[0],
            **storage.tensors,
            key_layers=tuple(self.key_cache[:, layer] for layer in range(self.num_layers)),
            value_layers=tuple(self.value_cache[:, layer] for layer in range(self.num_layers)),
            write_kernel=masked_kv_write,
            attention_kernel=triton_paged_attention,
        )

    def _require_live_batch(self, batch: _PreparedKVBatch) -> None:
        self._require_usable()
        if batch.owner is not self:
            raise ValueError("prepared KV batch belongs to a different cache manager")
        if batch.storage is not None:
            storage = batch.storage
            if (
                storage.owner is not self
                or storage.failed
                or not storage.in_use
                or storage.generation != batch.generation
            ):
                raise RuntimeError("stale or failed persistent KV generation")
            if batch.allocation_generation != self._allocation_generation:
                raise RuntimeError("stale prepared KV allocation generation")
            if any(getattr(batch, name) is not tensor for name, tensor in storage.tensors.items()):
                raise ValueError("persistent descriptor changed its borrowed tensor storage")
            return
        for request_id, allocation in batch.allocations:
            if self._allocations.get(request_id) is not allocation:
                raise RuntimeError(f"stale prepared KV batch for request {request_id!r}")

    def _validate_tensor(self, tensor, batch_size, name, *, query=False):
        if tensor.ndim != 3 or tensor.shape[0] != batch_size or tensor.shape[-1] != self.head_dim:
            raise ValueError(f"{name} must have shape [batch, heads, {self.head_dim}]")
        if query:
            if tensor.shape[1] == 0 or tensor.shape[1] % self.num_kv_heads:
                raise ValueError("query heads must be a positive multiple of KV heads")
        elif tensor.shape[1] != self.num_kv_heads:
            raise ValueError(f"{name} must have {self.num_kv_heads} KV heads")
        if tensor.device != self.device or tensor.dtype != self.dtype:
            raise ValueError(f"{name} must use device {self.device} and dtype {self.dtype}")

    @torch.no_grad()
    def write(
        self,
        layer: int,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write a packed batch before calling attend; future tokens stay masked.

        This compatibility adapter builds the full descriptor, including the
        attention tensors. The model reuses one prepared descriptor per traversal;
        standalone writes should not be substituted into its per-layer hot path.
        Address/ownership validation intentionally precedes tensor validation.
        """
        self._validate_layer(layer)
        batch = self._prepare_batch(request_ids, depths, positions)
        self._write_prepared(layer, batch, k, v)

    @torch.no_grad()
    def _write_prepared(
        self, layer: int, batch: _PreparedKVBatch, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        self._validate_layer(layer)
        self._require_live_batch(batch)
        if not batch.writable:
            raise ValueError("a read-only prepared KV batch cannot be written")
        self._validate_tensor(k, batch.row_count, "k")
        self._validate_tensor(v, batch.row_count, "v")
        if not batch.rows:
            return
        if len(batch.rows) == batch.row_count:
            self.key_cache[batch.write_blocks, layer, batch.write_offsets] = k
            self.value_cache[batch.write_blocks, layer, batch.write_offsets] = v
        elif self.backend == "triton":
            from vllm_lt.kernels.triton_kv_write import masked_kv_write

            masked_kv_write(
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                batch.write_blocks,
                batch.write_offsets,
                k,
                v,
                batch.active,
            )
        else:
            # Diagnostic eager Torch padding only: Python indexing may transfer
            # indices per layer. The production Triton path stays above.
            # Select live source rows before indexing addresses: -1 must never
            # alias a real page or token through advanced indexing.
            live = list(batch.live_rows)
            blocks, offsets = batch.write_blocks[live], batch.write_offsets[live]
            self.key_cache[blocks, layer, offsets] = k[live]
            self.value_cache[blocks, layer, offsets] = v[live]
        for allocation, depth, position in batch.rows:
            allocation.written[depth][layer].add(position)

    def _require_prefix(self, allocation, layer, depth, length):
        written = allocation.written[depth][layer]
        if length > written.prefix:
            raise RuntimeError(
                f"uninitialized KV history at layer {layer}, depth {depth}, "
                f"context length {length}; "
                "write preceding tokens and finalize early exits before attention"
            )

    @torch.no_grad()
    def attend(
        self,
        layer: int,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Apply causal GQA at each row's own position and recurrence depth."""
        self._validate_layer(layer)
        batch = self._prepare_batch(request_ids, depths, positions, for_write=False)
        return self._attend_prepared(layer, batch, q)

    @torch.no_grad()
    def _attend_prepared(
        self, layer: int, batch: _PreparedKVBatch, q: torch.Tensor
    ) -> torch.Tensor:
        self._validate_layer(layer)
        self._require_live_batch(batch)
        self._validate_tensor(q, batch.row_count, "q", query=True)
        if not batch.rows:
            return torch.zeros_like(q)
        for allocation, depth, position in batch.rows:
            self._require_prefix(allocation, layer, depth, position + 1)
        attention = triton_paged_attention if self.backend == "triton" else torch_paged_attention
        args = (
            q,
            self.key_cache[:, layer],
            self.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
        )
        if len(batch.rows) == batch.row_count:
            return attention(*args)
        return attention(*args, active=batch.active)

    @torch.no_grad()
    def finalize_token(self, request_id: str, position: int, exit_depth: int) -> None:
        """Propagate the final executed depth's K/V into every unexecuted depth."""
        allocation = self._get_allocation(request_id)
        self._validate_depth(exit_depth)
        self._validate_position(allocation, position)
        for layer in range(self.num_layers):
            if position not in allocation.written[exit_depth][layer]:
                raise RuntimeError(
                    "cannot finalize a token before every layer has written its exit depth"
                )
        logical_page, offset = divmod(position, self.block_size)
        source = allocation.block_tables[exit_depth][logical_page]
        destinations = [
            allocation.block_tables[depth][logical_page]
            for depth in range(exit_depth + 1, self.max_loops)
        ]
        if destinations:
            destination_indices = torch.tensor(destinations, device=self.device, dtype=torch.long)
            self.key_cache[destination_indices, :, offset] = self.key_cache[
                source, :, offset
            ].unsqueeze(0)
            self.value_cache[destination_indices, :, offset] = self.value_cache[
                source, :, offset
            ].unsqueeze(0)
        for depth in range(exit_depth + 1, self.max_loops):
            for layer in range(self.num_layers):
                allocation.written[depth][layer].add(position)

    @torch.no_grad()
    def read(self, layer: int, request_id: str, depth: int, length: int | None = None):
        """Materialize a fully initialized prefix as [token, KV head, dimension]."""
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        if length is None:
            length = allocation.written[depth][layer].prefix
        if (
            not isinstance(length, Integral)
            or isinstance(length, bool)
            or not 0 <= length <= allocation.max_tokens
        ):
            raise ValueError("read length exceeds the request's reserved token budget")
        self._require_prefix(allocation, layer, depth, length)
        positions = torch.arange(length, device=self.device)
        table = torch.tensor(allocation.block_tables[depth], device=self.device, dtype=torch.long)
        blocks = table[positions // self.block_size]
        offsets = positions % self.block_size
        return self.key_cache[blocks, layer, offsets], self.value_cache[blocks, layer, offsets]
