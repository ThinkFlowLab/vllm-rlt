"""Depth-indexed physical KV pages with LAST-EXITED token propagation.

Each request owns a separate block table for every recurrence depth. When a
token exits, its final K/V states are copied into all deeper planes. Subsequent
tokens may therefore loop further without encountering holes or reading the
wrong depth's history. These are physical copies, not cross-depth aliases.

The initial admission policy reserves a request's complete token budget at all
depths. It is conservative, but admitted requests cannot deadlock on KV growth.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral

import torch

from vllm_lt.kernels.paged_attention import torch_paged_attention, triton_paged_attention


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
        return True

    def free(self, request_id: str) -> None:
        """Release only this request's pages; repeated cleanup is harmless."""
        allocation = self._allocations.pop(request_id, None)
        if allocation is not None:
            for table in allocation.block_tables:
                self._free_blocks.extend(reversed(table))

    def get_block_table(self, request_id: str, depth: int) -> tuple[int, ...]:
        self._validate_depth(depth)
        return self._get_allocation(request_id).block_tables[depth]

    def _get_allocation(self, request_id: str) -> _Allocation:
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

    def _prepare_batch(
        self,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        *,
        for_write: bool = True,
    ) -> _PreparedKVBatch:
        """Build layer-independent addresses once; do not initialize any KV slot."""
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
        tables = []
        for allocation, depth, _ in rows:
            table = allocation.block_tables[depth][:width]
            tables.append(list(table) + [-1] * (width - len(table)))
        allocations = dict(zip(request_ids, (allocation for allocation, _, _ in rows)))
        return _PreparedKVBatch(
            owner=self,
            rows=rows,
            allocations=tuple(allocations.items()),
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

    def _require_live_batch(self, batch: _PreparedKVBatch) -> None:
        if batch.owner is not self:
            raise ValueError("prepared KV batch belongs to a different cache manager")
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
