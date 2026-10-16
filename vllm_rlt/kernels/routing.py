"""GPU-resident request routing; host descriptors may be mapped pinned memory."""

import triton
import triton.language as tl


@triton.jit
def slots_kernel(D, Slots, N, SIZE, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    slot = tl.load(D + row * 3, row < N, 0)
    tl.store(Slots + row, slot, row < SIZE)


@triton.jit
def metadata_kernel(
    D,
    Tables,
    Slots,
    Positions,
    Lengths,
    Blocks,
    Offsets,
    OutTables,
    N,
    SIZE,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    PLANES: tl.constexpr,
    OUT_WIDTH,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    valid = row < N
    slot = tl.load(D + row * 3, valid, 0)
    depth = tl.load(D + row * 3 + 1, valid, 0)
    pos = tl.load(D + row * 3 + 2, valid, 0)
    plane = depth if PLANES > 1 else 0
    base = (slot * PLANES + plane) * WIDTH
    block = tl.load(Tables + base + pos // PAGE, valid, 0)
    tl.store(Slots + row, slot)
    tl.store(Positions + row, pos)
    tl.store(Lengths + row, tl.where(valid, pos + 1, 0))
    tl.store(Blocks + row, block, valid)
    tl.store(Offsets + row, pos % PAGE, valid)
    for start in range(tl.cdiv(OUT_WIDTH, TILE)):
        col = start * TILE + tl.arange(0, TILE)
        value = tl.load(Tables + base + col, valid & (col < OUT_WIDTH), 0)
        tl.store(OutTables + row * WIDTH + col, value, col < OUT_WIDTH)


@triton.jit
def gather_kernel(
    Pool,
    Slots,
    Out,
    N,
    H: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    slot = tl.load(Slots + row)
    value = tl.load(Pool + slot * IN_STRIDE + col, (row < N) & (col < H), 0)
    tl.store(Out + row * OUT_STRIDE + col, value, col < H)


@triton.jit
def scatter_kernel(
    In,
    Slots,
    Pool,
    H: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    slot = tl.load(Slots + row)
    value = tl.load(In + row * IN_STRIDE + col, col < H, 0)
    tl.store(Pool + slot * OUT_STRIDE + col, value, col < H)


@triton.jit
def gather_segments_kernel(
    Pool,
    Segments,
    Out,
    POOL_ROWS: tl.constexpr,
    H: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
):
    segment = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(2) * TILE + tl.arange(0, TILE)
    slot = tl.load(Segments + segment * 5)
    offset = tl.load(Segments + segment * 5 + 1)
    count = tl.load(Segments + segment * 5 + 2)
    out_start = tl.load(Segments + segment * 5 + 3)
    mask = (rows[:, None] < count) & (cols[None, :] < H)
    source = tl.load(
        Pool + (slot * POOL_ROWS + offset + rows[:, None]) * IN_STRIDE + cols[None, :],
        mask,
        0,
    )
    tl.store(Out + (out_start + rows[:, None]) * OUT_STRIDE + cols[None, :], source, mask)


@triton.jit
def scatter_segments_kernel(
    In,
    Segments,
    Pool,
    POOL_ROWS: tl.constexpr,
    H: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
):
    segment = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(2) * TILE + tl.arange(0, TILE)
    slot = tl.load(Segments + segment * 5)
    offset = tl.load(Segments + segment * 5 + 1)
    count = tl.load(Segments + segment * 5 + 2)
    in_start = tl.load(Segments + segment * 5 + 3)
    keep_intermediate = tl.load(Segments + segment * 5 + 4) != 0
    mask = (rows[:, None] < count) & (cols[None, :] < H) & keep_intermediate
    value = tl.load(
        In + (in_start + rows[:, None]) * IN_STRIDE + cols[None, :],
        mask,
        0,
    )
    tl.store(
        Pool + (slot * POOL_ROWS + offset + rows[:, None]) * OUT_STRIDE + cols[None, :],
        value,
        mask,
    )


@triton.jit
def finalize_kernel(
    D,
    Tables,
    K,
    V,
    WIDTH: tl.constexpr,
    PLANES: tl.constexpr,
    PAGE: tl.constexpr,
    LAYERS: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    LAYER_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row, target = tl.program_id(0), tl.program_id(1)
    slot = tl.load(D + row * 3)
    depth = tl.load(D + row * 3 + 1)
    pos = tl.load(D + row * 3 + 2)
    if target > depth:
        source = tl.load(Tables + (slot * PLANES + depth) * WIDTH + pos // PAGE)
        dest = tl.load(Tables + (slot * PLANES + target) * WIDTH + pos // PAGE)
        x = tl.program_id(2) * TILE + tl.arange(0, TILE)
        offset = (x // CHANNELS) * LAYER_STRIDE + (pos % PAGE) * TOKEN_STRIDE + x % CHANNELS
        src = source.to(tl.int64) * BLOCK_STRIDE + offset
        dst = dest.to(tl.int64) * BLOCK_STRIDE + offset
        k = tl.load(K + src, x < LAYERS * CHANNELS, 0)
        v = tl.load(V + src, x < LAYERS * CHANNELS, 0)
        tl.store(K + dst, k, x < LAYERS * CHANNELS)
        tl.store(V + dst, v, x < LAYERS * CHANNELS)


@triton.jit
def prepare_prefill_tokens_kernel(Sources, Out, TILE: tl.constexpr):
    segment = tl.program_id(0)
    index = tl.program_id(1) * TILE + tl.arange(0, TILE)
    pointer = tl.load(Sources + segment * 4).to(tl.pointer_type(tl.int64))
    start = tl.load(Sources + segment * 4 + 1)
    count = tl.load(Sources + segment * 4 + 2)
    row = tl.load(Sources + segment * 4 + 3)
    token = tl.load(pointer + start + index, index < count, 0)
    tl.store(Out + row + index, token, index < count)


@triton.jit
def mixed_hidden_kernel(
    Sources,
    Hidden,
    STRIDE: tl.constexpr,
    H: tl.constexpr,
    SCATTER: tl.constexpr,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
):
    segment = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(2) * TILE + tl.arange(0, TILE)
    pointer_type = tl.pointer_type(Hidden.dtype.element_ty)
    count = tl.load(Sources + segment * 7 + 3)
    row = tl.load(Sources + segment * 7 + 4)
    mask = (rows[:, None] < count) & (cols[None, :] < H)
    batch_ptr = Hidden + (row + rows[:, None]) * STRIDE + cols[None, :]
    if SCATTER:
        chunk = tl.load(Sources + segment * 7 + 5)
        last = tl.load(Sources + segment * 7 + 6)
        value = tl.load(batch_ptr, mask, 0)
        if chunk != 0:
            tl.store(chunk.to(pointer_type) + rows[:, None] * H + cols[None, :], value, mask)
        if last != 0 and tl.program_id(1) == 0:
            last_value = tl.load(Hidden + (row + count - 1) * STRIDE + cols, cols < H, 0)
            tl.store(last.to(pointer_type) + cols, last_value, cols < H)
    else:
        source = tl.load(Sources + segment * 7).to(pointer_type)
        stride = tl.load(Sources + segment * 7 + 1)
        col_stride = tl.load(Sources + segment * 7 + 2)
        value = tl.load(source + rows[:, None] * stride + cols[None, :] * col_stride, mask, 0)
        tl.store(batch_ptr, value, mask)
