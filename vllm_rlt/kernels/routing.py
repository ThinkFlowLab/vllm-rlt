"""GPU-resident request routing; host descriptors may be mapped pinned memory."""

import triton
import triton.language as tl


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
