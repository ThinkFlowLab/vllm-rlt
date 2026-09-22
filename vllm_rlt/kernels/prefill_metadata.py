"""Expand compact, staged Prefill metadata without per-depth host transfers."""

import triton
import triton.language as tl


@triton.jit
def stage_prefill(Source, Target, Target32, N, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.load(Source + i, i < N, 0)
    tl.store(Target + i, value, i < N)
    tl.store(Target32 + i, value, i < N)


@triton.jit
def expand_prefill(
    Positions,
    Sequences,
    Tables,
    Blocks,
    Offsets,
    N,
    WIDTH,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(Positions + i, i < N, 0)
    seq = tl.load(Sequences + i, i < N, 0)
    block = tl.load(Tables + seq * WIDTH + pos // PAGE, i < N, 0)
    tl.store(Blocks + i, block, i < N)
    tl.store(Offsets + i, pos % PAGE, i < N)
