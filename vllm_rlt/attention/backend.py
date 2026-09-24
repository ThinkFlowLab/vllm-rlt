"""Select an explicit paged Attention implementation without fallback."""

from dataclasses import dataclass

import torch

from vllm_rlt.kernels.flash_attention import FLASH_BACKENDS, FlashPagedAttention
from vllm_rlt.kernels.paged_attention import torch_paged_attention, triton_paged_attention


@dataclass(frozen=True)
class BackendCapabilities:
    """Features relevant to execution routing, independent of kernel class."""

    packed_prefill: bool


def backend_capabilities(implementation) -> BackendCapabilities:
    return BackendCapabilities(packed_prefill=getattr(implementation, "generation", None) == 4)


def validate_backend_name(backend: str) -> None:
    if backend not in {"torch", "triton", *FLASH_BACKENDS}:
        raise ValueError("unknown attention backend")


def create_backend(
    backend: str, device: torch.device, dtype: torch.dtype, head_dim: int, block_size: int
):
    """Return the requested kernel; Flash validates its own hardware and package.

    The returned callable accepts query, per-layer key/value cache, physical
    page tables and causal lengths. Only FA4 also exposes packed ``prefill``.
    """
    validate_backend_name(backend)
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("the Triton attention backend requires a CUDA or ROCm device")
        if head_dim > 256:
            raise ValueError("the Triton attention backend supports head_dim <= 256")
        return triton_paged_attention
    if backend in FLASH_BACKENDS:
        return FlashPagedAttention(device, dtype, head_dim, block_size, backend)
    return torch_paged_attention
