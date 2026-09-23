"""Attention backend selection and kernel metadata semantics (M8)."""

from vllm_rlt.attention.backend import (
    BackendCapabilities,
    backend_capabilities,
    create_backend,
    validate_backend_name,
)
from vllm_rlt.attention.metadata import AttentionMetadata, AttentionRows, build_attention_metadata

__all__ = [
    "AttentionMetadata",
    "AttentionRows",
    "BackendCapabilities",
    "backend_capabilities",
    "build_attention_metadata",
    "create_backend",
    "validate_backend_name",
]
