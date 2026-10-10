"""Attention backend selection and kernel metadata semantics (M8)."""

from vllm_rlt.attention.backend import (
    AttentionBackend,
    BackendCapabilities,
    backend_capabilities,
    create_backend,
    validate_backend_name,
)
from vllm_rlt.attention.metadata import (
    AttentionMetadata,
    AttentionMetadataValues,
    AttentionRows,
    build_attention_metadata,
    plan_attention_metadata,
)

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "AttentionMetadataValues",
    "AttentionRows",
    "BackendCapabilities",
    "backend_capabilities",
    "build_attention_metadata",
    "create_backend",
    "plan_attention_metadata",
    "validate_backend_name",
]
