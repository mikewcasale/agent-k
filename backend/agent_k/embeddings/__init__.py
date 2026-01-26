"""Embedding utilities for Agent-K.

@notice: |
    Embedding utilities for Agent-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.embeddings
    provides:
        - agent_k.embeddings
    pattern: embeddings-package

@agent-guidance:
    do:
        - "Use agent_k.embeddings as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from .embedder import DEFAULT_MODEL, embed_documents, embed_query, get_embedder
from .retriever import RAGRetriever, RetrievalResult
from .store import InMemoryVectorStore, VectorRecord, VectorStore

__all__ = (
    "DEFAULT_MODEL",
    "embed_documents",
    "embed_query",
    "get_embedder",
    "RAGRetriever",
    "RetrievalResult",
    "InMemoryVectorStore",
    "VectorRecord",
    "VectorStore",
)
