"""Embedding utilities for Agent-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Local imports (core first, then alphabetical)
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
