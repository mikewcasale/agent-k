"""RAG retrieval patterns.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from dataclasses import dataclass
from typing import Any

# Third-party (alphabetical)
import numpy as np

# Local imports (core first, then alphabetical)
from .embedder import embed_query

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = ("RAGRetriever", "RetrievalResult")


# =============================================================================
# Section 9: Dataclasses
# =============================================================================
@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Single retrieval result."""

    content: str
    score: float
    metadata: dict[str, Any]


# =============================================================================
# Section 11: Classes
# =============================================================================
class RAGRetriever:
    """Simple RAG retriever with cosine similarity."""

    def __init__(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")
        if metadata is not None and len(metadata) != len(documents):
            raise ValueError("metadata length must match documents")
        self._documents = documents
        self._embeddings = np.array(embeddings, dtype=float)
        if not documents:
            self._embeddings = np.zeros((0, 0), dtype=float)
            self._norms = np.array([])
            self._metadata = metadata or []
            return
        if self._embeddings.ndim != 2 or self._embeddings.shape[0] != len(documents):
            raise ValueError("embeddings must be a 2D array aligned with documents")
        norms = np.linalg.norm(self._embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError("embeddings contain zero-norm vectors")
        self._norms = norms
        self._metadata = metadata or [{} for _ in documents]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve top-k relevant documents.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            Ranked retrieval results.
        """
        if top_k <= 0 or not self._documents:
            return []

        query_embedding = await embed_query(query)
        query_vec = np.array(query_embedding, dtype=float)
        if query_vec.ndim != 1:
            raise ValueError("query embedding must be a 1D vector")
        if query_vec.shape[0] != self._embeddings.shape[1]:
            raise ValueError("query embedding dimension mismatch")

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        similarities = np.dot(self._embeddings, query_vec) / (self._norms * query_norm)

        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            RetrievalResult(
                content=self._documents[i],
                score=float(similarities[i]),
                metadata=self._metadata[i],
            )
            for i in top_indices
        ]
