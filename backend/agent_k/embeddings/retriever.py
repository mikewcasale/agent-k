"""RAG retrieval patterns.

@notice: |
    RAG retrieval patterns.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.embeddings.retriever
    provides:
        - agent_k.embeddings.retriever:RAGRetriever
        - agent_k.embeddings.retriever:RetrievalResult
    pattern: retriever

@agent-guidance:
    do:
        - "Use agent_k.embeddings.retriever as the canonical home for this capability."
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

from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np

from agent_k.core.sage import Doc, Range

from .embedder import embed_query

__all__ = ("RAGRetriever", "RetrievalResult")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Single retrieval result.

    @notice: |
        Single retrieval result.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: output-model
            rationale: "Stable retrieval payload for RAG consumption."
            violations: "Ad-hoc dicts make ranking hard to interpret."
    """

    content: str
    score: float
    metadata: dict[str, Any]


class RAGRetriever:
    """Simple RAG retriever with cosine similarity.

    @notice: |
        Simple RAG retriever with cosine similarity.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: retriever
            rationale: "Encapsulates retrieval logic for RAG flows."
            violations: "Scattered retrieval logic leads to inconsistent ranking."
    """

    def __init__(
        self,
        documents: Annotated[list[str], Doc("Documents to index.")],
        embeddings: Annotated[list[list[float]], Doc("Embedding vectors aligned to documents.")],
        metadata: Annotated[list[dict[str, Any]] | None, Doc("Optional metadata aligned to documents.")] = None,
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
        query: Annotated[str, Doc("Search query.")],
        top_k: Annotated[int, Doc("Number of results to return."), Range(1, 1000)] = 5,
    ) -> list[RetrievalResult]:
        """Retrieve top-k relevant documents.

        @notice: |
            Computes cosine similarity and returns the top matches.
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
            RetrievalResult(content=self._documents[i], score=float(similarities[i]), metadata=self._metadata[i])
            for i in top_indices
        ]
