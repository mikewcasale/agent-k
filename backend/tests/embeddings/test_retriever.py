"""Tests for the RAG retriever.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

import agent_k.embeddings.retriever as retriever_module
from agent_k.embeddings.retriever import RAGRetriever

__all__ = ()

pytestmark = pytest.mark.anyio


class TestRAGRetriever:
    """Tests for RAGRetriever validation."""

    def test_init_length_mismatch(self) -> None:
        """Constructor should reject mismatched lengths."""
        with pytest.raises(ValueError, match='length mismatch'):
            RAGRetriever(['doc'], embeddings=[])

    def test_init_zero_norm_embeddings(self) -> None:
        """Constructor should reject zero-norm embeddings."""
        with pytest.raises(ValueError, match='zero-norm'):
            RAGRetriever(['doc'], embeddings=[[0.0, 0.0]])

    async def test_query_dimension_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Query embeddings must match stored dimensions."""

        async def fake_embed_query(_query: str) -> list[float]:
            return [1.0, 0.0]

        monkeypatch.setattr(retriever_module, 'embed_query', fake_embed_query)

        retriever = RAGRetriever(['doc'], embeddings=[[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match='dimension mismatch'):
            await retriever.retrieve('query')

    async def test_retrieve_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retriever should return ranked results."""

        async def fake_embed_query(_query: str) -> list[float]:
            return [1.0, 0.0]

        monkeypatch.setattr(retriever_module, 'embed_query', fake_embed_query)

        retriever = RAGRetriever(
            ['alpha', 'beta', 'gamma'],
            embeddings=[[1.0, 0.0], [0.6, 0.2], [0.0, 1.0]],
            metadata=[{'id': 1}, {'id': 2}, {'id': 3}],
        )
        results = await retriever.retrieve('query', top_k=2)

        assert [result.content for result in results] == ['alpha', 'beta']
        assert results[0].metadata['id'] == 1

    async def test_retrieve_top_k_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retriever should respect top_k limits."""

        async def fake_embed_query(_query: str) -> list[float]:
            return [1.0, 0.0]

        monkeypatch.setattr(retriever_module, 'embed_query', fake_embed_query)

        retriever = RAGRetriever(['alpha', 'beta', 'gamma'], embeddings=[[1.0, 0.0], [0.6, 0.2], [0.0, 1.0]])
        results = await retriever.retrieve('query', top_k=1)

        assert len(results) == 1
        assert results[0].content == 'alpha'

    async def test_retrieve_empty_documents(self) -> None:
        """Retriever should handle empty document sets."""
        retriever = RAGRetriever([], embeddings=[])

        results = await retriever.retrieve('query', top_k=3)

        assert results == []
