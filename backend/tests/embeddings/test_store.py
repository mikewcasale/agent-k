"""Tests for vector store utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.embeddings.store import InMemoryVectorStore, VectorRecord

__all__ = ()


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore behavior."""

    def test_query_dimension_mismatch(self) -> None:
        """Query should reject vectors with mismatched dimensions."""
        store = InMemoryVectorStore()
        store.upsert([VectorRecord(record_id="a", vector=[1.0, 0.0])])

        with pytest.raises(ValueError, match="same dimension"):
            store.query([1.0, 0.0, 0.0])

    def test_upsert_deduplicates_records(self) -> None:
        """Upsert should overwrite records with the same id."""
        store = InMemoryVectorStore()
        store.upsert([VectorRecord(record_id="a", vector=[1.0, 0.0])])
        store.upsert([VectorRecord(record_id="a", vector=[0.0, 1.0], metadata={"v": 2})])

        results = store.query([0.0, 1.0], top_k=5)

        assert len(results) == 1
        assert results[0].record_id == "a"
        assert results[0].vector == [0.0, 1.0]
        assert results[0].metadata["v"] == 2

    def test_query_ranking_order(self) -> None:
        """Query should return results ordered by similarity."""
        store = InMemoryVectorStore()
        store.upsert(
            [
                VectorRecord(record_id="a", vector=[1.0, 0.0]),
                VectorRecord(record_id="b", vector=[0.6, 0.2]),
                VectorRecord(record_id="c", vector=[0.0, 1.0]),
            ]
        )

        results = store.query([1.0, 0.0], top_k=3)

        assert [record.record_id for record in results] == ["a", "b", "c"]

    def test_query_zero_norm_vector(self) -> None:
        """Zero-norm vectors should not crash similarity scoring."""
        store = InMemoryVectorStore()
        store.upsert(
            [
                VectorRecord(record_id="a", vector=[1.0, 0.0]),
                VectorRecord(record_id="zero", vector=[0.0, 0.0]),
            ]
        )

        results = store.query([1.0, 0.0], top_k=2)

        assert [record.record_id for record in results] == ["a", "zero"]
