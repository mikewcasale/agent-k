"""Vector store abstractions.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Protocol

__all__ = ("InMemoryVectorStore", "VectorRecord", "VectorStore")


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Single vector record."""

    record_id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Protocol for vector stores."""

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update vector records."""
        ...

    def query(self, vector: list[float], top_k: int = 5) -> list[VectorRecord]:
        """Return the top matching records for a query vector."""
        ...


class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update vector records."""
        existing = {record.record_id: record for record in self._records}
        for record in records:
            existing[record.record_id] = record
        self._records = list(existing.values())

    def query(self, vector: list[float], top_k: int = 5) -> list[VectorRecord]:
        """Return the top matching records for a query vector."""
        scored = [(record, _cosine_similarity(vector, record.vector)) for record in self._records]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [record for record, _score in scored[:top_k]]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    dot = sum(left_val * right_val for left_val, right_val in zip(left, right, strict=False))
    left_norm = sqrt(sum(left_val * left_val for left_val in left))
    right_norm = sqrt(sum(right_val * right_val for right_val in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
