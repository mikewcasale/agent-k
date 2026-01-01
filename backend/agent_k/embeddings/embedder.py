"""Embedding utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from functools import lru_cache
from importlib import import_module
from typing import Final, Protocol, cast, runtime_checkable

__all__ = ("DEFAULT_MODEL", "embed_documents", "embed_query", "get_embedder")

DEFAULT_MODEL: Final[str] = "openai:text-embedding-3-small"


@runtime_checkable
class EmbedQueryResult(Protocol):
    """Protocol for embed_query results."""

    embedding: list[float]


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""

    def __init__(self, model: str) -> None: ...

    async def embed_documents(self, documents: list[str]) -> dict[str, list[float]]:
        """Embed multiple documents."""
        ...

    async def embed_query(self, query: str) -> EmbedQueryResult:
        """Embed a single query string."""
        ...


@lru_cache(maxsize=4)
def get_embedder(model: str = DEFAULT_MODEL) -> Embedder:
    """Get cached embedder instance.

    Args:
        model: Embedding model identifier.

    Returns:
        Configured Embedder instance.
    """
    embedder_cls = _resolve_embedder_class()
    return embedder_cls(model)


async def embed_documents(
    documents: list[str],
    model: str = DEFAULT_MODEL,
) -> list[list[float]]:
    """Embed multiple documents for indexing.

    Args:
        documents: Documents to embed.
        model: Embedding model.

    Returns:
        List of embedding vectors.
    """
    embedder = get_embedder(model)
    result = await embedder.embed_documents(documents)
    return [result[doc] for doc in documents]


async def embed_query(
    query: str,
    model: str = DEFAULT_MODEL,
) -> list[float]:
    """Embed query for similarity search.

    Args:
        query: Search query.
        model: Embedding model.

    Returns:
        Query embedding vector.
    """
    embedder = get_embedder(model)
    result = await embedder.embed_query(query)
    return result.embedding


def _resolve_embedder_class() -> type[Embedder]:
    """Resolve the Embedder implementation from pydantic_ai."""
    module = import_module("pydantic_ai")
    embedder_cls = getattr(module, "Embedder", None)
    if embedder_cls is None:  # pragma: no cover - optional dependency
        raise RuntimeError("pydantic_ai Embedder is not available")
    return cast("type[Embedder]", embedder_cls)
