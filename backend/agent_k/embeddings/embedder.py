"""Embedding utilities.

@notice: |
    Embedding utilities.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.embeddings.embedder
    provides:
        - agent_k.embeddings.embedder:Embedder
        - agent_k.embeddings.embedder:get_embedder
        - agent_k.embeddings.embedder:embed_documents
        - agent_k.embeddings.embedder:embed_query
    pattern: embedder

@agent-guidance:
    do:
        - "Use agent_k.embeddings.embedder as the canonical home for this capability."
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

from functools import lru_cache
from importlib import import_module
from typing import Annotated, Final, Protocol, cast, runtime_checkable

from agent_k.core.sage import Doc

__all__ = ("DEFAULT_MODEL", "embed_documents", "embed_query", "get_embedder")

DEFAULT_MODEL: Final[str] = "openai:text-embedding-3-small"


@runtime_checkable
class EmbedQueryResult(Protocol):
    """Protocol for embed_query results.

    @pattern:
        name: protocol-interface
        rationale: "Defines the embedding result contract."
        violations: "Divergent payloads break embeddings consumers."
    """

    embedding: list[float]


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers.

    @pattern:
        name: protocol-interface
        rationale: "Standard contract for embedding providers."
        violations: "Embedder implementations without this contract diverge."
    """

    def __init__(self, model: str) -> None: ...

    async def embed_documents(self, documents: list[str]) -> dict[str, list[float]]:
        """Embed multiple documents."""
        ...

    async def embed_query(self, query: str) -> EmbedQueryResult:
        """Embed a single query string."""
        ...


@lru_cache(maxsize=4)
def get_embedder(model: Annotated[str, Doc("Embedding model identifier.")] = DEFAULT_MODEL) -> Embedder:
    """Get cached embedder instance.

    @notice: |
        Returns a cached Embedder instance for the model.
    """
    embedder_cls = _resolve_embedder_class()
    return embedder_cls(model)


async def embed_documents(
    documents: Annotated[list[str], Doc("Documents to embed.")],
    model: Annotated[str, Doc("Embedding model identifier.")] = DEFAULT_MODEL,
) -> list[list[float]]:
    """Embed multiple documents for indexing.

    @notice: |
        Returns embeddings aligned to the input document order.
    """
    embedder = get_embedder(model)
    result = await embedder.embed_documents(documents)
    return [result[doc] for doc in documents]


async def embed_query(
    query: Annotated[str, Doc("Search query to embed.")],
    model: Annotated[str, Doc("Embedding model identifier.")] = DEFAULT_MODEL,
) -> list[float]:
    """Embed query for similarity search.

    @notice: |
        Returns a single embedding vector for the query.
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
