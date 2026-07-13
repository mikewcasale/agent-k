"""Search tool helpers for AGENT-K agents.

@notice: |
    Search tool helpers for AGENT-K agents.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.toolsets.search
    provides:
        - agent_k.toolsets.search:build_kaggle_search_query
        - agent_k.toolsets.search:build_scholarly_query
        - agent_k.toolsets.search:create_web_search_tool
        - agent_k.toolsets.search:prepare_web_search
        - agent_k.toolsets.search:create_web_fetch_tool
        - agent_k.toolsets.search:prepare_web_fetch
    pattern: toolset

@similar:
    - id: agent_k.toolsets.browser
        when: "Use for browsing fetched pages rather than constructing queries."

@agent-guidance:
    do:
        - "Use agent_k.toolsets.search as the canonical home for this capability."
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

from typing import Annotated, Any, Literal, cast

from pydantic_ai import RunContext
from pydantic_ai.builtin_tools import WebFetchTool, WebSearchTool, WebSearchUserLocation

from agent_k.core.sage import Doc

try:  # pragma: no cover - optional dependency
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:  # pragma: no cover - optional dependency
    OpenAIChatModel = None  # type: ignore[misc,assignment]

__all__ = (
    "build_kaggle_search_query",
    "build_scholarly_query",
    "create_web_fetch_tool",
    "create_web_search_tool",
    "prepare_web_fetch",
    "prepare_web_search",
)


def build_kaggle_search_query(query: Annotated[str, Doc("Search query text.")]) -> str:
    """Build a Kaggle-scoped web search query.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Prefixes the query with site restrictions for Kaggle.
    """
    return f"site:kaggle.com {query}".strip()


def build_scholarly_query(
    topic: Annotated[str, Doc("Topic to search for.")],
    source: Annotated[str, Doc("Source filter: arxiv, paperswithcode, or all.")] = "all",
) -> str:
    """Build a web search query for academic sources.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Adds site restrictions for scholarly sources.
    """
    if source == "arxiv":
        return f"site:arxiv.org {topic}".strip()
    if source == "paperswithcode":
        return f"site:paperswithcode.com {topic}".strip()
    return f"(site:arxiv.org OR site:paperswithcode.com) {topic}".strip()


def create_web_search_tool(
    *,
    search_context_size: Annotated[Literal["low", "medium", "high"], Doc("Search context size.")] = "medium",
    user_location: Annotated[WebSearchUserLocation | None, Doc("Optional user location context.")] = None,
    blocked_domains: Annotated[list[str] | None, Doc("Domains to exclude from search.")] = None,
    allowed_domains: Annotated[list[str] | None, Doc("Domains to allow for search.")] = None,
    max_uses: Annotated[int | None, Doc("Maximum tool uses per run.")] = None,
) -> WebSearchTool:
    """Create a WebSearchTool with explicit configuration.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Builds a configured WebSearchTool instance.
    """
    return WebSearchTool(
        search_context_size=search_context_size,
        user_location=user_location,
        blocked_domains=blocked_domains,
        allowed_domains=allowed_domains,
        max_uses=max_uses,
    )


async def prepare_web_search(
    ctx: Annotated[RunContext[Any], Doc("Run context for tool preparation.")],
) -> WebSearchTool | None:
    """Prepare WebSearchTool dynamically based on RunContext.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns a WebSearchTool when provider and deps allow.
    """
    if ctx.model.system not in {"anthropic", "openai", "google", "groq"}:
        return None
    if OpenAIChatModel is not None and isinstance(ctx.model, OpenAIChatModel):
        return None
    if getattr(ctx.deps, "offline_mode", False):
        return None

    user_location = _coerce_user_location(getattr(ctx.deps, "user_location", None))
    blocked_domains = getattr(ctx.deps, "blocked_domains", None)
    allowed_domains = getattr(ctx.deps, "allowed_domains", None)
    max_uses = getattr(ctx.deps, "search_budget", None)

    return create_web_search_tool(
        user_location=user_location, blocked_domains=blocked_domains, allowed_domains=allowed_domains, max_uses=max_uses
    )


def create_web_fetch_tool(
    *,
    allowed_domains: Annotated[list[str] | None, Doc("Domains allowed for fetch.")] = None,
    blocked_domains: Annotated[list[str] | None, Doc("Domains blocked for fetch.")] = None,
    max_uses: Annotated[int | None, Doc("Maximum tool uses per run.")] = None,
    enable_citations: Annotated[bool, Doc("Whether to include citations.")] = True,
    max_content_tokens: Annotated[int | None, Doc("Maximum tokens for fetched content.")] = None,
) -> WebFetchTool:
    """Create a WebFetchTool with explicit configuration.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Builds a configured WebFetchTool instance.
    """
    return WebFetchTool(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        max_uses=max_uses,
        enable_citations=enable_citations,
        max_content_tokens=max_content_tokens,
    )


async def prepare_web_fetch(
    ctx: Annotated[RunContext[Any], Doc("Run context for tool preparation.")],
) -> WebFetchTool | None:
    """Prepare WebFetchTool dynamically based on RunContext.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns a WebFetchTool when provider and deps allow.
    """
    if ctx.model.system not in {"anthropic", "google"}:
        return None
    if getattr(ctx.deps, "offline_mode", False):
        return None

    allowed_domains = getattr(ctx.deps, "allowed_domains", None)
    blocked_domains = getattr(ctx.deps, "blocked_domains", None)
    max_uses = getattr(ctx.deps, "fetch_budget", None)

    return create_web_fetch_tool(allowed_domains=allowed_domains, blocked_domains=blocked_domains, max_uses=max_uses)


def _coerce_user_location(value: Any) -> WebSearchUserLocation | None:
    if value is None:
        return None
    if isinstance(value, dict):
        cleaned = {
            key: val
            for key, val in value.items()
            if key in {"city", "country", "region", "timezone"} and isinstance(val, str)
        }
        return cast("WebSearchUserLocation", cleaned) if cleaned else None

    def _as_str(entry: Any) -> str | None:
        return entry if isinstance(entry, str) else None

    city = _as_str(getattr(value, "city", None))
    country = _as_str(getattr(value, "country", None))
    region = _as_str(getattr(value, "region", None))
    timezone = _as_str(getattr(value, "timezone", None))
    if not (city or country or region or timezone):
        return None

    data = {
        key: val
        for key, val in {"city": city, "country": country, "region": region, "timezone": timezone}.items()
        if val
    }
    return cast("WebSearchUserLocation", data) if data else None
