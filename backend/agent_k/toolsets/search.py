"""Search tool helpers for AGENT-K agents.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from typing import Any, Literal, cast

# Third-party (alphabetical)
from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.builtin_tools import WebFetchTool, WebSearchTool, WebSearchUserLocation

try:  # pragma: no cover - optional dependency
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:  # pragma: no cover - optional dependency
    OpenAIChatModel = None  # type: ignore[misc,assignment]

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "build_kaggle_search_query",
    "build_scholarly_query",
    "create_web_fetch_tool",
    "create_web_search_tool",
    "prepare_web_fetch",
    "prepare_web_search",
)

# =============================================================================
# Section 12: Functions
# =============================================================================


def build_kaggle_search_query(query: str) -> str:
    """Build a Kaggle-scoped web search query."""
    return f"site:kaggle.com {query}".strip()


def build_scholarly_query(topic: str, source: str = "all") -> str:
    """Build a web search query for academic sources."""
    if source == "arxiv":
        return f"site:arxiv.org {topic}".strip()
    if source == "paperswithcode":
        return f"site:paperswithcode.com {topic}".strip()
    return f"site:arxiv.org OR site:paperswithcode.com {topic}".strip()


def create_web_search_tool(
    *,
    search_context_size: Literal["low", "medium", "high"] = "medium",
    user_location: WebSearchUserLocation | None = None,
    blocked_domains: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    max_uses: int | None = None,
) -> WebSearchTool:
    """Create a WebSearchTool with explicit configuration."""
    return WebSearchTool(
        search_context_size=search_context_size,
        user_location=user_location,
        blocked_domains=blocked_domains,
        allowed_domains=allowed_domains,
        max_uses=max_uses,
    )


async def prepare_web_search(ctx: RunContext[Any]) -> WebSearchTool | None:
    """Prepare WebSearchTool dynamically based on RunContext."""
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
        user_location=user_location,
        blocked_domains=blocked_domains,
        allowed_domains=allowed_domains,
        max_uses=max_uses,
    )


def create_web_fetch_tool(
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    max_uses: int | None = None,
    enable_citations: bool = True,
    max_content_tokens: int | None = None,
) -> WebFetchTool:
    """Create a WebFetchTool with explicit configuration."""
    return WebFetchTool(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        max_uses=max_uses,
        enable_citations=enable_citations,
        max_content_tokens=max_content_tokens,
    )


async def prepare_web_fetch(ctx: RunContext[Any]) -> WebFetchTool | None:
    """Prepare WebFetchTool dynamically based on RunContext."""
    if ctx.model.system not in {"anthropic", "google"}:
        return None
    if getattr(ctx.deps, "offline_mode", False):
        return None

    allowed_domains = getattr(ctx.deps, "allowed_domains", None)
    blocked_domains = getattr(ctx.deps, "blocked_domains", None)
    max_uses = getattr(ctx.deps, "fetch_budget", None)

    return create_web_fetch_tool(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        max_uses=max_uses,
    )


def _coerce_user_location(value: Any) -> WebSearchUserLocation | None:
    if value is None:
        return None
    if isinstance(value, dict):
        cleaned = {
            key: val
            for key, val in value.items()
            if key in {"city", "country", "region", "timezone"} and isinstance(val, str)
        }
        return cast(WebSearchUserLocation, cleaned) if cleaned else None

    def _as_str(entry: Any) -> str | None:
        return entry if isinstance(entry, str) else None

    city = _as_str(getattr(value, "city", None))
    country = _as_str(getattr(value, "country", None))
    region = _as_str(getattr(value, "region", None))
    timezone = _as_str(getattr(value, "timezone", None))
    if not any([city, country, region, timezone]):
        return None

    data: dict[str, str] = {}
    if city:
        data["city"] = city
    if country:
        data["country"] = country
    if region:
        data["region"] = region
    if timezone:
        data["timezone"] = timezone
    return cast(WebSearchUserLocation, data) if data else None
