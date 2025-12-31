"""Lobbyist agent - competition discovery for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

# Third-party (alphabetical)
import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.agents import register_agent
from agent_k.agents.base import universal_tool_preparation
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.core.models import Competition  # noqa: TC001
from agent_k.infra.providers import get_model
from agent_k.toolsets import (
    AgentKMemoryTool,
    create_memory_backend,
    create_production_toolset,
    kaggle_toolset,
    prepare_memory_tool,
    prepare_web_search,
    register_memory_tool,
)

if TYPE_CHECKING:
    import httpx

    from agent_k.core.protocols import PlatformAdapter
    from agent_k.ui.ag_ui import EventEmitter

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "DiscoveryResult",
    "LobbyistDeps",
    "LobbyistSettings",
    "LOBBYIST_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "lobbyist_agent",
)

# =============================================================================
# Section 3: Constants
# =============================================================================
SCHEMA_VERSION: Final[str] = "1.0.0"


# =============================================================================
# Section 4: Settings
# =============================================================================
class LobbyistSettings(BaseSettings):
    """Configuration for the Lobbyist agent."""

    model_config = SettingsConfigDict(
        env_prefix="LOBBYIST_",
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model identifier for discovery tasks",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for discovery prompts",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum tokens for responses",
    )

    tool_retries: int = Field(
        default=2,
        ge=0,
        description="Tool retry attempts",
    )
    output_retries: int = Field(
        default=1,
        ge=0,
        description="Output validation retry attempts",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        description="Maximum competitions to return",
    )

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings for the configured model."""
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


# =============================================================================
# Section 5: Dependencies
# =============================================================================
@dataclass
class LobbyistDeps:
    """Dependencies for the Lobbyist agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    search_cache: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Section 6: Output Types
# =============================================================================
class DiscoveryResult(BaseModel):
    """Result of competition discovery."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    competitions: list[Competition] = Field(
        default_factory=list,
        description="Discovered competitions matching criteria",
    )
    total_searched: int = Field(
        default=0,
        ge=0,
        description="Total competitions scanned",
    )
    filters_applied: list[str] = Field(
        default_factory=list,
        description="Filters applied during discovery",
    )


# =============================================================================
# Section 7: System Prompt
# =============================================================================
LOBBYIST_SYSTEM_PROMPT: Final[str] = """You are the LOBBYIST agent in the AGENT-K system.

Your mission is to discover Kaggle competitions that match the user's criteria.

WORKFLOW:
1. Parse the user's natural language request to extract search criteria
2. Use WebSearch to find recent Kaggle competitions and trends
3. Use search_kaggle_competitions to query the Kaggle API
4. Use get_competition_details for promising matches
5. Use score_competition_fit to rank candidates
6. Return a structured DiscoveryResult with your findings

IMPORTANT:
- Always consider prize pool, deadline, and team constraints
- Prefer competitions with active communities and good documentation
- Flag any competitions with unusual rules or requirements
- Search both web and API - web search may find newer competitions
"""


# =============================================================================
# Section 8: Agent Singleton
# =============================================================================
settings = LobbyistSettings()
try:
    _memory_backend: AgentKMemoryTool | None = create_memory_backend()
except RuntimeError:  # pragma: no cover - optional dependency
    _memory_backend = None

_builtin_tools: list[Any] = [prepare_web_search]
if _memory_backend is not None:
    _builtin_tools.append(prepare_memory_tool)

lobbyist_toolset: FunctionToolset[LobbyistDeps] = FunctionToolset(id="lobbyist")

lobbyist_agent: Agent[LobbyistDeps, DiscoveryResult] = Agent(
    model=get_model(settings.model),
    deps_type=LobbyistDeps,
    output_type=DiscoveryResult,
    instructions=LOBBYIST_SYSTEM_PROMPT,
    name="lobbyist",
    model_settings=settings.model_settings,
    retries=settings.tool_retries,
    output_retries=settings.output_retries,
    builtin_tools=_builtin_tools,
    toolsets=[
        create_production_toolset(
            [
                lobbyist_toolset,
                cast(FunctionToolset[LobbyistDeps], kaggle_toolset),
            ]
        ),
    ],
    prepare_tools=universal_tool_preparation,
    instrument=True,
)

register_agent("lobbyist", lobbyist_agent)

if _memory_backend is not None:
    register_memory_tool(lobbyist_agent, _memory_backend)


# =============================================================================
# Section 9: Tools
# =============================================================================
@lobbyist_toolset.tool
async def search_kaggle_competitions(
    ctx: RunContext[LobbyistDeps],
    categories: list[str],
    keywords: list[str] | None = None,
    min_prize: int | None = None,
) -> list[dict[str, Any]]:
    """Search Kaggle for competitions matching criteria."""
    with logfire.span(
        "lobbyist.search_kaggle",
        categories=categories,
        keywords=keywords,
    ):
        await ctx.deps.event_emitter.emit_tool_start(
            task_id="discovery_search",
            tool_call_id=f"kaggle_search_{id(ctx)}",
            tool_type="kaggle_mcp",
            operation="competitions.list",
        )

        adapter = ctx.deps.platform_adapter
        competitions: list[dict[str, Any]] = []

        async for comp in adapter.search_competitions(
            categories=categories,
            keywords=keywords,
            min_prize=min_prize,
            active_only=True,
        ):
            competitions.append(comp.model_dump())
            ctx.deps.search_cache[comp.id] = comp

        await ctx.deps.event_emitter.emit_tool_result(
            task_id="discovery_search",
            tool_call_id=f"kaggle_search_{id(ctx)}",
            result={"count": len(competitions)},
            duration_ms=0,
        )

        return competitions


@lobbyist_toolset.tool
async def get_competition_details(
    ctx: RunContext[LobbyistDeps],
    competition_id: str,
) -> dict[str, Any]:
    """Get detailed information about a specific competition."""
    with logfire.span("lobbyist.get_details", competition_id=competition_id):
        adapter = ctx.deps.platform_adapter
        competition = await adapter.get_competition(competition_id)
        return competition.model_dump()


@lobbyist_toolset.tool
async def score_competition_fit(
    ctx: RunContext[LobbyistDeps],
    competition_id: str,
    target_domains: list[str],
    min_days_remaining: int,
    target_percentile: float,
) -> dict[str, Any]:
    """Score how well a competition fits the mission criteria."""
    competition = ctx.deps.search_cache.get(competition_id)
    if not competition:
        return {"score": 0.0, "reason": "Competition not in cache"}

    score = 0.0
    reasons: list[str] = []

    if any(domain.lower() in " ".join(competition.tags).lower() for domain in target_domains):
        score += 0.4
        reasons.append("matches_domain")

    days_remaining = competition.days_remaining
    if days_remaining >= min_days_remaining:
        score += 0.3
        reasons.append("sufficient_time")

    if competition.prize_pool and competition.prize_pool >= 10000:
        score += 0.2
        reasons.append("good_prize")

    score += min(0.1, target_percentile / 100.0)
    reasons.append("target_percentile")

    return {
        "competition_id": competition_id,
        "score": round(score, 2),
        "reasons": reasons,
        "days_remaining": days_remaining,
    }


# =============================================================================
# Section 10: Validators
# =============================================================================
@lobbyist_agent.output_validator
async def validate_discovery_result(
    ctx: RunContext[LobbyistDeps],
    result: DiscoveryResult,
) -> DiscoveryResult:
    """Validate discovery results meet minimum requirements."""
    if ctx.partial_output:
        return result
    if not result.competitions:
        raise ModelRetry("No competitions found. Broaden criteria and try again.")
    return result


# =============================================================================
# Section 11: Dynamic Instructions
# =============================================================================
@lobbyist_agent.instructions
async def add_search_context(ctx: RunContext[LobbyistDeps]) -> str:
    """Add cached search results to context."""
    if ctx.deps.search_cache:
        return f"Previously found competitions: {list(ctx.deps.search_cache.keys())}"
    return ""
