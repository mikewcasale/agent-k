"""Lobbyist agent - competition discovery for AGENT-K.

@notice: |
    Lobbyist agent - competition discovery for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.agents.lobbyist
    provides:
        - agent_k.agents.lobbyist:LobbyistAgent
        - agent_k.agents.lobbyist:LobbyistDeps
        - agent_k.agents.lobbyist:LobbyistSettings
        - agent_k.agents.lobbyist:DiscoveryResult
        - agent_k.agents.lobbyist:lobbyist_agent
    consumes:
        - agent_k.core.protocols:PlatformAdapter
        - agent_k.ui.agui:EventEmitter
        - agent_k.toolsets.kaggle:kaggle_toolset
    pattern: agent-singleton

@similar:
    - id: agent_k.agents.scientist
        when: "Use for research/analysis, not discovery."

@agent-guidance:
    do:
        - "Use agent_k.agents.lobbyist as the canonical home for this capability."
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_k.agents import register_agent
from agent_k.agents.base import MemoryMixin, universal_tool_preparation
from agent_k.agents.prompts import LOBBYIST_SYSTEM_PROMPT
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.core.models import Competition
from agent_k.core.sage import Doc, Range
from agent_k.infra.providers import get_model
from agent_k.toolsets import create_production_toolset, kaggle_toolset, prepare_memory_tool, prepare_web_search

if TYPE_CHECKING:
    import httpx

    from agent_k.core.protocols import PlatformAdapter
    from agent_k.ui.agui import EventEmitter

__all__ = (
    "DiscoveryResult",
    "LobbyistAgent",
    "LobbyistDeps",
    "LobbyistSettings",
    "LOBBYIST_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "lobbyist_agent",
)

SCHEMA_VERSION: Final[str] = "1.0.0"


class LobbyistSettings(BaseSettings):
    """Configuration for the Lobbyist agent.

    @notice: |
        Configuration for the Lobbyist agent.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: settings
            rationale: "Centralizes discovery agent configuration."
            violations: "Ad-hoc per-run overrides lead to inconsistent behavior."
    """

    model_config = SettingsConfigDict(env_prefix="LOBBYIST_", env_file=".env", extra="ignore", validate_default=True)
    model: str = Field(default=DEFAULT_MODEL, description="Model identifier for discovery tasks")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature for discovery prompts")
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens for responses")
    tool_retries: int = Field(default=2, ge=0, description="Tool retry attempts")
    output_retries: int = Field(default=1, ge=0, description="Output validation retry attempts")
    max_results: int = Field(default=50, ge=1, description="Maximum competitions to return")

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings for the configured model."""
        return ModelSettings(temperature=self.temperature, max_tokens=self.max_tokens)


@dataclass
class LobbyistDeps:
    """Dependencies for the Lobbyist agent.

    @notice: |
        Dependencies for the Lobbyist agent.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: dependency-container
            rationale: "Groups runtime services for discovery tools."
            violations: "Hidden globals make tests and tooling brittle."

        @collaborators:
            required:
                - httpx:AsyncClient
                - agent_k.core.protocols:PlatformAdapter
                - agent_k.ui.agui:EventEmitter
            optional:
                - agent_k.toolsets.memory:AgentKMemoryTool
            injection: constructor
            lifecycle: "Allocated per agent run."
    """

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    search_cache: dict[str, Any] = field(default_factory=dict)


class DiscoveryResult(BaseModel):
    """Result of competition discovery.

    @notice: |
        Result of competition discovery.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: output-model
            rationale: "Stable schema for discovery outputs."
            violations: "Ad-hoc dict outputs are hard to validate."
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    competitions: list[Competition] = Field(
        default_factory=list, description="Discovered competitions matching criteria"
    )
    total_searched: int = Field(default=0, ge=0, description="Total competitions scanned")
    filters_applied: list[str] = Field(default_factory=list, description="Filters applied during discovery")


class LobbyistAgent(MemoryMixin):
    """Lobbyist agent encapsulating competition discovery functionality.

    @notice: |
        Discovers Kaggle competitions matching mission criteria.
        Use the module-level lobbyist_agent or agent registry.

    @dev: |
        Registers discovery tools on initialization and wires memory/toolsets.

    @pattern:
        name: agent-singleton
        rationale: "Single instance ensures consistent memory/tool registration."
        violations: "Multiple instances cause duplicated tool registrations."

    @collaborators:
        required:
            - agent_k.core.protocols:PlatformAdapter
            - agent_k.ui.agui:EventEmitter
        optional:
            - httpx:AsyncClient
        injection: deps via RunContext
        lifecycle: "Module-level singleton at import time."

    @concurrency:
        model: asyncio
        safe: false
        reason: "Mutates internal caches and tool registrations."

    @invariants:
        - "self._agent is initialized after __init__ completes."
        - "self._toolset registers discovery tools exactly once."
    """

    def __init__(self, settings: Annotated[LobbyistSettings | None, Doc("Optional settings override.")] = None) -> None:
        """Initialize the Lobbyist agent.

        @notice: |
            Builds the agent singleton and registers tools.

        @dev: |
            Initializes memory backend, toolset, and pydantic-ai Agent.

        @state-changes:
            - self._settings
            - self._toolset
            - self._agent
        """
        self._settings = settings or LobbyistSettings()
        self._toolset: FunctionToolset[LobbyistDeps] = FunctionToolset(id="lobbyist")
        self._memory_backend = self._init_memory_backend()
        self._register_tools()
        self._agent = self._create_agent()
        register_agent("lobbyist", self._agent)
        self._setup_memory()

    @property
    def agent(self) -> Agent[LobbyistDeps, DiscoveryResult]:
        """Return the underlying pydantic-ai Agent."""
        return self._agent

    @property
    def settings(self) -> LobbyistSettings:
        """Return current settings."""
        return self._settings

    async def search_kaggle_competitions(
        self,
        ctx: RunContext[LobbyistDeps],
        categories: Annotated[list[str], Doc("Competition categories to search.")],
        keywords: Annotated[list[str] | None, Doc("Optional keyword filters.")] = None,
        min_prize: Annotated[int | None, Doc("Minimum prize pool in USD."), Range(0, 1_000_000_000)] = None,
    ) -> list[dict[str, Any]]:
        """Search Kaggle for competitions matching criteria.

        @notice: |
            Queries the platform adapter for competition listings.

        @dev: |
            Emits telemetry events before and after the search.

        @effects:
            io:
                - Kaggle API requests
            state:
                - ctx.deps.search_cache
        """
        with logfire.span("lobbyist.search_kaggle", categories=categories, keywords=keywords):
            await ctx.deps.event_emitter.emit_tool_start(
                task_id="discovery_search",
                tool_call_id=f"kaggle_search_{id(ctx)}",
                tool_type="kaggle_mcp",
                operation="competitions.list",
            )

            adapter = ctx.deps.platform_adapter
            competitions: list[dict[str, Any]] = []

            async for comp in adapter.search_competitions(
                categories=categories, keywords=keywords, min_prize=min_prize, active_only=True
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

    async def get_competition_details(
        self, ctx: RunContext[LobbyistDeps], competition_id: Annotated[str, Doc("Competition identifier (slug).")]
    ) -> dict[str, Any]:
        """Get detailed information about a specific competition.

        @notice: |
            Fetches the competition details via the platform adapter.

        @effects:
            io:
                - Kaggle API request
        """
        with logfire.span("lobbyist.get_details", competition_id=competition_id):
            adapter = ctx.deps.platform_adapter
            competition = await adapter.get_competition(competition_id)
            return competition.model_dump()

    async def score_competition_fit(
        self,
        ctx: RunContext[LobbyistDeps],
        competition_id: Annotated[str, Doc("Competition identifier to score.")],
        target_domains: Annotated[list[str], Doc("Target subject domains for scoring.")],
        min_days_remaining: Annotated[int, Doc("Minimum days remaining required."), Range(0, 3650)],
        target_percentile: Annotated[float, Doc("Target percentile for fit scoring."), Range(0.0, 100.0)],
    ) -> dict[str, Any]:
        """Score how well a competition fits the mission criteria.

        @notice: |
            Computes a heuristic fit score from cached competition metadata.

        @dev: |
            Returns a structured dict with score and reasoning.

        @effects:
            state:
                - none
        """
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

    def _create_agent(self) -> Agent[LobbyistDeps, DiscoveryResult]:
        """Create the underlying pydantic-ai agent.

        @factory-for:
            id: agent_k.agents.lobbyist:LobbyistAgent
            rationale: "Centralizes agent wiring and toolset preparation."
            singleton: true
            cache-key: "module"

        @canonical-home:
            for:
                - "lobbyist agent construction"
            notes: "Use LobbyistAgent() or module singleton."
        """
        builtin_tools: list[Any] = [prepare_web_search]
        if self._memory_backend is not None:
            builtin_tools.append(prepare_memory_tool)

        agent: Agent[LobbyistDeps, DiscoveryResult] = Agent(
            model=get_model(self._settings.model),
            deps_type=LobbyistDeps,
            output_type=DiscoveryResult,
            instructions=LOBBYIST_SYSTEM_PROMPT,
            name="lobbyist",
            model_settings=self._settings.model_settings,
            retries=self._settings.tool_retries,
            output_retries=self._settings.output_retries,
            builtin_tools=builtin_tools,
            toolsets=[
                create_production_toolset([self._toolset, cast("FunctionToolset[LobbyistDeps]", kaggle_toolset)])
            ],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

        agent.output_validator(self._validate_discovery_result)
        agent.instructions(self._add_search_context)
        return agent

    def _register_tools(self) -> None:
        """Register all discovery tools with the toolset.

        @effects:
            state:
                - self._toolset
        """
        self._toolset.tool(self.search_kaggle_competitions)
        self._toolset.tool(self.get_competition_details)
        self._toolset.tool(self.score_competition_fit)

    async def _validate_discovery_result(
        self, ctx: RunContext[LobbyistDeps], result: DiscoveryResult
    ) -> DiscoveryResult:
        """Validate discovery results meet minimum requirements."""
        if ctx.partial_output:
            return result
        if not result.competitions:
            raise ModelRetry("No competitions found. Broaden criteria and try again.")
        return result

    async def _add_search_context(self, ctx: RunContext[LobbyistDeps]) -> str:
        """Add cached search results to context."""
        return f"Previously found competitions: {list(ctx.deps.search_cache.keys())}" if ctx.deps.search_cache else ""


# Module-level singleton for backward compatibility
lobbyist_agent_instance = LobbyistAgent()
lobbyist_agent = lobbyist_agent_instance.agent
