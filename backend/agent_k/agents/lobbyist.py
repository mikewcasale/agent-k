"""Lobbyist agent - competition discovery for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

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
from agent_k.agents.prompts import LOBBYIST_SYSTEM_PROMPT
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.core.models import Competition
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

__all__ = (
    'DiscoveryResult',
    'LobbyistAgent',
    'LobbyistDeps',
    'LobbyistSettings',
    'LOBBYIST_SYSTEM_PROMPT',
    'SCHEMA_VERSION',
    'lobbyist_agent',
)

SCHEMA_VERSION: Final[str] = '1.0.0'


class LobbyistSettings(BaseSettings):
    """Configuration for the Lobbyist agent."""

    model_config = SettingsConfigDict(env_prefix='LOBBYIST_', env_file='.env', extra='ignore', validate_default=True)

    model: str = Field(default=DEFAULT_MODEL, description='Model identifier for discovery tasks')
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description='Sampling temperature for discovery prompts')
    max_tokens: int = Field(default=2048, ge=1, description='Maximum tokens for responses')

    tool_retries: int = Field(default=2, ge=0, description='Tool retry attempts')
    output_retries: int = Field(default=1, ge=0, description='Output validation retry attempts')
    max_results: int = Field(default=50, ge=1, description='Maximum competitions to return')

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings for the configured model."""
        return ModelSettings(temperature=self.temperature, max_tokens=self.max_tokens)


@dataclass
class LobbyistDeps:
    """Dependencies for the Lobbyist agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    search_cache: dict[str, Any] = field(default_factory=dict)


class DiscoveryResult(BaseModel):
    """Result of competition discovery."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    competitions: list[Competition] = Field(
        default_factory=list, description='Discovered competitions matching criteria'
    )
    total_searched: int = Field(default=0, ge=0, description='Total competitions scanned')
    filters_applied: list[str] = Field(default_factory=list, description='Filters applied during discovery')


class LobbyistAgent:
    """Lobbyist agent encapsulating competition discovery functionality."""

    def __init__(self, settings: LobbyistSettings | None = None) -> None:
        """Initialize the Lobbyist agent.

        Args:
            settings: Configuration for the agent. Uses defaults if not provided.
        """
        self._settings = settings or LobbyistSettings()
        self._toolset: FunctionToolset[LobbyistDeps] = FunctionToolset(id='lobbyist')
        self._memory_backend = self._init_memory_backend()
        self._register_tools()
        self._agent = self._create_agent()
        register_agent('lobbyist', self._agent)
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
        categories: list[str],
        keywords: list[str] | None = None,
        min_prize: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search Kaggle for competitions matching criteria."""
        with logfire.span('lobbyist.search_kaggle', categories=categories, keywords=keywords):
            await ctx.deps.event_emitter.emit_tool_start(
                task_id='discovery_search',
                tool_call_id=f'kaggle_search_{id(ctx)}',
                tool_type='kaggle_mcp',
                operation='competitions.list',
            )

            adapter = ctx.deps.platform_adapter
            competitions: list[dict[str, Any]] = []

            async for comp in adapter.search_competitions(
                categories=categories, keywords=keywords, min_prize=min_prize, active_only=True
            ):
                competitions.append(comp.model_dump())
                ctx.deps.search_cache[comp.id] = comp

            await ctx.deps.event_emitter.emit_tool_result(
                task_id='discovery_search',
                tool_call_id=f'kaggle_search_{id(ctx)}',
                result={'count': len(competitions)},
                duration_ms=0,
            )

            return competitions

    async def get_competition_details(self, ctx: RunContext[LobbyistDeps], competition_id: str) -> dict[str, Any]:
        """Get detailed information about a specific competition."""
        with logfire.span('lobbyist.get_details', competition_id=competition_id):
            adapter = ctx.deps.platform_adapter
            competition = await adapter.get_competition(competition_id)
            return competition.model_dump()

    async def score_competition_fit(
        self,
        ctx: RunContext[LobbyistDeps],
        competition_id: str,
        target_domains: list[str],
        min_days_remaining: int,
        target_percentile: float,
    ) -> dict[str, Any]:
        """Score how well a competition fits the mission criteria."""
        competition = ctx.deps.search_cache.get(competition_id)
        if not competition:
            return {'score': 0.0, 'reason': 'Competition not in cache'}

        score = 0.0
        reasons: list[str] = []

        if any(domain.lower() in ' '.join(competition.tags).lower() for domain in target_domains):
            score += 0.4
            reasons.append('matches_domain')

        days_remaining = competition.days_remaining
        if days_remaining >= min_days_remaining:
            score += 0.3
            reasons.append('sufficient_time')

        if competition.prize_pool and competition.prize_pool >= 10000:
            score += 0.2
            reasons.append('good_prize')

        score += min(0.1, target_percentile / 100.0)
        reasons.append('target_percentile')

        return {
            'competition_id': competition_id,
            'score': round(score, 2),
            'reasons': reasons,
            'days_remaining': days_remaining,
        }

    def _init_memory_backend(self) -> AgentKMemoryTool | None:
        try:
            return create_memory_backend()
        except RuntimeError:  # pragma: no cover - optional dependency
            return None

    def _create_agent(self) -> Agent[LobbyistDeps, DiscoveryResult]:
        """Create the underlying pydantic-ai agent."""
        builtin_tools: list[Any] = [prepare_web_search]
        if self._memory_backend is not None:
            builtin_tools.append(prepare_memory_tool)

        agent: Agent[LobbyistDeps, DiscoveryResult] = Agent(
            model=get_model(self._settings.model),
            deps_type=LobbyistDeps,
            output_type=DiscoveryResult,
            instructions=LOBBYIST_SYSTEM_PROMPT,
            name='lobbyist',
            model_settings=self._settings.model_settings,
            retries=self._settings.tool_retries,
            output_retries=self._settings.output_retries,
            builtin_tools=builtin_tools,
            toolsets=[
                create_production_toolset([self._toolset, cast('FunctionToolset[LobbyistDeps]', kaggle_toolset)])
            ],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

        agent.output_validator(self._validate_discovery_result)
        agent.instructions(self._add_search_context)

        return agent

    def _setup_memory(self) -> None:
        """Set up memory tool if available."""
        if self._memory_backend is None:
            return
        register_memory_tool(self._agent, self._memory_backend)

    def _register_tools(self) -> None:
        """Register all discovery tools with the toolset."""
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
            raise ModelRetry('No competitions found. Broaden criteria and try again.')
        return result

    async def _add_search_context(self, ctx: RunContext[LobbyistDeps]) -> str:
        """Add cached search results to context."""
        if ctx.deps.search_cache:
            return f'Previously found competitions: {list(ctx.deps.search_cache.keys())}'
        return ''


# Module-level singleton for backward compatibility
lobbyist_agent_instance = LobbyistAgent()
lobbyist_agent = lobbyist_agent_instance.agent
