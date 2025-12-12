"""Lobbyist agent for competition discovery.

Uses WebSearchTool builtin for web discovery per spec Section 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.builtin_tools import WebSearchTool

from ...core.constants import DEFAULT_MODEL, DISCOVERY_TIMEOUT_SECONDS
from ...core.models import Competition, CompetitionType
from ...core.protocols import PlatformAdapter
from ...ui.ag_ui.event_stream import EventEmitter

__all__ = ['LobbyistAgent', 'LobbyistDeps', 'DiscoveryResult']


# =============================================================================
# Dependency Container (Per Spec Section 7.1)
# =============================================================================
@dataclass
class LobbyistDeps:
    """Dependencies for the Lobbyist agent.
    
    The dependency container holds all external services required by the agent.
    Using a dataclass ensures type safety and clear documentation of requirements.
    """
    
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    search_cache: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Output Model (Per Spec Section 4.4)
# =============================================================================
class DiscoveryResult(BaseModel):
    """Result of competition discovery process."""
    
    model_config = {'frozen': True}
    
    competitions: list[Competition] = Field(
        default_factory=list,
        description='Discovered competitions matching criteria',
    )
    total_searched: int = Field(default=0)
    filters_applied: list[str] = Field(default_factory=list)


# =============================================================================
# Agent Factory (Per Spec Section 7.2)
# =============================================================================
def create_lobbyist_agent(
    model: str = DEFAULT_MODEL,
) -> Agent[LobbyistDeps, DiscoveryResult]:
    """Create and configure the Lobbyist agent.
    
    Per spec Section 7, uses WebSearchTool builtin for web discovery.
    """
    agent = Agent(
        model,
        deps_type=LobbyistDeps,
        output_type=DiscoveryResult,
        instructions=_get_lobbyist_instructions(),
        builtin_tools=[WebSearchTool()],  # Builtin tool per spec
        retries=2,
        name='lobbyist',  # For Logfire tracing
    )
    
    # =========================================================================
    # Dynamic Instructions (Per Spec Section 7.2)
    # =========================================================================
    @agent.instructions
    async def add_search_context(ctx: RunContext[LobbyistDeps]) -> str:
        """Add cached search results to context."""
        if ctx.deps.search_cache:
            return f"Previously found competitions: {list(ctx.deps.search_cache.keys())}"
        return ""
    
    # =========================================================================
    # Custom Tools (Domain-Specific)
    # =========================================================================
    @agent.tool
    async def search_kaggle_competitions(
        ctx: RunContext[LobbyistDeps],
        categories: list[str],
        keywords: list[str] | None = None,
        min_prize: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search Kaggle for competitions matching criteria.
        
        This tool uses the platform adapter (which may use MCPServerTool
        internally) to query the Kaggle API.
        
        Args:
            ctx: Run context with dependencies.
            categories: Competition categories to search.
            keywords: Optional keywords for filtering.
            min_prize: Minimum prize pool in USD.
        
        Returns:
            List of competition data dictionaries.
        """
        with logfire.span(
            'lobbyist.search_kaggle',
            categories=categories,
            keywords=keywords,
        ):
            # Emit tool start event
            await ctx.deps.event_emitter.emit_tool_start(
                task_id='discovery_search',
                tool_call_id=f'kaggle_search_{id(ctx)}',
                tool_type='kaggle_mcp',
                operation='competitions.list',
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
                
                # Cache results
                ctx.deps.search_cache[comp.id] = comp
            
            # Emit tool result event
            await ctx.deps.event_emitter.emit_tool_result(
                task_id='discovery_search',
                tool_call_id=f'kaggle_search_{id(ctx)}',
                result={'count': len(competitions)},
                duration_ms=0,  # Would be calculated
            )
            
            return competitions
    
    @agent.tool
    async def get_competition_details(
        ctx: RunContext[LobbyistDeps],
        competition_id: str,
    ) -> dict[str, Any]:
        """Get detailed information about a specific competition.
        
        Args:
            ctx: Run context with dependencies.
            competition_id: Unique identifier for the competition.
        
        Returns:
            Competition details as dictionary.
        """
        with logfire.span('lobbyist.get_details', competition_id=competition_id):
            adapter = ctx.deps.platform_adapter
            competition = await adapter.get_competition(competition_id)
            return competition.model_dump()
    
    @agent.tool
    async def score_competition_fit(
        ctx: RunContext[LobbyistDeps],
        competition_id: str,
        target_domains: list[str],
        min_days_remaining: int,
        target_percentile: float,
    ) -> dict[str, Any]:
        """Score how well a competition fits the mission criteria.
        
        Args:
            ctx: Run context.
            competition_id: Competition to score.
            target_domains: Desired domain areas.
            min_days_remaining: Minimum days until deadline.
            target_percentile: Target leaderboard percentile.
        
        Returns:
            Fit score and breakdown.
        """
        competition = ctx.deps.search_cache.get(competition_id)
        if not competition:
            return {'score': 0.0, 'reason': 'Competition not in cache'}
        
        # Calculate fit score
        score = 0.0
        factors = {}
        
        # Days remaining factor
        if competition.days_remaining >= min_days_remaining:
            score += 0.3
            factors['deadline'] = 'OK'
        else:
            factors['deadline'] = f'Only {competition.days_remaining} days'
        
        # Domain alignment
        comp_tags = set(competition.tags) if competition.tags else set()
        target_set = set(target_domains)
        overlap = comp_tags & target_set
        if overlap:
            score += 0.4 * (len(overlap) / len(target_set))
            factors['domain'] = list(overlap)
        
        # Prize factor
        if competition.prize_pool and competition.prize_pool > 0:
            score += 0.2
            factors['prize'] = f'${competition.prize_pool:,}'
        
        # Type factor
        if competition.competition_type in (
            CompetitionType.FEATURED,
            CompetitionType.RESEARCH,
        ):
            score += 0.1
            factors['type'] = competition.competition_type.value
        
        return {
            'competition_id': competition_id,
            'score': round(score, 3),
            'factors': factors,
        }
    
    # =========================================================================
    # Output Validator (Per Spec Section 7.3)
    # =========================================================================
    @agent.output_validator
    async def validate_discovery_result(
        ctx: RunContext[LobbyistDeps],
        result: DiscoveryResult,
    ) -> DiscoveryResult:
        """Validate discovery results meet minimum requirements."""
        if not result.competitions:
            logfire.warn('lobbyist.no_competitions_found')
        return result
    
    return agent


def _get_lobbyist_instructions() -> str:
    """Return system instructions for the Lobbyist agent."""
    return """You are the LOBBYIST agent in the AGENT-K multi-agent system.

Your mission is to discover Kaggle competitions that match the user's criteria.

AVAILABLE TOOLS:
- WebSearch (builtin): Search the web for competition information
- search_kaggle_competitions: Query Kaggle API for competitions
- get_competition_details: Get full details for a specific competition
- score_competition_fit: Score how well a competition matches criteria

DISCOVERY WORKFLOW:
1. Parse the user's natural language request to extract search criteria
2. Use WebSearch to find recent Kaggle competitions and trends
3. Use search_kaggle_competitions to query the Kaggle API
4. Use get_competition_details for promising matches
5. Use score_competition_fit to rank candidates
6. Return structured DiscoveryResult with your findings

IMPORTANT:
- Always consider prize pool, deadline, and team constraints
- Prefer competitions with active communities and good documentation
- Flag any competitions with unusual rules or requirements
- Search both web and API - web search may find newer competitions
"""


# =============================================================================
# Public Interface Class (Per Spec Section 7.4)
# =============================================================================
class LobbyistAgent:
    """High-level interface for the Lobbyist agent.
    
    Provides clean API for discovery operations while encapsulating
    the underlying Pydantic-AI agent with builtin tools.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        timeout: int = DISCOVERY_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the Lobbyist agent.
        
        Args:
            model: Model identifier for the underlying LLM.
            timeout: Maximum time for discovery operations in seconds.
        """
        self._timeout = timeout
        self._agent = create_lobbyist_agent(model)
    
    async def run(
        self,
        prompt: str,
        *,
        deps: LobbyistDeps,
    ) -> DiscoveryResult:
        """Execute competition discovery based on natural language prompt.
        
        Args:
            prompt: Natural language description of desired competitions.
            deps: Dependency container with required services.
        
        Returns:
            DiscoveryResult containing matched competitions.
        """
        with logfire.span('lobbyist.run', prompt=prompt[:100]):
            result = await self._agent.run(prompt, deps=deps)
            return result.output
