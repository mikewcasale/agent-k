"""Devstral-compatible Lobbyist agent for competition discovery.

This version uses custom tools instead of builtin tools that require
specific model provider support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ...core.constants import DEVSTRAL_MODEL
from ...core.models import Competition, CompetitionType
from ...core.protocols import PlatformAdapter
from ...infra.models import get_model

__all__ = ['DevstralLobbyistAgent', 'LobbyistDeps', 'DiscoveryResult']


# =============================================================================
# Dependency Container
# =============================================================================
@dataclass
class LobbyistDeps:
    """Dependencies for the Lobbyist agent."""
    
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: Any = None
    search_cache: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Output Model
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
# Agent Factory for Devstral
# =============================================================================
def create_devstral_lobbyist_agent(
    model: str = DEVSTRAL_MODEL,
) -> Agent[LobbyistDeps, DiscoveryResult]:
    """Create Devstral-compatible Lobbyist agent.
    
    This version does NOT use builtin tools (WebSearchTool, etc.)
    since they aren't supported with OpenAI-compatible models.
    Instead, it uses custom tools for Kaggle API access.
    """
    resolved_model = get_model(model)
    
    agent = Agent(
        resolved_model,
        deps_type=LobbyistDeps,
        output_type=DiscoveryResult,
        instructions=_get_lobbyist_instructions(),
        retries=2,
        name='lobbyist-devstral',
    )
    
    # =========================================================================
    # Custom Tools (Kaggle API access)
    # =========================================================================
    @agent.tool
    async def search_kaggle_competitions(
        ctx: RunContext[LobbyistDeps],
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search Kaggle for competitions matching criteria.
        
        Args:
            ctx: Run context with dependencies.
            categories: Competition categories (e.g., ['Playground', 'Featured']).
            keywords: Keywords to filter competitions.
            min_prize: Minimum prize pool in USD.
        
        Returns:
            List of competition data dictionaries.
        """
        with logfire.span(
            'lobbyist.search_kaggle',
            categories=categories,
            keywords=keywords,
        ):
            adapter = ctx.deps.platform_adapter
            competitions: list[dict[str, Any]] = []
            
            try:
                async for comp in adapter.search_competitions(
                    categories=categories,
                    keywords=keywords,
                    min_prize=min_prize,
                    active_only=True,
                ):
                    comp_dict = {
                        'id': comp.id,
                        'title': comp.title,
                        'type': comp.competition_type.value,
                        'metric': comp.metric.value,
                        'days_remaining': comp.days_remaining,
                        'prize_pool': comp.prize_pool,
                        'tags': list(comp.tags) if comp.tags else [],
                    }
                    competitions.append(comp_dict)
                    ctx.deps.search_cache[comp.id] = comp
                    
                    # Limit results
                    if len(competitions) >= 20:
                        break
            except Exception as e:
                logfire.error('search_failed', error=str(e))
                return [{'error': str(e)}]
            
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
            # Check cache first
            if competition_id in ctx.deps.search_cache:
                comp = ctx.deps.search_cache[competition_id]
                return {
                    'id': comp.id,
                    'title': comp.title,
                    'type': comp.competition_type.value,
                    'metric': comp.metric.value,
                    'metric_direction': comp.metric_direction,
                    'days_remaining': comp.days_remaining,
                    'deadline': str(comp.deadline),
                    'prize_pool': comp.prize_pool,
                    'max_team_size': comp.max_team_size,
                    'tags': list(comp.tags) if comp.tags else [],
                }
            
            try:
                adapter = ctx.deps.platform_adapter
                comp = await adapter.get_competition(competition_id)
                ctx.deps.search_cache[competition_id] = comp
                return {
                    'id': comp.id,
                    'title': comp.title,
                    'type': comp.competition_type.value,
                    'metric': comp.metric.value,
                    'days_remaining': comp.days_remaining,
                    'prize_pool': comp.prize_pool,
                }
            except Exception as e:
                return {'error': str(e)}
    
    @agent.tool_plain
    async def score_competition_fit(
        competition_id: str,
        target_domains: list[str],
        min_days_remaining: int,
        competition_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Score how well a competition fits the mission criteria.
        
        Args:
            competition_id: Competition to score.
            target_domains: Desired domain areas.
            min_days_remaining: Minimum days until deadline.
            competition_data: Competition data dictionary.
        
        Returns:
            Fit score and breakdown.
        """
        score = 0.0
        factors = {}
        
        # Days remaining factor
        days = competition_data.get('days_remaining', 0)
        if days >= min_days_remaining:
            score += 0.3
            factors['deadline'] = f'OK ({days} days)'
        else:
            factors['deadline'] = f'Too soon ({days} days)'
        
        # Domain alignment
        tags = set(competition_data.get('tags', []))
        target_set = set(target_domains)
        overlap = tags & target_set
        if overlap:
            score += 0.4 * (len(overlap) / max(len(target_set), 1))
            factors['domain'] = list(overlap)
        else:
            factors['domain'] = 'No match'
        
        # Prize factor
        prize = competition_data.get('prize_pool')
        if prize and prize > 0:
            score += 0.2
            factors['prize'] = f'${prize:,}'
        else:
            factors['prize'] = 'None'
        
        # Type factor
        comp_type = competition_data.get('type', '')
        if comp_type in ('featured', 'research', 'playground'):
            score += 0.1
            factors['type'] = comp_type
        
        return {
            'competition_id': competition_id,
            'score': round(score, 3),
            'factors': factors,
        }
    
    return agent


def _get_lobbyist_instructions() -> str:
    """Return system instructions for the Lobbyist agent."""
    return """You are the LOBBYIST agent in the AGENT-K multi-agent system.

Your mission is to discover Kaggle competitions that match the user's criteria.

AVAILABLE TOOLS:
- search_kaggle_competitions: Query Kaggle API for competitions
- get_competition_details: Get full details for a specific competition
- score_competition_fit: Score how well a competition matches criteria

DISCOVERY WORKFLOW:
1. Parse the user's request to extract search criteria
2. Use search_kaggle_competitions to find matching competitions
3. Use get_competition_details for promising matches
4. Use score_competition_fit to rank candidates
5. Return a DiscoveryResult with your findings

OUTPUT FORMAT:
Return a JSON object with:
- competitions: List of Competition objects matching criteria
- total_searched: Number of competitions examined
- filters_applied: List of filters used

IMPORTANT:
- Focus on active competitions with reasonable deadlines
- Prefer competitions with clear evaluation metrics
- Consider the user's experience level (beginner-friendly if not specified)
"""


# =============================================================================
# Public Interface Class
# =============================================================================
class DevstralLobbyistAgent:
    """Devstral-compatible Lobbyist agent.
    
    This version uses custom tools instead of builtin tools.
    """
    
    def __init__(
        self,
        model: str = DEVSTRAL_MODEL,
        *,
        timeout: int = 300,
    ) -> None:
        self._timeout = timeout
        self._agent = create_devstral_lobbyist_agent(model)
    
    async def run(
        self,
        prompt: str,
        *,
        deps: LobbyistDeps,
    ) -> DiscoveryResult:
        """Execute competition discovery."""
        with logfire.span('lobbyist.run', prompt=prompt[:100]):
            result = await self._agent.run(prompt, deps=deps)
            return result.output

