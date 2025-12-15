"""Kaggle toolset for AGENT-K agents.

Provides Kaggle API operations as a pydantic-ai toolset that works
with any model provider (including OpenAI-compatible like Devstral).

Uses FunctionToolset to properly integrate with pydantic-ai's tool system.
"""
from __future__ import annotations

from typing import Any

import logfire
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from ..adapters.kaggle import KaggleAdapter
from ..core.models import Competition

__all__ = ['create_kaggle_toolset']


def create_kaggle_toolset(adapter: KaggleAdapter) -> FunctionToolset[Any]:
    """Create a Kaggle toolset wrapping the Kaggle adapter.
    
    This creates a FunctionToolset with tools that call the Kaggle API.
    Works with any model provider including OpenAI-compatible endpoints.
    
    Example:
        >>> adapter = KaggleAdapter(config)
        >>> toolset = create_kaggle_toolset(adapter)
        >>> agent = Agent('devstral:local', toolsets=[toolset])
    """
    toolset: FunctionToolset[Any] = FunctionToolset(id='kaggle')
    
    # Cache for competition data
    _cache: dict[str, Competition] = {}
    
    @toolset.tool
    async def kaggle_search_competitions(
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search Kaggle for active competitions.
        
        Args:
            categories: Filter by competition type: Featured, Research, Playground, Getting Started, Community
            keywords: Keywords to search for in competition titles/descriptions
            min_prize: Minimum prize pool in USD
        
        Returns:
            List of active competitions matching the criteria.
        """
        with logfire.span('kaggle_search_competitions', categories=categories, keywords=keywords):
            competitions = []
            count = 0
            
            async for comp in adapter.search_competitions(
                categories=categories,
                keywords=keywords,
                min_prize=min_prize,
                active_only=True,
            ):
                _cache[comp.id] = comp
                competitions.append({
                    'id': comp.id,
                    'title': comp.title,
                    'type': comp.competition_type.value,
                    'metric': comp.metric.value,
                    'days_remaining': comp.days_remaining,
                    'prize_pool': comp.prize_pool,
                    'tags': list(comp.tags) if comp.tags else [],
                    'is_active': comp.is_active,
                })
                count += 1
                if count >= 20:
                    break
            
            return competitions
    
    @toolset.tool
    async def kaggle_get_competition(
        competition_id: str,
    ) -> dict[str, Any]:
        """Get detailed information about a specific Kaggle competition.
        
        Args:
            competition_id: The competition slug/ID (e.g., "titanic")
        
        Returns:
            Detailed competition information.
        """
        with logfire.span('kaggle_get_competition', competition_id=competition_id):
            # Check cache first
            if competition_id in _cache:
                comp = _cache[competition_id]
            else:
                try:
                    comp = await adapter.get_competition(competition_id)
                    _cache[competition_id] = comp
                except Exception as e:
                    return {'error': str(e)}
            
            return {
                'id': comp.id,
                'title': comp.title,
                'description': comp.description[:500] if comp.description else None,
                'type': comp.competition_type.value,
                'metric': comp.metric.value,
                'metric_direction': comp.metric_direction,
                'days_remaining': comp.days_remaining,
                'deadline': comp.deadline.isoformat(),
                'prize_pool': comp.prize_pool,
                'max_team_size': comp.max_team_size,
                'max_daily_submissions': comp.max_daily_submissions,
                'tags': list(comp.tags) if comp.tags else [],
            }
    
    @toolset.tool
    async def kaggle_get_leaderboard(
        competition_id: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get the current leaderboard for a competition.
        
        Args:
            competition_id: The competition slug/ID
            limit: Maximum number of entries to return (default: 20)
        
        Returns:
            Leaderboard entries with ranks and scores.
        """
        with logfire.span('kaggle_get_leaderboard', competition_id=competition_id):
            try:
                entries = await adapter.get_leaderboard(competition_id, limit=limit)
                return {
                    'competition_id': competition_id,
                    'total_entries': len(entries),
                    'entries': [
                        {
                            'rank': e.rank,
                            'team_name': e.team_name,
                            'score': e.score,
                        }
                        for e in entries
                    ],
                }
            except Exception as e:
                return {'error': str(e)}
    
    @toolset.tool
    async def kaggle_list_datasets(
        competition_id: str,
    ) -> dict[str, Any]:
        """List available datasets for a competition.
        
        Args:
            competition_id: The competition slug/ID
        
        Returns:
            List of data files available for the competition.
        """
        with logfire.span('kaggle_list_datasets', competition_id=competition_id):
            try:
                response = await adapter._request(
                    'GET',
                    f'/competitions/data/list/{competition_id}',
                )
                if response.status_code != 200:
                    return {'error': f'Failed to list datasets: {response.status_code}'}
                
                files = response.json()
                return {
                    'competition_id': competition_id,
                    'files': [
                        {
                            'name': f.get('name'),
                            'size': f.get('totalBytes'),
                            'description': f.get('description'),
                        }
                        for f in files
                    ],
                }
            except Exception as e:
                return {'error': str(e)}
    
    return toolset
