"""Agent-specific tools for the Lobbyist."""
from __future__ import annotations

from typing import Any

from ...core.models import Competition, LeaderboardEntry
from ...core.protocols import PlatformAdapter

__all__ = ['discover_competitions', 'fetch_leaderboard']


async def discover_competitions(
    adapter: PlatformAdapter,
    **kwargs: Any,
) -> list[Competition]:
    """Discover competitions using the provided adapter."""
    results: list[Competition] = []
    async for competition in adapter.search_competitions(**kwargs):
        results.append(competition)
    return results


async def fetch_leaderboard(
    adapter: PlatformAdapter,
    competition_id: str,
    *,
    limit: int = 100,
) -> list[LeaderboardEntry]:
    """Fetch leaderboard for a competition."""
    return await adapter.get_leaderboard(competition_id, limit=limit)
