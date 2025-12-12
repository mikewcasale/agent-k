"""Tools for the LYCURGUS orchestrator."""
from __future__ import annotations

from .agent import LycurgusOrchestrator
from ...core.models import MissionCriteria, MissionResult

__all__ = ['orchestrate']


async def orchestrate(
    orchestrator: LycurgusOrchestrator,
    competition_id: str,
    criteria: MissionCriteria | None = None,
) -> MissionResult:
    """Convenience helper to execute a mission."""
    return await orchestrator.execute_mission(competition_id, criteria=criteria)
