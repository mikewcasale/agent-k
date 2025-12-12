"""Evolution orchestration service."""
from __future__ import annotations

from ..core.models import MissionResult

__all__ = ['EvolutionService']


class EvolutionService:
    """Service coordinating evolutionary runs."""
    
    async def record_result(self, result: MissionResult) -> None:
        """Persist mission result."""
        return None
