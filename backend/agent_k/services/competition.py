"""Competition management service."""
from __future__ import annotations

from ..core.models import Competition

__all__ = ['CompetitionService']


class CompetitionService:
    """Service for managing competitions."""
    
    async def list_active(self) -> list[Competition]:
        """Return active competitions."""
        return []
