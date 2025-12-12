"""OpenEvolve integration adapter."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ...core.exceptions import CompetitionNotFoundError, SubmissionError
from ...core.models import Competition, LeaderboardEntry, Submission
from ...core.protocols import PlatformAdapter

__all__ = ['OpenEvolveAdapter']


@dataclass
class OpenEvolveAdapter(PlatformAdapter):
    """Adapter stub for OpenEvolve integration."""
    
    _platform_name: str = 'openevolve'
    
    @property
    def platform_name(self) -> str:
        return self._platform_name
    
    async def authenticate(self) -> bool:
        return True
    
    async def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        if False:
            yield Competition.model_validate({})  # pragma: no cover
        raise NotImplementedError
    
    async def get_competition(self, competition_id: str) -> Competition:
        raise CompetitionNotFoundError(competition_id)
    
    async def get_leaderboard(
        self,
        competition_id: str,
        *,
        limit: int = 100,
    ) -> list[LeaderboardEntry]:
        return []
    
    async def submit(
        self,
        competition_id: str,
        file_path: str,
        message: str = '',
    ) -> Submission:
        raise SubmissionError('Submission not supported for OpenEvolve stub')
    
    async def get_submission_status(
        self,
        competition_id: str,
        submission_id: str,
    ) -> Submission:
        raise SubmissionError('Submission status not implemented')
    
    async def download_data(
        self,
        competition_id: str,
        destination: str,
    ) -> list[str]:
        raise NotImplementedError
