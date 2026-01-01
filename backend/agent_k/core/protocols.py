"""Protocol definitions for platform adapters.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .models import Competition, LeaderboardEntry, Submission

__all__ = ("PlatformAdapter",)


@runtime_checkable
class PlatformAdapter(Protocol):
    """Protocol for platform adapters.

    Platform adapters provide the interface between AGENT-K and competition
    platforms (e.g., Kaggle, DrivenData). Implementations must be async-first
    and handle authentication, rate limiting, and error recovery internally.

    Example Implementation:
        >>> class KaggleAdapter:
        ...     async def search_competitions(
        ...         self, categories: list[str], **kwargs
        ...     ) -> AsyncIterator[Competition]:
        ...         async for comp in self._api.list_competitions():
        ...             yield Competition.model_validate(comp)
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Unique identifier for the platform."""
        ...

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the platform.

        Returns:
            True if authentication successful.

        Raises:
            AuthenticationError: If authentication fails.
        """
        ...

    @abstractmethod
    def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        """Search for competitions matching criteria.

        Args:
            categories: Filter by competition categories.
            keywords: Keywords to search in title/description.
            min_prize: Minimum prize pool filter.
            active_only: Only return active competitions.

        Yields:
            Competition entities matching criteria.
        """
        ...

    @abstractmethod
    async def get_competition(self, competition_id: str) -> Competition:
        """Get detailed competition information.

        Args:
            competition_id: Unique competition identifier.

        Returns:
            Full Competition entity.

        Raises:
            CompetitionNotFoundError: If competition doesn't exist.
        """
        ...

    @abstractmethod
    async def get_leaderboard(
        self,
        competition_id: str,
        *,
        limit: int = 100,
    ) -> list[LeaderboardEntry]:
        """Get competition leaderboard.

        Args:
            competition_id: Target competition.
            limit: Maximum entries to return.

        Returns:
            List of leaderboard entries sorted by rank.
        """
        ...

    @abstractmethod
    async def submit(
        self,
        competition_id: str,
        file_path: str,
        message: str = "",
    ) -> Submission:
        """Submit solution to competition.

        Args:
            competition_id: Target competition.
            file_path: Path to submission file.
            message: Optional submission message.

        Returns:
            Submission entity with initial status.

        Raises:
            SubmissionError: If submission fails.
            DeadlinePassedError: If competition has ended.
        """
        ...

    @abstractmethod
    async def get_submission_status(
        self,
        competition_id: str,
        submission_id: str,
    ) -> Submission:
        """Get status of a submission.

        Args:
            competition_id: Competition containing submission.
            submission_id: Submission to check.

        Returns:
            Updated Submission entity.
        """
        ...

    @abstractmethod
    async def download_data(
        self,
        competition_id: str,
        destination: str,
    ) -> list[str]:
        """Download competition data files.

        Args:
            competition_id: Target competition.
            destination: Directory to download files to.

        Returns:
            List of downloaded file paths.
        """
        ...
