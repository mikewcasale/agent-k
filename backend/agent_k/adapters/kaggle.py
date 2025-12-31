"""Kaggle platform adapter implementation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import asyncio
import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

# Third-party (alphabetical)
import httpx
import logfire
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.core.exceptions import (
    AuthenticationError,
    CompetitionNotFoundError,
    PlatformConnectionError,
    RateLimitError,
    SubmissionError,
)
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry, Submission
from agent_k.core.protocols import PlatformAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ('KaggleAdapter', 'KaggleSettings', 'SCHEMA_VERSION')

SCHEMA_VERSION: Final[str] = '1.0.0'


class KaggleSettings(BaseSettings):
    """Settings for Kaggle adapter.

    Environment variables are prefixed with KAGGLE_.
    """

    model_config = SettingsConfigDict(env_prefix='KAGGLE_', env_file='.env', extra='ignore', validate_default=True)
    username: str = Field(..., description='Kaggle API username')
    api_key: str = Field(..., description='Kaggle API key')
    base_url: str = Field(default='https://www.kaggle.com/api/v1', description='Base URL for Kaggle API')
    timeout: int = Field(default=30, ge=1, description='HTTP timeout in seconds')
    max_retries: int = Field(default=3, ge=0, description='Maximum retry attempts for failed requests')
    rate_limit_delay: float = Field(default=1.0, ge=0.0, description='Delay between rate-limited requests (seconds)')


@dataclass
class KaggleAdapter(PlatformAdapter):
    """Kaggle platform adapter.

    Implements PlatformAdapter protocol for Kaggle competition platform.
    Uses the Kaggle API for operations and handles authentication,
    rate limiting, and error recovery.

    Example:
        >>> config = KaggleSettings(username='my_username', api_key='my_api_key')
        >>> async with KaggleAdapter(config) as adapter:
        ...     async for comp in adapter.search_competitions(['featured']):
        ...         print(comp.title)
    """

    config: KaggleSettings
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _authenticated: bool = field(default=False, init=False)
    _rate_limit_semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout, auth=(self.config.username, self.config.api_key)
        )
        self._rate_limit_semaphore = asyncio.Semaphore(5)

    async def __aenter__(self) -> KaggleAdapter:
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self._client.aclose()

    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return 'kaggle'

    async def authenticate(self) -> bool:
        """Authenticate with Kaggle API."""
        with logfire.span('kaggle.authenticate'):
            try:
                response = await self._request('GET', '/competitions/list', params={'page': 1})
                self._authenticated = response.status_code == 200
                return self._authenticated
            except httpx.HTTPError as exc:
                raise AuthenticationError('kaggle', f'Authentication failed: {exc}') from exc

    async def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        """Search Kaggle competitions."""
        with logfire.span('kaggle.search_competitions', categories=categories):
            page = 1
            while True:
                params: dict[str, Any] = {'page': page}
                if categories:
                    params['category'] = categories[0]  # Kaggle API supports single category
                if keywords:
                    params['search'] = ' '.join(keywords)

                response = await self._request('GET', '/competitions/list', params=params)
                data = response.json()

                if not data:
                    break

                for item in data:
                    # Skip if item is not a dict (malformed data)
                    if not isinstance(item, dict):
                        logfire.warning('skipping_malformed_competition', item=str(item)[:100])
                        continue

                    try:
                        competition = self._parse_competition(item)
                    except Exception as exc:
                        logfire.warning('failed_to_parse_competition', error=str(exc))
                        continue

                    # Apply filters
                    if active_only and not competition.is_active:
                        continue
                    if min_prize and (competition.prize_pool or 0) < min_prize:
                        continue
                    if categories and competition.competition_type.value not in categories:
                        continue

                    yield competition

                page += 1

    async def get_competition(self, competition_id: str) -> Competition:
        """Get competition details from Kaggle."""
        with logfire.span('kaggle.get_competition', competition_id=competition_id):
            response = await self._request('GET', f'/competitions/data/list/{competition_id}')

            if response.status_code == 404:
                raise CompetitionNotFoundError(competition_id)

            response.raise_for_status()
            # Note: Kaggle API requires additional call for full details
            list_response = await self._request('GET', '/competitions/list', params={'search': competition_id})

            for item in list_response.json():
                if item.get('ref') == competition_id:
                    return self._parse_competition(item)

            raise CompetitionNotFoundError(competition_id)

    async def get_leaderboard(self, competition_id: str, *, limit: int = 100) -> list[LeaderboardEntry]:
        """Get competition leaderboard."""
        with logfire.span('kaggle.get_leaderboard', competition_id=competition_id):
            response = await self._request('GET', f'/competitions/{competition_id}/leaderboard/download')
            response.raise_for_status()

            entries: list[LeaderboardEntry] = []
            reader = csv.reader(io.StringIO(response.text))
            if next(reader, None) is None:
                return entries

            for i, row in enumerate(reader, start=1):
                if i > limit:
                    break
                if not row:
                    continue
                team_name = row[1] if len(row) > 1 else 'Unknown'
                score = 0.0
                if len(row) > 2:
                    try:
                        score = float(row[2])
                    except ValueError:
                        score = 0.0
                entries.append(LeaderboardEntry(rank=i, team_name=team_name, score=score))

            return entries

    async def submit(self, competition_id: str, file_path: str, message: str = '') -> Submission:
        """Submit solution to Kaggle competition."""
        with logfire.span('kaggle.submit', competition_id=competition_id):
            path = Path(file_path)
            if not path.exists():
                raise SubmissionError(competition_id, f'Submission file not found: {file_path}')

            with open(path, 'rb') as f:
                files = {'file': (path.name, f, 'text/csv')}
                data = {'message': message}

                response = await self._request(
                    'POST', f'/competitions/submissions/url/{competition_id}', data=data, files=files
                )

            if response.status_code != 200:
                raise SubmissionError(competition_id, f'Submission failed: {response.text}')

            result = response.json()
            return Submission(
                id=result.get('ref', 'unknown'), competition_id=competition_id, file_name=path.name, status='pending'
            )

    async def get_submission_status(self, competition_id: str, submission_id: str) -> Submission:
        """Get submission status from Kaggle."""
        with logfire.span('kaggle.get_submission_status', submission_id=submission_id):
            response = await self._request('GET', f'/competitions/submissions/list/{competition_id}')
            response.raise_for_status()

            for item in response.json():
                if item.get('ref') == submission_id:
                    return Submission(
                        id=submission_id,
                        competition_id=competition_id,
                        file_name=item.get('fileName', ''),
                        status='complete' if item.get('hasPublicScore') else 'pending',
                        public_score=item.get('publicScore'),
                    )

            raise SubmissionError(competition_id, f'Submission not found: {submission_id}', submission_id=submission_id)

    async def download_data(self, competition_id: str, destination: str) -> list[str]:
        """Download competition data files."""
        with logfire.span('kaggle.download_data', competition_id=competition_id):
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)

            # List available files
            response = await self._request('GET', f'/competitions/data/list/{competition_id}')
            response.raise_for_status()

            downloaded: list[str] = []
            for file_info in response.json():
                file_name = file_info.get('name', '')
                file_url = file_info.get('url', '')

                if file_url:
                    file_path = dest_path / file_name
                    async with self._client.stream('GET', file_url) as file_response:
                        file_response.raise_for_status()
                        with file_path.open('wb') as handle:
                            async for chunk in file_response.aiter_bytes():
                                handle.write(chunk)
                    downloaded.append(str(file_path))

            return downloaded

    # =========================================================================
    # Private Methods
    # =========================================================================
    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Make rate-limited request to Kaggle API."""
        async with self._rate_limit_semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    response = await self._client.request(method, path, **kwargs)

                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        raise RateLimitError('kaggle', 'Rate limit exceeded', retry_after=retry_after)

                    return response

                except httpx.HTTPError as exc:
                    if attempt == self.config.max_retries - 1:
                        raise PlatformConnectionError('kaggle', f'Kaggle API error: {exc}') from exc
                    await asyncio.sleep(self.config.rate_limit_delay * (attempt + 1))

            raise PlatformConnectionError('kaggle', 'Max retries exceeded')

    def _parse_competition(self, data: dict[str, Any]) -> Competition:
        """Parse Kaggle API response into Competition model."""
        # Map Kaggle category to our enum
        category_map = {
            'Featured': CompetitionType.FEATURED,
            'Research': CompetitionType.RESEARCH,
            'Getting Started': CompetitionType.GETTING_STARTED,
            'Playground': CompetitionType.PLAYGROUND,
            'Community': CompetitionType.COMMUNITY,
        }

        # Map Kaggle metric to our enum
        metric_map = {
            'accuracy': EvaluationMetric.ACCURACY,
            'auc': EvaluationMetric.AUC,
            'logloss': EvaluationMetric.LOG_LOSS,
            'rmse': EvaluationMetric.RMSE,
            'mae': EvaluationMetric.MAE,
        }

        # Parse tags - they may be strings or dicts with 'name' key
        raw_tags = data.get('tags', [])
        if raw_tags and isinstance(raw_tags[0], dict):
            tags = frozenset(t.get('name', str(t)) for t in raw_tags if t)
        else:
            tags = frozenset(str(t) for t in raw_tags if t)

        # Parse competition ID - extract slug from ref or URL
        # ref is typically a full URL like https://www.kaggle.com/competitions/slug
        comp_id_raw = data.get('ref', '') or data.get('url', '') or str(data.get('id', ''))
        if 'competitions/' in comp_id_raw:
            # Extract slug from URL
            comp_id = comp_id_raw.split('competitions/')[-1].rstrip('/').split('/')[0]
        elif '/' in comp_id_raw:
            # Extract slug from path
            comp_id = comp_id_raw.rstrip('/').split('/')[-1]
        else:
            comp_id = comp_id_raw

        # Parse prize pool - may be integer, string number, or text like "Swag"
        prize_raw = data.get('reward')
        prize_pool = None
        if prize_raw is not None:
            if isinstance(prize_raw, int):
                prize_pool = prize_raw
            elif isinstance(prize_raw, str):
                # Try to extract numeric value
                match = re.search(r'[\d,]+', prize_raw.replace(',', ''))
                if match:
                    try:
                        prize_pool = int(match.group().replace(',', ''))
                    except ValueError:
                        prize_pool = None

        return Competition(
            id=comp_id,
            title=data.get('title', ''),
            description=data.get('description'),
            competition_type=category_map.get(data.get('category', ''), CompetitionType.COMMUNITY),
            metric=metric_map.get(data.get('evaluationMetric', 'accuracy').lower(), EvaluationMetric.ACCURACY),
            deadline=datetime.fromisoformat(data.get('deadline', '2099-12-31T23:59:59+00:00').replace('Z', '+00:00')),
            prize_pool=prize_pool,
            max_team_size=data.get('maxTeamSize', 1),
            max_daily_submissions=data.get('maxDailySubmissions', 5),
            tags=tags,
            url=data.get('url'),
        )
