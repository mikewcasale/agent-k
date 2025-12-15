"""Kaggle platform adapter implementation.

This adapter provides integration with Kaggle's competition platform
via the official API and MCP server.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import logfire
from pydantic import BaseModel

from ...core.exceptions import (
    AuthenticationError,
    CompetitionNotFoundError,
    PlatformConnectionError,
    RateLimitError,
    SubmissionError,
)
from ...core.models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry, Submission
from ...core.protocols import PlatformAdapter

__all__ = ['KaggleAdapter', 'KaggleConfig']


class KaggleConfig(BaseModel):
    """Configuration for Kaggle adapter."""
    
    username: str
    api_key: str
    base_url: str = 'https://www.kaggle.com/api/v1'
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0


@dataclass
class KaggleAdapter:
    """Kaggle platform adapter.
    
    Implements PlatformAdapter protocol for Kaggle competition platform.
    Uses the Kaggle API for operations and handles authentication,
    rate limiting, and error recovery.
    
    Example:
        >>> config = KaggleConfig(
        ...     username='my_username',
        ...     api_key='my_api_key',
        ... )
        >>> async with KaggleAdapter(config) as adapter:
        ...     async for comp in adapter.search_competitions(['featured']):
        ...         print(comp.title)
    """
    
    config: KaggleConfig
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _authenticated: bool = field(default=False, init=False)
    _rate_limit_semaphore: asyncio.Semaphore = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            auth=(self.config.username, self.config.api_key),
        )
        self._rate_limit_semaphore = asyncio.Semaphore(5)
    
    async def __aenter__(self) -> KaggleAdapter:
        await self.authenticate()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()
    
    @property
    def platform_name(self) -> str:
        return 'kaggle'
    
    async def authenticate(self) -> bool:
        """Authenticate with Kaggle API."""
        with logfire.span('kaggle.authenticate'):
            try:
                response = await self._request('GET', '/competitions/list', params={'page': 1})
                self._authenticated = response.status_code == 200
                return self._authenticated
            except httpx.HTTPError as e:
                raise AuthenticationError(f'Kaggle authentication failed: {e}') from e
    
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
                    except Exception as e:
                        logfire.warning('failed_to_parse_competition', error=str(e))
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
            list_response = await self._request(
                'GET', '/competitions/list',
                params={'search': competition_id},
            )
            
            for item in list_response.json():
                if item.get('ref') == competition_id:
                    return self._parse_competition(item)
            
            raise CompetitionNotFoundError(competition_id)
    
    async def get_leaderboard(
        self,
        competition_id: str,
        *,
        limit: int = 100,
    ) -> list[LeaderboardEntry]:
        """Get competition leaderboard."""
        with logfire.span('kaggle.get_leaderboard', competition_id=competition_id):
            response = await self._request(
                'GET',
                f'/competitions/{competition_id}/leaderboard/download',
            )
            response.raise_for_status()
            
            entries: list[LeaderboardEntry] = []
            # Parse CSV response
            lines = response.text.strip().split('\n')
            for i, line in enumerate(lines[1:limit + 1], start=1):  # Skip header
                parts = line.split(',')
                entries.append(LeaderboardEntry(
                    rank=i,
                    team_name=parts[1] if len(parts) > 1 else 'Unknown',
                    score=float(parts[2]) if len(parts) > 2 else 0.0,
                ))
            
            return entries
    
    async def submit(
        self,
        competition_id: str,
        file_path: str,
        message: str = '',
    ) -> Submission:
        """Submit solution to Kaggle competition."""
        with logfire.span('kaggle.submit', competition_id=competition_id):
            path = Path(file_path)
            if not path.exists():
                raise SubmissionError(f'Submission file not found: {file_path}')
            
            with open(path, 'rb') as f:
                files = {'file': (path.name, f, 'text/csv')}
                data = {'message': message}
                
                response = await self._request(
                    'POST',
                    f'/competitions/submissions/url/{competition_id}',
                    data=data,
                    files=files,
                )
            
            if response.status_code != 200:
                raise SubmissionError(f'Submission failed: {response.text}')
            
            result = response.json()
            return Submission(
                id=result.get('ref', 'unknown'),
                competition_id=competition_id,
                file_name=path.name,
                status='pending',
            )
    
    async def get_submission_status(
        self,
        competition_id: str,
        submission_id: str,
    ) -> Submission:
        """Get submission status from Kaggle."""
        with logfire.span('kaggle.get_submission_status', submission_id=submission_id):
            response = await self._request(
                'GET',
                f'/competitions/submissions/list/{competition_id}',
            )
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
            
            raise SubmissionError(f'Submission not found: {submission_id}')
    
    async def download_data(
        self,
        competition_id: str,
        destination: str,
    ) -> list[str]:
        """Download competition data files."""
        with logfire.span('kaggle.download_data', competition_id=competition_id):
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # List available files
            response = await self._request(
                'GET',
                f'/competitions/data/list/{competition_id}',
            )
            response.raise_for_status()
            
            downloaded: list[str] = []
            for file_info in response.json():
                file_name = file_info.get('name', '')
                file_url = file_info.get('url', '')
                
                if file_url:
                    file_response = await self._client.get(file_url)
                    file_response.raise_for_status()
                    
                    file_path = dest_path / file_name
                    file_path.write_bytes(file_response.content)
                    downloaded.append(str(file_path))
            
            return downloaded
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make rate-limited request to Kaggle API."""
        async with self._rate_limit_semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    response = await self._client.request(method, path, **kwargs)
                    
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        raise RateLimitError(
                            'Kaggle rate limit exceeded',
                            retry_after=retry_after,
                        )
                    
                    return response
                    
                except httpx.HTTPError as e:
                    if attempt == self.config.max_retries - 1:
                        raise PlatformConnectionError(f'Kaggle API error: {e}') from e
                    await asyncio.sleep(self.config.rate_limit_delay * (attempt + 1))
            
            raise PlatformConnectionError('Max retries exceeded')
    
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
                import re
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
            competition_type=category_map.get(
                data.get('category', ''),
                CompetitionType.COMMUNITY,
            ),
            metric=metric_map.get(
                data.get('evaluationMetric', 'accuracy').lower(),
                EvaluationMetric.ACCURACY,
            ),
            deadline=datetime.fromisoformat(
                data.get('deadline', '2099-12-31T23:59:59+00:00').replace('Z', '+00:00')
            ),
            prize_pool=prize_pool,
            max_team_size=data.get('maxTeamSize', 1),
            max_daily_submissions=data.get('maxDailySubmissions', 5),
            tags=tags,
            url=data.get('url'),
        )
