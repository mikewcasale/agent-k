"""Kaggle platform adapter implementation.

@notice: |
    Kaggle platform adapter implementation.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.adapters.kaggle
    provides:
        - agent_k.adapters.kaggle:KaggleAdapter
        - agent_k.adapters.kaggle:KaggleSettings
    pattern: adapter

@similar:
    - id: agent_k.toolsets.kaggle
        when: "Toolset wrappers over this adapter for agent tools."

@agent-guidance:
    do:
        - "Use agent_k.adapters.kaggle as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import csv
import io
import random
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import quote

import httpx
import logfire
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_k.core.exceptions import (
    AuthenticationError,
    CompetitionNotFoundError,
    CompetitionRulesNotAcceptedError,
    PlatformConnectionError,
    RateLimitError,
    SubmissionError,
)
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry, Submission
from agent_k.core.protocols import PlatformAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ("KaggleAdapter", "KaggleSettings", "SCHEMA_VERSION")

SCHEMA_VERSION: Final[str] = "1.0.0"
_COMPETITION_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"kaggle\.com/competitions/([a-zA-Z0-9-]+)")
# Status codes we treat as transient and retry with exponential backoff.
# 429 is handled separately so we can honor Retry-After. 5xx entries here
# cover Kaggle's usual upstream/edge-cache blips (502/503/504) and generic
# 500 responses that Kaggle briefly emits during deploys.
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({500, 502, 503, 504})
# Upper bound on any single retry sleep. Prevents a hostile or misconfigured
# Retry-After header (e.g. 3600s) from pinning a mission for an hour before
# we even attempt again — we would rather surface the error to the caller.
_MAX_RETRY_SLEEP_SECONDS: Final[float] = 30.0


class KaggleSettings(BaseSettings):
    """Settings for Kaggle adapter.

        Environment variables are prefixed with KAGGLE_.

    @notice: |
        Settings for Kaggle adapter.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: settings
            rationale: "Centralizes Kaggle adapter configuration."
            violations: "Ad-hoc config makes auth inconsistent."
    """

    model_config = SettingsConfigDict(env_prefix="KAGGLE_", env_file=".env", extra="ignore", validate_default=True)
    username: str = Field(..., description="Kaggle API username")
    """Kaggle API username for authentication.

    @source: "KAGGLE_USERNAME environment variable."
    @sensitivity: internal
    @logging: "Avoid logging full username in telemetry."
    """
    api_key: str = Field(..., description="Kaggle API key")
    """Kaggle API key for authentication.

    @source: "KAGGLE_KEY environment variable."
    @sensitivity: secret
    @logging: "MUST NOT be logged."
    """
    base_url: str = Field(default="https://www.kaggle.com/api/v1", description="Base URL for Kaggle API")
    timeout: int = Field(default=30, ge=1, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts for failed requests")
    rate_limit_delay: float = Field(default=1.0, ge=0.0, description="Delay between rate-limited requests (seconds)")
    dry_run: bool = Field(default=False, description="Skip Kaggle submissions when enabled")


@dataclass
class KaggleAdapter(PlatformAdapter):
    """Kaggle platform adapter.

    Implements PlatformAdapter protocol for Kaggle competition platform.
    Uses the Kaggle API for operations and handles authentication,
    rate limiting, and error recovery.

    Example:
        >>> config = KaggleSettings(username="my_username", api_key="my_api_key")
        >>> async with KaggleAdapter(config) as adapter:
        ...     async for comp in adapter.search_competitions(["featured"]):
        ...         print(comp.title)

    @notice: |
        Async adapter for Kaggle competition APIs.

    @dev: |
        Wraps HTTP calls with retry/backoff and normalizes responses.

    @pattern:
        name: adapter
        rationale: "Encapsulates Kaggle-specific API behavior."
        violations: "Direct HTTP use scatters auth/rate-limit logic."

    @implements:
        - agent_k.core.protocols:PlatformAdapter

    @inheritdoc:
        - agent_k.core.protocols:PlatformAdapter

    @collaborators:
        required:
            - httpx:AsyncClient
        injection: dataclass
        lifecycle: "Use as async context manager per mission."

    @concurrency:
        model: asyncio
        safe: false
        reason: "Mutates internal auth and rate-limit state."

    @invariants:
        - "self._client is initialized after __post_init__."
    """

    config: KaggleSettings
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _authenticated: bool = field(default=False, init=False)
    _rate_limit_semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _dry_run: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout, auth=(self.config.username, self.config.api_key)
        )
        self._rate_limit_semaphore = asyncio.Semaphore(5)
        self._dry_run = self.config.dry_run

    async def __aenter__(self) -> KaggleAdapter:
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self._client.aclose()

    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return "kaggle"

    async def authenticate(self) -> bool:
        """Authenticate with Kaggle API."""
        with logfire.span("kaggle.authenticate"):
            try:
                response = await self._request("GET", "/competitions/list", params={"page": 1})
                self._authenticated = response.status_code == 200
                return self._authenticated
            except httpx.HTTPError as exc:
                raise AuthenticationError("kaggle", f"Authentication failed: {exc}") from exc

    async def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        """Search Kaggle competitions."""
        with logfire.span("kaggle.search_competitions", categories=categories):
            page = 1
            while True:
                params: dict[str, Any] = {"page": page}
                if categories:
                    params["category"] = categories[0]  # Kaggle API supports single category
                if keywords:
                    params["search"] = " ".join(keywords)

                response = await self._request("GET", "/competitions/list", params=params)
                data = response.json()

                if not data:
                    break

                for item in data:
                    # Skip if item is not a dict (malformed data)
                    if not isinstance(item, dict):
                        logfire.warning("skipping_malformed_competition", item=str(item)[:100])
                        continue

                    try:
                        competition = self._parse_competition(item)
                    except Exception as exc:
                        logfire.warning("failed_to_parse_competition", error=str(exc))
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
        with logfire.span("kaggle.get_competition", competition_id=competition_id):
            response = await self._request("GET", f"/competitions/data/list/{competition_id}")

            if response.status_code == 404:
                raise CompetitionNotFoundError(competition_id)

            response.raise_for_status()
            # Note: Kaggle API requires additional call for full details
            list_response = await self._request("GET", "/competitions/list", params={"search": competition_id})

            for item in list_response.json():
                try:
                    competition = self._parse_competition(item)
                except Exception as exc:
                    logfire.warning("failed_to_parse_competition", error=str(exc))
                    continue
                if competition.id == competition_id:
                    return competition

            raise CompetitionNotFoundError(competition_id)

    async def get_competition_by_url(self, url: str) -> Competition:
        """Get competition details from a Kaggle competition URL."""
        match = _COMPETITION_URL_PATTERN.search(url)
        if not match:
            raise CompetitionNotFoundError(url)
        return await self.get_competition(match.group(1))

    async def get_leaderboard(self, competition_id: str, *, limit: int = 100) -> list[LeaderboardEntry]:
        """Get competition leaderboard."""
        with logfire.span("kaggle.get_leaderboard", competition_id=competition_id):
            response = await self._request("GET", f"/competitions/{competition_id}/leaderboard/download")
            self._raise_rules_not_accepted(response, competition_id)
            response.raise_for_status()

            entries: list[LeaderboardEntry] = []
            content = response.content
            if response.headers.get("content-type", "").startswith("application/zip") or content[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    csv_name = next((name for name in archive.namelist() if name.lower().endswith(".csv")), None)
                    if not csv_name:
                        return entries
                    csv_text = archive.read(csv_name).decode("utf-8", errors="ignore")
            else:
                csv_text = response.text

            reader = csv.reader(io.StringIO(csv_text))
            if next(reader, None) is None:
                return entries

            for i, row in enumerate(reader, start=1):
                if i > limit:
                    break
                if not row:
                    continue
                team_name = row[1] if len(row) > 1 else "Unknown"
                score = 0.0
                if len(row) > 2:
                    try:
                        score = float(row[2])
                    except ValueError:
                        score = 0.0
                entries.append(LeaderboardEntry(rank=i, team_name=team_name, score=score))

            return entries

    async def submit(self, competition_id: str, file_path: str, message: str = "") -> Submission:
        """Submit solution to Kaggle competition."""
        with logfire.span("kaggle.submit", competition_id=competition_id):
            path = Path(file_path)
            if not path.exists():
                raise SubmissionError(competition_id, f"Submission file not found: {file_path}")
            if self._dry_run:
                logfire.info("kaggle_submit_dry_run", competition_id=competition_id, file_name=path.name)
                return Submission(
                    id="dry-run",
                    competition_id=competition_id,
                    file_name=path.name,
                    status="complete",
                    error_message="dry-run",
                )

            content_length = path.stat().st_size
            last_modified = int(path.stat().st_mtime)
            start_fields = {
                "competitionName": (None, competition_id),
                "contentLength": (None, str(content_length)),
                "lastModifiedEpochSeconds": (None, str(last_modified)),
                "fileName": (None, path.name),
            }
            start_response = await self._request("POST", "/competitions/submission-url", files=start_fields)
            if start_response.status_code != 200:
                raise SubmissionError(competition_id, f"Submission start failed: {start_response.text}")

            start_payload = start_response.json()
            token = start_payload.get("token")
            create_url = start_payload.get("createUrl") or start_payload.get("create_url")
            if not token or not create_url:
                raise SubmissionError(competition_id, f"Invalid submission upload response: {start_payload}")

            payload = path.read_bytes()
            async with httpx.AsyncClient(timeout=self.config.timeout) as upload_client:
                upload_response = await upload_client.put(create_url, content=payload)
            if upload_response.status_code not in {200, 201}:
                raise SubmissionError(competition_id, f"Upload failed: {upload_response.text}")

            submit_fields = {
                "competitionName": (None, competition_id),
                "blobFileTokens": (None, token),
                "submissionDescription": (None, message),
            }
            submit_response = await self._request(
                "POST", f"/competitions/submissions/submit/{competition_id}", files=submit_fields
            )
            if submit_response.status_code != 200:
                raise SubmissionError(competition_id, f"Submission failed: {submit_response.text}")

            result = submit_response.json()
            return Submission(
                id=str(result.get("ref", "unknown")),
                competition_id=competition_id,
                file_name=path.name,
                status="pending",
            )

    async def get_submission_status(self, competition_id: str, submission_id: str) -> Submission:
        """Get submission status from Kaggle."""
        with logfire.span("kaggle.get_submission_status", submission_id=submission_id):
            if self._dry_run:
                return Submission(
                    id=submission_id,
                    competition_id=competition_id,
                    file_name="dry-run",
                    status="complete",
                    error_message="dry-run",
                )
            response = await self._request("GET", f"/competitions/submissions/list/{competition_id}")
            response.raise_for_status()

            for item in response.json():
                if item.get("ref") == submission_id:
                    return Submission(
                        id=submission_id,
                        competition_id=competition_id,
                        file_name=item.get("fileName", ""),
                        status="complete" if item.get("hasPublicScore") else "pending",
                        public_score=item.get("publicScore"),
                    )

            return Submission(
                id=submission_id, competition_id=competition_id, file_name="", status="pending", public_score=None
            )

    async def download_data(self, competition_id: str, destination: str) -> list[str]:
        """Download competition data files."""
        with logfire.span("kaggle.download_data", competition_id=competition_id):
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)

            # List available files
            response = await self._request("GET", f"/competitions/data/list/{competition_id}")
            self._raise_rules_not_accepted(response, competition_id)
            response.raise_for_status()

            payload = response.json()
            files = payload.get("files", []) if isinstance(payload, dict) else payload

            downloaded: list[str] = []
            for file_info in files:
                if isinstance(file_info, str):
                    file_name = file_info
                    file_url = ""
                else:
                    file_name = file_info.get("name") or file_info.get("nameNullable") or ""
                    file_url = file_info.get("url", "")

                if not file_name:
                    continue

                if not file_url:
                    file_url = f"/competitions/data/download/{competition_id}/{quote(file_name)}"

                file_path = dest_path / file_name
                async with self._client.stream("GET", file_url, follow_redirects=True) as file_response:
                    self._raise_rules_not_accepted(file_response, competition_id)
                    file_response.raise_for_status()
                    with file_path.open("wb") as handle:
                        async for chunk in file_response.aiter_bytes():
                            handle.write(chunk)
                downloaded.append(str(file_path))

            return downloaded

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Make a rate-limited request to the Kaggle API with retry/backoff.

        Retries on the failure modes Kaggle exhibits in practice:

        * ``httpx.HTTPError`` — connection resets, DNS blips, timeouts.
        * HTTP 429 — honors ``Retry-After`` when present, capped by
          ``_MAX_RETRY_SLEEP_SECONDS`` so a large header value can't pin the
          mission before we surface the error.
        * Transient 5xx (500/502/503/504) — Kaggle's edge/upstream cache
          returns these for a few seconds around deploys; the previous
          implementation returned them straight to the caller, forcing the
          calling code to interpret raw ``response.status_code`` for retryable
          upstream errors.

        Sleeps use exponential backoff with jitter to avoid synchronizing
        retries across the semaphore-bounded pool of concurrent requests.
        After ``config.max_retries`` attempts, raises the appropriate typed
        exception: ``RateLimitError`` on 429 (preserves the previous contract),
        ``PlatformConnectionError`` on network or 5xx exhaustion.
        """
        attempts = max(1, self.config.max_retries)
        async with self._rate_limit_semaphore:
            for attempt in range(1, attempts + 1):
                delay: float | None = None
                status_code: int | None = None
                retry_after: int | None = None
                error: Exception | None = None

                try:
                    response = await self._client.request(method, path, **kwargs)
                except httpx.HTTPError as exc:
                    error = exc
                    if attempt >= attempts:
                        raise PlatformConnectionError("kaggle", f"Kaggle API error: {exc}") from exc
                    delay = self._retry_delay(attempt)
                else:
                    if response.status_code == 429:
                        status_code = 429
                        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                        if attempt >= attempts:
                            raise RateLimitError("kaggle", "Rate limit exceeded", retry_after=retry_after)
                        delay = self._rate_limit_sleep(attempt, retry_after)
                    elif response.status_code in _RETRYABLE_STATUS_CODES:
                        status_code = response.status_code
                        if attempt >= attempts:
                            raise PlatformConnectionError(
                                "kaggle", f"Kaggle API returned {response.status_code} after {attempts} attempts"
                            )
                        delay = self._retry_delay(attempt)
                    else:
                        return response

                logfire.warning(
                    "kaggle_request_retry",
                    method=method,
                    path=path,
                    attempt=attempt,
                    max_attempts=attempts,
                    delay_seconds=delay,
                    status_code=status_code,
                    retry_after=retry_after,
                    error=str(error) if error else None,
                )
                await asyncio.sleep(delay or 0.0)

            # Defensive: the loop above always returns or raises on the final
            # attempt. This keeps the type checker happy and traces any bug
            # that would otherwise exit the loop silently.
            raise PlatformConnectionError("kaggle", "Max retries exceeded")

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at ``_MAX_RETRY_SLEEP_SECONDS``.

        Uses "full jitter" between ``0.5 * base`` and ``base`` so a burst of
        concurrent retries doesn't synchronize on the same wake-up.
        """
        base = self.config.rate_limit_delay
        if base <= 0:
            return 0.0
        exponential = min(base * (2 ** (attempt - 1)), _MAX_RETRY_SLEEP_SECONDS)
        return random.uniform(exponential * 0.5, exponential)

    def _rate_limit_sleep(self, attempt: int, retry_after: int | None) -> float:
        """Sleep for a 429 retry.

        Prefers the server's ``Retry-After`` hint when present, but bounded so
        an unusually large value can't stall the caller for minutes. Falls
        back to exponential backoff when the header is missing or invalid.
        """
        if retry_after is None or retry_after <= 0:
            return self._retry_delay(attempt)
        return min(float(retry_after), _MAX_RETRY_SLEEP_SECONDS)

    def _raise_rules_not_accepted(self, response: httpx.Response, competition_id: str) -> None:
        if response.status_code != 403:
            return
        text = response.text.lower()
        if "accept" in text and "rules" in text:
            raise CompetitionRulesNotAcceptedError(competition_id)

    def _parse_competition(self, data: dict[str, Any]) -> Competition:
        """Parse Kaggle API response into Competition model."""
        # Map Kaggle category to our enum
        category_map = {
            "Featured": CompetitionType.FEATURED,
            "Research": CompetitionType.RESEARCH,
            "Getting Started": CompetitionType.GETTING_STARTED,
            "Playground": CompetitionType.PLAYGROUND,
            "Community": CompetitionType.COMMUNITY,
        }

        # Map Kaggle metric to our enum
        metric_map = {
            "accuracy": EvaluationMetric.ACCURACY,
            "auc": EvaluationMetric.AUC,
            "logloss": EvaluationMetric.LOG_LOSS,
            "rmse": EvaluationMetric.RMSE,
            "mae": EvaluationMetric.MAE,
            "rmsle": EvaluationMetric.RMSLE,
        }
        metric_raw = str(data.get("evaluationMetric", "accuracy")).strip()
        metric_key = metric_raw.lower()
        metric = metric_map.get(metric_key)
        if metric is None:
            if "logarithmic" in metric_key or "rmsle" in metric_key:
                metric = EvaluationMetric.RMSLE
            elif "mean squared" in metric_key or "rmse" in metric_key:
                metric = EvaluationMetric.RMSE
            elif "mean absolute" in metric_key or "mae" in metric_key:
                metric = EvaluationMetric.MAE
            elif "log loss" in metric_key or "logloss" in metric_key:
                metric = EvaluationMetric.LOG_LOSS
            elif "auc" in metric_key:
                metric = EvaluationMetric.AUC
            else:
                metric = EvaluationMetric.ACCURACY
        metric_direction_map = {
            EvaluationMetric.ACCURACY: "maximize",
            EvaluationMetric.AUC: "maximize",
            EvaluationMetric.F1: "maximize",
            EvaluationMetric.LOG_LOSS: "minimize",
            EvaluationMetric.RMSE: "minimize",
            EvaluationMetric.MAE: "minimize",
            EvaluationMetric.RMSLE: "minimize",
            EvaluationMetric.MAP: "maximize",
            EvaluationMetric.NDCG: "maximize",
        }
        metric_direction = metric_direction_map.get(metric, "maximize")

        # Parse tags - they may be strings or dicts with 'name' key
        raw_tags = data.get("tags", [])
        if raw_tags and isinstance(raw_tags[0], dict):
            tags = frozenset(t.get("name", str(t)) for t in raw_tags if t)
        else:
            tags = frozenset(str(t) for t in raw_tags if t)

        # Parse competition ID - extract slug from ref or URL
        # ref is typically a full URL like https://www.kaggle.com/competitions/slug
        comp_id_raw = data.get("ref", "") or data.get("url", "") or str(data.get("id", ""))
        if "competitions/" in comp_id_raw:
            # Extract slug from URL
            comp_id = comp_id_raw.split("competitions/")[-1].rstrip("/").split("/")[0]
        elif "/" in comp_id_raw:
            # Extract slug from path
            comp_id = comp_id_raw.rstrip("/").split("/")[-1]
        else:
            comp_id = comp_id_raw

        # Parse prize pool - may be integer, string number, or text like "Swag"
        prize_raw = data.get("reward")
        prize_pool = None
        if prize_raw is not None:
            if isinstance(prize_raw, int):
                prize_pool = prize_raw
            elif isinstance(prize_raw, str):
                # Try to extract numeric value
                match = re.search(r"[\d,]+", prize_raw.replace(",", ""))
                if match:
                    try:
                        prize_pool = int(match.group().replace(",", ""))
                    except ValueError:
                        prize_pool = None

        return Competition(
            id=comp_id,
            title=data.get("title", ""),
            description=data.get("description"),
            competition_type=category_map.get(data.get("category", ""), CompetitionType.COMMUNITY),
            metric=metric,
            metric_direction=metric_direction,
            deadline=datetime.fromisoformat(data.get("deadline", "2099-12-31T23:59:59+00:00").replace("Z", "+00:00")),
            prize_pool=prize_pool,
            max_team_size=data.get("maxTeamSize", 1),
            max_daily_submissions=data.get("maxDailySubmissions", 5),
            tags=tags,
            url=data.get("url"),
        )


def _parse_retry_after(raw: str | None) -> int | None:
    """Parse a ``Retry-After`` header into whole seconds.

    Accepts the integer-seconds form Kaggle emits in practice. HTTP-date
    values are ignored (returned as ``None``) so the caller falls back to
    exponential backoff rather than misinterpreting a date as seconds.
    """
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds
