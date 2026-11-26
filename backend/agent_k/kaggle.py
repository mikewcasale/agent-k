"""Kaggle HTTP integration with explicit authentication and bounded downloads.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import csv
import io
import re
import zipfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import (
    AuthenticationError,
    CompetitionNotFoundError,
    CompetitionRulesNotAcceptedError,
    RateLimitError,
    SubmissionError,
)
from .models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry, Submission

__all__ = ("KaggleAdapter", "KaggleSettings", "parse_metric", "validate_slug")


class KaggleSettings(BaseSettings):
    """Credentials read only when constructing a worker adapter."""

    model_config = SettingsConfigDict(env_prefix="KAGGLE_", extra="ignore", populate_by_name=True)
    username: str
    api_key: SecretStr = Field(validation_alias=AliasChoices("KAGGLE_KEY", "KAGGLE_API_KEY"))
    base_url: str = "https://www.kaggle.com/api/v1"
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_download_bytes: int = Field(default=20_000_000_000, ge=1)


class _CompetitionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ref: str
    title: str
    description: str | None = None
    category: str = "Community"
    evaluationMetric: str
    deadline: datetime
    reward: str | int | None = None
    maxTeamSize: int = 1
    maxDailySubmissions: int = 5
    tags: list[str | _TagPayload] = Field(default_factory=list)
    url: str | None = None


class _TagPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str


class _FilePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(validation_alias=AliasChoices("name", "nameNullable"))


class _FileListing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    files: list[_FilePayload]


class _UploadPayload(BaseModel):
    token: str
    create_url: str = Field(validation_alias=AliasChoices("createUrl", "create_url"))


class _SubmissionReceipt(BaseModel):
    ref: str | int


class _SubmissionPayload(BaseModel):
    ref: str | int
    fileName: str
    hasPublicScore: bool = False
    publicScore: float | None = None
    privateScore: float | None = None
    errorDescription: str | None = None


@dataclass
class KaggleAdapter:
    """Mission-scoped adapter using an injected HTTP client.

    The caller owns and closes the client. Non-idempotent submission requests are
    never retried automatically; an uncertain receipt requires reconciliation.
    """

    config: KaggleSettings
    client: httpx.AsyncClient

    @property
    def platform_name(self) -> str:
        """Return the external platform identifier."""
        return "kaggle"

    async def authenticate(self) -> bool:
        """Check credentials and fail immediately on rejected authentication."""
        response = await self._request("GET", "/competitions/list")
        if response.status_code in {401, 403}:
            raise AuthenticationError("kaggle")
        response.raise_for_status()
        return True

    async def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        """Yield supported competitions matching explicit discovery criteria."""
        response = await self._request("GET", "/competitions/list")
        response.raise_for_status()
        for item in response.json():
            try:
                competition = _parse_competition(_CompetitionPayload.model_validate(item))
            except ValueError:
                continue
            if active_only and not competition.is_active:
                continue
            if categories and competition.competition_type.value not in categories:
                continue
            if min_prize is not None and (competition.prize_pool or 0) < min_prize:
                continue
            if keywords and not any(word.lower() in competition.title.lower() for word in keywords):
                continue
            yield competition

    async def get_competition(self, competition_id: str) -> Competition:
        """Fetch exact metadata; unsupported or unknown metrics fail closed."""
        slug = validate_slug(competition_id)
        response = await self._request("GET", f"/competitions/list?search={quote(slug)}")
        response.raise_for_status()
        for item in response.json():
            payload = _CompetitionPayload.model_validate(item)
            if validate_slug(payload.ref) == slug:
                return _parse_competition(payload)
        raise CompetitionNotFoundError(slug)

    async def get_competition_by_url(self, url: str) -> Competition:
        """Resolve an exact Kaggle competition URL."""
        return await self.get_competition(validate_slug(url))

    async def get_leaderboard(
        self,
        competition_id: str,
        *,
        limit: int = 100,
    ) -> list[LeaderboardEntry]:
        """Read leaderboard values without inventing missing scores or ranks."""
        slug = validate_slug(competition_id)
        response = await self._request("GET", f"/competitions/{slug}/leaderboard/download")
        self._check_rules(response, slug)
        response.raise_for_status()
        content = response.content
        if zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = [name for name in archive.namelist() if name.endswith(".csv")]
                if len(names) != 1:
                    raise ValueError("Leaderboard archive must contain exactly one CSV")
                info = archive.getinfo(names[0])
                if info.file_size > 50_000_000:
                    raise ValueError("Leaderboard archive exceeds size limit")
                content = archive.read(info)
        rows = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        entries: list[LeaderboardEntry] = []
        for row in rows:
            if len(entries) >= limit:
                break
            entries.append(
                LeaderboardEntry(
                    rank=int(row["Rank"]),
                    team_name=row["TeamName"],
                    score=float(row["Score"]),
                ),
            )
        return entries

    async def download_data(self, competition_id: str, destination: str) -> list[str]:
        """Download files beneath a private destination using atomic replacement."""
        slug = validate_slug(competition_id)
        root = Path(destination).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        response = await self._request("GET", f"/competitions/data/list/{slug}")
        self._check_rules(response, slug)
        response.raise_for_status()
        raw = response.json()
        listing = _FileListing.model_validate({"files": raw} if isinstance(raw, list) else raw)
        downloaded: list[str] = []
        total_bytes = 0
        for item in listing.files:
            path = (root / item.name).resolve()
            if not path.is_relative_to(root) or path == root or Path(item.name).is_absolute():
                raise ValueError("Download name escapes dataset directory")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = path.with_suffix(path.suffix + ".partial")
            url = f"{self.config.base_url}/competitions/data/download/{slug}/{quote(item.name)}"
            try:
                async with self.client.stream(
                    "GET",
                    url,
                    auth=(self.config.username, self.config.api_key.get_secret_value()),
                    timeout=self.config.timeout,
                    follow_redirects=True,
                ) as stream:
                    stream.raise_for_status()
                    with temporary.open("xb") as handle:
                        temporary.chmod(0o600)
                        async for chunk in stream.aiter_bytes():
                            total_bytes += len(chunk)
                            if total_bytes > self.config.max_download_bytes:
                                raise ValueError("Download exceeds configured byte limit")
                            handle.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            downloaded.append(str(path))
        return downloaded

    async def submit(
        self,
        competition_id: str,
        file_path: str,
        message: str = "",
    ) -> Submission:
        """Upload one explicitly authorized submission, without automatic retries."""
        slug = validate_slug(competition_id)
        path = Path(file_path)
        response = await self._request(
            "POST",
            "/competitions/submission-url",
            fields={
                "competitionName": slug,
                "contentLength": str(path.stat().st_size),
                "lastModifiedEpochSeconds": str(int(path.stat().st_mtime)),
                "fileName": path.name,
            },
        )
        response.raise_for_status()
        upload = _UploadPayload.model_validate(response.json())
        parsed = urlparse(upload.create_url)
        if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com":
            raise SubmissionError(slug, "Untrusted submission upload endpoint")
        uploaded = await self.client.put(
            upload.create_url,
            content=path.read_bytes(),
            auth=httpx.Auth(),
            timeout=self.config.timeout,
        )
        uploaded.raise_for_status()
        receipt = await self._request(
            "POST",
            f"/competitions/submissions/submit/{slug}",
            fields={
                "competitionName": slug,
                "blobFileTokens": upload.token,
                "submissionDescription": message,
            },
        )
        receipt.raise_for_status()
        reference = _SubmissionReceipt.model_validate(receipt.json())
        return Submission(id=str(reference.ref), competition_id=slug, file_name=path.name)

    async def get_submission_status(self, competition_id: str, submission_id: str) -> Submission:
        """Return only scores actually reported by Kaggle."""
        slug = validate_slug(competition_id)
        response = await self._request("GET", f"/competitions/submissions/list/{slug}")
        response.raise_for_status()
        for raw in response.json():
            item = _SubmissionPayload.model_validate(raw)
            if str(item.ref) != submission_id:
                continue
            status: Literal["pending", "complete", "error"] = "pending"
            if item.errorDescription:
                status = "error"
            elif item.hasPublicScore:
                status = "complete"
            return Submission(
                id=submission_id,
                competition_id=slug,
                file_name=item.fileName,
                status=status,
                public_score=item.publicScore,
                private_score=item.privateScore,
                error_message=item.errorDescription,
            )
        raise SubmissionError(slug, "Submission receipt not found", submission_id=submission_id)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        fields: dict[str, str] | None = None,
    ) -> httpx.Response:
        attempts = self.config.max_retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                response = await self.client.request(
                    method,
                    self.config.base_url + path,
                    auth=(self.config.username, self.config.api_key.get_secret_value()),
                    files={key: (None, value) for key, value in fields.items()} if fields else None,
                    timeout=self.config.timeout,
                )
            except httpx.TransportError:
                if attempt + 1 == attempts:
                    raise
                await asyncio.sleep(min(2**attempt, 30))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt + 1 == attempts:
                if response.status_code == 429:
                    raise RateLimitError("kaggle", "Kaggle rate limit exhausted")
                return response
            raw_delay = response.headers.get("Retry-After", str(2**attempt))
            delay = min(float(raw_delay), 60) if raw_delay.isdigit() else min(2**attempt, 30)
            await asyncio.sleep(max(0, delay))
        raise RuntimeError("HTTP retry loop terminated without a response")

    @staticmethod
    def _check_rules(response: httpx.Response, slug: str) -> None:
        if response.status_code == 403 and "rules" in response.text.lower():
            raise CompetitionRulesNotAcceptedError(slug)


def validate_slug(value: str) -> str:
    """Accept a competition slug or canonical Kaggle URL, rejecting other paths."""
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError("Expected a Kaggle competition URL")
        match = re.fullmatch(r"/competitions/([A-Za-z0-9-]+)/?", parsed.path)
        if match is None:
            raise ValueError("Invalid Kaggle competition URL")
        value = match.group(1)
    if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", value):
        raise ValueError("Invalid competition slug")
    return value


def parse_metric(value: str) -> EvaluationMetric:
    """Normalize known metric names without an accuracy fallback."""
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    aliases = {
        "accuracy": EvaluationMetric.ACCURACY,
        "categorizationaccuracy": EvaluationMetric.ACCURACY,
        "auc": EvaluationMetric.AUC,
        "rocauc": EvaluationMetric.AUC,
        "logloss": EvaluationMetric.LOG_LOSS,
        "logarithmicloss": EvaluationMetric.LOG_LOSS,
        "f1": EvaluationMetric.F1,
        "f1score": EvaluationMetric.F1,
        "rmse": EvaluationMetric.RMSE,
        "rootmeansquarederror": EvaluationMetric.RMSE,
        "mae": EvaluationMetric.MAE,
        "meanabsoluteerror": EvaluationMetric.MAE,
        "rmsle": EvaluationMetric.RMSLE,
        "rootmeansquaredlogarithmicerror": EvaluationMetric.RMSLE,
    }
    if key not in aliases:
        raise ValueError(f"Unsupported competition metric: {value}")
    return aliases[key]


def _parse_competition(payload: _CompetitionPayload) -> Competition:
    metric = parse_metric(payload.evaluationMetric)
    category = payload.category.lower().replace(" ", "_")
    reward = str(payload.reward or "")
    prize = int(re.sub(r"[^0-9]", "", reward)) if re.search(r"\d", reward) else None
    tags = frozenset(tag if isinstance(tag, str) else tag.name for tag in payload.tags)
    return Competition(
        id=validate_slug(payload.ref),
        title=payload.title,
        description=payload.description,
        competition_type=CompetitionType(category),
        metric=metric,
        metric_direction="maximize"
        if metric
        in {
            EvaluationMetric.ACCURACY,
            EvaluationMetric.AUC,
            EvaluationMetric.F1,
        }
        else "minimize",
        deadline=payload.deadline,
        prize_pool=prize,
        max_team_size=payload.maxTeamSize,
        max_daily_submissions=payload.maxDailySubmissions,
        tags=tags,
        url=payload.url,
    )
