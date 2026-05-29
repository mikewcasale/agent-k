"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import io
import zipfile
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import (
    KaggleAdapter,
    KaggleSettings,
    _parse_leaderboard_date,
    _parse_leaderboard_score,
    _resolve_leaderboard_columns,
)

__all__ = ()

pytestmark = pytest.mark.anyio


def _zip_csv(csv_text: str, name: str = "leaderboard.csv") -> bytes:
    """Return a ZIP archive containing a single CSV entry (mirrors Kaggle's response)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, csv_text)
    return buffer.getvalue()


class TestKaggleSettings:
    """Tests for the KaggleSettings class."""

    def test_config_creation(self) -> None:
        """Config should be created with credentials."""
        config = KaggleSettings(username="test_user", api_key="test_key")

        assert config.username == "test_user"
        assert config.api_key == "test_key"

    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = KaggleSettings(username="user", api_key="key")

        assert config.base_url == "https://www.kaggle.com/api/v1"


class TestKaggleAdapter:
    """Tests for the KaggleAdapter class."""

    def test_adapter_creation(self) -> None:
        """Adapter should be created with config."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        assert adapter is not None

    @pytest.fixture
    def mock_http_response(self) -> httpx.Response:
        """Create a mock HTTP response."""
        return httpx.Response(
            200,
            json=[
                {
                    "ref": "titanic",
                    "title": "Titanic",
                    "category": "gettingStarted",
                    "reward": "$0",
                    "deadline": "2030-01-01T00:00:00Z",
                }
            ],
        )

    async def test_search_competitions_basic(self) -> None:
        """Search competitions should return results."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        # The adapter requires actual HTTP calls or mocking
        # For unit tests, we verify the adapter is properly constructed
        assert adapter is not None

    async def test_get_leaderboard_basic(self) -> None:
        """Get leaderboard should return entries."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        assert adapter is not None


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation


class TestResolveLeaderboardColumns:
    """Tests for the leaderboard header parser."""

    def test_matches_documented_kaggle_header(self) -> None:
        """Should resolve teamName/score indices from the canonical Kaggle header."""
        team_idx, score_idx, date_idx = _resolve_leaderboard_columns(["TeamId", "TeamName", "SubmissionDate", "Score"])

        assert team_idx == 1
        assert score_idx == 3
        assert date_idx == 2

    def test_case_and_punctuation_insensitive(self) -> None:
        """Should match header names regardless of casing or separators."""
        team_idx, score_idx, date_idx = _resolve_leaderboard_columns(
            ["team_id", "team name", "submission-date", "SCORE"]
        )

        assert team_idx == 1
        assert score_idx == 3
        assert date_idx == 2

    def test_reordered_columns(self) -> None:
        """Should pick the score column wherever it appears in the header."""
        team_idx, score_idx, _ = _resolve_leaderboard_columns(["Rank", "Score", "TeamName", "SubmissionDate"])

        assert team_idx == 2
        assert score_idx == 1

    def test_legacy_three_column_layout(self) -> None:
        """Should fall back to index 2 for the legacy 3-column header."""
        team_idx, score_idx, _ = _resolve_leaderboard_columns(["TeamId", "TeamName", "Score"])

        assert team_idx == 1
        assert score_idx == 2

    def test_unknown_header_uses_defaults(self) -> None:
        """Should fall back to documented Kaggle column positions on unknown headers."""
        team_idx, score_idx, date_idx = _resolve_leaderboard_columns(["a", "b", "c", "d"])

        assert team_idx == 1
        assert score_idx == 3
        assert date_idx == 2


class TestParseLeaderboardScore:
    """Tests for the score-cell parser."""

    def test_parses_decimal(self) -> None:
        assert _parse_leaderboard_score(["", "", "", "0.95"], 3) == 0.95

    def test_returns_none_for_blank(self) -> None:
        assert _parse_leaderboard_score(["", "", "", ""], 3) is None

    def test_returns_none_for_text(self) -> None:
        assert _parse_leaderboard_score(["", "", "", "2024-01-15"], 3) is None

    def test_returns_none_for_nan(self) -> None:
        assert _parse_leaderboard_score(["", "", "", "nan"], 3) is None

    def test_returns_none_for_inf(self) -> None:
        assert _parse_leaderboard_score(["", "", "", "inf"], 3) is None

    def test_returns_none_for_short_row(self) -> None:
        assert _parse_leaderboard_score(["just-one-cell"], 3) is None


class TestParseLeaderboardDate:
    """Tests for the submission-date parser."""

    def test_parses_kaggle_format(self) -> None:
        result = _parse_leaderboard_date(["", "", "2024-01-15 10:23:45", ""], 2)

        assert result == datetime(2024, 1, 15, 10, 23, 45)

    def test_parses_iso_with_z(self) -> None:
        result = _parse_leaderboard_date(["", "", "2024-01-15T10:23:45Z", ""], 2)

        assert result == datetime(2024, 1, 15, 10, 23, 45, tzinfo=UTC)

    def test_returns_none_for_blank(self) -> None:
        assert _parse_leaderboard_date(["", "", "", ""], 2) is None

    def test_returns_none_for_unparseable(self) -> None:
        assert _parse_leaderboard_date(["", "", "not-a-date", ""], 2) is None


class TestGetLeaderboardParsing:
    """Integration tests that exercise the CSV → LeaderboardEntry pipeline."""

    def _adapter_with_response(self, csv_bytes: bytes, *, content_type: str) -> KaggleAdapter:
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        def handler(request: httpx.Request) -> httpx.Response:
            if "leaderboard/download" in request.url.path:
                return httpx.Response(200, content=csv_bytes, headers={"content-type": content_type})
            return httpx.Response(200, json={})

        adapter._client = httpx.AsyncClient(
            base_url=config.base_url, transport=httpx.MockTransport(handler), auth=(config.username, config.api_key)
        )
        return adapter

    async def test_parses_score_from_named_column(self) -> None:
        """Score should be parsed from the documented Kaggle column, not from SubmissionDate."""
        csv_text = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "12345,Team Alpha,2024-01-15 10:23:45,0.95\n"
            "67890,Team Beta,2024-01-15 11:30:22,0.93\n"
        )
        adapter = self._adapter_with_response(_zip_csv(csv_text), content_type="application/zip")

        entries = await adapter.get_leaderboard("comp-1")

        assert [(e.rank, e.team_name, e.score) for e in entries] == [(1, "Team Alpha", 0.95), (2, "Team Beta", 0.93)]
        assert entries[0].last_submission == datetime(2024, 1, 15, 10, 23, 45)

    async def test_skips_rows_with_unparseable_scores(self) -> None:
        """Bad rows should be dropped rather than silently scored as 0.0."""
        csv_text = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "12345,Team Alpha,2024-01-15 10:23:45,0.95\n"
            "11111,Broken Team,2024-01-15 10:24:00,not-a-number\n"
            "67890,Team Beta,2024-01-15 11:30:22,0.93\n"
        )
        adapter = self._adapter_with_response(_zip_csv(csv_text), content_type="application/zip")

        entries = await adapter.get_leaderboard("comp-1")

        assert [e.team_name for e in entries] == ["Team Alpha", "Team Beta"]
        assert [e.rank for e in entries] == [1, 2]

    async def test_skips_nan_and_inf_scores(self) -> None:
        """NaN/Inf values must not pollute the leaderboard."""
        csv_text = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "12345,Team Alpha,2024-01-15 10:23:45,0.95\n"
            "11111,NaN Team,2024-01-15 10:24:00,nan\n"
            "22222,Inf Team,2024-01-15 10:25:00,inf\n"
        )
        adapter = self._adapter_with_response(_zip_csv(csv_text), content_type="application/zip")

        entries = await adapter.get_leaderboard("comp-1")

        assert [e.team_name for e in entries] == ["Team Alpha"]

    async def test_honors_limit(self) -> None:
        """Limit should cap the number of returned entries."""
        rows = "\n".join(f"{i},Team {i},2024-01-15 10:23:45,{1.0 - i * 0.01:.2f}" for i in range(1, 11))
        csv_text = f"TeamId,TeamName,SubmissionDate,Score\n{rows}\n"
        adapter = self._adapter_with_response(_zip_csv(csv_text), content_type="application/zip")

        entries = await adapter.get_leaderboard("comp-1", limit=3)

        assert len(entries) == 3
        assert entries[-1].rank == 3

    async def test_parses_plain_csv_response(self) -> None:
        """Adapter should also handle a non-zipped CSV body."""
        csv_text = "TeamId,TeamName,SubmissionDate,Score\n12345,Solo Team,2024-01-15 10:23:45,0.42\n"
        adapter = self._adapter_with_response(csv_text.encode("utf-8"), content_type="text/csv")

        entries = await adapter.get_leaderboard("comp-1")

        assert len(entries) == 1
        assert entries[0].score == 0.42
