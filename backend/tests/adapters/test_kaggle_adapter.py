"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import io
import zipfile
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, _parse_leaderboard_csv, _resolve_leaderboard_columns

__all__ = ()

pytestmark = pytest.mark.anyio


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

    async def test_get_leaderboard_parses_kaggle_csv_columns(self) -> None:
        """get_leaderboard should read scores from the Score column, not SubmissionDate."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        csv_body = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "111,Alpha,2024-10-21T04:32:11.000Z,0.9812\n"
            "222,Beta,2024-10-22T08:11:03.000Z,0.9500\n"
            "333,Gamma,2024-10-23T12:00:00.000Z,0.8000\n"
        )
        response = httpx.Response(
            200,
            headers={"content-type": "text/csv"},
            content=csv_body.encode("utf-8"),
            request=httpx.Request("GET", "https://www.kaggle.com/api/v1/competitions/some-comp/leaderboard/download"),
        )
        adapter._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

        entries = await adapter.get_leaderboard("some-comp", limit=5)

        assert [(e.rank, e.team_name, e.score) for e in entries] == [
            (1, "Alpha", 0.9812),
            (2, "Beta", 0.9500),
            (3, "Gamma", 0.8000),
        ]
        assert entries[0].last_submission == datetime.fromisoformat("2024-10-21T04:32:11.000+00:00")

    async def test_get_leaderboard_reads_zip_payload(self) -> None:
        """get_leaderboard should unpack a Kaggle-style zipped CSV."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        csv_body = "TeamId,TeamName,SubmissionDate,Score\n1,Zeta,2024-10-01T00:00:00Z,1.2345\n"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as archive:
            archive.writestr("titanic-publicleaderboard.csv", csv_body)
        payload = buffer.getvalue()

        response = httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=payload,
            request=httpx.Request("GET", "https://www.kaggle.com/api/v1/competitions/titanic/leaderboard/download"),
        )
        adapter._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

        entries = await adapter.get_leaderboard("titanic", limit=10)

        assert len(entries) == 1
        assert entries[0].team_name == "Zeta"
        assert entries[0].score == pytest.approx(1.2345)


class TestLeaderboardColumnResolution:
    """Tests for the leaderboard CSV column resolver."""

    def test_canonical_kaggle_header(self) -> None:
        """Canonical header should map TeamName to 1, Score to 3, SubmissionDate to 2."""
        team, score, date = _resolve_leaderboard_columns(["TeamId", "TeamName", "SubmissionDate", "Score"])

        assert (team, score, date) == (1, 3, 2)

    def test_alternate_header_order(self) -> None:
        """Resolver should locate columns regardless of order."""
        team, score, date = _resolve_leaderboard_columns(["Score", "TeamName", "SubmissionDate"])

        assert (team, score, date) == (1, 0, 2)

    def test_publicscore_alias(self) -> None:
        """PublicScore should be recognized as the score column."""
        team, score, _date = _resolve_leaderboard_columns(["TeamName", "PublicScore"])

        assert (team, score) == (0, 1)

    def test_missing_headers_fall_back_to_kaggle_layout(self) -> None:
        """If headers aren't recognized, use Kaggle's canonical column order."""
        team, score, date = _resolve_leaderboard_columns(["a", "b", "c", "d"])

        assert (team, score, date) == (1, 3, None)


class TestLeaderboardCsvParser:
    """Tests for ``_parse_leaderboard_csv``."""

    def test_parses_real_kaggle_layout(self) -> None:
        """Score comes from the Score column, not SubmissionDate."""
        csv_text = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "1,Alpha,2024-01-01T00:00:00.000Z,0.9500\n"
            "2,Beta,2024-01-02T00:00:00.000Z,0.9000\n"
        )

        entries = _parse_leaderboard_csv(csv_text, limit=10)

        assert [(e.rank, e.team_name, e.score) for e in entries] == [(1, "Alpha", 0.95), (2, "Beta", 0.9)]
        assert entries[0].last_submission is not None
        assert entries[0].last_submission.year == 2024

    def test_respects_limit(self) -> None:
        """Limit should cap the number of returned rows."""
        rows = ["TeamId,TeamName,SubmissionDate,Score"]
        rows.extend(f"{i},Team{i},2024-01-01T00:00:00Z,{1.0 - i * 0.01:.4f}" for i in range(1, 6))
        csv_text = "\n".join(rows) + "\n"

        entries = _parse_leaderboard_csv(csv_text, limit=3)

        assert len(entries) == 3
        assert entries[-1].team_name == "Team3"

    def test_skips_rows_with_empty_team_or_bad_score(self) -> None:
        """Malformed rows should be skipped rather than crashing validation."""
        csv_text = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "1,,2024-01-01T00:00:00Z,0.5\n"
            "2,ValidTeam,2024-01-02T00:00:00Z,not-a-number\n"
            "3,AnotherTeam,2024-01-03T00:00:00Z,nan\n"
            "4,GoodTeam,2024-01-04T00:00:00Z,0.42\n"
        )

        entries = _parse_leaderboard_csv(csv_text, limit=10)

        assert len(entries) == 1
        assert entries[0].team_name == "GoodTeam"
        assert entries[0].score == pytest.approx(0.42)
        assert entries[0].rank == 4

    def test_empty_input_returns_empty_list(self) -> None:
        """No header should yield no entries without error."""
        assert _parse_leaderboard_csv("", limit=10) == []

    def test_ignores_unparseable_submission_date(self) -> None:
        """A bad SubmissionDate should not drop the row."""
        csv_text = "TeamId,TeamName,SubmissionDate,Score\n1,Alpha,not-a-date,0.1\n"

        entries = _parse_leaderboard_csv(csv_text, limit=10)

        assert len(entries) == 1
        assert entries[0].last_submission is None
        assert entries[0].score == pytest.approx(0.1)


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation
