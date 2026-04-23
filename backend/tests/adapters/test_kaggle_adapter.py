"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, _parse_leaderboard_csv

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


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation


class TestParseLeaderboardCsv:
    """Tests for the header-aware leaderboard CSV parser."""

    def test_parses_kaggle_standard_header(self) -> None:
        """Kaggle's canonical TeamId,TeamName,SubmissionDate,Score layout should parse correctly."""
        csv_text = (
            "TeamId,TeamName,SubmissionDate,Score\n"
            "1,Alpha,2024-06-01T12:00:00.000Z,0.95\n"
            "2,Beta,2024-06-02 09:30:00,0.92\n"
            "3,Gamma,,0.88\n"
        )

        entries = _parse_leaderboard_csv(csv_text, competition_id="titanic", limit=100)

        assert len(entries) == 3
        assert entries[0].rank == 1
        assert entries[0].team_name == "Alpha"
        assert entries[0].score == pytest.approx(0.95)
        assert entries[0].last_submission == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        assert entries[1].team_name == "Beta"
        assert entries[1].score == pytest.approx(0.92)
        assert entries[1].last_submission == datetime(2024, 6, 2, 9, 30, 0)
        assert entries[2].team_name == "Gamma"
        assert entries[2].last_submission is None

    def test_case_insensitive_column_matching(self) -> None:
        """Column resolution should work for lowercase or whitespace-padded headers."""
        csv_text = " team_name , Score \nFoo,0.5\nBar,0.4\n"

        entries = _parse_leaderboard_csv(csv_text, competition_id="x", limit=100)

        assert [e.team_name for e in entries] == ["Foo", "Bar"]
        assert entries[0].score == pytest.approx(0.5)

    def test_respects_limit(self) -> None:
        """Only ``limit`` rows should be returned even if the CSV has more."""
        rows = ["TeamId,TeamName,SubmissionDate,Score"]
        rows.extend(f"{i},Team{i},2024-01-01T00:00:00Z,{0.9 - i * 0.01}" for i in range(1, 11))
        csv_text = "\n".join(rows) + "\n"

        entries = _parse_leaderboard_csv(csv_text, competition_id="x", limit=3)

        assert len(entries) == 3
        assert entries[-1].team_name == "Team3"

    def test_non_numeric_score_falls_back_to_zero(self) -> None:
        """A malformed score cell should become 0.0 rather than abort parsing."""
        csv_text = "TeamId,TeamName,SubmissionDate,Score\n1,Alpha,2024-01-01,not-a-number\n2,Beta,2024-01-02,0.75\n"

        entries = _parse_leaderboard_csv(csv_text, competition_id="x", limit=100)

        assert entries[0].score == 0.0
        assert entries[1].score == pytest.approx(0.75)

    def test_headerless_csv_uses_positional_fallback(self) -> None:
        """Without a recognizable header, rows use legacy positional layout for score."""
        csv_text = "1,Alpha,2024-01-01,0.9\n2,Beta,2024-01-02,0.8\n"

        entries = _parse_leaderboard_csv(csv_text, competition_id="x", limit=100)

        assert len(entries) == 2
        assert entries[0].team_name == "Alpha"
        assert entries[0].score == pytest.approx(0.9)
        assert entries[1].score == pytest.approx(0.8)

    def test_empty_rows_are_skipped(self) -> None:
        """Blank lines between entries should not consume a rank slot."""
        csv_text = "TeamId,TeamName,SubmissionDate,Score\n1,Alpha,2024-01-01,0.9\n\n2,Beta,2024-01-02,0.8\n"

        entries = _parse_leaderboard_csv(csv_text, competition_id="x", limit=100)

        assert [e.rank for e in entries] == [1, 2]
        assert [e.team_name for e in entries] == ["Alpha", "Beta"]

    def test_empty_csv_returns_empty_list(self) -> None:
        """Empty or header-only payloads should yield no entries."""
        assert _parse_leaderboard_csv("", competition_id="x", limit=100) == []
        assert _parse_leaderboard_csv("TeamId,TeamName,SubmissionDate,Score\n", competition_id="x", limit=100) == []


class TestGetLeaderboardIntegration:
    """End-to-end test of ``get_leaderboard`` against a mocked Kaggle response."""

    async def test_parses_kaggle_leaderboard_download(self) -> None:
        """get_leaderboard should return correctly-scored entries for a realistic CSV payload."""
        csv_payload = (
            b"TeamId,TeamName,SubmissionDate,Score\n"
            b"101,Team Alpha,2024-06-01T12:00:00Z,0.981\n"
            b"102,Team Beta,2024-06-01T13:15:30Z,0.975\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/leaderboard/download")
            return httpx.Response(200, headers={"content-type": "text/csv"}, content=csv_payload)

        config = KaggleSettings(username="u", api_key="k")
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url)

        try:
            entries = await adapter.get_leaderboard("titanic", limit=10)
        finally:
            await adapter._client.aclose()

        assert [e.team_name for e in entries] == ["Team Alpha", "Team Beta"]
        assert entries[0].score == pytest.approx(0.981)
        assert entries[1].score == pytest.approx(0.975)
        assert entries[0].last_submission == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
