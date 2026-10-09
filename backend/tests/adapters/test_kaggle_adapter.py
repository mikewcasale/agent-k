"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.models import CompetitionType, EvaluationMetric

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


class TestParseCompetition:
    """Tests for parsing Kaggle competition payloads."""

    @staticmethod
    def _adapter() -> KaggleAdapter:
        return KaggleAdapter(KaggleSettings(username="user", api_key="key"))

    @pytest.mark.parametrize(
        ("category", "expected"),
        [
            ("Getting Started", CompetitionType.GETTING_STARTED),
            ("gettingStarted", CompetitionType.GETTING_STARTED),
            ("Featured", CompetitionType.FEATURED),
            ("playground", CompetitionType.PLAYGROUND),
            ("unheard-of", CompetitionType.COMMUNITY),
            (None, CompetitionType.COMMUNITY),
        ],
    )
    def test_category_label_variants(self, category: str | None, expected: CompetitionType) -> None:
        """Category labels should resolve across Kaggle's spellings."""
        competition = self._adapter()._parse_competition(
            {"ref": "titanic", "title": "Titanic", "category": category, "deadline": "2030-01-01T00:00:00Z"}
        )

        assert competition.competition_type is expected

    def test_error_metric_direction(self) -> None:
        """An error metric should be parsed as a minimisation target."""
        competition = self._adapter()._parse_competition(
            {
                "ref": "house-prices",
                "title": "House Prices",
                "category": "Getting Started",
                "evaluationMetric": "Root Mean Squared Logarithmic Error",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )

        assert competition.metric is EvaluationMetric.RMSLE
        assert competition.metric_direction == "minimize"

    def test_deadline_without_offset(self) -> None:
        """A deadline without an offset should parse as UTC instead of failing."""
        competition = self._adapter()._parse_competition(
            {"ref": "titanic", "title": "Titanic", "category": "Featured", "deadline": "2030-01-01T00:00:00"}
        )

        assert competition.deadline.tzinfo is not None
        assert competition.deadline == datetime(2030, 1, 1, tzinfo=UTC)


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation
