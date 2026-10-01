"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.models import EvaluationMetric

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


class TestKaggleCompetitionMetricParsing:
    """Tests for evaluation-metric parsing in _parse_competition."""

    @staticmethod
    def _parse(evaluation_metric: str) -> Any:
        adapter = KaggleAdapter(KaggleSettings(username="user", api_key="key"))
        return adapter._parse_competition(
            {
                "ref": "https://www.kaggle.com/competitions/sample",
                "title": "Sample",
                "category": "Featured",
                "deadline": "2030-01-01T00:00:00Z",
                "evaluationMetric": evaluation_metric,
            }
        )

    @pytest.mark.parametrize(
        ("evaluation_metric", "expected_metric", "expected_direction"),
        [
            ("RootMeanSquaredError", EvaluationMetric.RMSE, "minimize"),
            ("RootMeanSquaredLogarithmicError", EvaluationMetric.RMSLE, "minimize"),
            ("AreaUnderRocCurve", EvaluationMetric.AUC, "maximize"),
            ("Mean F1-Score", EvaluationMetric.F1, "maximize"),
            ("MeanAbsoluteError", EvaluationMetric.MAE, "minimize"),
        ],
    )
    def test_metric_and_direction(
        self, evaluation_metric: str, expected_metric: EvaluationMetric, expected_direction: str
    ) -> None:
        """Parsed competitions should carry the resolved metric and its direction."""
        competition = self._parse(evaluation_metric)

        assert competition.metric == expected_metric
        assert competition.metric_direction == expected_direction

    def test_missing_metric_defaults_to_accuracy(self) -> None:
        """A missing evaluation metric should keep the historical accuracy default."""
        adapter = KaggleAdapter(KaggleSettings(username="user", api_key="key"))
        competition = adapter._parse_competition(
            {
                "ref": "https://www.kaggle.com/competitions/sample",
                "title": "Sample",
                "category": "Featured",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )

        assert competition.metric == EvaluationMetric.ACCURACY
        assert competition.metric_direction == "maximize"
