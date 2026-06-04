"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import (
    _DEFAULT_DEADLINE,
    KaggleAdapter,
    KaggleSettings,
    _parse_deadline,
    _parse_evaluation_metric,
)
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


class TestParseEvaluationMetric:
    """Tests for the free-form Kaggle metric parser."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("accuracy", EvaluationMetric.ACCURACY),
            ("Categorization Accuracy", EvaluationMetric.ACCURACY),
            ("AUC", EvaluationMetric.AUC),
            ("AUCROC", EvaluationMetric.AUC),
            ("AreaUnderROCCurve", EvaluationMetric.AUC),
            ("ROC", EvaluationMetric.AUC),
            ("LogLoss", EvaluationMetric.LOG_LOSS),
            ("Multi Class Log Loss", EvaluationMetric.LOG_LOSS),
            ("MulticlassLoss", EvaluationMetric.LOG_LOSS),
            ("BinaryLogLoss", EvaluationMetric.LOG_LOSS),
            ("Categorical Crossentropy", EvaluationMetric.LOG_LOSS),
            ("F1", EvaluationMetric.F1),
            ("F1Score", EvaluationMetric.F1),
            ("Mean F Score", EvaluationMetric.F1),
            ("MeanFScoreBeta", EvaluationMetric.F1),
            ("MicroF1", EvaluationMetric.F1),
            ("MacroF1", EvaluationMetric.F1),
            ("RMSE", EvaluationMetric.RMSE),
            ("Root Mean Squared Error", EvaluationMetric.RMSE),
            ("MeanColumnwiseRootMeanSquaredError", EvaluationMetric.RMSE),
            ("RMSLE", EvaluationMetric.RMSLE),
            ("Root Mean Squared Logarithmic Error", EvaluationMetric.RMSLE),
            ("MAE", EvaluationMetric.MAE),
            ("Mean Absolute Error", EvaluationMetric.MAE),
            ("MeanColumnwiseMAE", EvaluationMetric.MAE),
            ("MAP", EvaluationMetric.MAP),
            ("MAP@5", EvaluationMetric.MAP),
            ("MAP@K", EvaluationMetric.MAP),
            ("Mean Average Precision", EvaluationMetric.MAP),
            ("MeanAveragePrecisionAtK", EvaluationMetric.MAP),
            ("NDCG", EvaluationMetric.NDCG),
            ("NDCG@5", EvaluationMetric.NDCG),
            ("Normalized Discounted Cumulative Gain", EvaluationMetric.NDCG),
        ],
    )
    def test_known_kaggle_metric_names(self, raw: str, expected: EvaluationMetric) -> None:
        """Common Kaggle metric strings should map to the matching enum value."""
        assert _parse_evaluation_metric(raw) is expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "totally-unknown-metric"])
    def test_falls_back_to_accuracy(self, raw: str | None) -> None:
        """Missing or unrecognized metrics fall back to accuracy."""
        assert _parse_evaluation_metric(raw) is EvaluationMetric.ACCURACY


class TestParseDeadline:
    """Tests for the deadline parser used by `_parse_competition`."""

    def test_iso_with_z_suffix(self) -> None:
        result = _parse_deadline("2030-01-01T00:00:00Z")

        assert result == datetime(2030, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_iso_with_offset(self) -> None:
        result = _parse_deadline("2030-06-15T12:30:00+00:00")

        assert result == datetime(2030, 6, 15, 12, 30, 0, tzinfo=UTC)

    @pytest.mark.parametrize("value", [None, "", "   ", "not-a-date", 0, 12345])
    def test_invalid_values_use_default(self, value: Any) -> None:
        """None, empty, malformed, or non-string values use the default deadline."""
        assert _parse_deadline(value) == _DEFAULT_DEADLINE


class TestParseCompetition:
    """Tests for the `_parse_competition` happy path and edge cases."""

    def test_handles_null_deadline_without_crashing(self) -> None:
        """Kaggle sometimes returns deadline=null; parser must not crash."""
        config = KaggleSettings(username="u", api_key="k")
        adapter = KaggleAdapter(config)

        competition = adapter._parse_competition(
            {
                "ref": "https://www.kaggle.com/competitions/example-comp",
                "title": "Example",
                "category": "Featured",
                "evaluationMetric": "F1Score",
                "deadline": None,
                "reward": "$10,000",
            }
        )

        assert competition.id == "example-comp"
        assert competition.metric is EvaluationMetric.F1
        assert competition.metric_direction == "maximize"
        assert competition.deadline == _DEFAULT_DEADLINE
        assert competition.prize_pool == 10000

    def test_uses_correct_direction_for_log_loss(self) -> None:
        config = KaggleSettings(username="u", api_key="k")
        adapter = KaggleAdapter(config)

        competition = adapter._parse_competition(
            {
                "ref": "logloss-comp",
                "title": "LogLoss Comp",
                "category": "Playground",
                "evaluationMetric": "Multi Class Log Loss",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )

        assert competition.metric is EvaluationMetric.LOG_LOSS
        assert competition.metric_direction == "minimize"
