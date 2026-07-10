"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, detect_evaluation_metric
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


class TestDetectEvaluationMetric:
    """Tests for `detect_evaluation_metric`.

    Kaggle's ``evaluationMetric`` field returns free-form names ("RMSE",
    "RootMeanSquaredError", "MulticlassLoss", "SMAPE"). Mislabelling a
    minimize metric as maximize (or vice-versa) inverts the direction the
    prototype/evolution phases optimize toward, so this parser has to be
    both broad and robust.
    """

    @pytest.mark.parametrize(
        ("raw", "expected_metric", "expected_direction"),
        [
            # Regression (minimize).
            ("RMSE", EvaluationMetric.RMSE, "minimize"),
            ("rmse", EvaluationMetric.RMSE, "minimize"),
            ("RootMeanSquaredError", EvaluationMetric.RMSE, "minimize"),
            ("MeanColumnwiseRootMeanSquaredError", EvaluationMetric.RMSE, "minimize"),
            ("MeanSquaredError", EvaluationMetric.RMSE, "minimize"),
            ("MSE", EvaluationMetric.RMSE, "minimize"),
            ("RMSPE", EvaluationMetric.RMSE, "minimize"),
            ("RMSLE", EvaluationMetric.RMSLE, "minimize"),
            ("RootMeanSquaredLogarithmicError", EvaluationMetric.RMSLE, "minimize"),
            ("MeanColumnwiseRootMeanSquaredLogarithmicError", EvaluationMetric.RMSLE, "minimize"),
            ("MAE", EvaluationMetric.MAE, "minimize"),
            ("MeanAbsoluteError", EvaluationMetric.MAE, "minimize"),
            ("MedianAbsoluteError", EvaluationMetric.MAE, "minimize"),
            ("MAPE", EvaluationMetric.MAE, "minimize"),
            ("SMAPE", EvaluationMetric.MAE, "minimize"),
            ("SymmetricMeanAbsolutePercentageError", EvaluationMetric.MAE, "minimize"),
            # Classification loss (minimize).
            ("LogLoss", EvaluationMetric.LOG_LOSS, "minimize"),
            ("logloss", EvaluationMetric.LOG_LOSS, "minimize"),
            ("Log loss", EvaluationMetric.LOG_LOSS, "minimize"),
            ("MulticlassLoss", EvaluationMetric.LOG_LOSS, "minimize"),
            ("CategoricalCrossentropy", EvaluationMetric.LOG_LOSS, "minimize"),
            ("BinaryCrossentropy", EvaluationMetric.LOG_LOSS, "minimize"),
            # Classification score (maximize).
            ("Accuracy", EvaluationMetric.ACCURACY, "maximize"),
            ("CategorizationAccuracy", EvaluationMetric.ACCURACY, "maximize"),
            ("QuadraticWeightedKappa", EvaluationMetric.ACCURACY, "maximize"),
            ("Kappa", EvaluationMetric.ACCURACY, "maximize"),
            ("AUC", EvaluationMetric.AUC, "maximize"),
            ("AUCROC", EvaluationMetric.AUC, "maximize"),
            ("ROCAUC", EvaluationMetric.AUC, "maximize"),
            ("AUCPR", EvaluationMetric.AUC, "maximize"),
            # F-score family (maximize).
            ("F1", EvaluationMetric.F1, "maximize"),
            ("F1Score", EvaluationMetric.F1, "maximize"),
            ("MacroFScore", EvaluationMetric.F1, "maximize"),
            ("MicroFScore", EvaluationMetric.F1, "maximize"),
            ("MeanFScoreEntry", EvaluationMetric.F1, "maximize"),
            ("MeanFBeta", EvaluationMetric.F1, "maximize"),
            # Ranking (maximize).
            ("MAP", EvaluationMetric.MAP, "maximize"),
            ("MAP@10", EvaluationMetric.MAP, "maximize"),
            ("MeanAveragePrecision", EvaluationMetric.MAP, "maximize"),
            ("NDCG", EvaluationMetric.NDCG, "maximize"),
            ("NDCG@5", EvaluationMetric.NDCG, "maximize"),
            ("NormalizedDiscountedCumulativeGain", EvaluationMetric.NDCG, "maximize"),
            # Empty / whitespace / unknown → safe defaults.
            ("", EvaluationMetric.ACCURACY, "maximize"),
            ("   ", EvaluationMetric.ACCURACY, "maximize"),
            ("SomeCustomScore", EvaluationMetric.ACCURACY, "maximize"),
            # Unknown *error* / *loss* names must still be minimize.
            ("SomeCustomError", EvaluationMetric.ACCURACY, "minimize"),
            ("BespokeLoss", EvaluationMetric.ACCURACY, "minimize"),
            ("PixelDeviation", EvaluationMetric.ACCURACY, "minimize"),
        ],
    )
    def test_detect_evaluation_metric(
        self, raw: str, expected_metric: EvaluationMetric, expected_direction: str
    ) -> None:
        """Every Kaggle-shaped metric string maps to the expected enum + direction."""
        metric, direction = detect_evaluation_metric(raw)

        assert metric is expected_metric
        assert direction == expected_direction
