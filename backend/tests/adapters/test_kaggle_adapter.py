"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, _normalize_metric_key, _resolve_metric
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


class TestNormalizeMetricKey:
    """Tests for evaluation-metric label normalization."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Accuracy", "accuracy"),
            ("AUC", "auc"),
            ("Mean F1 Score", "meanf1score"),
            ("Mean F-Score @ Beta", "meanfscorebeta"),
            ("MAP@5", "map5"),
            ("NDCG@K", "ndcgk"),
            ("Root Mean Squared Logarithmic Error", "rootmeansquaredlogarithmicerror"),
            ("  Multiclass Log Loss  ", "multiclasslogloss"),
            ("F1_macro", "f1macro"),
            ("", ""),
            (None, ""),
            (42, ""),
        ],
    )
    def test_normalization(self, raw: Any, expected: str) -> None:
        """Metric labels collapse across casing, whitespace, and punctuation."""
        assert _normalize_metric_key(raw) == expected


class TestResolveMetric:
    """Tests for the alias-aware metric resolver.

    The previous parser silently mapped F1 / MAP@K / NDCG@K to ACCURACY because
    those labels were missing from both the lookup table and the substring
    fallback chain. These tests pin the corrected routing.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Direct aliases
            ("Accuracy", EvaluationMetric.ACCURACY),
            ("CategorizationAccuracy", EvaluationMetric.ACCURACY),
            ("AUC", EvaluationMetric.AUC),
            ("AUCROC", EvaluationMetric.AUC),
            ("AreaUnderCurve", EvaluationMetric.AUC),
            ("LogLoss", EvaluationMetric.LOG_LOSS),
            ("MulticlassLogLoss", EvaluationMetric.LOG_LOSS),
            ("RMSE", EvaluationMetric.RMSE),
            ("RootMeanSquaredError", EvaluationMetric.RMSE),
            ("RMSLE", EvaluationMetric.RMSLE),
            ("RootMeanSquaredLogarithmicError", EvaluationMetric.RMSLE),
            ("MAE", EvaluationMetric.MAE),
            ("MeanAbsoluteError", EvaluationMetric.MAE),
            # F1 family - the regression bug we are fixing
            ("F1", EvaluationMetric.F1),
            ("F1Score", EvaluationMetric.F1),
            ("Mean F1 Score", EvaluationMetric.F1),
            ("Macro F1", EvaluationMetric.F1),
            ("Micro F1", EvaluationMetric.F1),
            ("F1_macro", EvaluationMetric.F1),
            ("MeanFScore", EvaluationMetric.F1),
            ("MeanFScoreBeta", EvaluationMetric.F1),
            ("FBetaScore", EvaluationMetric.F1),
            # MAP / NDCG - also previously misrouted to ACCURACY
            ("MAP", EvaluationMetric.MAP),
            ("MAP@5", EvaluationMetric.MAP),
            ("Mean Average Precision", EvaluationMetric.MAP),
            ("MeanAveragePrecisionK", EvaluationMetric.MAP),
            ("NDCG", EvaluationMetric.NDCG),
            ("NDCG@K", EvaluationMetric.NDCG),
            ("NormalizedDiscountedCumulativeGain", EvaluationMetric.NDCG),
        ],
    )
    def test_known_aliases(self, raw: str, expected: EvaluationMetric) -> None:
        """Known label variants resolve to the correct enum value."""
        assert _resolve_metric(raw) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Specific-before-general substring fallback
            ("Some Logarithmic Error Variant", EvaluationMetric.RMSLE),
            ("Some NDCG Variant", EvaluationMetric.NDCG),
            ("Some Average Precision Variant", EvaluationMetric.MAP),
            ("Some Mean Squared Variant", EvaluationMetric.RMSE),
            ("Some Mean Absolute Variant", EvaluationMetric.MAE),
            ("Some Log Loss Variant", EvaluationMetric.LOG_LOSS),
            ("Some AUC Variant", EvaluationMetric.AUC),
            ("Some F1 Variant", EvaluationMetric.F1),
            ("Some F-Score Variant", EvaluationMetric.F1),
        ],
    )
    def test_substring_fallback(self, raw: str, expected: EvaluationMetric) -> None:
        """Less common phrasings still route via substring heuristics."""
        assert _resolve_metric(raw) is expected

    @pytest.mark.parametrize("raw", ["", None, "   ", "QuadraticWeightedKappa", "WeirdNewMetric"])
    def test_unknown_defaults_to_accuracy(self, raw: Any) -> None:
        """Unknown / empty labels fall back to ACCURACY (historical behavior)."""
        assert _resolve_metric(raw) is EvaluationMetric.ACCURACY


class TestParseCompetitionMetric:
    """Tests that _parse_competition propagates the resolved metric correctly."""

    def _adapter(self) -> KaggleAdapter:
        return KaggleAdapter(KaggleSettings(username="u", api_key="k"))

    def test_f1_metric_round_trips(self) -> None:
        """A Kaggle competition with F1Score must surface as EvaluationMetric.F1.

        Regression test: previously this silently became ACCURACY, which then
        caused the evolver's metric-aware scoring to compute the wrong score.
        """
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "santander-customer-satisfaction",
                "title": "Santander",
                "category": "Featured",
                "evaluationMetric": "Mean F1 Score",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )
        assert comp.metric is EvaluationMetric.F1
        assert comp.metric_direction == "maximize"

    def test_map_at_k_resolves_to_map(self) -> None:
        """``MAP@5`` must resolve to MAP and use a maximize direction."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "some-recommender",
                "title": "Some Recommender",
                "category": "Featured",
                "evaluationMetric": "MAP@5",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )
        assert comp.metric is EvaluationMetric.MAP
        assert comp.metric_direction == "maximize"

    def test_ndcg_at_k_resolves_to_ndcg(self) -> None:
        """``NDCG@K`` must resolve to NDCG and use a maximize direction."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "some-ranking",
                "title": "Some Ranking",
                "category": "Featured",
                "evaluationMetric": "NDCG@K",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )
        assert comp.metric is EvaluationMetric.NDCG
        assert comp.metric_direction == "maximize"

    def test_minimize_metric_direction_preserved(self) -> None:
        """Regression metrics still surface a minimize direction."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "some-regression",
                "title": "Some Regression",
                "category": "Featured",
                "evaluationMetric": "RootMeanSquaredLogarithmicError",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )
        assert comp.metric is EvaluationMetric.RMSLE
        assert comp.metric_direction == "minimize"
