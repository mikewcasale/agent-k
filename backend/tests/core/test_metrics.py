"""Tests for generic evaluation-metric resolution.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.metrics import (
    CLASSIFICATION_METRICS,
    DEFAULT_METRIC,
    METRIC_DIRECTIONS,
    PROBABILITY_METRICS,
    normalize_metric_text,
    resolve_evaluation_metric,
)
from agent_k.core.models import EvaluationMetric

__all__ = ()


class TestNormalizeMetricText:
    """Tests for normalize_metric_text."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RootMeanSquaredError", "root mean squared error"),
            ("Root Mean Squared Error", "root mean squared error"),
            ("root_mean_squared_error", "root mean squared error"),
            ("Log-Loss", "log loss"),
            ("RMSE", "rmse"),
            ("Mean Average Precision @ 5", "mean average precision 5"),
            ("  Accuracy  ", "accuracy"),
            ("", ""),
        ],
    )
    def test_normalizes_spellings(self, raw: str, expected: str) -> None:
        """Different spellings of a metric should normalise identically."""
        assert normalize_metric_text(raw) == expected


class TestResolveEvaluationMetric:
    """Tests for resolve_evaluation_metric."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # camelCase slugs returned by the Kaggle API.
            ("RootMeanSquaredError", EvaluationMetric.RMSE),
            ("RootMeanSquaredLogarithmicError", EvaluationMetric.RMSLE),
            ("MeanAbsoluteError", EvaluationMetric.MAE),
            ("MultiClassLogLoss", EvaluationMetric.LOG_LOSS),
            ("AreaUnderRocCurve", EvaluationMetric.AUC),
            ("CategorizationAccuracy", EvaluationMetric.ACCURACY),
            # Spelled-out metric descriptions.
            ("Area Under the Receiver Operating Characteristic Curve", EvaluationMetric.AUC),
            ("Mean Columnwise Root Mean Squared Error", EvaluationMetric.RMSE),
            ("Mean Absolute Percentage Error", EvaluationMetric.MAE),
            ("Symmetric Mean Absolute Percentage Error", EvaluationMetric.MAE),
            ("Continuous Ranked Probability Score", EvaluationMetric.MAE),
            ("Normalized Discounted Cumulative Gain @ 5", EvaluationMetric.NDCG),
            ("Mean Average Precision @ 3", EvaluationMetric.MAP),
            ("Mean F1-Score", EvaluationMetric.F1),
            ("Macro F1", EvaluationMetric.F1),
            ("Dice Coefficient", EvaluationMetric.F1),
            ("Mean Intersection Over Union", EvaluationMetric.F1),
            ("Matthews Correlation Coefficient", EvaluationMetric.F1),
            ("Quantile Loss", EvaluationMetric.MAE),
            # Acronyms, including parenthesised forms.
            ("RMSE", EvaluationMetric.RMSE),
            ("rmsle", EvaluationMetric.RMSLE),
            ("AUC", EvaluationMetric.AUC),
            ("AUC-ROC", EvaluationMetric.AUC),
            ("Weighted RMSSE", EvaluationMetric.RMSE),
            ("Root Mean Squared Error (RMSE)", EvaluationMetric.RMSE),
            ("NDCG", EvaluationMetric.NDCG),
            ("MAP@5", EvaluationMetric.MAP),
            ("logLoss", EvaluationMetric.LOG_LOSS),
        ],
    )
    def test_resolves_known_metrics(self, raw: str, expected: EvaluationMetric) -> None:
        """Known metric spellings should resolve to the matching metric."""
        resolution = resolve_evaluation_metric(raw)

        assert resolution.metric == expected
        assert resolution.matched is True
        assert resolution.direction == METRIC_DIRECTIONS[expected]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RootMeanSquaredError", "minimize"),
            ("AreaUnderRocCurve", "maximize"),
            ("MultiClassLogLoss", "minimize"),
            ("Mean Average Precision @ 3", "maximize"),
        ],
    )
    def test_direction_follows_metric(self, raw: str, expected: str) -> None:
        """Direction should follow the resolved metric, not the raw text."""
        assert resolve_evaluation_metric(raw).direction == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "Quadratic Weighted Kappa", "Custom Platform Score"])
    def test_falls_back_for_unknown_metrics(self, raw: str | None) -> None:
        """Unrecognised metrics should fall back to the default metric."""
        resolution = resolve_evaluation_metric(raw)

        assert resolution.metric == DEFAULT_METRIC
        assert resolution.matched is False
        assert resolution.direction == METRIC_DIRECTIONS[DEFAULT_METRIC]

    def test_specific_phrases_win_over_generic_ones(self) -> None:
        """Logarithmic error must not be swallowed by the squared-error phrases."""
        assert resolve_evaluation_metric("Mean Squared Logarithmic Error").metric == EvaluationMetric.RMSLE
        assert resolve_evaluation_metric("Mean Squared Error").metric == EvaluationMetric.RMSE


class TestMetricTables:
    """Tests for the shared metric tables."""

    def test_every_metric_has_a_direction(self) -> None:
        """Direction lookup must cover the whole EvaluationMetric enum."""
        assert set(METRIC_DIRECTIONS) == set(EvaluationMetric)

    def test_probability_metrics_are_classification_metrics(self) -> None:
        """Probability-scored metrics are a subset of the classification metrics."""
        assert PROBABILITY_METRICS <= CLASSIFICATION_METRICS

    def test_classification_metrics_maximize_or_minimize_consistently(self) -> None:
        """Probability metrics minimise loss or maximise ranking quality."""
        assert METRIC_DIRECTIONS[EvaluationMetric.LOG_LOSS] == "minimize"
        assert METRIC_DIRECTIONS[EvaluationMetric.AUC] == "maximize"
