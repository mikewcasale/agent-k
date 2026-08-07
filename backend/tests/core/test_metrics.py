"""Tests for the evaluation metric registry.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.metrics import (
    CLASSIFICATION_METRICS,
    PROBA_METRICS,
    RANKING_METRICS,
    REGRESSION_METRICS,
    direction_for,
    is_classification_metric,
    is_ranking_metric,
    is_regression_metric,
    metric_direction,
    parse_metric,
    uses_probability,
)
from agent_k.core.models import EvaluationMetric

__all__ = ()


class TestParseMetricAliases:
    """Compact aliases and abbreviations should resolve exactly."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("accuracy", EvaluationMetric.ACCURACY),
            ("acc", EvaluationMetric.ACCURACY),
            ("AUC", EvaluationMetric.AUC),
            ("aucroc", EvaluationMetric.AUC),
            ("LogLoss", EvaluationMetric.LOG_LOSS),
            ("multiLogLoss", EvaluationMetric.MULTI_LOG_LOSS),
            ("multiclasslogloss", EvaluationMetric.MULTI_LOG_LOSS),
            ("F1", EvaluationMetric.F1),
            ("macroF1", EvaluationMetric.F1),
            ("BalancedAccuracy", EvaluationMetric.BALANCED_ACCURACY),
            ("MCC", EvaluationMetric.MCC),
            ("QWK", EvaluationMetric.QUADRATIC_KAPPA),
            ("QuadraticWeightedKappa", EvaluationMetric.QUADRATIC_KAPPA),
            ("RMSE", EvaluationMetric.RMSE),
            ("MSE", EvaluationMetric.RMSE),
            ("MAE", EvaluationMetric.MAE),
            ("RMSLE", EvaluationMetric.RMSLE),
            ("MedAE", EvaluationMetric.MEDAE),
            ("R2", EvaluationMetric.R2),
            ("rsquared", EvaluationMetric.R2),
            ("SMAPE", EvaluationMetric.SMAPE),
            ("MAPE", EvaluationMetric.MAPE),
            ("MCRMSE", EvaluationMetric.MCRMSE),
            ("Spearman", EvaluationMetric.SPEARMAN),
            ("PearsonCorrelation", EvaluationMetric.PEARSON),
            ("MAP", EvaluationMetric.MAP),
            ("NDCG", EvaluationMetric.NDCG),
            ("MRR", EvaluationMetric.MRR),
        ],
    )
    def test_alias_resolves(self, raw: str, expected: EvaluationMetric) -> None:
        metric, _ = parse_metric(raw)
        assert metric is expected


class TestParseMetricPatterns:
    """Verbose Kaggle metric names should resolve via pattern search."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Area Under ROC Curve", EvaluationMetric.AUC),
            ("Root Mean Squared Error", EvaluationMetric.RMSE),
            ("Root Mean Squared Logarithmic Error", EvaluationMetric.RMSLE),
            ("Mean Absolute Percentage Error", EvaluationMetric.MAPE),
            ("Symmetric Mean Absolute Percentage Error", EvaluationMetric.SMAPE),
            ("Median Absolute Error", EvaluationMetric.MEDAE),
            ("Matthews Correlation Coefficient", EvaluationMetric.MCC),
            ("Cohen's Kappa (Quadratic)", EvaluationMetric.QUADRATIC_KAPPA),
            ("Mean Columnwise Root Mean Squared Error", EvaluationMetric.MCRMSE),
            ("Multiclass Cross Entropy", EvaluationMetric.MULTI_LOG_LOSS),
            ("Binary Cross Entropy", EvaluationMetric.LOG_LOSS),
            ("Categorization Accuracy", EvaluationMetric.ACCURACY),
            ("Normalized Discounted Cumulative Gain", EvaluationMetric.NDCG),
            ("Mean Reciprocal Rank", EvaluationMetric.MRR),
            ("Mean Average Precision", EvaluationMetric.MAP),
            ("MAP@10", EvaluationMetric.MAP),
            ("Coefficient of Determination", EvaluationMetric.R2),
            ("Spearman rank correlation", EvaluationMetric.SPEARMAN),
        ],
    )
    def test_pattern_resolves(self, raw: str, expected: EvaluationMetric) -> None:
        metric, _ = parse_metric(raw)
        assert metric is expected


class TestParseMetricDirection:
    """parse_metric should return the canonical direction for each metric."""

    @pytest.mark.parametrize(
        ("metric", "direction"),
        [
            (EvaluationMetric.ACCURACY, "maximize"),
            (EvaluationMetric.AUC, "maximize"),
            (EvaluationMetric.LOG_LOSS, "minimize"),
            (EvaluationMetric.MULTI_LOG_LOSS, "minimize"),
            (EvaluationMetric.F1, "maximize"),
            (EvaluationMetric.BALANCED_ACCURACY, "maximize"),
            (EvaluationMetric.MCC, "maximize"),
            (EvaluationMetric.QUADRATIC_KAPPA, "maximize"),
            (EvaluationMetric.RMSE, "minimize"),
            (EvaluationMetric.MAE, "minimize"),
            (EvaluationMetric.RMSLE, "minimize"),
            (EvaluationMetric.MEDAE, "minimize"),
            (EvaluationMetric.SMAPE, "minimize"),
            (EvaluationMetric.MAPE, "minimize"),
            (EvaluationMetric.MCRMSE, "minimize"),
            (EvaluationMetric.R2, "maximize"),
            (EvaluationMetric.SPEARMAN, "maximize"),
            (EvaluationMetric.PEARSON, "maximize"),
            (EvaluationMetric.MAP, "maximize"),
            (EvaluationMetric.NDCG, "maximize"),
            (EvaluationMetric.MRR, "maximize"),
        ],
    )
    def test_direction(self, metric: EvaluationMetric, direction: str) -> None:
        assert direction_for(metric) == direction
        assert metric_direction(metric) == direction


class TestParseMetricFallback:
    """Unknown metric strings should fall back safely."""

    def test_empty_uses_default(self) -> None:
        metric, direction = parse_metric("")
        assert metric is EvaluationMetric.ACCURACY
        assert direction == "maximize"

    def test_none_uses_default(self) -> None:
        metric, direction = parse_metric(None)
        assert metric is EvaluationMetric.ACCURACY
        assert direction == "maximize"

    def test_unknown_uses_default_maximize_direction(self) -> None:
        metric, direction = parse_metric("completely unrelated metric name xyz")
        assert metric is EvaluationMetric.ACCURACY
        assert direction == "maximize"

    def test_default_override(self) -> None:
        metric, direction = parse_metric("unrelated", default=EvaluationMetric.RMSE)
        assert metric is EvaluationMetric.RMSE
        assert direction == "minimize"


class TestClassificationTaxonomy:
    """Classification/regression/ranking taxonomy helpers should agree with the sets."""

    def test_classification_set_disjoint_from_regression(self) -> None:
        assert not CLASSIFICATION_METRICS & REGRESSION_METRICS

    def test_ranking_set_disjoint_from_classification(self) -> None:
        assert not RANKING_METRICS & CLASSIFICATION_METRICS

    def test_ranking_set_disjoint_from_regression(self) -> None:
        assert not RANKING_METRICS & REGRESSION_METRICS

    def test_all_metrics_classified(self) -> None:
        union = CLASSIFICATION_METRICS | REGRESSION_METRICS | RANKING_METRICS
        assert set(EvaluationMetric) == union

    @pytest.mark.parametrize("metric", list(CLASSIFICATION_METRICS))
    def test_is_classification(self, metric: EvaluationMetric) -> None:
        assert is_classification_metric(metric)
        assert not is_regression_metric(metric)
        assert not is_ranking_metric(metric)

    @pytest.mark.parametrize("metric", list(REGRESSION_METRICS))
    def test_is_regression(self, metric: EvaluationMetric) -> None:
        assert is_regression_metric(metric)
        assert not is_classification_metric(metric)
        assert not is_ranking_metric(metric)

    @pytest.mark.parametrize("metric", list(RANKING_METRICS))
    def test_is_ranking(self, metric: EvaluationMetric) -> None:
        assert is_ranking_metric(metric)
        assert not is_classification_metric(metric)
        assert not is_regression_metric(metric)


class TestProbabilityMetrics:
    """Probability metrics must include AUC and all log-loss variants."""

    @pytest.mark.parametrize(
        "metric", [EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS, EvaluationMetric.MULTI_LOG_LOSS]
    )
    def test_uses_probability_true(self, metric: EvaluationMetric) -> None:
        assert uses_probability(metric)
        assert metric in PROBA_METRICS

    @pytest.mark.parametrize(
        "metric",
        [EvaluationMetric.ACCURACY, EvaluationMetric.F1, EvaluationMetric.RMSE, EvaluationMetric.QUADRATIC_KAPPA],
    )
    def test_uses_probability_false(self, metric: EvaluationMetric) -> None:
        assert not uses_probability(metric)
