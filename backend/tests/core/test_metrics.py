"""Tests for evaluation metric resolution.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.metrics import (
    METRIC_SPECS,
    describe_metric,
    is_classification_metric,
    normalize_metric_name,
    resolve_metric,
    uses_probability,
)
from agent_k.core.models import EvaluationMetric


class TestNormalizeMetricName:
    """Normalisation of platform metric identifiers."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RootMeanSquaredError", "root mean squared error"),
            ("MAP@3", "map 3"),
            ("AUC", "auc"),
            ("Mean F-Score (Beta=1)", "mean f score beta 1"),
            ("  rmsle  ", "rmsle"),
            ("", ""),
        ],
    )
    def test_normalizes_camel_case_and_punctuation(self, raw: str, expected: str) -> None:
        """Ensure CamelCase identifiers split into matchable words."""
        assert normalize_metric_name(raw) == expected


class TestResolveMetric:
    """Resolution of platform metric names onto the supported taxonomy."""

    @pytest.mark.parametrize(
        ("raw", "metric"),
        [
            ("RootMeanSquaredError", EvaluationMetric.RMSE),
            ("MeanColumnwiseRootMeanSquaredError", EvaluationMetric.RMSE),
            ("rmse", EvaluationMetric.RMSE),
            ("RootMeanSquaredLogarithmicError", EvaluationMetric.RMSLE),
            ("RMSLE", EvaluationMetric.RMSLE),
            ("MeanAbsoluteError", EvaluationMetric.MAE),
            ("SymmetricMeanAbsolutePercentageError", EvaluationMetric.MAE),
            ("ContinuousRankedProbabilityScore", EvaluationMetric.MAE),
            ("PinballLoss", EvaluationMetric.MAE),
            ("MulticlassLoss", EvaluationMetric.LOG_LOSS),
            ("logLoss", EvaluationMetric.LOG_LOSS),
            ("AreaUnderReceiverOperatingCharacteristicCurve", EvaluationMetric.AUC),
            ("AUC", EvaluationMetric.AUC),
            ("NormalizedGini", EvaluationMetric.AUC),
            ("MeanFScoreBeta1", EvaluationMetric.F1),
            ("MacroFScore", EvaluationMetric.F1),
            ("CategorizationAccuracy", EvaluationMetric.ACCURACY),
            ("QuadraticWeightedKappa", EvaluationMetric.ACCURACY),
            ("MatthewsCorrelationCoefficient", EvaluationMetric.ACCURACY),
            ("MeanAveragePrecision", EvaluationMetric.MAP),
            ("MAP@3", EvaluationMetric.MAP),
            ("NDCG@5", EvaluationMetric.NDCG),
            ("R2", EvaluationMetric.R2),
            ("CoefficientOfDetermination", EvaluationMetric.R2),
        ],
    )
    def test_recognizes_platform_metric_names(self, raw: str, metric: EvaluationMetric) -> None:
        """Ensure documented platform metric names map to the right metric."""
        resolved = resolve_metric(raw)
        assert resolved.metric is metric
        assert resolved.recognized is True
        assert resolved.raw == raw

    @pytest.mark.parametrize(
        ("raw", "direction"),
        [
            ("RootMeanSquaredError", "minimize"),
            ("SymmetricMeanAbsolutePercentageError", "minimize"),
            ("MulticlassLoss", "minimize"),
            ("QuadraticWeightedKappa", "maximize"),
            ("AreaUnderReceiverOperatingCharacteristicCurve", "maximize"),
            ("R2", "maximize"),
        ],
    )
    def test_direction_follows_metric_family(self, raw: str, direction: str) -> None:
        """Ensure the optimisation direction matches the resolved metric."""
        assert resolve_metric(raw).direction == direction

    @pytest.mark.parametrize(
        ("raw", "metric", "direction"),
        [
            ("WeightedHausdorffDistance", EvaluationMetric.RMSE, "minimize"),
            ("MeanBestErrorRate", EvaluationMetric.LOG_LOSS, "minimize"),
            ("CustomLeaderboardScore", EvaluationMetric.R2, "maximize"),
            ("MultiLabelClassificationCost", EvaluationMetric.LOG_LOSS, "minimize"),
            ("PerClassRecall", EvaluationMetric.ACCURACY, "maximize"),
        ],
    )
    def test_unrecognized_names_use_lexical_proxy(self, raw: str, metric: EvaluationMetric, direction: str) -> None:
        """Ensure metrics outside the taxonomy keep a sane direction and family."""
        resolved = resolve_metric(raw)
        assert resolved.recognized is False
        assert resolved.metric is metric
        assert resolved.direction == direction

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_blank_names_fall_back_to_accuracy(self, raw: str | None) -> None:
        """Ensure missing metric metadata degrades to the historical default."""
        resolved = resolve_metric(raw)
        assert resolved.metric is EvaluationMetric.ACCURACY
        assert resolved.direction == "maximize"
        assert resolved.recognized is False


class TestMetricSpecs:
    """Behavioural flags derived from the taxonomy."""

    def test_every_metric_has_a_spec(self) -> None:
        """Ensure the taxonomy covers all enum members."""
        assert set(METRIC_SPECS) == set(EvaluationMetric)

    @pytest.mark.parametrize(
        ("metric", "classification", "proba"),
        [
            (EvaluationMetric.ACCURACY, True, False),
            (EvaluationMetric.AUC, True, True),
            (EvaluationMetric.LOG_LOSS, True, True),
            (EvaluationMetric.F1, True, False),
            (EvaluationMetric.RMSE, False, False),
            (EvaluationMetric.R2, False, False),
            (EvaluationMetric.MAP, False, False),
        ],
    )
    def test_family_flags(self, metric: EvaluationMetric, classification: bool, proba: bool) -> None:
        """Ensure classification and probability flags match the metric family."""
        assert is_classification_metric(metric) is classification
        assert uses_probability(metric) is proba


class TestDescribeMetric:
    """Prompt rendering of metrics."""

    def test_appends_platform_name_when_informative(self) -> None:
        """Ensure approximated metrics keep the platform name for agents."""
        rendered = describe_metric(EvaluationMetric.MAE, "minimize", "SymmetricMeanAbsolutePercentageError")
        assert rendered == "mae (minimize), reported by the platform as 'SymmetricMeanAbsolutePercentageError'"

    @pytest.mark.parametrize("metric_name", [None, "", "RMSE", "rmse"])
    def test_omits_redundant_platform_name(self, metric_name: str | None) -> None:
        """Ensure the platform name is dropped when it repeats the metric value."""
        assert describe_metric(EvaluationMetric.RMSE, "minimize", metric_name) == "rmse (minimize)"
