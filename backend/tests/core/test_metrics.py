"""Tests for evaluation metric resolution.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.metrics import (
    CLASSIFICATION_METRICS,
    METRIC_DIRECTIONS,
    PROBABILITY_METRICS,
    RANKING_METRICS,
    REGRESSION_METRICS,
    is_classification_metric,
    metric_direction_for,
    resolve_evaluation_metric,
    uses_probability_output,
)
from agent_k.core.models import EvaluationMetric


def test_every_metric_has_a_direction() -> None:
    """Ensure no enum member can fall back to a default direction."""
    assert set(METRIC_DIRECTIONS) == set(EvaluationMetric)


def test_metric_families_partition_the_enum() -> None:
    """Ensure classification, regression, and ranking families cover the enum."""
    families = CLASSIFICATION_METRICS | REGRESSION_METRICS | RANKING_METRICS
    assert families == set(EvaluationMetric)
    assert not CLASSIFICATION_METRICS & REGRESSION_METRICS
    assert not REGRESSION_METRICS & RANKING_METRICS
    assert PROBABILITY_METRICS <= CLASSIFICATION_METRICS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Accuracy", EvaluationMetric.ACCURACY),
        ("Categorization Accuracy", EvaluationMetric.ACCURACY),
        ("Quadratic Weighted Kappa", EvaluationMetric.ACCURACY),
        ("Matthews Correlation Coefficient", EvaluationMetric.ACCURACY),
        ("AUC", EvaluationMetric.AUC),
        ("Area Under Receiver Operating Characteristic Curve", EvaluationMetric.AUC),
        ("Mean Columnwise AUC", EvaluationMetric.AUC),
        ("Normalized Gini Coefficient", EvaluationMetric.AUC),
        ("LogLoss", EvaluationMetric.LOG_LOSS),
        ("Multi Class Log Loss", EvaluationMetric.LOG_LOSS),
        ("Multiclass Logarithmic Loss", EvaluationMetric.LOG_LOSS),
        ("Mean Columnwise Log Loss", EvaluationMetric.LOG_LOSS),
        ("Macro F1 Score", EvaluationMetric.F1),
        ("Mean F-Score", EvaluationMetric.F1),
        ("Dice Coefficient", EvaluationMetric.F1),
        ("Intersection Over Union", EvaluationMetric.F1),
        ("RMSE", EvaluationMetric.RMSE),
        ("Root Mean Squared Error", EvaluationMetric.RMSE),
        ("Mean Columnwise Root Mean Squared Error", EvaluationMetric.RMSE),
        ("Weighted Root Mean Squared Scaled Error (WRMSSE)", EvaluationMetric.RMSE),
        ("Continuous Ranked Probability Score", EvaluationMetric.RMSE),
        ("MAE", EvaluationMetric.MAE),
        ("Mean Absolute Error", EvaluationMetric.MAE),
        ("Median Absolute Error", EvaluationMetric.MAE),
        ("Symmetric Mean Absolute Percentage Error", EvaluationMetric.MAE),
        ("RMSLE", EvaluationMetric.RMSLE),
        ("Root Mean Squared Logarithmic Error", EvaluationMetric.RMSLE),
        ("R2", EvaluationMetric.R2),
        ("Coefficient of Determination", EvaluationMetric.R2),
        ("Spearman Rank Correlation", EvaluationMetric.R2),
        ("MAP@3", EvaluationMetric.MAP),
        ("Mean Average Precision", EvaluationMetric.MAP),
        ("NDCG@5", EvaluationMetric.NDCG),
        ("Normalized Discounted Cumulative Gain", EvaluationMetric.NDCG),
    ],
)
def test_resolve_known_metric_names(raw: str, expected: EvaluationMetric) -> None:
    """Ensure real platform metric names resolve to the right metric."""
    resolution = resolve_evaluation_metric(raw)
    assert resolution.metric == expected
    assert resolution.matched is True
    assert resolution.raw == raw
    assert resolution.direction == METRIC_DIRECTIONS[expected]


@pytest.mark.parametrize("raw", ["Multi Class Log Loss", "Multiclass Logarithmic Loss", "Mean Columnwise Log Loss"])
def test_logarithmic_loss_is_not_a_regression_metric(raw: str) -> None:
    """Guard the ordering bug that read logarithmic loss as RMSLE."""
    resolution = resolve_evaluation_metric(raw)
    assert is_classification_metric(resolution.metric)
    assert uses_probability_output(resolution.metric)


@pytest.mark.parametrize("raw", ["Pinball Loss", "Weighted Quantile Loss", "Mean Columnwise Deviance"])
def test_unknown_error_metrics_are_minimized(raw: str) -> None:
    """Ensure unrecognised error metrics never get maximized."""
    resolution = resolve_evaluation_metric(raw)
    assert resolution.matched is False
    assert resolution.direction == "minimize"


@pytest.mark.parametrize("raw", ["Mean Reward", "Custom Points Total"])
def test_unknown_reward_metrics_are_maximized(raw: str) -> None:
    """Ensure unrecognised reward metrics default to maximization."""
    resolution = resolve_evaluation_metric(raw)
    assert resolution.matched is False
    assert resolution.direction == "maximize"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_missing_metric_name_falls_back_to_accuracy(raw: str | None) -> None:
    """Ensure an absent metric name resolves without raising."""
    resolution = resolve_evaluation_metric(raw)
    assert resolution.metric == EvaluationMetric.ACCURACY
    assert resolution.matched is False
    assert resolution.direction == "maximize"


def test_direction_helper_matches_table() -> None:
    """Ensure metric_direction_for agrees with the direction table."""
    for metric, direction in METRIC_DIRECTIONS.items():
        assert metric_direction_for(metric) == direction


def test_r2_is_a_maximized_regression_metric() -> None:
    """Ensure R2 competitions are treated as regression, not classification."""
    assert metric_direction_for(EvaluationMetric.R2) == "maximize"
    assert is_classification_metric(EvaluationMetric.R2) is False
    assert uses_probability_output(EvaluationMetric.R2) is False
