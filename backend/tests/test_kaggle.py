"""Validate Kaggle boundary normalization without simulated service responses.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.kaggle import parse_metric, validate_slug
from agent_k.models import EvaluationMetric, MissionCriteria

__all__ = ()


@pytest.mark.parametrize("metric", ["MAP", "NDCG", "Root Mean Squared Percentage Error", ""])
def test_unknown_metrics_cannot_become_accuracy(metric: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_metric(metric)


@pytest.mark.parametrize(
    "value", ["../dataset", "https://example.com/competitions/a", "a/b", "a?b"]
)
def test_slug_rejects_paths_and_other_domains(value: str) -> None:
    with pytest.raises(ValueError):
        validate_slug(value)


def test_metric_distinguishes_log_loss_and_logarithmic_regression() -> None:
    assert parse_metric("Logarithmic Loss") is EvaluationMetric.LOG_LOSS
    assert parse_metric("Root Mean Squared Logarithmic Error") is EvaluationMetric.RMSLE


def test_metric_accepts_kaggle_categorization_accuracy_name() -> None:
    assert parse_metric("Categorization Accuracy") is EvaluationMetric.ACCURACY


def test_mission_rejects_conflicting_constraints() -> None:
    with pytest.raises(ValueError, match="both targeted and excluded"):
        MissionCriteria(
            target_domains=frozenset({"tabular"}), exclude_domains=frozenset({"tabular"})
        )
