"""Tests for baseline scoring helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

import csv
import math
from typing import TYPE_CHECKING

import pytest

from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import _compute_baseline_score, _evaluate_metric, _prediction_value

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class TestPredictionValue:
    """Tests for prediction helpers."""

    def test_prediction_value_regression_mean(self) -> None:
        """Regression metrics should return the mean value."""
        prediction, numeric = _prediction_value(
            EvaluationMetric.RMSE,
            [1.0, 3.0],
            ["1", "3"],
            None,
        )

        assert prediction == pytest.approx(2.0)
        assert numeric == pytest.approx(2.0)

    def test_prediction_value_classification_mapping(self) -> None:
        """Classification metrics should use the majority label."""
        prediction, numeric = _prediction_value(
            EvaluationMetric.ACCURACY,
            [0.0, 1.0, 1.0],
            ["cat", "dog", "dog"],
            {"cat": 0, "dog": 1},
        )

        assert prediction == "dog"
        assert numeric == pytest.approx(1.0)

    def test_prediction_value_proba_clamps(self) -> None:
        """Probability metrics should clamp predictions to (0, 1)."""
        prediction, numeric = _prediction_value(
            EvaluationMetric.AUC,
            [2.0, 2.0],
            ["2", "2"],
            None,
        )

        assert prediction == pytest.approx(1 - 1e-3)
        assert numeric == pytest.approx(1 - 1e-3)


class TestEvaluateMetric:
    """Tests for metric evaluation helpers."""

    def test_accuracy_with_zero_predictions(self) -> None:
        """Accuracy should reflect zeros in the predictions."""
        score = _evaluate_metric(EvaluationMetric.ACCURACY, [0.0, 1.0, 0.0], prediction=0.0)

        assert score == pytest.approx(2 / 3)

    def test_f1_edge_case(self) -> None:
        """F1 should be zero when there are no true positives."""
        score = _evaluate_metric(EvaluationMetric.F1, [0.0, 0.0, 0.0], prediction=1.0)

        assert score == 0.0

    def test_auc_baseline(self) -> None:
        """AUC baseline should be constant."""
        score = _evaluate_metric(EvaluationMetric.AUC, [0.0, 1.0], prediction=0.5)

        assert score == 0.5

    def test_rmsle_filters_negative_values(self) -> None:
        """RMSLE should ignore negative values."""
        values = [1.0, -1.0, 3.0]
        prediction = 0.0
        score = _evaluate_metric(EvaluationMetric.RMSLE, values, prediction=prediction)

        expected = math.sqrt(
            (
                (math.log1p(1.0) - math.log1p(prediction)) ** 2
                + (math.log1p(3.0) - math.log1p(prediction)) ** 2
            )
            / 2
        )
        assert score == pytest.approx(expected)


class TestComputeBaselineScore:
    """Tests for baseline score computation."""

    def test_compute_baseline_score_multiple_columns(self, tmp_path: Path) -> None:
        """Baseline score should average per-column metrics."""
        train_path = tmp_path / "train.csv"
        _write_csv(
            train_path,
            ["id", "target_a", "target_b"],
            [["1", "0", "1"], ["2", "2", "1"]],
        )

        score = _compute_baseline_score(
            train_path=train_path,
            target_columns=["target_a", "target_b"],
            metric=EvaluationMetric.RMSE,
        )

        assert score == pytest.approx(0.5)
