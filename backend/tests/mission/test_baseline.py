"""Tests for baseline scoring helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import math
from typing import TYPE_CHECKING

import pytest

from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import _compute_baseline_score, _evaluate_metric, _load_target_values, _prediction_value

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
        prediction, numeric = _prediction_value(EvaluationMetric.RMSE, [1.0, 3.0], ["1", "3"], None)

        assert prediction == pytest.approx(2.0)
        assert numeric == pytest.approx(2.0)

    def test_prediction_value_classification_mapping(self) -> None:
        """Classification metrics should use the majority label."""
        prediction, numeric = _prediction_value(
            EvaluationMetric.ACCURACY, [0.0, 1.0, 1.0], ["cat", "dog", "dog"], {"cat": 0, "dog": 1}
        )

        assert prediction == "dog"
        assert numeric == pytest.approx(1.0)

    def test_prediction_value_proba_clamps(self) -> None:
        """Probability metrics should clamp predictions to (0, 1)."""
        prediction, numeric = _prediction_value(EvaluationMetric.AUC, [2.0, 2.0], ["2", "2"], None)

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
            ((math.log1p(1.0) - math.log1p(prediction)) ** 2 + (math.log1p(3.0) - math.log1p(prediction)) ** 2) / 2
        )
        assert score == pytest.approx(expected)


class TestComputeBaselineScore:
    """Tests for baseline score computation."""

    def test_compute_baseline_score_multiple_columns(self, tmp_path: Path) -> None:
        """Baseline score should average per-column metrics."""
        train_path = tmp_path / "train.csv"
        _write_csv(train_path, ["id", "target_a", "target_b"], [["1", "0", "1"], ["2", "2", "1"]])

        score = _compute_baseline_score(
            train_path=train_path, target_columns=["target_a", "target_b"], metric=EvaluationMetric.RMSE
        )

        assert score == pytest.approx(0.5)

    def test_compute_baseline_score_skips_empty_columns(self, tmp_path: Path) -> None:
        """A column with no usable targets should not be averaged in as a perfect score."""
        train_path = tmp_path / "train.csv"
        _write_csv(train_path, ["id", "target_a", "target_b"], [["1", "", "1"], ["2", "NA", "3"]])

        score = _compute_baseline_score(
            train_path=train_path, target_columns=["target_a", "target_b"], metric=EvaluationMetric.RMSE
        )

        assert score == pytest.approx(1.0)

    def test_compute_baseline_score_without_usable_targets(self, tmp_path: Path) -> None:
        """Baseline score should fall back to zero when no column is usable."""
        train_path = tmp_path / "train.csv"
        _write_csv(train_path, ["id", "target"], [["1", ""], ["2", "null"]])

        score = _compute_baseline_score(train_path=train_path, target_columns=["target"], metric=EvaluationMetric.RMSE)

        assert score == 0.0


class TestLoadTargetValues:
    """Tests for target-column loading."""

    def test_missing_cells_do_not_force_label_encoding(self, tmp_path: Path) -> None:
        """A numeric column with blank and NA cells should stay numeric."""
        train_path = tmp_path / "train.csv"
        _write_csv(train_path, ["id", "target"], [["1", "1.5"], ["2", ""], ["3", "3.5"], ["4", "NA"]])

        numeric_values, raw_values, mapping = _load_target_values(train_path, "target")

        assert mapping is None
        assert raw_values == ["1.5", "3.5"]
        assert numeric_values == pytest.approx([1.5, 3.5])

    def test_non_finite_cells_are_dropped(self, tmp_path: Path) -> None:
        """NaN and infinite targets should be dropped rather than poison the baseline."""
        train_path = tmp_path / "train.csv"
        _write_csv(train_path, ["id", "target"], [["1", "2"], ["2", "nan"], ["3", "inf"], ["4", "4"]])

        numeric_values, _raw_values, mapping = _load_target_values(train_path, "target")

        assert mapping is None
        assert numeric_values == pytest.approx([2.0, 4.0])

    def test_categorical_column_is_label_encoded(self, tmp_path: Path) -> None:
        """A genuinely categorical column should still be label encoded."""
        train_path = tmp_path / "train.csv"
        _write_csv(train_path, ["id", "target"], [["1", "cat"], ["2", "dog"], ["3", "cat"]])

        numeric_values, raw_values, mapping = _load_target_values(train_path, "target")

        assert mapping == {"cat": 0, "dog": 1}
        assert raw_values == ["cat", "dog", "cat"]
        assert numeric_values == pytest.approx([0.0, 1.0, 0.0])


class TestMulticlassBaseline:
    """Tests for multiclass baseline scoring."""

    def test_log_loss_uses_class_priors(self) -> None:
        """Multiclass log loss should equal the entropy of the label distribution."""
        values = [0.0, 0.0, 1.0, 2.0]

        score = _evaluate_metric(EvaluationMetric.LOG_LOSS, values, prediction=0.5)

        expected = -(0.5 * math.log(0.5) + 0.25 * math.log(0.25) + 0.25 * math.log(0.25))
        assert score == pytest.approx(expected)
        assert score > 0.0

    def test_log_loss_binary_is_unchanged(self) -> None:
        """Binary log loss should keep scoring the constant positive-class probability."""
        values = [0.0, 1.0, 1.0, 1.0]

        score = _evaluate_metric(EvaluationMetric.LOG_LOSS, values, prediction=0.75)

        expected = -(math.log(1 - 0.75) + 3 * math.log(0.75)) / 4
        assert score == pytest.approx(expected)

    def test_f1_is_macro_averaged_for_multiclass(self) -> None:
        """Multiclass F1 should macro-average a constant majority prediction."""
        values = [0.0, 0.0, 1.0, 2.0]

        score = _evaluate_metric(EvaluationMetric.F1, values, prediction=0.0)

        assert score == pytest.approx((2 * 0.5 / 1.5) / 3)

    def test_f1_multiclass_unpredicted_class_scores_zero(self) -> None:
        """Predicting a class absent from the labels should score zero."""
        score = _evaluate_metric(EvaluationMetric.F1, [0.0, 1.0, 2.0], prediction=9.0)

        assert score == 0.0

    def test_prediction_value_multiclass_proba_uses_majority_prior(self) -> None:
        """Probability metrics should not average label-encoded multiclass targets."""
        prediction, numeric = _prediction_value(
            EvaluationMetric.LOG_LOSS, [0.0, 0.0, 1.0, 2.0], ["a", "a", "b", "c"], {"a": 0, "b": 1, "c": 2}
        )

        assert prediction == pytest.approx(0.5)
        assert numeric == pytest.approx(0.5)
