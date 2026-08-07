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

    def test_medae_odd_and_even(self) -> None:
        """MedAE should use the median absolute deviation."""
        odd = _evaluate_metric(EvaluationMetric.MEDAE, [1.0, 4.0, 9.0], prediction=4.0)
        even = _evaluate_metric(EvaluationMetric.MEDAE, [1.0, 4.0, 9.0, 12.0], prediction=4.0)
        assert odd == pytest.approx(3.0)
        assert even == pytest.approx(4.0)

    def test_smape_skips_zero_denominators(self) -> None:
        """SMAPE zero-denominator entries should contribute zero, not raise."""
        score = _evaluate_metric(EvaluationMetric.SMAPE, [0.0, 100.0], prediction=0.0)
        # First entry: value=0, pred=0 -> 0. Second entry: |100|/(50) = 2.0. Average 1.0.
        assert score == pytest.approx(1.0)

    def test_mape_ignores_zero_targets(self) -> None:
        """MAPE should ignore zero targets to avoid division by zero."""
        score = _evaluate_metric(EvaluationMetric.MAPE, [0.0, 100.0, 200.0], prediction=100.0)
        # |100-100|/100 + |100-200|/200 = 0 + 0.5, divided by 2 (nonzero count) = 0.25
        assert score == pytest.approx(0.25)

    def test_r2_returns_zero_for_constant_target(self) -> None:
        """R^2 is undefined when ss_tot is zero; return 0.0."""
        score = _evaluate_metric(EvaluationMetric.R2, [3.0, 3.0, 3.0], prediction=3.0)
        assert score == 0.0

    def test_r2_perfect_prediction(self) -> None:
        """R^2 for a mean prediction equals 0 (no explanatory power)."""
        score = _evaluate_metric(EvaluationMetric.R2, [1.0, 2.0, 3.0], prediction=2.0)
        # ss_res = ((1-2)^2 + (2-2)^2 + (3-2)^2) = 2, ss_tot = same = 2, R2 = 0
        assert score == pytest.approx(0.0)

    def test_mcrmse_matches_rmse_for_single_column(self) -> None:
        """MCRMSE should equal RMSE for a single column of values."""
        values = [1.0, 2.0, 3.0]
        prediction = 2.0
        assert _evaluate_metric(EvaluationMetric.MCRMSE, values, prediction=prediction) == pytest.approx(
            _evaluate_metric(EvaluationMetric.RMSE, values, prediction=prediction)
        )

    def test_balanced_accuracy_uses_class_balance(self) -> None:
        """Balanced accuracy for the majority-only baseline should be 0.5."""
        # 3 positives, 1 negative; constant prediction=1 -> tpr=1, tnr=0 -> 0.5
        assert _evaluate_metric(EvaluationMetric.BALANCED_ACCURACY, [1.0, 1.0, 1.0, 0.0], prediction=1.0) == 0.5

    def test_correlation_metrics_baseline_zero(self) -> None:
        """Constant baselines yield zero correlation."""
        assert _evaluate_metric(EvaluationMetric.SPEARMAN, [1.0, 2.0, 3.0], prediction=2.0) == 0.0
        assert _evaluate_metric(EvaluationMetric.PEARSON, [1.0, 2.0, 3.0], prediction=2.0) == 0.0

    def test_multi_log_loss_uses_logistic_form(self) -> None:
        """Multi-class log loss baseline matches the binary log-loss formula."""
        score = _evaluate_metric(EvaluationMetric.MULTI_LOG_LOSS, [0.0, 1.0], prediction=0.5)
        expected = _evaluate_metric(EvaluationMetric.LOG_LOSS, [0.0, 1.0], prediction=0.5)
        assert score == pytest.approx(expected)

    def test_unknown_metric_returns_zero(self) -> None:
        """MCC and quadratic kappa constant-prediction baselines are zero."""
        assert _evaluate_metric(EvaluationMetric.MCC, [0.0, 1.0], prediction=1.0) == 0.0
        assert _evaluate_metric(EvaluationMetric.QUADRATIC_KAPPA, [0.0, 1.0], prediction=1.0) == 0.0


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
