"""Tests for Evolver submission scoring.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

try:
    from agent_k.agents.evolver import _normalize_metric_key, _score_submission
except TypeError as exc:
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise


def _write_submission(path: Path, frame: pd.DataFrame) -> Path:
    submission_path = path / "submission.csv"
    frame.to_csv(submission_path, index=False)
    return submission_path


@pytest.fixture
def binary_y_val() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3, 4], "target": [0, 1, 1, 0]})


class TestNormalizeMetricKey:
    """Canonical key mapping for metric labels."""

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("rmse", "rmse"),
            ("RMSE", "rmse"),
            ("log_loss", "logloss"),
            ("logLoss", "logloss"),
            ("ROC AUC", "auc"),
            ("roc-auc", "auc"),
            ("average_precision", "map"),
            ("F1_score", "f1"),
            ("R Squared", "r2"),
            ("mlogloss", "logloss"),
            ("multiclass_logloss", "logloss"),
            ("CrossEntropy", "logloss"),
            ("ndcg", "ndcg"),
            ("accuracy", "accuracy"),
        ],
    )
    def test_normalization(self, metric: str, expected: str) -> None:
        assert _normalize_metric_key(metric) == expected


class TestScoreSubmissionRegression:
    """Regression metrics work for any number of targets."""

    def test_rmse_perfect_predictions(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3], "target": [1.0, 2.0, 3.0]})
        submission = _write_submission(tmp_path, y_val)
        score = _score_submission(
            submission_path=submission, metric="rmse", id_column="id", target_columns=["target"], y_val=y_val
        )
        assert score == pytest.approx(0.0)

    def test_mse_alias(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2], "target": [1.0, 1.0]}))
        score = _score_submission(
            submission_path=submission, metric="mse", id_column="id", target_columns=["target"], y_val=y_val
        )
        assert score == pytest.approx(1.0)

    def test_mae(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 0.0, 0.0, 0.0]})
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [1.0, -1.0, 2.0, -2.0]}))
        score = _score_submission(
            submission_path=submission, metric="mae", id_column="id", target_columns=["target"], y_val=y_val
        )
        assert score == pytest.approx(1.5)

    def test_rmsle_clips_negatives(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2], "target": [-5.0, 0.0]}))
        score = _score_submission(
            submission_path=submission, metric="rmsle", id_column="id", target_columns=["target"], y_val=y_val
        )
        # Negative predictions get clipped to 0 → rmsle == 0 against y_true == 0.
        assert score == pytest.approx(0.0)

    def test_r2_perfect(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3], "target": [1.0, 2.0, 3.0]})
        submission = _write_submission(tmp_path, y_val)
        score = _score_submission(
            submission_path=submission, metric="r_squared", id_column="id", target_columns=["target"], y_val=y_val
        )
        assert score == pytest.approx(1.0)

    def test_rmse_multi_target(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "a": [0.0, 0.0], "b": [0.0, 0.0]})
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2], "a": [1.0, 1.0], "b": [1.0, 1.0]}))
        score = _score_submission(
            submission_path=submission, metric="rmse", id_column="id", target_columns=["a", "b"], y_val=y_val
        )
        assert score == pytest.approx(1.0)


class TestScoreSubmissionProbabilistic:
    """Single-target probabilistic and ranking metrics."""

    def test_logloss_clips_extremes(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 1.0, 1.0, 0.0]}))
        score = _score_submission(
            submission_path=submission, metric="log_loss", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        # Perfect-but-clipped predictions → near zero, but not exactly zero.
        assert 0.0 < score < 1e-4

    def test_logloss_multi_target_raises(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "a": [0, 1], "b": [1, 0]})
        submission = _write_submission(tmp_path, y_val)
        with pytest.raises(ValueError, match="single-target"):
            _score_submission(
                submission_path=submission, metric="logLoss", id_column="id", target_columns=["a", "b"], y_val=y_val
            )

    def test_auc_perfect(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.1, 0.9, 0.8, 0.2]}))
        score = _score_submission(
            submission_path=submission, metric="auc", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        assert score == pytest.approx(1.0)

    def test_roc_auc_alias(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.1, 0.9, 0.8, 0.2]}))
        score = _score_submission(
            submission_path=submission, metric="ROC_AUC", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        assert score == pytest.approx(1.0)

    def test_map_single_target(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.1, 0.9, 0.8, 0.2]}))
        score = _score_submission(
            submission_path=submission, metric="map", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        assert score == pytest.approx(1.0)

    def test_map_multi_target_raises(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "a": [0, 1], "b": [1, 0]})
        submission = _write_submission(tmp_path, y_val)
        with pytest.raises(ValueError, match="single-target"):
            _score_submission(
                submission_path=submission, metric="map", id_column="id", target_columns=["a", "b"], y_val=y_val
            )

    def test_ndcg_perfect_order(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4, 5], "rel": [0.0, 1.0, 2.0, 3.0, 4.0]})
        submission = _write_submission(tmp_path, y_val)
        score = _score_submission(
            submission_path=submission, metric="ndcg", id_column="id", target_columns=["rel"], y_val=y_val
        )
        assert score == pytest.approx(1.0)


class TestScoreSubmissionClassification:
    """Threshold-based classification metrics."""

    def test_accuracy_perfect(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 1.0, 1.0, 0.0]}))
        score = _score_submission(
            submission_path=submission, metric="accuracy", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        assert score == pytest.approx(1.0)

    def test_accuracy_threshold_at_half(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.49, 0.51, 0.50, 0.49]}))
        score = _score_submission(
            submission_path=submission, metric="accuracy", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        # Predictions: [0, 1, 1, 0]; truth: [0, 1, 1, 0] → all correct.
        assert score == pytest.approx(1.0)

    def test_accuracy_multi_target_averages(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "a": [0, 1], "b": [1, 0]})
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2], "a": [0.0, 1.0], "b": [0.0, 0.0]}))
        score = _score_submission(
            submission_path=submission, metric="accuracy", id_column="id", target_columns=["a", "b"], y_val=y_val
        )
        # Column a: 2/2 correct = 1.0, column b: 1/2 correct = 0.5 → mean 0.75.
        assert score == pytest.approx(0.75)

    def test_f1_perfect(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 1.0, 1.0, 0.0]}))
        score = _score_submission(
            submission_path=submission, metric="f1", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        assert score == pytest.approx(1.0)

    def test_f1_all_zero_returns_zero_division(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        # All predicted negatives → tp = 0 → f1 = 0.0 (zero_division=0 prevents warnings).
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 0.0, 0.0, 0.0]}))
        score = _score_submission(
            submission_path=submission, metric="f1_score", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        assert score == pytest.approx(0.0)

    def test_f1_multi_target_macro(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4], "a": [0, 1, 1, 0], "b": [1, 0, 1, 0]})
        submission = _write_submission(
            tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "a": [0.0, 1.0, 1.0, 0.0], "b": [1.0, 0.0, 1.0, 0.0]})
        )
        score = _score_submission(
            submission_path=submission, metric="f1", id_column="id", target_columns=["a", "b"], y_val=y_val
        )
        # Both columns perfect → 1.0 average.
        assert score == pytest.approx(1.0)


class TestScoreSubmissionErrors:
    """Validation errors for malformed submissions."""

    def test_unsupported_metric(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, binary_y_val)
        with pytest.raises(ValueError, match="Unsupported metric"):
            _score_submission(
                submission_path=submission,
                metric="not_a_metric",
                id_column="id",
                target_columns=["target"],
                y_val=binary_y_val,
            )

    def test_missing_id_column(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"target": [0, 1, 1, 0]}))
        with pytest.raises(ValueError, match="missing id column"):
            _score_submission(
                submission_path=submission,
                metric="accuracy",
                id_column="id",
                target_columns=["target"],
                y_val=binary_y_val,
            )

    def test_missing_target_column(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [1, 2, 3, 4]}))
        with pytest.raises(ValueError, match="missing target columns"):
            _score_submission(
                submission_path=submission,
                metric="accuracy",
                id_column="id",
                target_columns=["target"],
                y_val=binary_y_val,
            )

    def test_no_overlap_after_merge(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(tmp_path, pd.DataFrame({"id": [99, 100], "target": [0.0, 0.0]}))
        with pytest.raises(ValueError, match="No rows to score"):
            _score_submission(
                submission_path=submission, metric="rmse", id_column="id", target_columns=["target"], y_val=binary_y_val
            )

    def test_non_finite_predictions_replaced(self, tmp_path: Path, binary_y_val: pd.DataFrame) -> None:
        submission = _write_submission(
            tmp_path, pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, np.inf, np.nan, -np.inf]})
        )
        # Should not raise — non-finite values are replaced with 0 before scoring.
        score = _score_submission(
            submission_path=submission, metric="rmse", id_column="id", target_columns=["target"], y_val=binary_y_val
        )
        # Truth: [0, 1, 1, 0]; preds after sanitization: [0, 0, 0, 0]; squared errors: [0,1,1,0].
        assert score == pytest.approx(np.sqrt(0.5))
