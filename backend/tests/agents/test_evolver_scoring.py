"""Tests for evolver submission scoring across metrics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

__all__ = ()

try:
    from agent_k.agents.evolver import (
        _score_accuracy,
        _score_f1,
        _score_map,
        _score_ndcg,
        _score_submission,
        _to_class_predictions,
    )
except TypeError as exc:
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise


def _write_submission(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "submission.csv"
    df.to_csv(path, index=False)
    return path


class TestToClassPredictions:
    """Tests for probability-to-label coercion."""

    def test_binary_threshold_uses_actual_class_values(self) -> None:
        preds = np.array([0.1, 0.49, 0.5, 0.8])
        truth = np.array([0.0, 1.0, 0.0, 1.0])

        labels = _to_class_predictions(preds, truth)

        assert labels.tolist() == [0.0, 0.0, 1.0, 1.0]

    def test_multiclass_rounds_and_clips_to_known_range(self) -> None:
        preds = np.array([-0.4, 0.6, 1.4, 2.9, 9.0])
        truth = np.array([0.0, 1.0, 2.0])

        labels = _to_class_predictions(preds, truth)

        assert labels.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]

    def test_hard_label_predictions_pass_through(self) -> None:
        preds = np.array([0.0, 1.0, 0.0, 1.0])
        truth = np.array([0.0, 1.0])

        labels = _to_class_predictions(preds, truth)

        assert labels.tolist() == [0.0, 1.0, 0.0, 1.0]


class TestScoreAccuracy:
    """Tests for accuracy scorer over probability predictions."""

    def test_perfect_probabilities_score_one(self) -> None:
        truth = np.array([0.0, 1.0, 0.0, 1.0])
        preds = np.array([0.1, 0.9, 0.2, 0.7])

        assert _score_accuracy(truth, preds) == pytest.approx(1.0)

    def test_inverted_probabilities_score_zero(self) -> None:
        truth = np.array([0.0, 1.0, 0.0, 1.0])
        preds = np.array([0.9, 0.1, 0.8, 0.2])

        assert _score_accuracy(truth, preds) == pytest.approx(0.0)


class TestScoreF1:
    """Tests for F1 scorer over probability predictions."""

    def test_binary_perfect_f1_is_one(self) -> None:
        truth = np.array([0.0, 1.0, 0.0, 1.0])
        preds = np.array([0.0, 1.0, 0.0, 1.0])

        assert _score_f1(truth, preds) == pytest.approx(1.0)

    def test_multiclass_uses_weighted_average(self) -> None:
        truth = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
        preds = np.array([0.0, 1.0, 2.0, 0.0, 0.0])

        # 1 misclassification out of 5 - weighted F1 is bounded below 1.
        score = _score_f1(truth, preds)
        assert 0.0 < score < 1.0


class TestScoreMap:
    """Tests for mean average precision scorer."""

    def test_binary_perfect_ranking_scores_one(self) -> None:
        truth = np.array([0.0, 0.0, 1.0, 1.0])
        preds = np.array([0.1, 0.2, 0.8, 0.9])

        assert _score_map(truth, preds) == pytest.approx(1.0)

    def test_multiclass_raises(self) -> None:
        truth = np.array([0.0, 1.0, 2.0])
        preds = np.array([0.1, 0.5, 0.9])

        with pytest.raises(ValueError, match="binary"):
            _score_map(truth, preds)


class TestScoreNdcg:
    """Tests for NDCG scorer."""

    def test_perfect_ranking_scores_one(self) -> None:
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        preds = np.array([0.1, 0.2, 0.3, 0.4])

        assert _score_ndcg(truth, preds) == pytest.approx(1.0)


class TestScoreSubmissionMetricCoverage:
    """Integration tests across previously unsupported metrics."""

    def test_accuracy_metric_scores_submission(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
        submission = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0.1, 0.9, 0.2, 0.7]})
        path = _write_submission(tmp_path, submission)

        score = _score_submission(
            submission_path=path, metric="accuracy", id_column="id", target_columns=["label"], y_val=y_val
        )

        assert score == pytest.approx(1.0)

    def test_f1_metric_scores_submission(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
        submission = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0.0, 1.0, 0.0, 1.0]})
        path = _write_submission(tmp_path, submission)

        score = _score_submission(
            submission_path=path, metric="f1", id_column="id", target_columns=["label"], y_val=y_val
        )

        assert score == pytest.approx(1.0)

    def test_map_metric_scores_submission(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
        submission = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0.1, 0.2, 0.8, 0.9]})
        path = _write_submission(tmp_path, submission)

        score = _score_submission(
            submission_path=path, metric="map", id_column="id", target_columns=["label"], y_val=y_val
        )

        assert score == pytest.approx(1.0)

    def test_ndcg_metric_scores_submission(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2, 3, 4], "label": [1.0, 2.0, 3.0, 4.0]})
        submission = pd.DataFrame({"id": [1, 2, 3, 4], "label": [0.1, 0.2, 0.3, 0.4]})
        path = _write_submission(tmp_path, submission)

        score = _score_submission(
            submission_path=path, metric="ndcg", id_column="id", target_columns=["label"], y_val=y_val
        )

        assert score == pytest.approx(1.0)

    def test_unsupported_metric_still_raises(self, tmp_path: Path) -> None:
        y_val = pd.DataFrame({"id": [1, 2], "label": [0, 1]})
        submission = pd.DataFrame({"id": [1, 2], "label": [0.2, 0.8]})
        path = _write_submission(tmp_path, submission)

        with pytest.raises(ValueError, match="Unsupported metric"):
            _score_submission(
                submission_path=path, metric="mysterymetric", id_column="id", target_columns=["label"], y_val=y_val
            )
