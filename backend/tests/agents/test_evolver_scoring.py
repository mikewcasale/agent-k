"""Tests for evolution submission scoring across metric families.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

try:
    from agent_k.agents.evolver import _normalise_metric_key, _sample_submission_fill, _score_submission
except TypeError as exc:  # pragma: no cover - environment guard mirrored from test_evolver.py
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise


def _score(tmp_path: Path, *, metric: str, predictions: dict[str, list[Any]], truth: dict[str, list[Any]]) -> float:
    """Write a submission/ground-truth pair to disk and score it."""
    ids = list(range(1, len(next(iter(truth.values()))) + 1))
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": ids, **predictions}).to_csv(submission_path, index=False)
    return _score_submission(
        submission_path=submission_path,
        metric=metric,
        id_column="id",
        target_columns=list(truth),
        y_val=pd.DataFrame({"id": ids, **truth}),
    )


class TestMetricKeyNormalisation:
    """Tests for platform metric-name normalisation."""

    @pytest.mark.parametrize("spelling", ["logLoss", "log_loss", "Log Loss", "LOG-LOSS"])
    def test_log_loss_spellings_collapse(self, spelling: str) -> None:
        """Platform spellings of log loss should resolve to one key."""
        assert _normalise_metric_key(spelling) == "logloss"


class TestLabelMetrics:
    """Tests for metrics scored against hard class labels."""

    def test_accuracy_scores_integer_labels(self, tmp_path: Path) -> None:
        """Accuracy is a supported metric, not a scoring failure."""
        score = _score(
            tmp_path, metric="accuracy", predictions={"target": [0, 1, 1, 0]}, truth={"target": [0, 1, 0, 0]}
        )
        assert score == pytest.approx(0.75)

    def test_accuracy_scores_string_labels(self, tmp_path: Path) -> None:
        """Categorical labels must not be float-coerced."""
        score = _score(
            tmp_path,
            metric="accuracy",
            predictions={"target": ["cat", "dog", "dog", "cat"]},
            truth={"target": ["cat", "dog", "cat", "cat"]},
        )
        assert score == pytest.approx(0.75)

    def test_accuracy_survives_csv_dtype_drift(self, tmp_path: Path) -> None:
        """Integer labels read back as floats should still match."""
        score = _score(
            tmp_path, metric="accuracy", predictions={"target": [0.0, 1.0, 1.0, 0.0]}, truth={"target": [0, 1, 0, 0]}
        )
        assert score == pytest.approx(0.75)

    def test_f1_uses_binary_averaging_for_two_classes(self, tmp_path: Path) -> None:
        """Binary F1 uses the higher sorted class as the positive label."""
        score = _score(
            tmp_path,
            metric="f1",
            predictions={"target": ["cat", "dog", "dog", "cat"]},
            truth={"target": ["cat", "dog", "cat", "cat"]},
        )
        assert score == pytest.approx(2 / 3)

    def test_f1_uses_weighted_averaging_beyond_two_classes(self, tmp_path: Path) -> None:
        """Multiclass F1 falls back to weighted averaging."""
        from sklearn.metrics import f1_score

        truth = ["a", "b", "a", "c"]
        predictions = ["a", "b", "c", "a"]
        score = _score(tmp_path, metric="f1", predictions={"target": predictions}, truth={"target": truth})
        assert score == pytest.approx(f1_score(truth, predictions, average="weighted", zero_division=0))

    def test_f1_handles_single_class_columns(self, tmp_path: Path) -> None:
        """A degenerate single-class column scores instead of raising."""
        score = _score(tmp_path, metric="f1", predictions={"target": [1, 1, 1, 1]}, truth={"target": [1, 1, 1, 1]})
        assert score == pytest.approx(1.0)

    def test_multi_target_labels_are_averaged(self, tmp_path: Path) -> None:
        """Multi-target accuracy averages the per-column scores."""
        score = _score(
            tmp_path,
            metric="accuracy",
            predictions={"a": [0, 1, 1, 0], "b": [0, 1, 0, 0]},
            truth={"a": [0, 1, 0, 0], "b": [0, 1, 0, 0]},
        )
        assert score == pytest.approx((0.75 + 1.0) / 2)

    def test_continuous_predictions_raise_actionable_error(self, tmp_path: Path) -> None:
        """Probabilities under a label metric are a format error, not a zero."""
        with pytest.raises(ValueError, match="continuous scores that match no label"):
            _score(
                tmp_path,
                metric="accuracy",
                predictions={"target": [0.7, 0.2, 0.9, 0.1]},
                truth={"target": [0, 1, 0, 0]},
            )

    def test_wrong_but_discrete_predictions_score_zero(self, tmp_path: Path) -> None:
        """A wrong label is an inaccurate submission, not a malformed one."""
        score = _score(
            tmp_path,
            metric="accuracy",
            predictions={"target": ["fish", "fish", "fish", "fish"]},
            truth={"target": ["cat", "dog", "cat", "cat"]},
        )
        assert score == pytest.approx(0.0)


class TestProbabilityMetrics:
    """Tests for metrics scored against predicted probabilities."""

    def test_auc_matches_sklearn_for_numeric_labels(self, tmp_path: Path) -> None:
        """Numeric AUC behaviour is unchanged."""
        from sklearn.metrics import roc_auc_score

        truth = [0, 1, 0, 0]
        predictions = [0.7, 0.2, 0.9, 0.1]
        score = _score(tmp_path, metric="auc", predictions={"target": predictions}, truth={"target": truth})
        assert score == pytest.approx(roc_auc_score(truth, predictions))

    def test_auc_binarises_string_labels(self, tmp_path: Path) -> None:
        """Categorical targets binarise against the higher sorted class."""
        from sklearn.metrics import roc_auc_score

        predictions = [0.7, 0.2, 0.9, 0.1]
        score = _score(
            tmp_path, metric="auc", predictions={"target": predictions}, truth={"target": ["cat", "dog", "cat", "cat"]}
        )
        assert score == pytest.approx(roc_auc_score([0, 1, 0, 0], predictions))

    def test_log_loss_binarises_string_labels(self, tmp_path: Path) -> None:
        """Log loss on categorical targets scores instead of raising."""
        from sklearn.metrics import log_loss

        predictions = [0.7, 0.2, 0.9, 0.1]
        score = _score(
            tmp_path,
            metric="logLoss",
            predictions={"target": predictions},
            truth={"target": ["cat", "dog", "cat", "cat"]},
        )
        assert score == pytest.approx(log_loss([0, 1, 0, 0], predictions, labels=[0, 1]))

    def test_non_finite_predictions_are_neutralised(self, tmp_path: Path) -> None:
        """NaN probabilities become 0.5 rather than a confident negative."""
        score = _score(
            tmp_path, metric="logLoss", predictions={"target": [float("nan")] * 4}, truth={"target": [0, 1, 0, 1]}
        )
        assert score == pytest.approx(-np.log(0.5))

    def test_multiclass_targets_are_rejected(self, tmp_path: Path) -> None:
        """One probability column cannot express three classes."""
        with pytest.raises(ValueError, match="exactly 2 classes"):
            _score(
                tmp_path,
                metric="auc",
                predictions={"target": [0.7, 0.2, 0.9, 0.1]},
                truth={"target": ["a", "b", "c", "a"]},
            )

    def test_multi_target_probabilities_are_rejected(self, tmp_path: Path) -> None:
        """Multi-target probability scoring stays unsupported."""
        with pytest.raises(ValueError, match="single-target"):
            _score(
                tmp_path,
                metric="auc",
                predictions={"a": [0.7, 0.2, 0.9, 0.1], "b": [0.7, 0.2, 0.9, 0.1]},
                truth={"a": [0, 1, 0, 0], "b": [0, 1, 0, 0]},
            )


class TestRegressionMetrics:
    """Tests that continuous-metric behaviour is preserved."""

    @pytest.mark.parametrize(("metric", "expected"), [("rmse", 0.5), ("mae", 0.25)])
    def test_regression_metrics_are_unchanged(self, tmp_path: Path, metric: str, expected: float) -> None:
        """RMSE and MAE keep their previous values."""
        score = _score(
            tmp_path,
            metric=metric,
            predictions={"target": [1.0, 2.0, 3.0, 4.0]},
            truth={"target": [1.0, 2.0, 2.0, 4.0]},
        )
        assert score == pytest.approx(expected)

    def test_rmsle_clips_negative_predictions(self, tmp_path: Path) -> None:
        """RMSLE clips at zero so log1p stays defined."""
        from sklearn.metrics import mean_squared_log_error

        score = _score(
            tmp_path,
            metric="rmsle",
            predictions={"target": [-5.0, 2.0, 3.0, 4.0]},
            truth={"target": [1.0, 2.0, 2.0, 4.0]},
        )
        expected = float(np.sqrt(mean_squared_log_error([1.0, 2.0, 2.0, 4.0], [0.0, 2.0, 3.0, 4.0])))
        assert score == pytest.approx(expected)


class TestUnsupportedMetrics:
    """Tests for metrics the held-out split cannot score."""

    @pytest.mark.parametrize("metric", ["map", "ndcg"])
    def test_ranking_metrics_name_the_supported_set(self, tmp_path: Path, metric: str) -> None:
        """Ranking metrics raise an error that lists what is supported."""
        with pytest.raises(ValueError, match="supported: accuracy, auc, f1, logloss, mae, rmse, rmsle"):
            _score(tmp_path, metric=metric, predictions={"target": [1, 2, 3, 4]}, truth={"target": [1, 2, 3, 4]})


class TestSampleSubmissionFill:
    """Tests for the sample_submission placeholder value."""

    def test_numeric_targets_keep_zero(self) -> None:
        """Numeric columns keep the historical 0.0 placeholder."""
        assert _sample_submission_fill(pd.Series([1.5, 2.5, 3.5])) == 0.0

    def test_categorical_targets_use_the_modal_label(self) -> None:
        """String columns advertise a real label instead of a float."""
        assert _sample_submission_fill(pd.Series(["cat", "dog", "cat"])) == "cat"

    def test_empty_categorical_targets_fall_back_to_empty_string(self) -> None:
        """An all-null categorical column still yields a writable value."""
        assert _sample_submission_fill(pd.Series([None, None], dtype="object")) == ""
