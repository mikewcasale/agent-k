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
    from agent_k.agents.evolver import _normalize_label, _score_submission, _sorted_label_values
except TypeError as exc:
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise

_IDS = [1, 2, 3, 4, 5, 6]


def _write(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "submission.csv"
    frame.to_csv(path, index=False)
    return path


def _multiclass_truth() -> pd.DataFrame:
    return pd.DataFrame({"id": _IDS, "target": ["Class_1", "Class_2", "Class_3", "Class_1", "Class_2", "Class_3"]})


def _multiclass_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": _IDS,
            "Class_1": [0.8, 0.1, 0.1, 0.7, 0.2, 0.1],
            "Class_2": [0.1, 0.8, 0.2, 0.2, 0.7, 0.2],
            "Class_3": [0.1, 0.1, 0.7, 0.1, 0.1, 0.7],
        }
    )


class TestLabelNormalization:
    """Tests for class label and column name tokenization."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("Class_1", "1"), ("class 1", "1"), ("target_a", "a"), ("Pred-9", "9"), ("setosa", "setosa")],
    )
    def test_normalize_label(self, value: str, expected: str) -> None:
        """Prefixes and separators should be stripped from label tokens."""
        assert _normalize_label(value) == expected

    def test_normalize_label_keeps_bare_prefix(self) -> None:
        """A label that is only a prefix should survive untouched."""
        assert _normalize_label("class") == "class"

    def test_sorted_label_values_orders_numerically(self) -> None:
        """Numeric labels should sort by value rather than lexicographically."""
        assert _sorted_label_values([10, 2, 1]) == ["1", "2", "10"]

    def test_sorted_label_values_falls_back_to_text(self) -> None:
        """Non-numeric labels should sort as text."""
        assert _sorted_label_values(["b", "a"]) == ["a", "b"]


class TestClassProbabilityScoring:
    """Tests for one-probability-column-per-class submissions."""

    def test_multiclass_log_loss(self, tmp_path: Path) -> None:
        """A class-expanded submission should score against a single label column."""
        path = _write(tmp_path, _multiclass_probabilities())
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=_multiclass_truth(),
            submission_columns=["Class_1", "Class_2", "Class_3"],
        )
        assert score == pytest.approx(0.312164, abs=1e-5)

    def test_multiclass_auc(self, tmp_path: Path) -> None:
        """Perfectly ranked class probabilities should score a macro AUC of one."""
        path = _write(tmp_path, _multiclass_probabilities())
        score = _score_submission(
            submission_path=path,
            metric="auc",
            id_column="id",
            target_columns=["target"],
            y_val=_multiclass_truth(),
            submission_columns=["Class_1", "Class_2", "Class_3"],
        )
        assert score == pytest.approx(1.0)

    def test_numeric_class_columns_match_numeric_labels(self, tmp_path: Path) -> None:
        """Columns named after numeric labels should pair with those labels."""
        frame = _multiclass_probabilities().rename(columns={"Class_1": "1", "Class_2": "2", "Class_3": "3"})
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=pd.DataFrame({"id": _IDS, "target": [1, 2, 3, 1, 2, 3]}),
            submission_columns=["1", "2", "3"],
        )
        assert score == pytest.approx(0.312164, abs=1e-5)

    def test_labels_sorting_numerically_and_lexicographically_disagree(self, tmp_path: Path) -> None:
        """Numeric labels whose text order differs must keep probabilities on the right class."""
        frame = _multiclass_probabilities().rename(columns={"Class_1": "1", "Class_2": "2", "Class_3": "10"})
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=pd.DataFrame({"id": _IDS, "target": [1, 2, 10, 1, 2, 10]}),
            submission_columns=["1", "2", "10"],
        )
        assert score == pytest.approx(0.312164, abs=1e-5)

    def test_binary_class_columns_pick_the_greater_label_as_positive(self, tmp_path: Path) -> None:
        """A two-column probability submission should treat the greater label as positive."""
        frame = pd.DataFrame({"id": _IDS, "2": [0.9, 0.1, 0.8, 0.2, 0.3, 0.7], "10": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]})
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="auc",
            id_column="id",
            target_columns=["target"],
            y_val=pd.DataFrame({"id": _IDS, "target": [2, 10, 2, 10, 10, 2]}),
            submission_columns=["2", "10"],
        )
        assert score == pytest.approx(1.0)

    def test_unmatched_columns_pair_positionally(self, tmp_path: Path) -> None:
        """Columns that no label matches should pair with sorted labels in order."""
        frame = _multiclass_probabilities().rename(columns={"Class_1": "p_a", "Class_2": "p_b", "Class_3": "p_c"})
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=_multiclass_truth(),
            submission_columns=["p_a", "p_b", "p_c"],
        )
        assert score == pytest.approx(0.312164, abs=1e-5)

    def test_class_absent_from_split_is_scored_on_observed_classes(self, tmp_path: Path) -> None:
        """A class missing from the validation split should not break scoring."""
        path = _write(tmp_path, _multiclass_probabilities())
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=pd.DataFrame({"id": _IDS, "target": ["Class_1", "Class_2"] * 3}),
            submission_columns=["Class_1", "Class_2", "Class_3"],
        )
        assert score > 0.0

    def test_degenerate_rows_are_renormalized(self, tmp_path: Path) -> None:
        """Missing and all-zero probability rows should score instead of raising."""
        frame = _multiclass_probabilities()
        frame.loc[0, "Class_1"] = np.nan
        frame.loc[1, ["Class_1", "Class_2", "Class_3"]] = 0.0
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=_multiclass_truth(),
            submission_columns=["Class_1", "Class_2", "Class_3"],
        )
        assert np.isfinite(score)

    def test_zero_probabilities_still_score_auc(self, tmp_path: Path) -> None:
        """Exact zeros must not break the probability-sum check in macro AUC."""
        frame = pd.DataFrame(
            {
                "id": _IDS,
                "Class_1": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "Class_2": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                "Class_3": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            }
        )
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="auc",
            id_column="id",
            target_columns=["target"],
            y_val=_multiclass_truth(),
            submission_columns=["Class_1", "Class_2", "Class_3"],
        )
        assert score == pytest.approx(1.0)

    def test_submission_column_named_like_the_truth_alias(self, tmp_path: Path) -> None:
        """A class column colliding with the internal truth alias must not shadow it."""
        frame = _multiclass_probabilities().rename(columns={"Class_1": "_target_true"})
        path = _write(tmp_path, frame)
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=_multiclass_truth(),
            submission_columns=["_target_true", "Class_2", "Class_3"],
        )
        assert score == pytest.approx(0.312164, abs=1e-5)

    def test_unmappable_column_count_reports_a_clear_error(self, tmp_path: Path) -> None:
        """Unmatched columns outnumbering the observed classes cannot pair positionally."""
        frame = _multiclass_probabilities().rename(columns={"Class_1": "p_a", "Class_2": "p_b", "Class_3": "p_c"})
        path = _write(tmp_path, frame)
        with pytest.raises(ValueError, match="Cannot map"):
            _score_submission(
                submission_path=path,
                metric="logLoss",
                id_column="id",
                target_columns=["target"],
                y_val=pd.DataFrame({"id": _IDS, "target": ["Class_1", "Class_2"] * 3}),
                submission_columns=["p_a", "p_b", "p_c"],
            )

    def test_single_class_split_reports_a_clear_error(self, tmp_path: Path) -> None:
        """A validation split with one class cannot be scored on probabilities."""
        path = _write(tmp_path, _multiclass_probabilities())
        with pytest.raises(ValueError, match="single class"):
            _score_submission(
                submission_path=path,
                metric="logLoss",
                id_column="id",
                target_columns=["target"],
                y_val=pd.DataFrame({"id": _IDS, "target": ["Class_1"] * 6}),
                submission_columns=["Class_1", "Class_2", "Class_3"],
            )

    def test_label_metric_on_probability_matrix_reports_a_clear_error(self, tmp_path: Path) -> None:
        """A label metric cannot score a probability matrix."""
        path = _write(tmp_path, _multiclass_probabilities())
        with pytest.raises(ValueError, match="probability submission"):
            _score_submission(
                submission_path=path,
                metric="accuracy",
                id_column="id",
                target_columns=["target"],
                y_val=_multiclass_truth(),
                submission_columns=["Class_1", "Class_2", "Class_3"],
            )


class TestSingleColumnScoring:
    """Tests for submissions with one prediction column per target."""

    @pytest.mark.parametrize(("metric", "expected"), [("rmse", 0.141421), ("mae", 0.133333), ("rmsle", 0.034121)])
    def test_regression_metrics(self, tmp_path: Path, metric: str, expected: float) -> None:
        """Regression metrics should keep scoring without submission columns given."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "y": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1]}))
        score = _score_submission(
            submission_path=path,
            metric=metric,
            id_column="id",
            target_columns=["y"],
            y_val=pd.DataFrame({"id": _IDS, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
        )
        assert score == pytest.approx(expected, abs=1e-5)

    def test_renamed_submission_column(self, tmp_path: Path) -> None:
        """A submission column named differently from the train target should still score."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "prediction": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1]}))
        score = _score_submission(
            submission_path=path,
            metric="rmse",
            id_column="id",
            target_columns=["y"],
            y_val=pd.DataFrame({"id": _IDS, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
            submission_columns=["prediction"],
        )
        assert score == pytest.approx(0.141421, abs=1e-5)

    def test_multi_output_regression(self, tmp_path: Path) -> None:
        """Multi-output targets should average across prediction columns."""
        path = _write(
            tmp_path,
            pd.DataFrame({"id": _IDS, "a": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1], "b": [2.1, 3.1, 3.9, 5.2, 5.8, 7.1]}),
        )
        score = _score_submission(
            submission_path=path,
            metric="rmse",
            id_column="id",
            target_columns=["a", "b"],
            y_val=pd.DataFrame({"id": _IDS, "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}),
            submission_columns=["a", "b"],
        )
        assert score == pytest.approx(0.141421, abs=1e-5)

    @pytest.mark.parametrize("labels", [["no", "yes", "no", "yes", "yes", "no"], [0, 1, 0, 1, 1, 0]])
    def test_binary_auc_accepts_label_targets(self, tmp_path: Path, labels: list[object]) -> None:
        """Binary AUC should score string and numeric labels alike."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "target": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]}))
        score = _score_submission(
            submission_path=path,
            metric="auc",
            id_column="id",
            target_columns=["target"],
            y_val=pd.DataFrame({"id": _IDS, "target": labels}),
        )
        assert score == pytest.approx(1.0)

    def test_binary_log_loss_accepts_string_labels(self, tmp_path: Path) -> None:
        """Binary log loss should encode non-numeric labels instead of failing."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "target": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]}))
        score = _score_submission(
            submission_path=path,
            metric="logLoss",
            id_column="id",
            target_columns=["target"],
            y_val=pd.DataFrame({"id": _IDS, "target": ["no", "yes", "no", "yes", "yes", "no"]}),
        )
        assert score == pytest.approx(0.228393, abs=1e-5)

    def test_multiclass_labels_need_class_columns(self, tmp_path: Path) -> None:
        """One probability column cannot score a multiclass label column."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "target": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]}))
        with pytest.raises(ValueError, match="one probability column per class"):
            _score_submission(
                submission_path=path,
                metric="logLoss",
                id_column="id",
                target_columns=["target"],
                y_val=_multiclass_truth(),
            )

    def test_non_numeric_target_reports_the_offending_column(self, tmp_path: Path) -> None:
        """A regression metric on label targets should name the column it cannot cast."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "target": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]}))
        with pytest.raises(ValueError, match="target_true"):
            _score_submission(
                submission_path=path,
                metric="rmse",
                id_column="id",
                target_columns=["target"],
                y_val=_multiclass_truth(),
            )

    def test_missing_submission_column_reports_the_expected_name(self, tmp_path: Path) -> None:
        """A submission missing an expected column should name it."""
        path = _write(tmp_path, pd.DataFrame({"id": _IDS, "wrong": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]}))
        with pytest.raises(ValueError, match="prediction"):
            _score_submission(
                submission_path=path,
                metric="rmse",
                id_column="id",
                target_columns=["y"],
                y_val=pd.DataFrame({"id": _IDS, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
                submission_columns=["prediction"],
            )
