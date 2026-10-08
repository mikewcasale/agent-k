"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from agent_k.core.submission import SubmissionValidation
from agent_k.evolution.evaluator import _extract_error_feedback, _fitness_from_score, _submission_mutation_hint

__all__ = ()


def test_extract_error_feedback_import_error() -> None:
    stderr = "ModuleNotFoundError: No module named 'lightgbm'"
    stdout = "Baseline RMSE score: 0.1"
    feedback = _extract_error_feedback(stderr, stdout)

    assert "MUTATION HINT [ImportError]" in feedback
    assert "try/except fallback pattern" in feedback
    assert "from lightgbm import LGBMRegressor" in feedback


def test_extract_error_feedback_column_mismatch() -> None:
    stderr = "ValueError: columns are missing: {'Id'}"
    stdout = "Baseline RMSE score: 0.1"
    feedback = _extract_error_feedback(stderr, stdout)

    assert "MUTATION HINT [ColumnError]" in feedback
    assert "test features match train" in feedback
    assert "X_test = test_df[X.columns]" in feedback


def test_extract_error_feedback_missing_baseline() -> None:
    stderr = "NameError: name 'score' is not defined"
    stdout = ""
    feedback = _extract_error_feedback(stderr, stdout)

    assert "MUTATION HINT [MissingBaseline]" in feedback
    assert "Add baseline logging" in feedback
    assert "MUTATION HINT [NameError]" in feedback


def test_submission_mutation_hint_quotes_issues() -> None:
    validation = SubmissionValidation(
        valid=False,
        issues=("missing columns ['target']",),
        row_count=3,
        expected_row_count=3,
        columns=("id", "prediction"),
        expected_columns=("id", "target"),
    )
    hint = _submission_mutation_hint(validation)

    assert "MUTATION HINT [InvalidSubmission]" in hint
    assert "missing columns ['target']" in hint
    assert 'sample = pd.read_csv("sample_submission.csv")' in hint


def test_fitness_from_score_keeps_minimized_scores_positive() -> None:
    assert _fitness_from_score(0.5, "minimize") == 1.0 / 1.5
    assert _fitness_from_score(2.0, "minimize") < _fitness_from_score(0.5, "minimize")
    assert _fitness_from_score(None, "minimize") == 0.0
    assert _fitness_from_score(0.5, "minimize") > _fitness_from_score(None, "minimize")


def test_fitness_from_score_passes_through_maximized_scores() -> None:
    assert _fitness_from_score(0.75, "maximize") == 0.75
    assert _fitness_from_score(-1.0, "maximize") == 0.0
