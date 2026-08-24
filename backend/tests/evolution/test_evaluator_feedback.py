"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.evolution.evaluator import _compute_cv_variance, _extract_error_feedback

__all__ = ()


def test_compute_cv_variance_returns_zero_for_insufficient_folds() -> None:
    """Fewer than two fold scores cannot yield a variance signal."""
    assert _compute_cv_variance("Fold 1: 0.85") == 0.0
    assert _compute_cv_variance("") == 0.0


def test_compute_cv_variance_matches_manual_calculation() -> None:
    """Variance should match the population variance of parsed fold scores."""
    stdout = "Fold 1: 0.80\nFold 2: 0.90\nFold 3: 0.85"
    mean = (0.80 + 0.90 + 0.85) / 3
    expected = sum((s - mean) ** 2 for s in (0.80, 0.90, 0.85)) / 3

    assert _compute_cv_variance(stdout) == pytest.approx(expected)


def test_compute_cv_variance_handles_scientific_and_negative_scores() -> None:
    """Sci-notation and negative fold scores must contribute to variance."""
    stdout = "Fold 1: 1.0e-3\nFold 2: 2.0e-3\nFold 3: 3.0e-3"
    variance = _compute_cv_variance(stdout)

    # Should be strictly positive; the old regex would drop these entirely and
    # return 0.0, hiding stability differences between candidates.
    assert variance > 0.0

    negatives_stdout = "Fold 1: -0.5\nFold 2: -0.6"
    assert _compute_cv_variance(negatives_stdout) > 0.0


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
