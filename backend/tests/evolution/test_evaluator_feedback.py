"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.evolution.evaluator import _compute_cv_variance, _extract_error_feedback

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


class TestComputeCvVariance:
    """Tests for fold-score variance extraction from solution stdout."""

    def test_variance_with_decimal_folds(self) -> None:
        """Plain decimal fold scores yield the standard population variance."""
        stdout = "Fold 1: 0.80\nFold 2: 0.82\nFold 3: 0.78\n"
        # mean = 0.80, deviations 0.0, 0.02, -0.02 → variance ≈ 0.000266667
        assert _compute_cv_variance(stdout) == pytest.approx(0.0002666667, rel=1e-3)

    def test_variance_with_scientific_notation(self) -> None:
        """Scientific-notation fold scores must contribute to the variance."""
        stdout = "Fold 1: 1.0e-3\nFold 2: 2.0e-3\nFold 3: 3.0e-3\n"
        # mean = 2e-3, deviations -1e-3, 0, 1e-3 → variance ≈ 6.667e-7
        assert _compute_cv_variance(stdout) == pytest.approx(6.6667e-7, rel=1e-3)

    def test_variance_with_signed_folds(self) -> None:
        """Signed fold scores parse correctly (e.g. negative log-likelihood)."""
        stdout = "Fold 1: -0.5\nFold 2: -0.7\n"
        # mean = -0.6, deviations 0.1, -0.1 → variance = 0.01
        assert _compute_cv_variance(stdout) == pytest.approx(0.01)

    def test_variance_skips_non_finite_tokens(self) -> None:
        """nan/inf fold tokens are ignored without poisoning the variance."""
        stdout = "Fold 1: 0.5\nFold 2: nan\nFold 3: 0.7\nFold 4: inf\n"
        # nan/inf are not matched by the float regex, so only 0.5 and 0.7 remain.
        # mean = 0.6, deviations -0.1, 0.1 → variance = 0.01
        assert _compute_cv_variance(stdout) == pytest.approx(0.01)

    def test_variance_returns_zero_when_too_few_folds(self) -> None:
        """A single fold score is insufficient to compute variance."""
        assert _compute_cv_variance("Fold 1: 0.5\n") == 0.0
        assert _compute_cv_variance("no folds reported") == 0.0
