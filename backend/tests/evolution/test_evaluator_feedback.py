"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from agent_k.evolution.evaluator import _extract_error_feedback

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
