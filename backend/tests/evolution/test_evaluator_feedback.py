"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.evolution.evaluator import _extract_error_feedback, _fitness_from_score

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


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_fitness_from_score_clamps_non_finite_to_zero(bad_score: float, direction: str) -> None:
    """Non-finite cv scores must collapse to the 'no info' baseline fitness."""
    assert _fitness_from_score(bad_score, direction) == 0.0


def test_fitness_from_score_returns_zero_on_none() -> None:
    """None cv_score (parse miss) still maps to 0.0."""
    assert _fitness_from_score(None, "maximize") == 0.0
