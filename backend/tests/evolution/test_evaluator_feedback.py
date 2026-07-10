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


def test_extract_error_feedback_timeout_flag_fires_hint_on_empty_stderr() -> None:
    """SIGKILL leaves stderr empty; the timed_out flag must still surface the hint."""
    feedback = _extract_error_feedback("", "", timed_out=True)

    assert "MUTATION HINT [Timeout]" in feedback
    assert "Speed up execution" in feedback


def test_extract_error_feedback_timeout_not_duplicated() -> None:
    """When both the flag and stderr signal a timeout, only one hint is emitted."""
    stderr = "Process timed out after 120 seconds"
    feedback = _extract_error_feedback(stderr, "", timed_out=True)

    assert feedback.count("MUTATION HINT [Timeout]") == 1


def test_extract_error_feedback_stderr_only_timeout_still_fires() -> None:
    """Backwards-compat: stderr-based detection continues to work without the flag."""
    stderr = "asyncio.TimeoutError: task timed out"
    feedback = _extract_error_feedback(stderr, "Baseline RMSE score: 0.5")

    assert "MUTATION HINT [Timeout]" in feedback
