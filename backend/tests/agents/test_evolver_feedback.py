"""Tests for evolver error feedback helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.agents.evolver import _build_error_feedback

__all__ = ()


@pytest.mark.parametrize(
    ("stderr", "error", "timed_out", "returncode", "expected_category", "expected_snippet"),
    [
        (
            "ModuleNotFoundError: No module named 'lightgbm'",
            "Execution failed (exit 1)",
            False,
            1,
            "import_error",
            "try/except",
        ),
        (
            "ValueError: columns are missing: {'Id'}",
            "Execution failed (exit 1)",
            False,
            1,
            "column_mismatch",
            "align train/test",
        ),
        ("", "submission.csv not found after execution", False, 0, "missing_submission", "submission.csv"),
        (
            "",
            "Unable to score submission: submission.csv missing target columns: ['y']",
            False,
            0,
            "submission_schema",
            "sample_submission",
        ),
        ("", "Execution timed out", True, 1, "timeout", "timed out"),
    ],
)
def test_build_error_feedback(
    stderr: str, error: str, timed_out: bool, returncode: int, expected_category: str, expected_snippet: str
) -> None:
    category, feedback = _build_error_feedback(stderr=stderr, error=error, timed_out=timed_out, returncode=returncode)

    assert category == expected_category
    assert expected_snippet.lower() in feedback.lower()
