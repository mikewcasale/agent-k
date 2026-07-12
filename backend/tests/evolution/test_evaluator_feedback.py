"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

from agent_k.core.solution import ExecutionResult
from agent_k.evolution.evaluator import _extract_error_feedback, evaluate

if TYPE_CHECKING:
    from pathlib import Path

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


def test_extract_error_feedback_missing_submission_requires_code() -> None:
    """MissingSubmission hint fires only when code context is provided."""
    stderr = ""
    stdout = "Baseline RMSE score: 0.1\nFold 1: 0.09"

    without_code = _extract_error_feedback(stderr, stdout)
    with_code = _extract_error_feedback(stderr, stdout, code="import pandas as pd\n")

    assert "MUTATION HINT [MissingSubmission]" not in without_code
    assert "MUTATION HINT [MissingSubmission]" in with_code
    assert "submission_df.to_csv('submission.csv', index=False)" in with_code


def test_extract_error_feedback_clean_output_hints_only() -> None:
    """Clean stderr but incomplete stdout should emit only output-quality hints."""
    feedback = _extract_error_feedback(stderr="", stdout="", code="import pandas as pd\n")

    assert "MUTATION HINT [MissingBaseline]" in feedback
    assert "MUTATION HINT [MissingFolds]" in feedback
    assert "MUTATION HINT [MissingSubmission]" in feedback
    assert "MUTATION HINT [ImportError]" not in feedback
    assert "MUTATION HINT [SyntaxError]" not in feedback


def test_evaluate_emits_feedback_when_clean_exit_but_missing_baseline(tmp_path: Path) -> None:
    """evaluate() should emit error_feedback when returncode==0 but result is invalid.

    Regression: previously the feedback branch gated on ``returncode != 0``, so a
    solution that ran to completion but forgot to print its baseline score or
    write submission.csv returned an empty ``error_feedback`` artifact and the
    mutation LLM received no guidance about the missing output.
    """
    program = tmp_path / "solution.py"
    program.write_text("import pandas as pd\nprint('did nothing useful')\n", encoding="utf-8")

    result = ExecutionResult(returncode=0, stdout="did nothing useful\n", stderr="", runtime_ms=42, timed_out=False)

    async def _fake_execute_solution(*_args: object, **_kwargs: object) -> ExecutionResult:
        return result

    context_payload = json.dumps({"work_dir": str(tmp_path), "metric_direction": "minimize", "hints": []})
    previous = os.environ.get("AGENT_K_OPENEVOLVE_CONTEXT")
    os.environ["AGENT_K_OPENEVOLVE_CONTEXT"] = context_payload
    try:
        with patch("agent_k.core.solution.execute_solution", new=_fake_execute_solution):
            evaluation = evaluate(str(program))
    finally:
        if previous is None:
            os.environ.pop("AGENT_K_OPENEVOLVE_CONTEXT", None)
        else:
            os.environ["AGENT_K_OPENEVOLVE_CONTEXT"] = previous

    assert evaluation.metrics["valid"] == 0.0
    feedback = evaluation.artifacts["error_feedback"]
    assert feedback, "expected non-empty error_feedback when result is invalid"
    assert "MUTATION HINT [MissingBaseline]" in feedback
    assert "MUTATION HINT [MissingSubmission]" in feedback
