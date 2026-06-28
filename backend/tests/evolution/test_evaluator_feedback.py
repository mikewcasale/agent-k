"""Tests for OpenEvolve evaluator error feedback extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from agent_k.evolution.evaluator import _extract_error_feedback, _extract_warnings

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


def test_extract_warnings_parses_python_warnings_module() -> None:
    stderr = (
        "/opt/.venv/lib/python3.13/site-packages/sklearn/foo.py:42: "
        "ConvergenceWarning: lbfgs failed to converge (status=1):\n"
        "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n"
    )

    warnings = _extract_warnings(stderr)

    assert warnings == ["ConvergenceWarning: lbfgs failed to converge (status=1):"]


def test_extract_warnings_parses_library_bracket_format() -> None:
    stderr = (
        "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n"
        "[LightGBM] [Info] Number of positive: 100, number of negative: 50\n"
    )

    warnings = _extract_warnings(stderr)

    assert warnings == ["LightGBMWarning: No further splits with positive gain, best gain: -inf"]


def test_extract_warnings_deduplicates_repeated_emissions() -> None:
    stderr = "\n".join(
        f"/tmp/solution.py:{lineno}: DeprecationWarning: pass keyword 'silent=True'" for lineno in (12, 18, 24, 36)
    )

    warnings = _extract_warnings(stderr)

    assert warnings == ["DeprecationWarning: pass keyword 'silent=True'"]


def test_extract_warnings_dedupes_after_numeric_normalization() -> None:
    stderr = (
        "[LightGBM] [Warning] Stopped by early_stopping_rounds at iteration 50\n"
        "[LightGBM] [Warning] Stopped by early_stopping_rounds at iteration 73\n"
        "[LightGBM] [Warning] Stopped by early_stopping_rounds at iteration 91\n"
    )

    warnings = _extract_warnings(stderr)

    assert warnings == ["LightGBMWarning: Stopped by early_stopping_rounds at iteration 50"]


def test_extract_warnings_ignores_non_warning_lines() -> None:
    stderr = (
        "Reading input file...\n"
        "print('No warnings here')\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/x.py", line 1, in <module>\n'
        "    raise RuntimeError('boom')\n"
    )

    assert _extract_warnings(stderr) == []


def test_extract_warnings_caps_at_limit() -> None:
    suffixes = ("alpha", "beta", "gamma", "delta", "epsilon")
    stderr = "\n".join(f"foo.py:10: UserWarning: case-{name}" for name in suffixes)

    warnings = _extract_warnings(stderr, limit=3)

    assert warnings == ["UserWarning: case-alpha", "UserWarning: case-beta", "UserWarning: case-gamma"]


def test_extract_warnings_inline_category_without_path() -> None:
    stderr = "FutureWarning: pandas.read_csv accepts engine='c' which will be deprecated"

    warnings = _extract_warnings(stderr)

    assert warnings == ["FutureWarning: pandas.read_csv accepts engine='c' which will be deprecated"]


def test_extract_warnings_xgboost_style_generic() -> None:
    stderr = "[20:34:56] WARNING: src/learner.cc:553: No visible GPU is found, setting `gpu_id` to 0"

    warnings = _extract_warnings(stderr)

    assert warnings == ["Warning: src/learner.cc:553: No visible GPU is found, setting `gpu_id` to 0"]
