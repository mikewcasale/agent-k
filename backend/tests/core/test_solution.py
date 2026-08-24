"""Tests for solution execution utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pytest

from agent_k.core.solution import (
    _is_sensitive_env_key,
    _sanitize_env,
    execute_solution,
    parse_baseline_score,
    parse_fold_scores,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import TestEnv

__all__ = ()

pytestmark = pytest.mark.anyio


class TestParseBaselineScore:
    """Tests for baseline score parsing."""

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("Baseline RMSE score: 0.123", 0.123),
            ("baseline accuracy score: -1.5", -1.5),
            ("some text\nBaseline logLoss score: 1.2345\n", 1.2345),
            ("no score here", None),
            ("Baseline score: not-a-number", None),
            # Scientific notation: previously silently truncated (e.g. ``1.2345``
            # instead of ``1.2345e-05``), destroying fitness signal for metrics
            # like RMSLE on normalized targets.
            ("Baseline RMSLE score: 1.2345e-05", 1.2345e-05),
            ("Baseline MAE score: -3.2e2", -320.0),
            ("Baseline logLoss score: 4.5E+2", 450.0),
            # Missing metric label: previously failed to match.
            ("Baseline score: 0.834", 0.834),
            # Alternate separators / signs.
            ("Baseline score = 0.42", 0.42),
            ("Baseline MAE score: +1.5", 1.5),
            (".Baseline MAE score:.5", 0.5),
            # Non-finite values must be rejected so evolution cannot rank
            # crashed runs above real solutions.
            ("Baseline MAE score: nan", None),
            ("Baseline MAE score: inf", None),
            ("Baseline MAE score: -inf", None),
        ],
    )
    def test_parse_baseline_score(self, output: str, expected: float | None) -> None:
        """Baseline score parsing should handle common formats."""
        result = parse_baseline_score(output)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestParseFoldScores:
    """Tests for per-fold CV score parsing used by evolution stability metrics."""

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("Fold 1: 0.85\nFold 2: 0.83\nFold 3: 0.87", [0.85, 0.83, 0.87]),
            # Sci-notation and negatives (log-losses returned as negatives by
            # sklearn cross_val_score with ``neg_log_loss``).
            ("Fold 1: -1.2e-3\nFold 2: -1.4e-3", [-1.2e-3, -1.4e-3]),
            # Colon-only separator without whitespace.
            ("Fold 1:0.85\nFold 2:0.83", [0.85, 0.83]),
            # Whitespace-only separator.
            ("Fold 1 0.85\nFold 2 0.83", [0.85, 0.83]),
            # Equals separator.
            ("Fold 1 = 0.9\nFold 2 = 0.91", [0.9, 0.91]),
            # ``Fold 10.85`` (no separator) must not be misparsed as
            # ``Fold 1 -> 0.85``.
            ("Fold 10.85", []),
            # nan/inf must be dropped, not propagated into variance.
            ("Fold 1: 0.85\nFold 2: nan\nFold 3: 0.87", [0.85, 0.87]),
            # No matches.
            ("no folds here", []),
        ],
    )
    def test_parse_fold_scores(self, output: str, expected: list[float]) -> None:
        """Fold score parsing should tolerate common formats and reject junk."""
        assert parse_fold_scores(output) == pytest.approx(expected)


class TestEnvSanitization:
    """Tests for environment sanitization helpers."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [("OPENAI_API_KEY", True), ("kaggle_key", True), ("my_token", True), ("PATH", False), ("DATA_DIR", False)],
    )
    def test_is_sensitive_env_key(self, key: str, expected: bool) -> None:
        """Sensitive keys should be detected case-insensitively."""
        assert _is_sensitive_env_key(key) is expected

    def test_sanitize_env_filters_sensitive_keys(self, env: TestEnv, tmp_path: Path) -> None:
        """Sanitization should drop sensitive keys and set defaults."""
        env.set("KAGGLE_KEY", "secret")
        env.set("SAFE_VAR", "ok")

        sanitized = _sanitize_env({"EXTRA": "1"}, work_path=tmp_path)

        assert "KAGGLE_KEY" not in sanitized
        assert sanitized["SAFE_VAR"] == "ok"
        assert sanitized["EXTRA"] == "1"
        assert sanitized["HOME"] == str(tmp_path)
        assert sanitized["PYTHONNOUSERSITE"] == "1"
        assert sanitized["PYTHONDONTWRITEBYTECODE"] == "1"


class TestExecuteSolution:
    """Tests for execute_solution behavior."""

    async def test_execute_solution_nonzero_exit(self, tmp_path: Path) -> None:
        """Execution should capture non-zero return codes."""
        code = "import sys\nsys.exit(7)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=1)

        assert result.returncode == 7
        assert result.timed_out is False

    async def test_execute_solution_timeout(self, tmp_path: Path) -> None:
        """Execution should report timeouts."""
        code = "import time\ntime.sleep(1)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=0.2)

        assert result.timed_out is True
        assert result.returncode != 0
