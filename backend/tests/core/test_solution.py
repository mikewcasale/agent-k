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
    truncate_output,
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
        ],
    )
    def test_parse_baseline_score(self, output: str, expected: float | None) -> None:
        """Baseline score parsing should handle common formats."""
        result = parse_baseline_score(output)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestTruncateOutput:
    """Tests for the head+tail execution output truncation helper."""

    def test_returns_text_when_within_limit(self) -> None:
        """Text shorter than the limit should pass through unchanged."""
        text = "short output"
        assert truncate_output(text, 100) == text

    def test_returns_empty_for_non_positive_limit(self) -> None:
        """Zero or negative limits should return an empty string."""
        assert truncate_output("anything", 0) == ""
        assert truncate_output("anything", -5) == ""

    def test_preserves_tail_with_marker(self) -> None:
        """Tail should survive truncation so final scores/errors are kept."""
        text = "X" * 200 + "FINAL_SCORE=0.123"
        result = truncate_output(text, 80)

        assert len(result) <= 80
        assert result.endswith("FINAL_SCORE=0.123")
        assert "truncated" in result

    def test_preserves_head_and_tail(self) -> None:
        """Both ends of the text should appear in the truncated result."""
        text = "HEAD_MARKER" + "x" * 500 + "TAIL_MARKER"
        result = truncate_output(text, 120, tail_fraction=0.5)

        assert "HEAD_MARKER" in result or result.startswith("HEAD")
        assert result.endswith("TAIL_MARKER")
        assert len(result) <= 120

    def test_tail_fraction_one_yields_all_tail(self) -> None:
        """tail_fraction=1.0 should drop the head entirely (besides marker)."""
        text = "abc" + "x" * 500 + "ZZZZ"
        result = truncate_output(text, 60, tail_fraction=1.0)

        assert result.endswith("ZZZZ")
        assert len(result) <= 60
        assert not result.startswith("abc")

    def test_invalid_tail_fraction_raises(self) -> None:
        """tail_fraction outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="tail_fraction"):
            truncate_output("x" * 100, 50, tail_fraction=1.5)
        with pytest.raises(ValueError, match="tail_fraction"):
            truncate_output("x" * 100, 50, tail_fraction=-0.1)

    def test_extreme_small_limit_returns_tail_slice(self) -> None:
        """When max_length cannot fit the marker, return the trailing slice."""
        text = "x" * 100 + "END"
        result = truncate_output(text, 5)

        assert len(result) == 5
        assert result.endswith("END")

    def test_records_dropped_char_count(self) -> None:
        """Marker should report how many chars were dropped from the middle."""
        text = "x" * 1000
        result = truncate_output(text, 200)

        assert len(result) <= 200
        # 1000 original - (200 - marker_len) kept = dropped count
        # Marker template is "\n... [{count} chars truncated] ...\n"
        import re as _re

        match = _re.search(r"\[(\d+) chars truncated\]", result)
        assert match is not None
        dropped = int(match.group(1))
        kept = len(result) - len(match.group(0)) - len("\n...  ...\n")
        assert dropped + kept == 1000


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
