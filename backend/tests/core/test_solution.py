"""Tests for solution execution utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pytest

from agent_k.core.solution import (
    _BoundedCapture,
    _is_sensitive_env_key,
    _sanitize_env,
    execute_solution,
    parse_baseline_score,
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

    async def test_execute_solution_timeout_keeps_partial_output(self, tmp_path: Path) -> None:
        """Output written before a timeout should survive the kill."""
        code = (
            "import sys, time\n"
            "print('Baseline RMSE score: 0.5', flush=True)\n"
            "sys.stderr.write('WARNING: slow fold\\n')\n"
            "sys.stderr.flush()\n"
            "time.sleep(30)\n"
        )
        result = await execute_solution(code, tmp_path, timeout_seconds=1)

        assert result.timed_out is True
        assert parse_baseline_score(result.stdout) == pytest.approx(0.5)
        assert "WARNING: slow fold" in result.stderr

    async def test_execute_solution_caps_stdout(self, tmp_path: Path) -> None:
        """Runaway stdout should be capped to the configured budget."""
        max_output_bytes = 4096
        code = "for i in range(20000):\n    print('%06d' % i + 'x' * 64)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=60, max_output_bytes=max_output_bytes)

        assert result.returncode == 0
        encoded = result.stdout.encode("utf-8")
        assert len(encoded) < max_output_bytes * 2
        assert "bytes elided" in result.stdout
        assert result.stdout.startswith("000000")
        assert result.stdout.rstrip().endswith("x" * 64)

    async def test_execute_solution_keeps_short_output_intact(self, tmp_path: Path) -> None:
        """Output within the budget should not be elided."""
        code = "print('line one')\nprint('line two')\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=30)

        assert result.stdout == "line one\nline two\n"
        assert "elided" not in result.stdout


class TestBoundedCapture:
    """Tests for the bounded stdout/stderr buffer."""

    def test_under_budget_is_verbatim(self) -> None:
        """Captures below the budget should round-trip unchanged."""
        capture = _BoundedCapture(64)
        capture.feed(b"hello ")
        capture.feed(b"world")

        assert capture.result() == b"hello world"

    def test_over_budget_keeps_head_and_tail(self) -> None:
        """Captures above the budget should keep both ends and report the gap."""
        capture = _BoundedCapture(10)
        capture.feed(b"0123456789" * 3)

        result = capture.result()
        assert result.startswith(b"01234")
        assert result.endswith(b"56789")
        assert b"[20 bytes elided]" in result

    def test_chunked_feeds_match_single_feed(self) -> None:
        """Chunk boundaries should not change the captured window."""
        payload = bytes(range(256)) * 8
        single = _BoundedCapture(128)
        single.feed(payload)

        chunked = _BoundedCapture(128)
        for start in range(0, len(payload), 7):
            chunked.feed(payload[start : start + 7])

        assert chunked.result() == single.result()

    def test_zero_budget_drops_everything(self) -> None:
        """A zero budget should retain no payload bytes."""
        capture = _BoundedCapture(0)
        capture.feed(b"discarded")

        assert capture.result() == b"\n...[9 bytes elided]...\n"

    def test_empty_feed_is_ignored(self) -> None:
        """Empty chunks should not create an elision marker."""
        capture = _BoundedCapture(8)
        capture.feed(b"")

        assert capture.result() == b""
