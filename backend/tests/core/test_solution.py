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

    async def test_execute_solution_reports_exact_output_sizes(self, tmp_path: Path) -> None:
        """Output that fits the budget should be returned verbatim."""
        code = "import sys\nprint('hello')\nsys.stderr.write('oops\\n')\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=30)

        assert result.stdout == "hello\n"
        assert result.stderr == "oops\n"
        assert result.stdout_bytes == 6
        assert result.stderr_bytes == 5
        assert result.output_truncated is False

    async def test_execute_solution_bounds_runaway_output(self, tmp_path: Path) -> None:
        """A solution writing far past the budget should be capped without losing the score line."""
        code = (
            "import sys\n"
            "line = 'x' * 1000\n"
            "for _ in range(5_000):\n"
            "    print(line)\n"
            "    sys.stderr.write(line + '\\n')\n"
            "print('Baseline RMSE score: 0.25')\n"
        )
        result = await execute_solution(code, tmp_path, timeout_seconds=60, max_output_bytes=8_192)

        assert result.output_truncated is True
        assert result.stdout_bytes > 5_000_000
        assert result.stderr_bytes > 5_000_000
        assert len(result.stdout) < 10_000
        assert len(result.stderr) < 10_000
        assert parse_baseline_score(result.stdout) == pytest.approx(0.25)

    async def test_execute_solution_survives_full_pipe_buffer(self, tmp_path: Path) -> None:
        """Writing more than the OS pipe buffer must not deadlock the runner."""
        code = "print('y' * 4_000_000)\nimport sys\nsys.exit(0)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=60)

        assert result.returncode == 0
        assert result.timed_out is False
        assert result.stdout_bytes > 4_000_000

    async def test_execute_solution_bounds_output_on_timeout(self, tmp_path: Path) -> None:
        """A noisy solution that also times out should still be capped and reported."""
        code = "import sys\nwhile True:\n    sys.stdout.write('z' * 1000)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=1, max_output_bytes=4_096)

        assert result.timed_out is True
        assert len(result.stdout) < 6_000


class TestBoundedCapture:
    """Tests for the bounded output capture."""

    def test_keeps_short_streams_intact(self) -> None:
        """Streams under the limit should round-trip unchanged."""
        capture = _BoundedCapture(limit=1_024)
        capture.feed(b"first\n")
        capture.feed(b"second\n")

        assert capture.render() == "first\nsecond\n"
        assert capture.total_bytes == 13
        assert capture.truncated is False
        assert capture.dropped_bytes == 0

    def test_boundary_fill_is_not_truncated(self) -> None:
        """A stream exactly at the limit should not be reported as truncated."""
        capture = _BoundedCapture(limit=100)
        capture.feed(b"a" * 100)

        assert capture.truncated is False
        assert capture.render() == "a" * 100

    def test_keeps_head_and_tail_when_over_limit(self) -> None:
        """Over-limit streams should keep the head, the tail, and a dropped-byte notice."""
        capture = _BoundedCapture(limit=100)
        capture.feed(b"H" * 40)
        capture.feed(b"M" * 5_000)
        capture.feed(b"T" * 40)

        rendered = capture.render()

        assert capture.total_bytes == 5_080
        assert capture.truncated is True
        assert rendered.startswith("H" * 25)
        assert rendered.endswith("T" * 40)
        assert "bytes of output dropped" in rendered
        assert capture.dropped_bytes == 5_080 - 25 - 75

    def test_chunking_does_not_change_the_excerpt(self) -> None:
        """Byte-at-a-time feeding should match a single bulk feed."""
        payload = bytes(range(256)) * 40

        bulk = _BoundedCapture(limit=200)
        bulk.feed(payload)

        drip = _BoundedCapture(limit=200)
        for index in range(0, len(payload), 7):
            drip.feed(payload[index : index + 7])

        assert drip.render() == bulk.render()
        assert drip.total_bytes == bulk.total_bytes == len(payload)

    def test_excerpt_stays_bounded_across_many_chunks(self) -> None:
        """Two megabytes fed in chunks should still render close to the limit."""
        capture = _BoundedCapture(limit=1_000)
        for _ in range(2_000):
            capture.feed(b"q" * 1_000)

        assert capture.total_bytes == 2_000_000
        assert capture.truncated is True
        assert len(capture.render()) <= 1_100
