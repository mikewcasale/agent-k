"""Tests for solution execution utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import pytest

import agent_k.core.solution as solution_module
from agent_k.core.solution import (
    _execute_with_builtin_tool,
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


def _seed_inline_data_files(work_path: Path) -> None:
    """Write minimal CSVs so `_load_inline_files` does not bail out."""
    (work_path / "train.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (work_path / "test.csv").write_text("a,b\n3,4\n", encoding="utf-8")
    (work_path / "sample_submission.csv").write_text("a,b\n5,6\n", encoding="utf-8")


class _FailingBuiltinAgent:
    """Test double whose `run` always raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def run(self, _script: str) -> Any:
        raise self._exc


class _NoToolBuiltinAgent:
    """Test double whose `run` returns a payload that omits any tool call."""

    async def run(self, _script: str) -> Any:
        class _Run:
            @staticmethod
            def all_messages() -> list[Any]:
                return []

        return _Run()


class TestExecuteWithBuiltinToolObservability:
    """Failures in the OpenAI builtin code-execution tool must surface in logfire."""

    async def test_run_exception_is_logged_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the agent raises, log a warning and fall back via `None`."""
        _seed_inline_data_files(tmp_path)
        monkeypatch.setattr(solution_module, "_supports_code_execution", lambda _spec: True)
        monkeypatch.setattr(
            solution_module, "_get_code_execution_agent", lambda _spec: _FailingBuiltinAgent(RuntimeError("boom"))
        )

        warnings: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            solution_module.logfire, "warning", lambda event, **kwargs: warnings.append((event, kwargs))
        )

        result = await _execute_with_builtin_tool(
            "print('hi')", tmp_path, env=None, model_spec="openai:gpt-4o", max_inline_data_bytes=1_000_000
        )

        assert result is None
        assert any(event == "builtin_code_execution_failed" for event, _ in warnings)
        failed_kwargs = next(kwargs for event, kwargs in warnings if event == "builtin_code_execution_failed")
        assert failed_kwargs["error_type"] == "RuntimeError"
        assert failed_kwargs["model_spec"] == "openai:gpt-4o"
        assert failed_kwargs["error"] == "boom"
        assert isinstance(failed_kwargs["runtime_ms"], int)

    async def test_missing_tool_call_is_logged_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the model does not invoke the tool, log a warning and fall back via `None`."""
        _seed_inline_data_files(tmp_path)
        monkeypatch.setattr(solution_module, "_supports_code_execution", lambda _spec: True)
        monkeypatch.setattr(solution_module, "_get_code_execution_agent", lambda _spec: _NoToolBuiltinAgent())

        warnings: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            solution_module.logfire, "warning", lambda event, **kwargs: warnings.append((event, kwargs))
        )

        result = await _execute_with_builtin_tool(
            "print('hi')", tmp_path, env=None, model_spec="openai:gpt-4o", max_inline_data_bytes=1_000_000
        )

        assert result is None
        assert any(event == "builtin_code_execution_no_tool_call" for event, _ in warnings)
