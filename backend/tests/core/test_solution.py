"""Tests for solution execution utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from agent_k.core import solution as solution_module
from agent_k.core.solution import (
    _is_sensitive_env_key,
    _sanitize_env,
    _supports_code_execution,
    execute_solution,
    parse_baseline_score,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import TestEnv

__all__ = ()

pytestmark = pytest.mark.anyio

_REMOTE_MODEL_SPEC = "openai-responses:gpt-4o"


class _StubCodeExecutionAgent:
    """Stand-in for the remote code-execution agent.

    Sleeps for `delay_seconds`, then either raises or returns a response with no
    code-execution part, which is the shape the fallback path keys off.
    """

    def __init__(self, *, delay_seconds: float, error: Exception | None = None) -> None:
        self._delay_seconds = delay_seconds
        self._error = error
        self.calls = 0

    async def run(self, _script: str) -> Any:
        self.calls += 1
        await asyncio.sleep(self._delay_seconds)
        if self._error is not None:
            raise self._error
        return _StubRunResult()


class _StubRunResult:
    """Run result carrying no builtin code-execution return part."""

    def all_messages(self) -> list[Any]:
        return []


def _write_inline_data_files(work_path: Path) -> None:
    """Create the CSVs the remote path inlines before it calls the model."""
    (work_path / "train.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")
    (work_path / "test.csv").write_text("id\n2\n", encoding="utf-8")
    (work_path / "sample_submission.csv").write_text("id,target\n2,0.0\n", encoding="utf-8")


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


class TestSupportsCodeExecution:
    """Tests for remote code-execution provider gating."""

    @pytest.mark.parametrize(
        ("model_spec", "expected"),
        [
            ("openai-responses:gpt-4o", True),
            # `openai:` resolves to OpenAIChatModel, which rejects CodeExecutionTool.
            ("openai:gpt-4o", False),
            ("openrouter:openai/gpt-oss-120b:free", False),
            ("devstral:local", False),
            ("anthropic:claude-3-haiku-20240307", False),
        ],
    )
    def test_supports_code_execution(self, model_spec: str, expected: bool) -> None:
        """Only specs resolving to a builtin code runner should be accepted."""
        assert _supports_code_execution(model_spec) is expected

    async def test_unsupported_spec_skips_remote_agent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unsupported spec should never build or call the remote agent."""
        _write_inline_data_files(tmp_path)

        def fail(_model_spec: str) -> Any:  # pragma: no cover - must not be reached
            pytest.fail("remote agent was built for an unsupported model spec")

        monkeypatch.setattr(solution_module, "_get_code_execution_agent", fail)

        result = await execute_solution(
            "print('local')\n",
            tmp_path,
            timeout_seconds=10,
            use_builtin_code_execution=True,
            model_spec="openai:gpt-4o",
        )

        assert result.returncode == 0
        assert "local" in result.stdout


class TestRemoteCodeExecution:
    """Tests for the builtin code-execution path's time budget."""

    async def test_remote_timeout_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hung remote call should be cut off and reported as a timeout."""
        _write_inline_data_files(tmp_path)
        agent = _StubCodeExecutionAgent(delay_seconds=30)
        monkeypatch.setattr(solution_module, "_get_code_execution_agent", lambda _spec: agent)

        marker = tmp_path / "ran_locally.txt"
        code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n"
        result = await execute_solution(
            code, tmp_path, timeout_seconds=0.2, use_builtin_code_execution=True, model_spec=_REMOTE_MODEL_SPEC
        )

        assert agent.calls == 1
        assert result.timed_out is True
        assert result.returncode != 0
        assert result.runtime_ms < 30_000
        # The budget belongs to the remote attempt, so no local re-run may happen.
        assert not marker.exists()

    async def test_remote_failure_falls_back_locally(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fast remote failure should still leave budget for local execution."""
        _write_inline_data_files(tmp_path)
        agent = _StubCodeExecutionAgent(delay_seconds=0, error=RuntimeError("provider rejected the tool"))
        monkeypatch.setattr(solution_module, "_get_code_execution_agent", lambda _spec: agent)

        result = await execute_solution(
            "print('fallback ran')\n",
            tmp_path,
            timeout_seconds=10,
            use_builtin_code_execution=True,
            model_spec=_REMOTE_MODEL_SPEC,
        )

        assert agent.calls == 1
        assert result.returncode == 0
        assert "fallback ran" in result.stdout

    async def test_exhausted_budget_skips_local_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A remote failure that burns the whole budget must not double it locally."""
        _write_inline_data_files(tmp_path)
        agent = _StubCodeExecutionAgent(delay_seconds=0.3, error=RuntimeError("provider died late"))
        monkeypatch.setattr(solution_module, "_get_code_execution_agent", lambda _spec: agent)

        marker = tmp_path / "ran_locally.txt"
        code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n"
        result = await execute_solution(
            code, tmp_path, timeout_seconds=0.2, use_builtin_code_execution=True, model_spec=_REMOTE_MODEL_SPEC
        )

        assert agent.calls == 1
        assert result.timed_out is True
        assert not marker.exists()

    async def test_missing_remote_result_falls_back_locally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response without a code-execution part should fall back to local execution."""
        _write_inline_data_files(tmp_path)
        agent = _StubCodeExecutionAgent(delay_seconds=0)
        monkeypatch.setattr(solution_module, "_get_code_execution_agent", lambda _spec: agent)

        result = await execute_solution(
            "print('local after empty remote')\n",
            tmp_path,
            timeout_seconds=10,
            use_builtin_code_execution=True,
            model_spec=_REMOTE_MODEL_SPEC,
        )

        assert agent.calls == 1
        assert result.returncode == 0
        assert "local after empty remote" in result.stdout
