"""Tests for OpenEvolve evaluator stage 2 pass-through behavior.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_k.core.solution import ExecutionResult
from agent_k.evolution import evaluator as evaluator_module
from agent_k.evolution.evaluator import evaluate_stage2

__all__ = ()


def _make_execute_stub(*, returncode: int, stdout: str, stderr: str = "", write_submission: bool = True) -> Any:
    """Return an async stub for execute_solution that mimics a run.

    The stub optionally writes a submission.csv into the work directory so
    stage 2's validity gate (`submission_path.exists()`) passes.
    """

    async def _stub(
        code: str, work_path: Path, *, timeout_seconds: float | None = None, env: dict[str, str] | None = None
    ) -> ExecutionResult:
        if write_submission:
            (work_path / "submission.csv").write_text("Id,target\n1,0.5\n", encoding="utf-8")
        return ExecutionResult(returncode=returncode, stdout=stdout, stderr=stderr, runtime_ms=10, timed_out=False)

    return _stub


def _install_context(monkeypatch: pytest.MonkeyPatch, work_dir: Path, direction: str) -> None:
    """Set the context env var evaluate_stage2 reads on entry."""
    context = {"work_dir": str(work_dir), "metric_direction": direction}
    monkeypatch.setenv(evaluator_module._CONTEXT_ENV, json.dumps(context))


def _write_program(tmp_path: Path, marker: str = "print('hi')") -> Path:
    program = tmp_path / "solution.py"
    program.write_text(marker + "\n", encoding="utf-8")
    return program


def test_stage2_minimize_direction_passes_when_execution_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _install_context(monkeypatch, work_dir, direction="minimize")

    program_path = _write_program(tmp_path)

    stub = _make_execute_stub(returncode=0, stdout="Baseline RMSE score: 0.5\n")
    monkeypatch.setattr("agent_k.core.solution.execute_solution", stub)

    result = evaluate_stage2(str(program_path))

    assert result.metrics["valid"] == 1.0
    assert result.metrics["combined_score"] > 0.6, (
        "minimize-direction candidates with valid execution must clear the cascade "
        "stage 2 threshold (0.6) — this was the regression"
    )
    assert result.metrics["combined_score"] == pytest.approx(0.65)
    assert result.metrics["cv_score"] == pytest.approx(0.5)


def test_stage2_maximize_direction_passes_when_execution_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _install_context(monkeypatch, work_dir, direction="maximize")

    program_path = _write_program(tmp_path)

    stub = _make_execute_stub(returncode=0, stdout="Baseline AUC score: 0.72\n")
    monkeypatch.setattr("agent_k.core.solution.execute_solution", stub)

    result = evaluate_stage2(str(program_path))

    assert result.metrics["valid"] == 1.0
    assert result.metrics["combined_score"] == pytest.approx(0.65)
    assert result.metrics["cv_score"] == pytest.approx(0.72)


def test_stage2_maximize_low_score_still_passes_when_execution_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weak but valid maximize candidate should reach full evaluation.

    The old gate silently dropped anything with fitness <= 0.4 (e.g. an
    accuracy of 0.2 on the noisy 1000-row subset). The 1000-row estimate
    is too noisy to threshold on — full evaluation is where the
    accept/reject decision belongs.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _install_context(monkeypatch, work_dir, direction="maximize")

    program_path = _write_program(tmp_path)

    stub = _make_execute_stub(returncode=0, stdout="Baseline accuracy score: 0.2\n")
    monkeypatch.setattr("agent_k.core.solution.execute_solution", stub)

    result = evaluate_stage2(str(program_path))

    assert result.metrics["valid"] == 1.0
    assert result.metrics["combined_score"] == pytest.approx(0.65)


def test_stage2_fails_when_execution_returncode_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _install_context(monkeypatch, work_dir, direction="minimize")

    program_path = _write_program(tmp_path)

    stub = _make_execute_stub(
        returncode=1, stdout="", stderr="Traceback (most recent call last):\nZeroDivisionError: division by zero\n"
    )
    monkeypatch.setattr("agent_k.core.solution.execute_solution", stub)

    result = evaluate_stage2(str(program_path))

    assert result.metrics["valid"] == 0.0
    assert result.metrics["combined_score"] == 0.0
    assert "feedback" in result.artifacts


def test_stage2_fails_when_submission_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _install_context(monkeypatch, work_dir, direction="minimize")

    program_path = _write_program(tmp_path)

    stub = _make_execute_stub(returncode=0, stdout="Baseline RMSE score: 0.5\n", write_submission=False)
    monkeypatch.setattr("agent_k.core.solution.execute_solution", stub)

    result = evaluate_stage2(str(program_path))

    assert result.metrics["valid"] == 0.0
    assert result.metrics["combined_score"] == 0.0


def test_stage2_fails_when_baseline_unparseable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _install_context(monkeypatch, work_dir, direction="maximize")

    program_path = _write_program(tmp_path)

    stub = _make_execute_stub(returncode=0, stdout="ran with no baseline line\n")
    monkeypatch.setattr("agent_k.core.solution.execute_solution", stub)

    result = evaluate_stage2(str(program_path))

    assert result.metrics["valid"] == 0.0
    assert result.metrics["combined_score"] == 0.0
