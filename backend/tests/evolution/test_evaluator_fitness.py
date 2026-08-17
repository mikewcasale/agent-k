"""Tests for OpenEvolve evaluator fitness reporting.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from typing import TYPE_CHECKING

from agent_k.core.fitness import FITNESS_FLOOR
from agent_k.evolution.evaluator import _failure_metrics, _fitness_from_score, evaluate

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

__all__ = ()

_SCORING_PROGRAM = """
import csv

print("Baseline RMSE score: 2.000000")
print("Fold 1: 2.100000")
print("Fold 2: 1.900000")

with open("submission.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "target"])
    for row_id in range(3):
        writer.writerow([row_id, 0.5])
"""

_CRASHING_PROGRAM = """
import sys

print("Baseline RMSE score: 0.000001")
sys.exit(1)
"""


def _write_context(monkeypatch: pytest.MonkeyPatch, work_dir: Path, direction: str) -> None:
    payload = {"work_dir": str(work_dir), "timeout": 60, "validation_split": 0.2, "metric_direction": direction}
    monkeypatch.setenv("AGENT_K_OPENEVOLVE_CONTEXT", json.dumps(payload))


def _run(monkeypatch: pytest.MonkeyPatch, work_dir: Path, source: str, direction: str) -> dict[str, float]:
    _write_context(monkeypatch, work_dir, direction)
    program_path = work_dir / "program.py"
    program_path.write_text(source, encoding="utf-8")
    result = evaluate(str(program_path))
    return dict(result.metrics)


class TestFitnessFromScore:
    """Tests for the evaluator's score conversion."""

    def test_minimize_fitness_is_above_floor(self) -> None:
        """Minimize-direction scores stay above the failure floor."""
        assert _fitness_from_score(2.0, "minimize") > FITNESS_FLOOR

    def test_lower_minimize_score_wins(self) -> None:
        """A better (lower) minimize score reports higher fitness."""
        assert _fitness_from_score(0.5, "minimize") > _fitness_from_score(5.0, "minimize")

    def test_invalid_run_reports_floor(self) -> None:
        """An invalid run reports the floor even with a parsed score."""
        assert _fitness_from_score(0.0, "minimize", valid=False) == FITNESS_FLOOR

    def test_missing_score_reports_floor(self) -> None:
        """A missing score reports the floor."""
        assert _fitness_from_score(None, "maximize") == FITNESS_FLOOR

    def test_failure_metrics_match_floor(self) -> None:
        """Failure metrics pin fitness to the floor."""
        metrics = _failure_metrics()
        assert metrics["fitness"] == FITNESS_FLOOR
        assert metrics["combined_score"] == FITNESS_FLOOR


class TestEvaluateMinimizeDirection:
    """End-to-end evaluator runs for minimize-direction competitions."""

    def test_valid_solution_outranks_failure_floor(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A working minimize solution scores above the failure floor."""
        metrics = _run(monkeypatch, tmp_path, _SCORING_PROGRAM, "minimize")

        assert metrics["valid"] == 1.0
        assert metrics["cv_score"] == 2.0
        assert metrics["combined_score"] > FITNESS_FLOOR
        assert metrics["combined_score"] == metrics["fitness"]
        assert metrics["cv_variance"] > 0.0

    def test_crashing_solution_reports_floor(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A crashing solution reports the floor despite printing a score."""
        metrics = _run(monkeypatch, tmp_path, _CRASHING_PROGRAM, "minimize")

        assert metrics["valid"] == 0.0
        assert metrics["combined_score"] == FITNESS_FLOOR

    def test_valid_solution_beats_crashing_solution(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A real score always outranks a crash under minimize direction."""
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        crash_dir = tmp_path / "crash"
        crash_dir.mkdir()

        valid_metrics = _run(monkeypatch, valid_dir, _SCORING_PROGRAM, "minimize")
        crash_metrics = _run(monkeypatch, crash_dir, _CRASHING_PROGRAM, "minimize")

        assert valid_metrics["combined_score"] > crash_metrics["combined_score"]


class TestEvaluateMaximizeDirection:
    """End-to-end evaluator runs for maximize-direction competitions."""

    def test_valid_solution_reports_raw_score(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Maximize-direction fitness passes the score through."""
        metrics = _run(monkeypatch, tmp_path, _SCORING_PROGRAM, "maximize")

        assert metrics["valid"] == 1.0
        assert metrics["combined_score"] == 2.0
