"""Tests for the canonical fitness convention shared across the evolution loop.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agent_k.agents.evolver import evolver_agent_instance
from agent_k.core.strategy import fitness_to_score, score_to_fitness
from agent_k.evolution.evaluator import _failure_metrics, _fitness_from_score, evaluate
from agent_k.mission.nodes import _fitness_from_score as _mission_fitness_from_score

if TYPE_CHECKING:
    from agent_k.core.types import MetricDirection

__all__ = ()

_CONTEXT_ENV = "AGENT_K_OPENEVOLVE_CONTEXT"

_SCORING_SOLUTION = """
import csv

with open("train.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

values = [float(row["target"]) for row in rows]
mean = sum(values) / len(values)
rmse = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
print(f"Baseline RMSE score: {rmse:.6f}")

with open("test.csv", newline="", encoding="utf-8") as handle:
    test_rows = list(csv.DictReader(handle))

with open("submission.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "target"])
    for row in test_rows:
        writer.writerow([row["id"], f"{mean:.6f}"])
"""

_NO_SUBMISSION_SOLUTION = """
print("Baseline RMSE score: 0.100000")
"""


def _write_dataset(work_dir: Path) -> None:
    """Write a small train/test pair the solutions below can actually score."""
    (work_dir / "train.csv").write_text(
        "id,feature,target\n1,0.1,2.0\n2,0.4,4.0\n3,0.9,6.0\n4,1.3,8.0\n", encoding="utf-8"
    )
    (work_dir / "test.csv").write_text("id,feature\n5,1.7\n6,2.1\n", encoding="utf-8")


def _run_evaluate(work_dir: Path, code: str, direction: MetricDirection) -> dict[str, float]:
    program_path = work_dir / "program.py"
    program_path.write_text(code, encoding="utf-8")
    context = {"work_dir": str(work_dir), "timeout": 60, "validation_split": 0.2, "metric_direction": direction}
    previous = os.environ.get(_CONTEXT_ENV)
    os.environ[_CONTEXT_ENV] = json.dumps(context)
    try:
        return dict(evaluate(str(program_path)).metrics)
    finally:
        if previous is None:
            os.environ.pop(_CONTEXT_ENV, None)
        else:
            os.environ[_CONTEXT_ENV] = previous


@pytest.mark.parametrize("score", [0.0, 0.25, 1.0, 42.0])
def test_score_to_fitness_round_trip(score: float) -> None:
    """Both directions should invert back to the original score."""
    for direction in ("minimize", "maximize"):
        fitness = score_to_fitness(score, direction)
        assert fitness >= 0.0
        assert fitness_to_score(fitness, direction) == pytest.approx(score)


def test_score_to_fitness_orders_minimized_scores() -> None:
    """A lower minimized score must rank above a higher one."""
    assert score_to_fitness(0.2, "minimize") > score_to_fitness(0.9, "minimize")


def test_evaluator_matches_agent_and_mission_conventions() -> None:
    """The three fitness producers must agree so values survive process hops."""
    for direction in ("minimize", "maximize"):
        for score in (0.05, 0.5, 3.0):
            evaluator_fitness = _fitness_from_score(score, direction)
            assert evaluator_fitness == pytest.approx(evolver_agent_instance._fitness_from_score(score, direction))
            assert evaluator_fitness == pytest.approx(_mission_fitness_from_score(score, direction))


def test_evaluator_fitness_is_never_below_failure_fitness() -> None:
    """A scored minimized candidate must outrank a failed evaluation."""
    failure_fitness = _failure_metrics()["combined_score"]
    for score in (0.01, 1.0, 500.0):
        assert _fitness_from_score(score, "minimize") > failure_fitness


def test_evaluate_ranks_working_solution_above_failure(tmp_path: Path) -> None:
    """A solution that scores and submits beats one that only prints a score."""
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    _write_dataset(working_dir)
    working_metrics = _run_evaluate(working_dir, _SCORING_SOLUTION, "minimize")

    assert working_metrics["valid"] == 1.0
    assert working_metrics["cv_score"] > 0.0
    assert working_metrics["combined_score"] == pytest.approx(score_to_fitness(working_metrics["cv_score"], "minimize"))

    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    _write_dataset(incomplete_dir)
    incomplete_metrics = _run_evaluate(incomplete_dir, _NO_SUBMISSION_SOLUTION, "minimize")

    assert incomplete_metrics["valid"] == 0.0
    assert incomplete_metrics["combined_score"] == 0.0
    assert working_metrics["combined_score"] > incomplete_metrics["combined_score"]
    assert working_metrics["combined_score"] > _failure_metrics()["combined_score"]
