"""Tests for the canonical fitness scale used by the OpenEvolve evaluator.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from agent_k.core.strategy import score_to_fitness
from agent_k.evolution.evaluator import _failure_metrics, _fitness_from_score, evaluate, evaluate_stage2
from agent_k.mission.nodes import (
    _fitness_from_score as mission_fitness_from_score,
    _score_from_fitness as mission_score_from_fitness,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ()

_CONTEXT_ENV = "AGENT_K_OPENEVOLVE_CONTEXT"

_SCORING_SOLUTION = """
import csv

print("Baseline RMSE score: {score}")

with open("sample_submission.csv", newline="") as handle:
    rows = list(csv.reader(handle))

with open("submission.csv", "w", newline="") as handle:
    csv.writer(handle).writerows(rows)
"""

_NO_SUBMISSION_SOLUTION = """
print("Baseline RMSE score: 0.42")
"""

_PANDAS_SOLUTION = """
import pandas as pd
from sklearn.dummy import DummyRegressor

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

model = DummyRegressor()
model.fit(train_df[["feature"]], train_df["target"])
preds = model.predict(test_df[["feature"]])

print(f"Baseline RMSE score: {abs(float(preds.mean())):.6f}")
submission["target"] = preds[: len(submission)]
submission.to_csv("submission.csv", index=False)
"""


@pytest.fixture
def work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Provide a work directory holding real competition-shaped CSV files."""
    (tmp_path / "train.csv").write_text("id,feature,target\n1,1.0,2.0\n2,2.0,4.0\n3,3.0,6.0\n4,4.0,8.0\n")
    (tmp_path / "test.csv").write_text("id,feature\n5,5.0\n6,6.0\n")
    (tmp_path / "sample_submission.csv").write_text("id,target\n5,0.0\n6,0.0\n")
    monkeypatch.delenv(_CONTEXT_ENV, raising=False)
    yield tmp_path


def _set_context(monkeypatch: pytest.MonkeyPatch, work_dir: Path, *, metric_direction: str) -> None:
    payload: dict[str, Any] = {
        "work_dir": str(work_dir),
        "timeout": 60,
        "validation_split": 0.2,
        "metric_direction": metric_direction,
        "hints": [],
    }
    monkeypatch.setenv(_CONTEXT_ENV, json.dumps(payload))


def _write_program(work_dir: Path, code: str) -> str:
    program_path = work_dir / "program.py"
    program_path.write_text(code, encoding="utf-8")
    return str(program_path)


def test_fitness_from_score_matches_canonical_scale() -> None:
    """The evaluator must score solutions on the shared bounded fitness scale."""
    assert _fitness_from_score(0.42, "minimize") == score_to_fitness(0.42, "minimize")
    assert _fitness_from_score(0.91, "maximize") == score_to_fitness(0.91, "maximize")
    assert _fitness_from_score(0.42, "minimize") == mission_fitness_from_score(0.42, "minimize")


def test_fitness_from_score_keeps_minimize_fitness_above_failures() -> None:
    """A minimized score must outrank the fitness reported for a failed run."""
    failure_fitness = _failure_metrics()["combined_score"]

    for score in (0.0, 0.42, 7.5, 1_000.0):
        fitness = _fitness_from_score(score, "minimize")
        assert 0.0 < fitness <= 1.0
        assert fitness > failure_fitness


def test_fitness_from_score_orders_minimize_scores() -> None:
    """Lower scores must yield higher fitness for minimized metrics."""
    assert _fitness_from_score(0.2, "minimize") > _fitness_from_score(0.8, "minimize")
    assert _fitness_from_score(0.8, "maximize") > _fitness_from_score(0.2, "maximize")


def test_fitness_from_score_zero_for_invalid_runs() -> None:
    """A parsed score from a failed run must not earn any fitness."""
    assert _fitness_from_score(0.42, "minimize", valid=False) == 0.0
    assert _fitness_from_score(0.91, "maximize", valid=False) == 0.0
    assert _fitness_from_score(None, "minimize") == 0.0


def test_fitness_round_trips_through_mission_score_recovery() -> None:
    """The mission graph must recover the original score from evaluator fitness."""
    fitness = _fitness_from_score(0.42, "minimize")
    recovered = mission_score_from_fitness(fitness, "minimize")

    assert recovered is not None
    assert recovered == pytest.approx(0.42)


def test_evaluate_scores_minimize_solution_above_failure(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A working solution must outrank a failed evaluation for minimized metrics."""
    _set_context(monkeypatch, work_dir, metric_direction="minimize")
    program_path = _write_program(work_dir, _SCORING_SOLUTION.format(score="0.42"))

    result = evaluate(program_path)

    assert result.metrics["valid"] == 1.0
    assert result.metrics["cv_score"] == pytest.approx(0.42)
    assert result.metrics["combined_score"] == pytest.approx(score_to_fitness(0.42, "minimize"))
    assert result.metrics["combined_score"] > _failure_metrics()["combined_score"]
    assert result.metrics["fitness"] == result.metrics["combined_score"]


def test_evaluate_prefers_the_lower_minimize_score(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Between two working solutions the lower minimized score must win."""
    _set_context(monkeypatch, work_dir, metric_direction="minimize")

    better = evaluate(_write_program(work_dir, _SCORING_SOLUTION.format(score="0.20")))
    worse = evaluate(_write_program(work_dir, _SCORING_SOLUTION.format(score="0.80")))

    assert better.metrics["combined_score"] > worse.metrics["combined_score"]


def test_evaluate_withholds_fitness_without_a_submission(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A solution that prints a score but writes no submission earns no fitness."""
    _set_context(monkeypatch, work_dir, metric_direction="minimize")
    program_path = _write_program(work_dir, _NO_SUBMISSION_SOLUTION)

    result = evaluate(program_path)

    assert result.metrics["valid"] == 0.0
    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["fitness"] == 0.0


def test_evaluate_scores_maximize_solution_from_the_raw_score(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maximized metrics keep the raw score as fitness."""
    _set_context(monkeypatch, work_dir, metric_direction="maximize")
    program_path = _write_program(work_dir, _SCORING_SOLUTION.format(score="0.91"))

    result = evaluate(program_path)

    assert result.metrics["valid"] == 1.0
    assert result.metrics["combined_score"] == pytest.approx(0.91)


def test_evaluate_stage2_scores_minimize_solutions(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 2 must award positive fitness to a working minimized solution."""
    _set_context(monkeypatch, work_dir, metric_direction="minimize")
    program_path = _write_program(work_dir, _PANDAS_SOLUTION)

    result = evaluate_stage2(program_path)

    assert result.metrics["valid"] == 1.0
    assert result.metrics["combined_score"] > 0.0
    assert result.metrics["combined_score"] <= 0.65
