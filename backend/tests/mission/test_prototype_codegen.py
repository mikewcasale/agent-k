"""Tests for baseline prototype code generation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from agent_k.core.data import infer_competition_schema
from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import PrototypeNode

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

_LIGHTGBM_STRATEGY = "Train a LightGBM model with tuned hyperparameters"
_ROW_COUNT = 90
_TEST_ROWS = 20


@dataclass(frozen=True, slots=True)
class _Competition:
    """Minimal competition stand-in exposing the metric the generator reads."""

    metric: EvaluationMetric


@dataclass(frozen=True, slots=True)
class _Research:
    """Minimal research stand-in exposing strategy recommendations."""

    strategy_recommendations: list[str]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_temporal_competition(directory: Path) -> None:
    start = date(2021, 1, 1)
    rows = [
        [str(index), (start + timedelta(days=index)).isoformat(), f"store_{index % 3}", f"{10.0 + index * 0.5:.2f}"]
        for index in range(_ROW_COUNT + _TEST_ROWS)
    ]
    _write_csv(directory / "train.csv", ["id", "date", "store", "sales"], rows[:_ROW_COUNT])
    _write_csv(directory / "test.csv", ["id", "date", "store"], [row[:3] for row in rows[_ROW_COUNT:]])
    _write_csv(directory / "sample_submission.csv", ["id", "sales"], [[row[0], "0"] for row in rows[_ROW_COUNT:]])


def _generate(
    metric: EvaluationMetric, *, strategy: str, time_column: str | None = None, is_temporal: bool = False
) -> str:
    return PrototypeNode()._generate_prototype(
        _Competition(metric=metric),
        _Research(strategy_recommendations=[strategy]),
        target_columns=["sales"],
        train_target_columns=["sales"],
        id_column="id",
        time_column=time_column,
        is_temporal=is_temporal,
    )


@pytest.mark.parametrize(
    "strategy", [_LIGHTGBM_STRATEGY, "Fit a linear regression", "Use gradient boosting", "Start with a random forest"]
)
def test_generated_prototype_is_valid_python(strategy: str) -> None:
    """Every model branch must render an importable module, not an indented block."""
    code = _generate(EvaluationMetric.RMSE, strategy=strategy)

    compile(code, "<prototype>", "exec")
    assert not code.splitlines()[0].startswith(" ")


def test_non_temporal_prototype_uses_random_split() -> None:
    """Competitions without a time order keep the stratified random holdout."""
    code = _generate(EvaluationMetric.RMSE, strategy=_LIGHTGBM_STRATEGY)

    assert "train_test_split(" in code
    assert "TIME_ORDER_COLUMN" not in code


def test_temporal_prototype_uses_chronological_split() -> None:
    """Time-ordered competitions validate on the most recent rows."""
    code = _generate(EvaluationMetric.RMSE, strategy=_LIGHTGBM_STRATEGY, time_column="date", is_temporal=True)

    compile(code, "<prototype>", "exec")
    assert "TIME_COLUMN = 'date'" in code
    assert '_order = np.argsort(X[TIME_ORDER_COLUMN].to_numpy(), kind="stable")' in code
    assert "_expand_time_features" in code


def test_temporal_prototype_without_time_column_uses_row_order() -> None:
    """Forecasting competitions without a parsable date fall back to file order."""
    code = _generate(EvaluationMetric.RMSE, strategy=_LIGHTGBM_STRATEGY, is_temporal=True)

    compile(code, "<prototype>", "exec")
    assert "TIME_COLUMN = None" in code
    assert "_order = np.arange(len(X))" in code
    assert "_expand_time_features" not in code


def test_temporal_prototype_runs_and_writes_submission(tmp_path: Path) -> None:
    """The generated time-aware baseline trains, scores, and writes a submission."""
    _write_temporal_competition(tmp_path)
    schema = infer_competition_schema(tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "sample_submission.csv")
    assert schema.time_column == "date"

    code = _generate(
        EvaluationMetric.RMSE, strategy=_LIGHTGBM_STRATEGY, time_column=schema.time_column, is_temporal=True
    )
    solution_path = tmp_path / "solution.py"
    solution_path.write_text(code, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, solution_path.name], cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Baseline rmse score:" in result.stdout
    submission_path = tmp_path / "submission.csv"
    assert submission_path.exists()
    with submission_path.open("r", encoding="utf-8", newline="") as handle:
        submission_rows = list(csv.DictReader(handle))
    assert len(submission_rows) == _TEST_ROWS
    assert all(row["sales"] for row in submission_rows)
