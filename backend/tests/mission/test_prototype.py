"""Tests for prototype generation, including chronological validation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.mission.nodes import PrototypeNode

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

_TRAIN_DAYS = 60
_TEST_DAYS = 10


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _competition() -> Competition:
    return Competition(
        id="temporal-competition",
        title="Temporal Competition",
        description=None,
        competition_type=CompetitionType.FEATURED,
        metric=EvaluationMetric.RMSE,
        metric_direction="minimize",
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
        prize_pool=None,
        max_team_size=1,
        max_daily_submissions=5,
        tags=frozenset({"tabular"}),
        url=None,
    )


def _write_temporal_dataset(directory: Path) -> None:
    """Write a train/test/sample trio whose rows advance one day at a time."""
    start = date(2024, 1, 1)
    train_rows: list[list[str]] = []
    for offset in range(_TRAIN_DAYS):
        current = start + timedelta(days=offset)
        train_rows.append([str(offset), current.isoformat(), str(offset % 7), str(10.0 + offset * 0.5)])
    _write_csv(directory / "train.csv", ["id", "sale_date", "weekday", "sales"], train_rows)

    test_rows: list[list[str]] = []
    for offset in range(_TRAIN_DAYS, _TRAIN_DAYS + _TEST_DAYS):
        current = start + timedelta(days=offset)
        test_rows.append([str(offset), current.isoformat(), str(offset % 7)])
    _write_csv(directory / "test.csv", ["id", "sale_date", "weekday"], test_rows)

    _write_csv(directory / "sample_submission.csv", ["id", "sales"], [[row[0], "0"] for row in test_rows])


def _generate(time_column: str | None) -> str:
    return PrototypeNode()._generate_prototype(
        _competition(),
        None,
        target_columns=["sales"],
        train_target_columns=["sales"],
        id_column="id",
        time_column=time_column,
    )


def test_prototype_without_time_column_uses_random_split() -> None:
    """Non-temporal competitions keep the shuffled train/test split."""
    code = _generate(None)

    assert "TIME_COLUMN = None" in code
    assert "train_test_split(" in code


def test_prototype_with_time_column_splits_chronologically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated baseline validates on the latest rows, never on shuffled ones."""
    _write_temporal_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_K_VALIDATION_SPLIT", raising=False)

    code = _generate("sale_date")
    namespace: dict[str, Any] = {"__name__": "__prototype__"}
    exec(compile(code, "prototype.py", "exec"), namespace)

    x_train = namespace["X_train"]
    x_val = namespace["X_val"]
    ordinal = namespace["TIME_ORDINAL_COLUMN"]

    assert namespace["HAS_TIME_ORDER"] is True
    assert len(x_val) == _TRAIN_DAYS - int(_TRAIN_DAYS * 0.8)
    assert x_val[ordinal].min() > x_train[ordinal].max()
    assert "sale_date" not in x_train.columns
    assert "agent_k_time_dayofweek" in x_train.columns


def test_prototype_with_time_column_produces_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The temporal baseline runs end to end and writes one prediction per test row."""
    _write_temporal_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_K_VALIDATION_SPLIT", raising=False)

    code = _generate("sale_date")
    exec(compile(code, "prototype.py", "exec"), {"__name__": "__prototype__"})

    submission_path = tmp_path / "submission.csv"
    assert submission_path.exists()

    with submission_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == _TEST_DAYS
    assert all(row["sales"] for row in rows)
