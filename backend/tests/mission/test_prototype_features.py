"""Tests for the feature matrix the prototype generator emits.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import _IDENTIFIER_MIN_ROWS, _MAX_ONEHOT_CARDINALITY, PrototypeNode

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

_REGIONS: tuple[str, ...] = ("north", "south", "east", "west")
_SKU_COUNT: int = _MAX_ONEHOT_CARDINALITY + 5


def _generate(metric: EvaluationMetric = EvaluationMetric.RMSE) -> str:
    node = PrototypeNode()
    return node._generate_prototype(
        SimpleNamespace(metric=metric),
        SimpleNamespace(strategy_recommendations=["use gradient boosting"]),
        target_columns=["target"],
        train_target_columns=["target"],
        id_column="row_id",
    )


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_fixture(work_path: Path, *, train_rows: int, drop_test_column: str | None = None) -> None:
    """Write a train/test/sample trio with an id, a free-text, and a high-cardinality column."""
    _write_csv(
        work_path / "train.csv",
        ["row_id", "region", "sku", "note", "area", "target"],
        [
            [
                f"T{index:04d}",
                _REGIONS[index % len(_REGIONS)],
                f"sku-{index % _SKU_COUNT}",
                f"free text note number {index}",
                100 + (index * 7) % 90,
                1000 + (index * 13) % 500,
            ]
            for index in range(train_rows)
        ],
    )

    test_rows = 20
    test_header = ["area", "note", "sku", "region", "row_id"]
    test_values: list[list[object]] = [
        [
            100 + (index * 11) % 90,
            f"unseen note {index}",
            f"sku-{(index + 3) % _SKU_COUNT}",
            _REGIONS[(index + 1) % len(_REGIONS)],
            f"S{index:04d}",
        ]
        for index in range(test_rows)
    ]
    if drop_test_column is not None:
        position = test_header.index(drop_test_column)
        test_header.pop(position)
        for row in test_values:
            row.pop(position)
    _write_csv(work_path / "test.csv", test_header, test_values)

    _write_csv(
        work_path / "sample_submission.csv", ["row_id", "target"], [[f"S{index:04d}", 0] for index in range(test_rows)]
    )


def _run(work_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    solution_path = work_path / "solution.py"
    solution_path.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-I", solution_path.name],
        cwd=work_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


class TestPrototypeFeatureSelection:
    """Tests for identifier exclusion and categorical routing."""

    def test_id_column_is_never_a_feature(self) -> None:
        """The submission id column must not reach the model."""
        code = _generate()

        assert "excluded = [ID_COLUMN] if ID_COLUMN in frame.columns else []" in code
        assert "clf.predict(test_df)" not in code
        assert "clf.predict_proba(test_df)" not in code

    def test_identifier_and_high_cardinality_columns_are_routed(self, tmp_path: Path) -> None:
        """Row identifiers are dropped and high-cardinality categories avoid one-hot encoding."""
        _write_fixture(tmp_path, train_rows=60)

        result = _run(tmp_path, _generate())

        assert result.returncode == 0, result.stderr
        plan = next(line for line in result.stdout.splitlines() if line.startswith("Feature plan:"))
        assert "excluded=['row_id', 'note']" in plan
        assert "onehot=['region']" in plan
        assert "ordinal=['sku']" in plan
        assert (tmp_path / "submission.csv").exists()

    def test_submission_covers_every_test_row(self, tmp_path: Path) -> None:
        """Predictions are written for each row of the sample submission."""
        _write_fixture(tmp_path, train_rows=60)

        result = _run(tmp_path, _generate())

        assert result.returncode == 0, result.stderr
        with (tmp_path / "submission.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 20
        assert all(row["target"] for row in rows)

    def test_missing_test_column_is_imputed(self, tmp_path: Path) -> None:
        """A feature absent from test.csv is imputed instead of aborting the run."""
        _write_fixture(tmp_path, train_rows=60, drop_test_column="region")

        result = _run(tmp_path, _generate())

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "submission.csv").exists()

    def test_small_frames_keep_unique_categoricals(self, tmp_path: Path) -> None:
        """Below the identifier row floor only the declared id column is dropped."""
        _write_fixture(tmp_path, train_rows=_IDENTIFIER_MIN_ROWS - 1)

        result = _run(tmp_path, _generate())

        assert result.returncode == 0, result.stderr
        plan = next(line for line in result.stdout.splitlines() if line.startswith("Feature plan:"))
        assert "excluded=['row_id']" in plan
        assert "'note'" in plan

    @pytest.mark.parametrize("metric", [EvaluationMetric.ACCURACY, EvaluationMetric.AUC])
    def test_classification_metrics_still_produce_a_submission(self, metric: EvaluationMetric, tmp_path: Path) -> None:
        """Classification paths keep working with the narrowed feature matrix."""
        _write_fixture(tmp_path, train_rows=60)
        _write_csv(
            tmp_path / "train.csv",
            ["row_id", "region", "sku", "note", "area", "target"],
            [
                [
                    f"T{index:04d}",
                    _REGIONS[index % len(_REGIONS)],
                    f"sku-{index % _SKU_COUNT}",
                    f"free text note number {index}",
                    100 + (index * 7) % 90,
                    index % 2,
                ]
                for index in range(60)
            ],
        )

        result = _run(tmp_path, _generate(metric))

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "submission.csv").exists()
