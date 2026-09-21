"""Tests for competition data utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import zipfile
from typing import TYPE_CHECKING

from agent_k.core.data import infer_competition_schema, locate_data_files, stage_competition_data

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_infer_competition_schema_basic(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(train_path, ["id", "feature", "target"], [["1", "0.1", "0"]])
    _write_csv(test_path, ["id", "feature"], [["2", "0.2"]])
    _write_csv(sample_path, ["id", "target"], [["2", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.id_column == "id"
    assert schema.target_columns == ["target"]
    assert schema.train_target_columns == ["target"]


def test_infer_competition_schema_multiclass(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(train_path, ["id", "feature", "target"], [["1", "0.1", "class_a"]])
    _write_csv(test_path, ["id", "feature"], [["2", "0.2"]])
    _write_csv(sample_path, ["id", "Class_A", "Class_B"], [["2", "0.5", "0.5"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.id_column == "id"
    assert schema.target_columns == ["Class_A", "Class_B"]
    assert schema.train_target_columns == ["target"]


def test_locate_data_files_from_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(train_path, ["id", "target"], [["1", "0"]])
    _write_csv(test_path, ["id"], [["2"]])
    _write_csv(sample_path, ["id", "target"], [["2", "0"]])

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(train_path, arcname="train.csv")
        archive.write(test_path, arcname="test.csv")
        archive.write(sample_path, arcname="sample_submission.csv")

    located_train, located_test, located_sample = locate_data_files([zip_path])

    assert located_train.name == "train.csv"
    assert located_test.name == "test.csv"
    assert located_sample.name == "sample_submission.csv"


def test_stage_competition_data(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    dest_dir = tmp_path / "dest"

    train_path = source_dir / "train_data.csv"
    test_path = source_dir / "test_data.csv"
    sample_path = source_dir / "sample_submission.csv"

    _write_csv(train_path, ["id", "target"], [["1", "0"]])
    _write_csv(test_path, ["id"], [["2"]])
    _write_csv(sample_path, ["id", "target"], [["2", "0"]])

    staged = stage_competition_data(train_path, test_path, sample_path, dest_dir)

    assert staged["train"].name == "train.csv"
    assert staged["test"].name == "test.csv"
    assert staged["sample"].name == "sample_submission.csv"
    assert staged["train"].exists()
    assert staged["test"].exists()
    assert staged["sample"].exists()


def test_infer_competition_schema_detects_time_column(tmp_path: Path) -> None:
    """A date column shared by train and test becomes the time index."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(
        train_path,
        ["id", "date", "store", "sales"],
        [[str(index), f"2021-01-{index + 1:02d}", "a", str(index)] for index in range(10)],
    )
    _write_csv(
        test_path, ["id", "date", "store"], [[str(index), f"2021-02-{index + 1:02d}", "a"] for index in range(10)]
    )
    _write_csv(sample_path, ["id", "sales"], [["0", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.time_column == "date"


def test_infer_competition_schema_prefers_most_specific_time_column(tmp_path: Path) -> None:
    """A timestamp column outranks a coarser month column."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    header = ["id", "month_start", "timestamp", "target"]
    rows = [
        [str(index), f"2021-{index + 1:02d}-01", f"2021-01-{index + 1:02d} 08:30:00", str(index)] for index in range(9)
    ]
    _write_csv(train_path, header, rows)
    _write_csv(test_path, header[:-1], [row[:-1] for row in rows])
    _write_csv(sample_path, ["id", "target"], [["0", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.time_column == "timestamp"


def test_infer_competition_schema_ignores_integer_time_named_columns(tmp_path: Path) -> None:
    """Integer features named after time units are not treated as a time index."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    header = ["id", "day_of_week", "year_built", "target"]
    rows = [[str(index), str(index % 7), str(1990 + index), str(index)] for index in range(10)]
    _write_csv(train_path, header, rows)
    _write_csv(test_path, header[:-1], [row[:-1] for row in rows])
    _write_csv(sample_path, ["id", "target"], [["0", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.time_column is None


def test_infer_competition_schema_ignores_train_only_time_column(tmp_path: Path) -> None:
    """A date column missing from the test set cannot order predictions."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(
        train_path,
        ["id", "date", "target"],
        [[str(index), f"2021-01-{index + 1:02d}", str(index)] for index in range(10)],
    )
    _write_csv(test_path, ["id"], [[str(index)] for index in range(10)])
    _write_csv(sample_path, ["id", "target"], [["0", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.time_column is None
