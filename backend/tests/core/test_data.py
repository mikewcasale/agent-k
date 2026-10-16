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


def _temporal_rows(days: list[str]) -> list[list[str]]:
    return [[str(index), day, f"{index * 0.5}"] for index, day in enumerate(days)]


def test_infer_competition_schema_detects_forecast_split(tmp_path: Path) -> None:
    """A test set that starts after the train set marks the date column as the order column."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(
        train_path,
        ["id", "date", "target"],
        [
            ["1", "2021-03-01", "1.0"],
            ["2", "2021-01-01", "2.0"],
            ["3", "2021-02-01", "3.0"],
            ["4", "2021-04-01", "4.0"],
        ],
    )
    _write_csv(
        test_path, ["id", "date"], [["5", "2021-05-01"], ["6", "2021-06-01"], ["7", "2021-07-01"], ["8", "2021-08-01"]]
    )
    _write_csv(sample_path, ["id", "target"], [["5", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.datetime_columns == ["date"]
    assert schema.temporal_order_column == "date"


def test_infer_competition_schema_detects_sorted_train(tmp_path: Path) -> None:
    """A chronologically sorted train file marks the date column as the order column."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(
        train_path,
        ["id", "timestamp", "target"],
        _temporal_rows(["2021-01-01 00:00:00", "2021-01-01 01:00:00", "2021-01-01 02:00:00", "2021-01-01 03:00:00"]),
    )
    _write_csv(test_path, ["id", "timestamp"], [["9", "2021-01-01 04:00:00"]])
    _write_csv(sample_path, ["id", "target"], [["9", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.datetime_columns == ["timestamp"]
    assert schema.temporal_order_column == "timestamp"


def test_infer_competition_schema_ignores_unordered_dates(tmp_path: Path) -> None:
    """An unordered date attribute is reported but never becomes the order column."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(
        train_path,
        ["id", "birth_date", "target"],
        [
            ["1", "1980-04-02", "1.0"],
            ["2", "1955-11-30", "2.0"],
            ["3", "1992-07-14", "3.0"],
            ["4", "1964-01-09", "4.0"],
        ],
    )
    _write_csv(
        test_path,
        ["id", "birth_date"],
        [["5", "1971-02-02"], ["6", "1948-09-21"], ["7", "1999-12-03"], ["8", "1960-06-16"]],
    )
    _write_csv(sample_path, ["id", "target"], [["5", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.datetime_columns == ["birth_date"]
    assert schema.temporal_order_column is None


def test_infer_competition_schema_ignores_numeric_identifiers(tmp_path: Path) -> None:
    """Digit-only columns are never mistaken for dates."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _write_csv(
        train_path,
        ["id", "serial", "target"],
        [["1", "20210101", "1.0"], ["2", "20210102", "2.0"], ["3", "20210103", "3.0"], ["4", "20210104", "4.0"]],
    )
    _write_csv(test_path, ["id", "serial"], [["5", "20210105"]])
    _write_csv(sample_path, ["id", "target"], [["5", "0"]])

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.datetime_columns == []
    assert schema.temporal_order_column is None
