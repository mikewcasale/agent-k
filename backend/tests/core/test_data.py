"""Tests for competition data utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import gzip
import zipfile
from typing import TYPE_CHECKING

import pytest

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


def _write_dataset(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "train.csv", ["id", "feature", "target"], [["1", "9", "0"]])
    _write_csv(directory / "test.csv", ["id", "feature"], [["2", "8"]])
    _write_csv(directory / "sample_submission.csv", ["id", "target"], [["2", "0"]])


def test_locate_data_files_prefers_extracted_members_over_per_file_archives(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    archives: list[Path] = []
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        archive_path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(tmp_path / name, arcname=name)
        (tmp_path / name).unlink()
        archives.append(archive_path)

    train_path, test_path, sample_path = locate_data_files(archives)

    assert (train_path.name, test_path.name, sample_path.name) == ("train.csv", "test.csv", "sample_submission.csv")
    schema = infer_competition_schema(train_path, test_path, sample_path)
    assert schema.id_column == "id"
    assert schema.train_target_columns == ["target"]


def test_locate_data_files_expands_nested_archives(tmp_path: Path) -> None:
    inner_dir = tmp_path / "inner"
    _write_dataset(inner_dir)
    inner_archives: list[Path] = []
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        inner_archive = inner_dir / f"{name}.zip"
        with zipfile.ZipFile(inner_archive, "w") as archive:
            archive.write(inner_dir / name, arcname=name)
        (inner_dir / name).unlink()
        inner_archives.append(inner_archive)

    outer = tmp_path / "competition.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        for inner_archive in inner_archives:
            archive.write(inner_archive, arcname=inner_archive.name)
        inner_archive.unlink()

    train_path, test_path, sample_path = locate_data_files([outer])

    assert (train_path.name, test_path.name, sample_path.name) == ("train.csv", "test.csv", "sample_submission.csv")


def test_locate_data_files_expands_gzip_payloads(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    payloads: list[Path] = []
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        source = tmp_path / name
        payload = tmp_path / f"{name}.gz"
        with gzip.open(payload, "wb") as handle:
            handle.write(source.read_bytes())
        source.unlink()
        payloads.append(payload)

    train_path, test_path, sample_path = locate_data_files(payloads)

    assert (train_path.name, test_path.name, sample_path.name) == ("train.csv", "test.csv", "sample_submission.csv")
    assert infer_competition_schema(train_path, test_path, sample_path).id_column == "id"


def test_locate_data_files_is_order_independent(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _write_csv(tmp_path / "train_labels.csv", ["id", "label"], [["1", "0"]])
    (tmp_path / "extras").mkdir()
    _write_csv(tmp_path / "extras" / "test_metadata.csv", ["id", "note"], [["2", "x"]])

    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    forward = locate_data_files(sorted(files))
    reverse = locate_data_files(sorted(files, reverse=True))

    assert forward == reverse
    assert [path.name for path in forward] == ["train.csv", "test.csv", "sample_submission.csv"]


def test_locate_data_files_reports_missing_roles(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", ["id", "target"], [["1", "0"]])

    with pytest.raises(FileNotFoundError, match="test, sample"):
        locate_data_files([tmp_path / "train.csv"])


def test_infer_competition_schema_rejects_binary_headers(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    binary_path = tmp_path / "sample_submission.csv"
    binary_path.write_bytes(b"PK\x03\x04\x14\x00\x00\x00id,target\n")

    with pytest.raises(ValueError, match="not readable text"):
        infer_competition_schema(tmp_path / "train.csv", tmp_path / "test.csv", binary_path)


def test_locate_data_files_prefers_exact_name_over_shallower_partial(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", ["id", "feature", "target"], [["1", "9", "0"]])
    _write_csv(tmp_path / "sample_submission.csv", ["id", "target"], [["2", "0"]])
    _write_csv(tmp_path / "test_metadata.csv", ["id", "note"], [["2", "x"]])
    nested = tmp_path / "data"
    nested.mkdir()
    _write_csv(nested / "test.csv", ["id", "feature"], [["2", "8"]])

    _, test_path, _ = locate_data_files(sorted(path for path in tmp_path.rglob("*") if path.is_file()))

    assert test_path == nested / "test.csv"
