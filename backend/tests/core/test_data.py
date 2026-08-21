"""Tests for competition data utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import zipfile
from typing import TYPE_CHECKING

import pytest

from agent_k.core.data import _safe_extract_zip, infer_competition_schema, locate_data_files, stage_competition_data

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


def test_infer_competition_schema_strips_bom_and_whitespace(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    train_path.write_bytes(b"\xef\xbb\xbfid , feature, target\n1,0.1,0\n")
    test_path.write_bytes(b"\xef\xbb\xbfid , feature\n2,0.2\n")
    sample_path.write_bytes(b"\xef\xbb\xbfid , target\n2,0\n")

    schema = infer_competition_schema(train_path, test_path, sample_path)

    assert schema.id_column == "id"
    assert schema.target_columns == ["target"]
    assert schema.train_target_columns == ["target"]


def test_locate_data_files_prefers_canonical_over_auxiliary(tmp_path: Path) -> None:
    names = ["train_labels.csv", "train.csv", "test_metadata.csv", "test.csv", "sample_submission.csv"]
    for name in names:
        _write_csv(tmp_path / name, ["id", "target"], [["1", "0"]])

    # Directory iteration order is filesystem-dependent; selection must not be.
    for order in (names, sorted(names), list(reversed(names))):
        train, test, sample = locate_data_files([tmp_path / name for name in order])

        assert train.name == "train.csv"
        assert test.name == "test.csv"
        assert sample.name == "sample_submission.csv"


def test_locate_data_files_handles_camel_case_and_nesting(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_csv(nested / "train.csv", ["id", "target"], [["1", "0"]])
    _write_csv(tmp_path / "train_labels.csv", ["id", "target"], [["1", "0"]])
    _write_csv(tmp_path / "test.csv", ["id"], [["2"]])
    _write_csv(tmp_path / "sampleSubmission.csv", ["id", "target"], [["2", "0"]])

    train, test, sample = locate_data_files(sorted(tmp_path.rglob("*")))

    assert train.name == "train.csv"
    assert train.parent.name == "nested"
    assert test.name == "test.csv"
    assert sample.name == "sampleSubmission.csv"


def test_locate_data_files_survives_corrupt_zip(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", ["id", "target"], [["1", "0"]])
    _write_csv(tmp_path / "test.csv", ["id"], [["2"]])
    _write_csv(tmp_path / "sample_submission.csv", ["id", "target"], [["2", "0"]])
    (tmp_path / "partial_download.zip").write_bytes(b"not a real zip archive")

    train, test, sample = locate_data_files(sorted(tmp_path.iterdir()))

    assert train.name == "train.csv"
    assert test.name == "test.csv"
    assert sample.name == "sample_submission.csv"


def test_locate_data_files_ignores_archive_itself(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_csv(source / "train.csv", ["id", "target"], [["1", "0"]])
    _write_csv(source / "test.csv", ["id"], [["2"]])
    _write_csv(source / "sample_submission.csv", ["id", "target"], [["2", "0"]])

    zip_path = tmp_path / "train_and_test.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            archive.write(source / name, arcname=name)

    train, test, sample = locate_data_files([zip_path])

    # The archive name contains "train" and "test" but is not a data file.
    assert train.name == "train.csv"
    assert test.name == "test.csv"
    assert sample.name == "sample_submission.csv"


def test_locate_data_files_falls_back_to_loose_match(tmp_path: Path) -> None:
    for name in ("my_train_file.dat", "my_test_file.dat", "submission_format.dat"):
        _write_csv(tmp_path / name, ["id", "target"], [["1", "0"]])

    train, test, sample = locate_data_files(sorted(tmp_path.iterdir()))

    assert train.name == "my_train_file.dat"
    assert test.name == "my_test_file.dat"
    assert sample.name == "submission_format.dat"


def test_locate_data_files_reports_missing_roles(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", ["id", "target"], [["1", "0"]])

    with pytest.raises(FileNotFoundError) as excinfo:
        locate_data_files(sorted(tmp_path.iterdir()))

    message = str(excinfo.value)
    assert "sample" in message
    assert "test" in message


def test_safe_extract_zip_rejects_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escaped.csv", "id,target\n1,0\n")

    destination = tmp_path / "dest"
    destination.mkdir()

    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(zip_path, destination)

    assert not (tmp_path / "escaped.csv").exists()


def test_safe_extract_zip_skips_already_extracted(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    _write_csv(source, ["id", "target"], [["1", "0"]])

    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source, arcname="train.csv")

    destination = tmp_path / "dest"
    destination.mkdir()

    first = _safe_extract_zip(zip_path, destination)
    extracted = destination / "train.csv"
    marker = extracted.stat().st_mtime_ns

    second = _safe_extract_zip(zip_path, destination)

    assert first == second
    assert extracted.stat().st_mtime_ns == marker


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
