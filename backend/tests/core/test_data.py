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


def _make_zip_with_member(zip_path: Path, member_name: str, payload: bytes = b"x") -> None:
    """Write a zip whose archive entries can include traversal/absolute names."""
    with zipfile.ZipFile(zip_path, "w") as archive:
        info = zipfile.ZipInfo(filename=member_name)
        archive.writestr(info, payload)


def test_safe_extract_zip_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    """Sibling directories sharing the destination's prefix must be rejected.

    Regression for the prior `str.startswith` check, which permitted
    destination=/tmp/data to extract into /tmp/data2/...
    """
    destination = tmp_path / "data"
    destination.mkdir()
    sibling = tmp_path / "data2"
    sibling.mkdir()

    zip_path = tmp_path / "evil.zip"
    _make_zip_with_member(zip_path, "../data2/sneak.csv", payload=b"id\n1\n")

    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(zip_path, destination)

    assert not (sibling / "sneak.csv").exists()


def test_safe_extract_zip_rejects_parent_traversal(tmp_path: Path) -> None:
    destination = tmp_path / "stage"
    destination.mkdir()

    zip_path = tmp_path / "evil.zip"
    _make_zip_with_member(zip_path, "../escape.csv", payload=b"x\n")

    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(zip_path, destination)

    assert not (tmp_path / "escape.csv").exists()


def test_safe_extract_zip_rejects_absolute_path(tmp_path: Path) -> None:
    destination = tmp_path / "stage"
    destination.mkdir()
    outside = tmp_path / "outside.csv"

    zip_path = tmp_path / "evil.zip"
    _make_zip_with_member(zip_path, str(outside), payload=b"x\n")

    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(zip_path, destination)

    assert not outside.exists()


def test_safe_extract_zip_allows_legitimate_nested_paths(tmp_path: Path) -> None:
    destination = tmp_path / "stage"
    destination.mkdir()

    zip_path = tmp_path / "ok.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/dir/data.csv", b"id\n1\n")
        archive.writestr("top.csv", b"id\n2\n")

    extracted = _safe_extract_zip(zip_path, destination)

    nested = destination / "nested" / "dir" / "data.csv"
    top = destination / "top.csv"
    assert nested.exists()
    assert top.exists()
    assert {p.resolve() for p in extracted} == {nested.resolve(), top.resolve()}
