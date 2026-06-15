"""Tests for competition data utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import stat
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


def test_locate_data_files_rejects_zip_sibling_prefix_traversal(tmp_path: Path) -> None:
    """Reject zip entries that resolve to a sibling sharing the destination prefix."""
    zip_path = tmp_path / "data.zip"
    sibling_name = f"{tmp_path.name}SIBLING"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"../{sibling_name}/evil.csv", "id,target\n1,0\n")

    sibling_target = tmp_path.parent / sibling_name / "evil.csv"
    with pytest.raises(ValueError, match="escapes destination"):
        locate_data_files([zip_path])
    assert not sibling_target.exists()


def test_locate_data_files_rejects_zip_symlink_member(tmp_path: Path) -> None:
    """Reject zip members marked as symlinks to prevent escape via link following."""
    zip_path = tmp_path / "data.zip"
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret")

    with zipfile.ZipFile(zip_path, "w") as archive:
        info = zipfile.ZipInfo("mylink")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o755) << 16
        archive.writestr(info, str(outside))

    try:
        with pytest.raises(ValueError, match="symlink"):
            locate_data_files([zip_path])
        assert not (tmp_path / "mylink").is_symlink()
    finally:
        outside.unlink(missing_ok=True)


def test_locate_data_files_rejects_zip_absolute_member(tmp_path: Path) -> None:
    """Reject zip entries with absolute filenames."""
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        info = zipfile.ZipInfo("/etc/evil.csv")
        archive.writestr(info, "id,target\n1,0\n")

    with pytest.raises(ValueError, match="absolute path"):
        locate_data_files([zip_path])
