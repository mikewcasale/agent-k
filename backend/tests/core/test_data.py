"""Tests for competition data utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
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


def _touch_csv(path: Path) -> None:
    _write_csv(path, ["id", "target"], [["1", "0"]])


def test_locate_data_files_prefers_exact_test_over_train_test(tmp_path: Path) -> None:
    """`train_test_split.csv` must not shadow the real `test.csv`."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    decoy_path = tmp_path / "train_test_split.csv"

    for path in (train_path, test_path, sample_path, decoy_path):
        _touch_csv(path)

    # Force the decoy to come first in the iteration order to mimic an
    # unfortunate filesystem walk order.
    paths = [decoy_path, test_path, train_path, sample_path]
    located_train, located_test, located_sample = locate_data_files(paths)

    assert located_train == train_path
    assert located_test == test_path
    assert located_sample == sample_path


def test_locate_data_files_does_not_reuse_same_file_for_two_roles(tmp_path: Path) -> None:
    """A single file matching multiple tokens must not satisfy both roles."""
    combo_path = tmp_path / "train_test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    _touch_csv(combo_path)
    _touch_csv(sample_path)

    with pytest.raises(FileNotFoundError):
        locate_data_files([combo_path, sample_path])


def test_locate_data_files_assigns_combo_file_to_strongest_role(tmp_path: Path) -> None:
    """When `train_test.csv` is the only train candidate, it falls to train and a separate test.csv wins test."""
    combo_path = tmp_path / "train_test.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    for path in (combo_path, test_path, sample_path):
        _touch_csv(path)

    located_train, located_test, located_sample = locate_data_files([combo_path, test_path, sample_path])

    assert located_train == combo_path
    assert located_test == test_path
    assert located_sample == sample_path


def test_locate_data_files_ignores_substring_inside_unrelated_token(tmp_path: Path) -> None:
    """`testimony.csv` must not be picked as the test file."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    unrelated_path = tmp_path / "testimony.csv"

    for path in (train_path, test_path, sample_path, unrelated_path):
        _touch_csv(path)

    located_train, located_test, located_sample = locate_data_files(
        [unrelated_path, train_path, sample_path, test_path]
    )

    assert located_test == test_path
    assert located_train == train_path
    assert located_sample == sample_path


def test_locate_data_files_picks_canonical_sample_over_submission_format(tmp_path: Path) -> None:
    """`sample_submission.csv` outranks a generic `submission_format.csv`."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    decoy_path = tmp_path / "submission_format.csv"

    for path in (train_path, test_path, sample_path, decoy_path):
        _touch_csv(path)

    _, _, located_sample = locate_data_files([decoy_path, train_path, test_path, sample_path])

    assert located_sample == sample_path


def test_locate_data_files_falls_back_to_submission_when_sample_missing(tmp_path: Path) -> None:
    """When no `sample_submission.csv` exists, a `submission.csv` is acceptable."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "submission.csv"

    for path in (train_path, test_path, sample_path):
        _touch_csv(path)

    _, _, located_sample = locate_data_files([train_path, test_path, sample_path])

    assert located_sample == sample_path


def test_locate_data_files_camelcase_sample_submission(tmp_path: Path) -> None:
    """Older Kaggle competitions use `sampleSubmission.csv`."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sampleSubmission.csv"

    for path in (train_path, test_path, sample_path):
        _touch_csv(path)

    _, _, located_sample = locate_data_files([train_path, test_path, sample_path])

    assert located_sample == sample_path


def test_locate_data_files_skips_non_data_suffixes(tmp_path: Path) -> None:
    """README and other non-data files must not be selected even when their name contains a token."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    readme_path = tmp_path / "train_README.md"
    readme_path.write_text("notes about train", encoding="utf-8")

    for path in (train_path, test_path, sample_path):
        _touch_csv(path)

    located_train, located_test, located_sample = locate_data_files([readme_path, train_path, test_path, sample_path])

    assert located_train == train_path
    assert located_test == test_path
    assert located_sample == sample_path
