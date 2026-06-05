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


def test_locate_data_files_rejects_substring_lookalikes(tmp_path: Path) -> None:
    """``pretrained_features.csv`` must not be misidentified as the train file."""
    pretrained = tmp_path / "pretrained_features.csv"
    test_metadata = tmp_path / "test_metadata.csv"
    train_file = tmp_path / "train.csv"
    real_test_file = tmp_path / "test.csv"
    sample_file = tmp_path / "sample_submission.csv"

    for path in (pretrained, test_metadata, train_file, real_test_file, sample_file):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    located_train, located_test, located_sample = locate_data_files(
        [pretrained, test_metadata, train_file, real_test_file, sample_file]
    )

    assert located_train == train_file
    assert located_test == real_test_file
    assert located_sample == sample_file


def test_locate_data_files_prefers_exact_stem_over_decorated(tmp_path: Path) -> None:
    """``train.csv`` wins over ``train_metadata.csv`` even when the latter appears first."""
    train_decorated = tmp_path / "train_metadata.csv"
    train_canonical = tmp_path / "train.csv"
    test_decorated = tmp_path / "test_features.csv"
    test_canonical = tmp_path / "test.csv"
    sample_file = tmp_path / "sample_submission.csv"

    for path in (train_decorated, train_canonical, test_decorated, test_canonical, sample_file):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    located_train, located_test, located_sample = locate_data_files(
        [train_decorated, test_decorated, train_canonical, test_canonical, sample_file]
    )

    assert located_train == train_canonical
    assert located_test == test_canonical
    assert located_sample == sample_file


def test_locate_data_files_does_not_double_assign(tmp_path: Path) -> None:
    """No file may be returned for two different roles."""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    sample_file = tmp_path / "submission.csv"

    for path in (train_file, test_file, sample_file):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    located_train, located_test, located_sample = locate_data_files([train_file, test_file, sample_file])

    assert {located_train, located_test, located_sample} == {train_file, test_file, sample_file}


def test_locate_data_files_handles_word_boundary_train_suffix(tmp_path: Path) -> None:
    """``features_train.csv`` is matched as the train file when no canonical name exists."""
    train_file = tmp_path / "features_train.csv"
    test_file = tmp_path / "features_test.csv"
    sample_file = tmp_path / "sample_submission.csv"

    for path in (train_file, test_file, sample_file):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    located_train, located_test, located_sample = locate_data_files([train_file, test_file, sample_file])

    assert located_train == train_file
    assert located_test == test_file
    assert located_sample == sample_file


def test_locate_data_files_falls_back_to_bare_submission(tmp_path: Path) -> None:
    """When no ``sample_submission`` file exists, plain ``submission.csv`` is used."""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    sample_file = tmp_path / "submission.csv"

    for path in (train_file, test_file, sample_file):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    _, _, located_sample = locate_data_files([train_file, test_file, sample_file])

    assert located_sample == sample_file


def test_locate_data_files_ignores_non_data_extensions(tmp_path: Path) -> None:
    """Scripts and other non-data extensions containing role tokens must not be returned."""
    script = tmp_path / "train_model.py"
    script.write_text("# script", encoding="utf-8")
    notes = tmp_path / "test_notes.md"
    notes.write_text("# notes", encoding="utf-8")
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    sample_file = tmp_path / "sample_submission.csv"

    for path in (train_file, test_file, sample_file):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    located_train, located_test, located_sample = locate_data_files(
        [script, notes, train_file, test_file, sample_file]
    )

    assert located_train == train_file
    assert located_test == test_file
    assert located_sample == sample_file


def test_locate_data_files_deterministic_across_input_order(tmp_path: Path) -> None:
    """Selection is independent of input order."""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    sample_file = tmp_path / "sample_submission.csv"
    train_extra = tmp_path / "train_v2.csv"
    test_extra = tmp_path / "test_v2.csv"

    for path in (train_file, test_file, sample_file, train_extra, test_extra):
        _write_csv(path, ["id", "target"], [["1", "0"]])

    forward = locate_data_files([train_extra, test_extra, train_file, test_file, sample_file])
    reverse = locate_data_files([sample_file, test_file, train_file, test_extra, train_extra])

    assert forward == reverse
    assert forward == (train_file, test_file, sample_file)


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
