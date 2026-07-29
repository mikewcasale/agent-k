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


def test_locate_data_files_prefers_exact_stem_over_substring(tmp_path: Path) -> None:
    """`pick('test')` must ignore unrelated files like latest_scores.csv."""
    latest = tmp_path / "latest_scores.csv"
    manifest = tmp_path / "manifest.csv"
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    sample = tmp_path / "sample_submission.csv"
    for path in (latest, manifest, train, test, sample):
        _write_csv(path, ["id", "value"], [["1", "0"]])

    train_path, test_path, sample_path = locate_data_files([latest, manifest, train, test, sample])

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"
    assert sample_path.name == "sample_submission.csv"


def test_locate_data_files_exact_stem_wins_over_prefix(tmp_path: Path) -> None:
    """train.csv must win over train_extra.csv even when listed later."""
    extra_train = tmp_path / "train_extra.csv"
    real_train = tmp_path / "train.csv"
    extra_test = tmp_path / "test_v2.csv"
    real_test = tmp_path / "test.csv"
    sample = tmp_path / "sample_submission.csv"
    for path in (extra_train, real_train, extra_test, real_test, sample):
        _write_csv(path, ["id", "value"], [["1", "0"]])

    train_path, test_path, _ = locate_data_files([extra_train, extra_test, real_train, real_test, sample])

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"


def test_locate_data_files_prefix_fallback(tmp_path: Path) -> None:
    """When no exact stem match exists, prefer `<token>_...` or `<token>-...` stems."""
    train = tmp_path / "train_data.csv"
    test = tmp_path / "test_data.csv"
    sample = tmp_path / "sample_submission.csv"
    for path in (train, test, sample):
        _write_csv(path, ["id", "value"], [["1", "0"]])

    train_path, test_path, sample_path = locate_data_files([train, test, sample])

    assert train_path.name == "train_data.csv"
    assert test_path.name == "test_data.csv"
    assert sample_path.name == "sample_submission.csv"


def test_locate_data_files_ignores_test_substring_in_unrelated_names(tmp_path: Path) -> None:
    """The word 'test' inside 'latest' or 'greatest' must not steal the test slot."""
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    latest = tmp_path / "latest.csv"
    greatest = tmp_path / "greatest.csv"
    sample = tmp_path / "sample_submission.csv"
    for path in (train, test, latest, greatest, sample):
        _write_csv(path, ["id", "value"], [["1", "0"]])

    # Put decoys first — the fix must still pick test.csv over the decoys.
    _, test_path, _ = locate_data_files([latest, greatest, train, test, sample])
    assert test_path.name == "test.csv"


def test_locate_data_files_prefers_data_extensions(tmp_path: Path) -> None:
    """A non-data file with a matching stem (train.txt) must lose to train.csv."""
    train_txt = tmp_path / "train.txt"
    train_csv = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    sample = tmp_path / "sample_submission.csv"
    train_txt.write_text("noise")
    for path in (train_csv, test, sample):
        _write_csv(path, ["id", "value"], [["1", "0"]])

    train_path, _, _ = locate_data_files([train_txt, train_csv, test, sample])
    assert train_path.name == "train.csv"


def test_safe_extract_zip_rejects_prefix_confusion(tmp_path: Path) -> None:
    """A zip entry that resolves to a sibling directory must not extract."""
    destination = tmp_path / "foo"
    destination.mkdir()
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../foobar/leak.csv", "evil")

    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(zip_path, destination)

    # And ensure the extraction did not create the sibling directory.
    assert not (tmp_path / "foobar").exists()


def test_safe_extract_zip_rejects_absolute_path(tmp_path: Path) -> None:
    """A zip entry with a fully-qualified path outside destination must not extract."""
    destination = tmp_path / "dst"
    destination.mkdir()
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../outside.csv", "evil")

    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(zip_path, destination)


def test_safe_extract_zip_allows_nested_paths(tmp_path: Path) -> None:
    """Legitimate nested entries inside the destination extract normally."""
    destination = tmp_path / "dst"
    destination.mkdir()
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("train.csv", "id,y\n1,2\n")
        archive.writestr("nested/test.csv", "id\n3\n")

    extracted = _safe_extract_zip(zip_path, destination)

    extracted_names = sorted(str(p.relative_to(destination)) for p in extracted)
    assert extracted_names == sorted(["nested/test.csv", "train.csv"])
    assert (destination / "train.csv").read_text() == "id,y\n1,2\n"
    assert (destination / "nested" / "test.csv").read_text() == "id\n3\n"


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
