"""Tests for generic feature modality inference.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from pathlib import Path

from agent_k.core.modality import FeatureModality, infer_data_modality


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_infer_data_modality_classifies_numeric_and_categorical(tmp_path: Path) -> None:
    """Numeric columns and low-cardinality strings are separated."""
    train_path = _write_csv(
        tmp_path / "train.csv",
        ["id", "age", "grade", "target"],
        [[str(index), str(index * 1.5), "a" if index % 2 else "b", "1"] for index in range(20)],
    )

    modality = infer_data_modality(train_path, exclude_columns=("id", "target"))

    assert modality.sampled_rows == 20
    assert modality.columns_of(FeatureModality.NUMERIC) == ("age",)
    assert modality.columns_of(FeatureModality.CATEGORICAL) == ("grade",)
    assert modality.has_tabular_features is True


def test_infer_data_modality_detects_free_text(tmp_path: Path) -> None:
    """Long, high-cardinality string columns are classified as free text."""
    rows = [
        [str(index), f"the quick brown fox number {index} jumped over the lazy dog again", "0"] for index in range(30)
    ]
    train_path = _write_csv(tmp_path / "train.csv", ["id", "comment", "target"], rows)

    modality = infer_data_modality(train_path, exclude_columns=("id", "target"))

    assert modality.text_columns == ("comment",)
    assert modality.has_tabular_features is False


def test_infer_data_modality_detects_media_references(tmp_path: Path) -> None:
    """Columns of media paths are classified as file references with a media kind."""
    rows = [[str(index), f"images/train/{index}.jpg", str(index)] for index in range(15)]
    train_path = _write_csv(tmp_path / "train.csv", ["id", "image_path", "target"], rows)

    modality = infer_data_modality(train_path, exclude_columns=("id", "target"))

    assert modality.file_reference_columns == ("image_path",)
    assert modality.media_kinds == frozenset({"image"})


def test_infer_data_modality_detects_datetime_columns(tmp_path: Path) -> None:
    """ISO-like date strings are classified as datetime rather than categorical."""
    rows = [[str(index), f"2026-01-{index + 1:02d} 08:30", "3.5"] for index in range(20)]
    train_path = _write_csv(tmp_path / "train.csv", ["id", "observed_at", "target"], rows)

    modality = infer_data_modality(train_path, exclude_columns=("id", "target"))

    assert modality.columns_of(FeatureModality.DATETIME) == ("observed_at",)


def test_infer_data_modality_reports_empty_columns(tmp_path: Path) -> None:
    """Columns without any populated value are reported as empty."""
    rows = [[str(index), "", "1"] for index in range(10)]
    train_path = _write_csv(tmp_path / "train.csv", ["id", "blank", "target"], rows)

    modality = infer_data_modality(train_path, exclude_columns=("id", "target"))

    assert modality.columns_of(FeatureModality.EMPTY) == ("blank",)
    assert modality.has_tabular_features is False


def test_infer_data_modality_bounds_sampled_rows(tmp_path: Path) -> None:
    """Only the requested number of head rows is read."""
    rows = [[str(index), str(index), "1"] for index in range(500)]
    train_path = _write_csv(tmp_path / "train.csv", ["id", "value", "target"], rows)

    modality = infer_data_modality(train_path, exclude_columns=("id", "target"), sample_rows=25)

    assert modality.sampled_rows == 25


def test_infer_data_modality_handles_missing_file(tmp_path: Path) -> None:
    """A missing training file yields an empty result instead of raising."""
    modality = infer_data_modality(tmp_path / "absent.csv")

    assert modality.columns == ()
    assert modality.sampled_rows == 0
