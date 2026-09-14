"""Tests for tabular format detection, reading, and CSV normalization.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import gzip
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from agent_k.core.data import (
    TabularFormat,
    count_tabular_rows,
    detect_tabular_source,
    infer_competition_schema,
    normalize_to_csv,
    read_tabular,
    read_tabular_header,
    stage_competition_data,
)
from agent_k.core.hints import build_dataset_profile

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _frame(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(rows)),
            "feature": [float(index) / 2 for index in range(rows)],
            "category": ["a" if index % 2 == 0 else "b" for index in range(rows)],
            "target": [float(index) * 1.5 for index in range(rows)],
        }
    )


def test_detect_parquet_by_magic_bytes(tmp_path: Path) -> None:
    # Kaggle archives routinely ship columnar payloads under a `.csv` name.
    path = tmp_path / "train.csv"
    _frame().to_parquet(path, index=False)

    source = detect_tabular_source(path)

    assert source.format is TabularFormat.PARQUET
    assert source.compression is None


def test_detect_gzip_csv_by_magic_bytes(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        _frame().to_csv(handle, index=False)

    source = detect_tabular_source(path)

    assert source.format is TabularFormat.DELIMITED
    assert source.compression == "gzip"


def test_detect_tsv_by_suffix(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv"
    _frame().to_csv(path, index=False, sep="\t")

    source = detect_tabular_source(path)

    assert source.format is TabularFormat.DELIMITED
    assert source.delimiter == "\t"


def test_detect_compressed_tsv_uses_inner_suffix(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        _frame().to_csv(handle, index=False, sep="\t")

    source = detect_tabular_source(path)

    assert source.format is TabularFormat.DELIMITED
    assert source.delimiter == "\t"
    assert source.compression == "gzip"


def test_detect_binary_payload_is_unknown(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    path.write_bytes(b"\x89PNG\x00\x01\x02\x03")

    assert detect_tabular_source(path).format is TabularFormat.UNKNOWN


def test_detect_plain_csv(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    _frame().to_csv(path, index=False)

    source = detect_tabular_source(path)

    assert source.format is TabularFormat.DELIMITED
    assert source.delimiter == ","
    assert source.compression is None


def test_read_tabular_parquet_honors_nrows(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _frame(rows=50).to_parquet(path, index=False)

    frame = read_tabular(path, nrows=7)

    assert len(frame.index) == 7
    assert list(frame.columns) == ["id", "feature", "category", "target"]


def test_read_tabular_parquet_zero_rows_keeps_columns(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _frame().to_parquet(path, index=False)

    frame = read_tabular(path, nrows=0)

    assert frame.empty
    assert list(frame.columns) == ["id", "feature", "category", "target"]


def test_read_tabular_feather(tmp_path: Path) -> None:
    path = tmp_path / "train.feather"
    _frame(rows=4).to_feather(path)

    frame = read_tabular(path)

    assert len(frame.index) == 4


def test_read_tabular_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")

    with pytest.raises(ValueError, match="Unsupported tabular file format"):
        read_tabular(path)


def test_read_tabular_header_matches_across_formats(tmp_path: Path) -> None:
    expected = ["id", "feature", "category", "target"]
    csv_path = tmp_path / "train.csv"
    parquet_path = tmp_path / "train.parquet"
    tsv_path = tmp_path / "train.tsv"
    gzip_path = tmp_path / "train.csv.gz"
    _frame().to_csv(csv_path, index=False)
    _frame().to_parquet(parquet_path, index=False)
    _frame().to_csv(tsv_path, index=False, sep="\t")
    with gzip.open(gzip_path, "wt", encoding="utf-8", newline="") as handle:
        _frame().to_csv(handle, index=False)

    assert read_tabular_header(csv_path) == expected
    assert read_tabular_header(parquet_path) == expected
    assert read_tabular_header(tsv_path) == expected
    assert read_tabular_header(gzip_path) == expected


def test_count_tabular_rows_matches_across_formats(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    parquet_path = tmp_path / "train.parquet"
    gzip_path = tmp_path / "train.csv.gz"
    _frame(rows=12).to_csv(csv_path, index=False)
    _frame(rows=12).to_parquet(parquet_path, index=False)
    with gzip.open(gzip_path, "wt", encoding="utf-8", newline="") as handle:
        _frame(rows=12).to_csv(handle, index=False)

    assert count_tabular_rows(csv_path) == 12
    assert count_tabular_rows(parquet_path) == 12
    assert count_tabular_rows(gzip_path) == 12


def test_normalize_to_csv_links_plain_csv(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    destination = tmp_path / "staged" / "train.csv"
    destination.parent.mkdir()
    _frame().to_csv(source, index=False)

    converted = normalize_to_csv(source, destination)

    assert converted is False
    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_normalize_to_csv_converts_parquet(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    destination = tmp_path / "staged" / "train.csv"
    _frame(rows=6).to_parquet(source, index=False)

    converted = normalize_to_csv(source, destination)

    assert converted is True
    pd.testing.assert_frame_equal(pd.read_csv(destination), _frame(rows=6))


def test_normalize_to_csv_rejects_unknown_format(tmp_path: Path) -> None:
    source = tmp_path / "train.bin"
    source.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")

    with pytest.raises(ValueError, match="Unsupported competition data format"):
        normalize_to_csv(source, tmp_path / "train.csv")


def test_stage_competition_data_normalizes_parquet(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train_path = source_dir / "train.parquet"
    test_path = source_dir / "test.parquet"
    sample_path = source_dir / "sample_submission.csv"
    train_frame = _frame(rows=8)
    train_frame.to_parquet(train_path, index=False)
    train_frame.drop(columns=["target"]).to_parquet(test_path, index=False)
    train_frame[["id", "target"]].to_csv(sample_path, index=False)

    staged = stage_competition_data(train_path, test_path, sample_path, tmp_path / "work")

    # Every downstream consumer reads the staged files with pd.read_csv.
    pd.testing.assert_frame_equal(pd.read_csv(staged["train"]), train_frame)
    assert list(pd.read_csv(staged["test"]).columns) == ["id", "feature", "category"]

    schema = infer_competition_schema(staged["train"], staged["test"], staged["sample"])
    assert schema.id_column == "id"
    assert schema.target_columns == ["target"]
    assert schema.train_target_columns == ["target"]


def test_stage_competition_data_normalizes_gzip_and_tsv(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train_path = source_dir / "train.csv.gz"
    test_path = source_dir / "test.tsv"
    sample_path = source_dir / "sample_submission.csv"
    train_frame = _frame(rows=4)
    with gzip.open(train_path, "wt", encoding="utf-8", newline="") as handle:
        train_frame.to_csv(handle, index=False)
    train_frame.drop(columns=["target"]).to_csv(test_path, index=False, sep="\t")
    train_frame[["id", "target"]].to_csv(sample_path, index=False)

    staged = stage_competition_data(train_path, test_path, sample_path, tmp_path / "work")

    pd.testing.assert_frame_equal(pd.read_csv(staged["train"]), train_frame)
    assert list(pd.read_csv(staged["test"]).columns) == ["id", "feature", "category"]


def test_build_dataset_profile_reads_parquet(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    sample_path = tmp_path / "sample_submission.parquet"
    train_frame = _frame(rows=30)
    train_frame.to_parquet(train_path, index=False)
    train_frame.drop(columns=["target"]).to_parquet(test_path, index=False)
    train_frame[["id", "target"]].to_parquet(sample_path, index=False)

    profile = build_dataset_profile(train_path, test_path, sample_path)

    assert profile.row_count == 30
    assert set(profile.columns) == {"id", "feature", "category", "target"}
