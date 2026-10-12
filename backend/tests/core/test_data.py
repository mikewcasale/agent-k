"""Tests for competition data utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import gzip
import shutil
import zipfile
from typing import TYPE_CHECKING

from agent_k.core.data import ensure_csv, infer_competition_schema, locate_data_files, stage_competition_data

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


def test_locate_data_files_converts_parquet(tmp_path: Path) -> None:
    """Parquet competition files should be converted to readable CSV."""
    import pandas as pd

    frame = pd.DataFrame({"id": [1, 2, 3], "feature": [0.5, 1.5, 2.5], "target": [0, 1, 0]})
    frame.to_parquet(tmp_path / "train.parquet")
    pd.DataFrame({"id": [4], "feature": [3.5]}).to_parquet(tmp_path / "test.parquet")
    pd.DataFrame({"id": [4], "target": [0]}).to_parquet(tmp_path / "sample_submission.parquet")

    train_path, test_path, sample_path = locate_data_files(
        [tmp_path / "train.parquet", tmp_path / "test.parquet", tmp_path / "sample_submission.parquet"]
    )

    assert train_path.suffix == ".csv"
    pd.testing.assert_frame_equal(pd.read_csv(train_path), frame)

    schema = infer_competition_schema(train_path, test_path, sample_path)
    assert schema.id_column == "id"
    assert schema.target_columns == ["target"]
    assert schema.train_target_columns == ["target"]


def test_locate_data_files_converts_gzipped_csv(tmp_path: Path) -> None:
    """Gzipped CSV files should be decompressed into plain CSV."""
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["id", "target"], [["1", "0"], ["2", "1"]])
    with csv_path.open("rb") as source, gzip.open(tmp_path / "train.csv.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    csv_path.unlink()

    train_path = ensure_csv(tmp_path / "train.csv.gz")

    assert train_path.read_text(encoding="utf-8").splitlines() == ["id,target", "1,0", "2,1"]


def test_ensure_csv_converts_zipped_and_tab_delimited(tmp_path: Path) -> None:
    """Zipped CSV and TSV sources should both become comma-delimited CSV."""
    inner = tmp_path / "payload.csv"
    _write_csv(inner, ["id", "target"], [["1", "0"]])
    with zipfile.ZipFile(tmp_path / "train.csv.zip", "w") as archive:
        archive.write(inner, arcname="train.csv")

    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text("id\tfeature\n7\t0.5\n", encoding="utf-8")

    assert ensure_csv(tmp_path / "train.csv.zip").read_text(encoding="utf-8").splitlines() == ["id,target", "1,0"]
    assert ensure_csv(tsv_path).read_text(encoding="utf-8").splitlines() == ["id,feature", "7,0.5"]


def test_ensure_csv_passes_through_plain_csv(tmp_path: Path) -> None:
    """Plain CSV sources are returned untouched so staging can hard-link."""
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, ["id", "target"], [["1", "0"]])

    assert ensure_csv(csv_path) == csv_path
    assert list(tmp_path.iterdir()) == [csv_path]


def test_ensure_csv_reuses_cached_conversion(tmp_path: Path) -> None:
    """Repeated conversions reuse the cached CSV instead of rewriting it."""
    import pandas as pd

    pd.DataFrame({"id": [1], "target": [0]}).to_parquet(tmp_path / "train.parquet")

    first = ensure_csv(tmp_path / "train.parquet")
    first_stat = first.stat()
    second = ensure_csv(tmp_path / "train.parquet")

    assert second == first
    assert second.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert second.stat().st_ino == first_stat.st_ino


def test_ensure_csv_falls_back_to_source_on_failure(tmp_path: Path) -> None:
    """A corrupt container must not abort the mission; the source path is returned."""
    broken = tmp_path / "train.parquet"
    broken.write_bytes(b"not a parquet file")

    assert ensure_csv(broken) == broken
    assert not (tmp_path / "train.agentk.csv").exists()


def test_stage_competition_data_from_parquet(tmp_path: Path) -> None:
    """Canonical staged filenames must hold CSV content even for parquet sources."""
    import pandas as pd

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pd.DataFrame({"id": [1], "feature": [0.5], "target": [0]}).to_parquet(source_dir / "train.parquet")
    pd.DataFrame({"id": [2], "feature": [1.5]}).to_parquet(source_dir / "test.parquet")
    pd.DataFrame({"id": [2], "target": [0]}).to_parquet(source_dir / "sample_submission.parquet")

    staged = stage_competition_data(
        source_dir / "train.parquet",
        source_dir / "test.parquet",
        source_dir / "sample_submission.parquet",
        tmp_path / "dest",
    )

    assert list(pd.read_csv(staged["train"]).columns) == ["id", "feature", "target"]
    assert list(pd.read_csv(staged["sample"]).columns) == ["id", "target"]


def test_ensure_csv_sniffs_delimiter_for_text_files(tmp_path: Path) -> None:
    """Delimiter is sniffed for generic .txt data files."""
    text_path = tmp_path / "train.txt"
    text_path.write_text("id|feature|target\n1|0.5|0\n2|1.5|1\n", encoding="utf-8")

    converted = ensure_csv(text_path)

    assert converted.read_text(encoding="utf-8").splitlines() == ["id,feature,target", "1,0.5,0", "2,1.5,1"]
