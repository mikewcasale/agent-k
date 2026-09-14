"""Competition data utilities for AGENT-K.

@notice: |
    Competition data utilities for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.data
    provides:
        - agent_k.core.data
    pattern: data-access

@agent-guidance:
    do:
        - "Use agent_k.core.data as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import os
import shutil
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import logfire
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = (
    "CompetitionSchema",
    "TabularFormat",
    "TabularSource",
    "count_tabular_rows",
    "detect_tabular_source",
    "infer_competition_schema",
    "locate_data_files",
    "normalize_to_csv",
    "read_tabular",
    "read_tabular_header",
    "stage_competition_data",
)

type CompressionName = Literal["gzip", "bz2", "zip", "xz", "zstd"]
"""Compression containers pandas can read directly."""

_MAGIC_LENGTH: Final[int] = 8
_PARQUET_MAGIC: Final[bytes] = b"PAR1"
_FEATHER_MAGIC: Final[tuple[bytes, ...]] = (b"ARROW1", b"FEA1")
_COMPRESSION_MAGIC: Final[tuple[tuple[bytes, CompressionName], ...]] = (
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"PK\x03\x04", "zip"),
)
_COMPRESSION_SUFFIXES: Final[dict[str, CompressionName]] = {
    ".gz": "gzip",
    ".gzip": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".zst": "zstd",
    ".zip": "zip",
}
_PARQUET_SUFFIXES: Final[frozenset[str]] = frozenset({".parquet", ".pq"})
_FEATHER_SUFFIXES: Final[frozenset[str]] = frozenset({".feather", ".arrow", ".ipc"})
_TAB_SUFFIXES: Final[frozenset[str]] = frozenset({".tsv", ".tab"})


class TabularFormat(StrEnum):
    """Tabular file formats recognized in competition downloads.

    @notice: |
        Tabular file formats recognized in competition downloads.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum keeps format handling explicit across readers."
            violations: "Suffix string checks drift between call sites."
    """

    DELIMITED = "delimited"
    PARQUET = "parquet"
    FEATHER = "feather"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TabularSource:
    """Detected format details for a competition data file.

    @notice: |
        Detected format details for a competition data file.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: detection-model
            rationale: "Bundles format, delimiter, and compression for readers."
            violations: "Loose tuples make reader selection ambiguous."
    """

    path: Path
    format: TabularFormat
    delimiter: str = ","
    compression: CompressionName | None = None


@dataclass(frozen=True, slots=True)
class CompetitionSchema:
    """Schema details inferred from competition data files.

    @notice: |
        Schema details inferred from competition data files.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: schema-model
            rationale: "Captures inferred column layout for downstream steps."
            violations: "Ad-hoc dicts drift from actual data shape."
    """

    id_column: str
    target_columns: list[str]
    train_target_columns: list[str]


def detect_tabular_source(path: Path) -> TabularSource:
    """Detect the tabular format of a competition data file.

    @notice: |
        Classifies a file as delimited text, Parquet, or Feather.

    @dev: |
        Content magic bytes win over the file suffix, because Kaggle archives
        routinely ship compressed or columnar payloads under a `.csv` name.
        The suffix only supplies the delimiter and a fallback classification.
    """
    magic = _read_magic(path)
    suffix = _logical_suffix(path)
    compression = next((name for prefix, name in _COMPRESSION_MAGIC if magic.startswith(prefix)), None)
    if compression is None:
        compression = _COMPRESSION_SUFFIXES.get(path.suffix.lower())

    if magic.startswith(_PARQUET_MAGIC) or (compression is None and suffix in _PARQUET_SUFFIXES):
        return TabularSource(path=path, format=TabularFormat.PARQUET)
    if magic.startswith(_FEATHER_MAGIC) or (compression is None and suffix in _FEATHER_SUFFIXES):
        return TabularSource(path=path, format=TabularFormat.FEATHER)
    if suffix in _PARQUET_SUFFIXES or suffix in _FEATHER_SUFFIXES:
        # Columnar payload inside a compressed container: needs manual extraction first.
        return TabularSource(path=path, format=TabularFormat.UNKNOWN)
    if compression is None and b"\x00" in magic:
        return TabularSource(path=path, format=TabularFormat.UNKNOWN)

    delimiter = "\t" if suffix in _TAB_SUFFIXES else ","
    return TabularSource(path=path, format=TabularFormat.DELIMITED, delimiter=delimiter, compression=compression)


def read_tabular(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a competition data file regardless of its on-disk format.

    @notice: |
        Loads delimited text, Parquet, or Feather files into a DataFrame.

    @dev: |
        Parquet reads honor `nrows` through Arrow record batches so that
        profiling a multi-gigabyte file never materializes the whole table.

    @errors:
        terminal:
            - ValueError
    """
    source = detect_tabular_source(path)
    if source.format is TabularFormat.DELIMITED:
        return pd.read_csv(
            path, sep=source.delimiter, compression=source.compression or "infer", nrows=nrows, low_memory=False
        )
    if source.format is TabularFormat.PARQUET:
        return _read_parquet(path, nrows=nrows)
    if source.format is TabularFormat.FEATHER:
        frame = pd.read_feather(path)
        return frame.head(nrows) if nrows is not None else frame
    raise ValueError(f"Unsupported tabular file format: {path.name}")


def read_tabular_header(path: Path) -> list[str]:
    """Read the column names of a competition data file.

    @notice: |
        Returns column names without materializing the full table.

    @dev: |
        Parquet headers come from Arrow file metadata; delimited files are
        read one row at a time.

    @errors:
        terminal:
            - ValueError
    """
    source = detect_tabular_source(path)
    if source.format is TabularFormat.DELIMITED and source.compression is None:
        return _read_delimited_header(path, source.delimiter)
    if source.format is TabularFormat.PARQUET:
        import pyarrow.parquet as pq

        names: list[str] = list(pq.ParquetFile(path).schema_arrow.names)
        return names
    return [str(column) for column in read_tabular(path, nrows=0).columns]


def count_tabular_rows(path: Path) -> int:
    """Count the data rows of a competition data file.

    @notice: |
        Returns the number of rows excluding any header row.

    @dev: |
        Parquet row counts come from footer metadata, so no data pages are read.
        Uncompressed delimited files are counted by line to avoid a full parse.
    """
    source = detect_tabular_source(path)
    if source.format is TabularFormat.PARQUET:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if source.format is TabularFormat.UNKNOWN:
        return 0
    if source.format is TabularFormat.DELIMITED and source.compression is None:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    return int(len(read_tabular(path).index))


def normalize_to_csv(source: Path, destination: Path) -> bool:
    """Materialize a competition data file as comma-delimited CSV.

    @notice: |
        Links plain CSV sources and converts every other supported format.

    @dev: |
        Returns True when a conversion was performed. Nested Parquet values
        (lists and structs) are serialized with their pandas string form.

    @effects:
        io:
            - local filesystem writes

    @errors:
        terminal:
            - ValueError
    """
    detected = detect_tabular_source(source)
    if detected.format is TabularFormat.UNKNOWN:
        raise ValueError(f"Unsupported competition data format: {source.name}")

    if detected.format is TabularFormat.DELIMITED and detected.compression is None and detected.delimiter == ",":
        _link_or_copy(source, destination)
        return False

    frame = read_tabular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    logfire.info(
        "competition_data_normalized",
        source=str(source),
        destination=str(destination),
        source_format=str(detected.format),
        compression=detected.compression or "none",
        rows=int(len(frame.index)),
    )
    return True


def infer_competition_schema(train_path: Path, test_path: Path, sample_path: Path) -> CompetitionSchema:
    """Infer competition schema from train/test/sample submission files.

    @notice: |
        Analyzes data headers to determine ID column and target columns.

    @dev: |
        Compares train vs test headers to identify target columns.
        Falls back to sample submission columns if no difference found.
    """
    train_header = read_tabular_header(train_path)
    test_header = read_tabular_header(test_path)
    sample_header = read_tabular_header(sample_path)

    if len(sample_header) < 2:
        raise ValueError("Sample submission missing required columns")

    id_column = sample_header[0]
    target_columns = sample_header[1:]

    train_target_columns = [
        column for column in train_header if column not in test_header and column != id_column
    ] or list(target_columns)

    return CompetitionSchema(
        id_column=id_column, target_columns=list(target_columns), train_target_columns=train_target_columns
    )


def locate_data_files(paths: Iterable[str | Path]) -> tuple[Path, Path, Path]:
    """Locate train/test/sample files from downloaded data.

    @notice: |
        Finds train, test, and sample submission files from a list of paths.

    @dev: |
        Automatically extracts ZIP files and searches for files by name pattern.
        Raises FileNotFoundError if required files are not found.
    """
    files: list[Path] = []

    for path_value in paths:
        path = Path(path_value)
        files.append(path)
        if path.suffix.lower() == ".zip" and path.exists():
            files.extend(_safe_extract_zip(path, path.parent))

    def pick(token: str) -> Path | None:
        for path in files:
            if token in path.name.lower():
                return path
        return None

    train_path = pick("train")
    test_path = pick("test")
    sample_path = pick("sample_submission") or pick("submission")

    if not train_path or not test_path or not sample_path:
        raise FileNotFoundError("Required competition data files not found")

    return train_path, test_path, sample_path


def stage_competition_data(
    train_path: Path, test_path: Path, sample_path: Path, destination: Path, *, competition_id: str | None = None
) -> dict[str, Path]:
    """Stage competition data into canonical filenames.

    @notice: |
        Copies or links data files to a destination with standardized names.

    @dev: |
        Creates train.csv, test.csv, sample_submission.csv in destination.
        Plain CSV sources are hard linked when possible to save disk space;
        Parquet, Feather, tab-separated, and compressed sources are converted
        so downstream `pd.read_csv` consumers see real CSV content.

    @errors:
        terminal:
            - ValueError
    """
    destination.mkdir(parents=True, exist_ok=True)

    staged = {
        "train": destination / "train.csv",
        "test": destination / "test.csv",
        "sample": destination / "sample_submission.csv",
    }

    normalize_to_csv(train_path, staged["train"])
    normalize_to_csv(test_path, staged["test"])
    normalize_to_csv(sample_path, staged["sample"])

    if competition_id:
        competition_dir = destination / competition_id
        competition_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(staged["train"], competition_dir / staged["train"].name)
        _link_or_copy(staged["test"], competition_dir / staged["test"].name)
        _link_or_copy(staged["sample"], competition_dir / staged["sample"].name)

    return staged


def _read_magic(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(_MAGIC_LENGTH)
    except OSError:
        return b""


def _logical_suffix(path: Path) -> str:
    for suffix in reversed([value.lower() for value in path.suffixes]):
        if suffix not in _COMPRESSION_SUFFIXES:
            return suffix
    return ""


def _read_parquet(path: Path, *, nrows: int | None) -> pd.DataFrame:
    if nrows is None:
        return pd.read_parquet(path)

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    batch = next(iter(parquet_file.iter_batches(batch_size=max(nrows, 1))), None)
    if batch is None:
        return pd.DataFrame(columns=list(parquet_file.schema_arrow.names))
    frame: pd.DataFrame = batch.to_pandas()
    return frame.head(nrows)


def _read_delimited_header(path: Path, delimiter: str) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader, [])


def _safe_extract_zip(archive_path: Path, destination: Path) -> list[Path]:
    extracted: list[Path] = []
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename.endswith("/"):
                continue
            target_path = (destination / member.filename).resolve()
            if not str(target_path).startswith(str(destination_resolved)):
                raise ValueError(f"Zip entry escapes destination: {member.filename}")
            archive.extract(member, destination)
            extracted.append(target_path)

    return extracted


def _link_or_copy(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
