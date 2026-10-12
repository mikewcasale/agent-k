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

import bz2
import csv
import gzip
import io
import lzma
import os
import shutil
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final
from uuid import uuid4

import logfire

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

__all__ = ("CompetitionSchema", "ensure_csv", "infer_competition_schema", "locate_data_files", "stage_competition_data")

CONVERTED_CSV_MARKER: Final[str] = ".agentk"
"""Suffix marker applied to CSV files converted from another container format."""

_PARQUET_BATCH_ROWS: Final[int] = 50_000
_SNIFF_BYTES: Final[int] = 65_536
_DELIMITED_SUFFIXES: Final[dict[str, str | None]] = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".txt": None}
_PARQUET_SUFFIXES: Final[frozenset[str]] = frozenset({".parquet", ".pq"})
_FEATHER_SUFFIXES: Final[frozenset[str]] = frozenset({".feather", ".arrow"})
_STREAM_OPENERS: Final[dict[str, Callable[..., Any]]] = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
_COMPRESSED_SUFFIXES: Final[frozenset[str]] = frozenset({*_STREAM_OPENERS, ".zip"})


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


def infer_competition_schema(train_path: Path, test_path: Path, sample_path: Path) -> CompetitionSchema:
    """Infer competition schema from train/test/sample submission files.

    @notice: |
        Analyzes CSV headers to determine ID column and target columns.

    @dev: |
        Compares train vs test headers to identify target columns.
        Falls back to sample submission columns if no difference found.
    """
    train_header = _read_header(train_path)
    test_header = _read_header(test_path)
    sample_header = _read_header(sample_path)

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
        Non-CSV containers (Parquet, Feather, TSV, compressed CSV) are converted
        to real CSV next to the source so downstream readers stay format-agnostic.
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

    return ensure_csv(train_path), ensure_csv(test_path), ensure_csv(sample_path)


def ensure_csv(source: Path, *, fallback_dir: Path | None = None) -> Path:
    """Return a path to a plain CSV rendering of a competition data file.

    @notice: |
        Converts Parquet, Feather, TSV, and compressed CSV files into real CSV.

    @dev: |
        Plain `.csv` sources are returned unchanged so callers keep the cheap
        hard-link staging path. Conversions are written atomically next to the
        source (or into `fallback_dir` when the source directory is read-only)
        and reused while the cached CSV is newer than the source, so repeated
        staging during evolution does not re-convert. Any conversion failure
        falls back to the original path rather than aborting the mission.
    """
    suffix = source.suffix.lower()
    compression = suffix if suffix in _COMPRESSED_SUFFIXES else None
    inner = Path(source.stem) if compression else source
    inner_suffix = inner.suffix.lower()

    if compression is None and inner_suffix == ".csv":
        return source

    convertible = inner_suffix in _DELIMITED_SUFFIXES or (
        compression is None and inner_suffix in _PARQUET_SUFFIXES | _FEATHER_SUFFIXES
    )
    if not convertible:
        return source

    target = _conversion_target(source, inner.stem, fallback_dir)
    if target is None:
        return source
    if _conversion_is_fresh(source, target):
        return target

    try:
        _convert_to_csv(source, target, compression=compression, inner_suffix=inner_suffix)
    except Exception as exc:
        logfire.warning("csv_conversion_failed", source=str(source), error=str(exc))
        return source

    logfire.info("csv_conversion_completed", source=str(source), target=str(target))
    return target


def _conversion_target(source: Path, stem: str, fallback_dir: Path | None) -> Path | None:
    name = f"{stem}{CONVERTED_CSV_MARKER}.csv"
    if os.access(source.parent, os.W_OK):
        return source.parent / name
    if fallback_dir is None:
        return None
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir / name


def _conversion_is_fresh(source: Path, target: Path) -> bool:
    try:
        target_stat = target.stat()
        source_stat = source.stat()
    except OSError:
        return False
    return target_stat.st_size > 0 and target_stat.st_mtime >= source_stat.st_mtime


def _convert_to_csv(source: Path, target: Path, *, compression: str | None, inner_suffix: str) -> None:
    temporary = target.with_name(f"{target.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        if inner_suffix in _PARQUET_SUFFIXES:
            _write_parquet_as_csv(source, temporary)
        elif inner_suffix in _FEATHER_SUFFIXES:
            _write_feather_as_csv(source, temporary)
        else:
            _write_delimited_as_csv(
                source, temporary, compression=compression, delimiter=_DELIMITED_SUFFIXES[inner_suffix]
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_as_csv(source: Path, target: Path) -> None:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pa_parquet

    parquet_file = pa_parquet.ParquetFile(source)
    schema = parquet_file.schema_arrow
    with pa_csv.CSVWriter(str(target), schema) as writer:
        for batch in parquet_file.iter_batches(batch_size=_PARQUET_BATCH_ROWS):
            writer.write_table(pa.Table.from_batches([batch], schema))


def _write_feather_as_csv(source: Path, target: Path) -> None:
    import pyarrow.csv as pa_csv
    import pyarrow.feather as pa_feather

    pa_csv.write_csv(pa_feather.read_table(source), str(target))


def _write_delimited_as_csv(source: Path, target: Path, *, compression: str | None, delimiter: str | None) -> None:
    with _open_text(source, compression) as handle, target.open("w", encoding="utf-8", newline="") as output:
        rows: Iterable[str] = handle
        resolved = delimiter
        if resolved is None:
            sample = handle.read(_SNIFF_BYTES) + handle.readline()
            resolved = _sniff_delimiter(sample)
            rows = chain(io.StringIO(sample, newline=""), handle)
        writer = csv.writer(output)
        writer.writerows(csv.reader(rows, delimiter=resolved))


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


@contextmanager
def _open_text(source: Path, compression: str | None) -> Iterator[IO[str]]:
    if compression == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one member in {source.name}, found {len(members)}")
            with (
                archive.open(members[0]) as binary,
                io.TextIOWrapper(binary, encoding="utf-8", errors="replace", newline="") as handle,
            ):
                yield handle
        return

    opener = _STREAM_OPENERS.get(compression or "")
    if opener is None:
        with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            yield handle
        return

    with opener(source, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        yield handle


def stage_competition_data(
    train_path: Path, test_path: Path, sample_path: Path, destination: Path, *, competition_id: str | None = None
) -> dict[str, Path]:
    """Stage competition data into canonical filenames.

    @notice: |
        Copies or links data files to a destination with standardized names.

    @dev: |
        Creates train.csv, test.csv, sample_submission.csv in destination.
        Sources that are not already plain CSV are converted first so the
        canonical filenames always hold CSV content.
        Uses hard links when possible to save disk space.
    """
    destination.mkdir(parents=True, exist_ok=True)

    staged = {
        "train": destination / "train.csv",
        "test": destination / "test.csv",
        "sample": destination / "sample_submission.csv",
    }

    _link_or_copy(ensure_csv(train_path, fallback_dir=destination), staged["train"])
    _link_or_copy(ensure_csv(test_path, fallback_dir=destination), staged["test"])
    _link_or_copy(ensure_csv(sample_path, fallback_dir=destination), staged["sample"])

    if competition_id:
        competition_dir = destination / competition_id
        competition_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(staged["train"], competition_dir / staged["train"].name)
        _link_or_copy(staged["test"], competition_dir / staged["test"].name)
        _link_or_copy(staged["sample"], competition_dir / staged["sample"].name)

    return staged


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
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
