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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ("CompetitionSchema", "infer_competition_schema", "locate_data_files", "stage_competition_data")

SAMPLE_ROW_LIMIT: Final[int] = 400
"""Maximum number of data rows sampled per file when inferring column types."""

MIN_DATETIME_SAMPLES: Final[int] = 4
"""Minimum number of non-empty values required before a column can read as temporal."""

DATETIME_MATCH_RATIO: Final[float] = 0.8
"""Fraction of sampled values that must parse before a column is treated as temporal."""

_DATETIME_SEPARATORS: Final[tuple[str, ...]] = ("-", "/", ":")
_DATETIME_FORMATS: Final[tuple[str | None, ...]] = (
    None,
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y/%m",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
)


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
    datetime_columns: list[str] = field(default_factory=list)
    temporal_order_column: str | None = None


def infer_competition_schema(train_path: Path, test_path: Path, sample_path: Path) -> CompetitionSchema:
    """Infer competition schema from train/test/sample submission files.

    @notice: |
        Analyzes CSV headers to determine ID column, target columns, and temporal layout.

    @dev: |
        Compares train vs test headers to identify target columns.
        Falls back to sample submission columns if no difference found.
        Samples data rows to detect datetime feature columns and, when the data is
        ordered or split chronologically, the column that defines that order.
    """
    train_header, train_rows = _read_sample(train_path)
    test_header, test_rows = _read_sample(test_path)
    sample_header = _read_header(sample_path)

    if len(sample_header) < 2:
        raise ValueError("Sample submission missing required columns")

    id_column = sample_header[0]
    target_columns = sample_header[1:]

    train_target_columns = [
        column for column in train_header if column not in test_header and column != id_column
    ] or list(target_columns)

    datetime_columns, temporal_order_column = _infer_temporal_layout(
        train_header=train_header,
        train_rows=train_rows,
        test_header=test_header,
        test_rows=test_rows,
        excluded={id_column, *train_target_columns, *target_columns},
    )

    return CompetitionSchema(
        id_column=id_column,
        target_columns=list(target_columns),
        train_target_columns=train_target_columns,
        datetime_columns=datetime_columns,
        temporal_order_column=temporal_order_column,
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
        Uses hard links when possible to save disk space.
    """
    destination.mkdir(parents=True, exist_ok=True)

    staged = {
        "train": destination / "train.csv",
        "test": destination / "test.csv",
        "sample": destination / "sample_submission.csv",
    }

    _link_or_copy(train_path, staged["train"])
    _link_or_copy(test_path, staged["test"])
    _link_or_copy(sample_path, staged["sample"])

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


def _read_sample(path: Path, *, limit: int = SAMPLE_ROW_LIMIT) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows: list[list[str]] = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return header, rows


def _parse_datetime(value: str, fmt: str | None) -> datetime | None:
    if not any(separator in value for separator in _DATETIME_SEPARATORS):
        return None
    try:
        parsed = datetime.fromisoformat(value) if fmt is None else datetime.strptime(value, fmt)
    except ValueError:
        return None
    return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def _column_datetimes(values: list[str]) -> list[datetime] | None:
    """Parse a sampled column with a single format, or return None when it is not temporal."""
    if len(values) < MIN_DATETIME_SAMPLES:
        return None
    for fmt in _DATETIME_FORMATS:
        parsed = [_parse_datetime(value, fmt) for value in values]
        matched = [item for item in parsed if item is not None]
        if len(matched) / len(values) < DATETIME_MATCH_RATIO:
            continue
        return matched if len(set(matched)) > 1 else None
    return None


def _column_values(header: list[str], rows: list[list[str]], column: str) -> list[str]:
    try:
        index = header.index(column)
    except ValueError:
        return []
    return [row[index].strip() for row in rows if index < len(row) and row[index].strip()]


def _is_non_decreasing(values: list[datetime]) -> bool:
    return all(earlier <= later for earlier, later in zip(values, values[1:], strict=False))


def _infer_temporal_layout(
    *,
    train_header: list[str],
    train_rows: list[list[str]],
    test_header: list[str],
    test_rows: list[list[str]],
    excluded: set[str],
) -> tuple[list[str], str | None]:
    """Detect datetime feature columns and the column that orders the data in time.

    @notice: |
        Returns the detected datetime columns and the temporal ordering column.

    @dev: |
        A column only becomes the ordering column when the sampled train values are
        non-decreasing in file order, or when every sampled test value is later than
        every sampled train value (a forecast split). This keeps unordered date
        attributes, such as a date of birth, from switching validation to a
        chronological holdout.
    """
    datetime_columns: list[str] = []
    order_column: str | None = None
    order_rank = 0

    for column in train_header:
        if column in excluded:
            continue
        train_values = _column_datetimes(_column_values(train_header, train_rows, column))
        if train_values is None:
            continue
        datetime_columns.append(column)

        test_values = _column_datetimes(_column_values(test_header, test_rows, column))
        is_forecast_split = test_values is not None and min(test_values) > max(train_values)
        rank = 2 if is_forecast_split else 1 if _is_non_decreasing(train_values) else 0
        if rank > order_rank:
            order_rank = rank
            order_column = column

    return datetime_columns, order_column


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
