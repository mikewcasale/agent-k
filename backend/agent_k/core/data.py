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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ("CompetitionSchema", "infer_competition_schema", "locate_data_files", "stage_competition_data")

_TIME_COLUMN_TOKENS: Final[tuple[str, ...]] = (
    "timestamp",
    "datetime",
    "date",
    "time",
    "period",
    "month",
    "week",
    "year",
    "day",
)
"""Name tokens that make a column a candidate time index."""

_TIME_SAMPLE_ROWS: Final[int] = 500
_TIME_PARSE_RATIO: Final[float] = 0.9
_TIME_MIN_LENGTH: Final[int] = 6
_COMPACT_DATE_LENGTH: Final[int] = 8
_TIME_SEPARATORS: Final[tuple[str, ...]] = ("-", "/", ":")
_COMPACT_DATE_MAX_RANK: Final[int] = _TIME_COLUMN_TOKENS.index("date")
"""Compact ``YYYYMMDD`` values are only trusted for columns named after a full date."""

_TIME_FORMATS: Final[tuple[str, ...]] = (
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
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
    time_column: str | None = None


def infer_competition_schema(train_path: Path, test_path: Path, sample_path: Path) -> CompetitionSchema:
    """Infer competition schema from train/test/sample submission files.

    @notice: |
        Analyzes CSV headers to determine ID column and target columns.

    @dev: |
        Compares train vs test headers to identify target columns.
        Falls back to sample submission columns if no difference found.
        Also resolves a time index column when train and test share a
        column whose sampled values parse as dates or timestamps.
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

    time_column = _infer_time_column(
        train_path,
        train_header=train_header,
        test_header=test_header,
        excluded={id_column, *target_columns, *train_target_columns},
    )

    return CompetitionSchema(
        id_column=id_column,
        target_columns=list(target_columns),
        train_target_columns=train_target_columns,
        time_column=time_column,
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


def _infer_time_column(
    train_path: Path, *, train_header: Sequence[str], test_header: Sequence[str], excluded: set[str]
) -> str | None:
    """Resolve the column that orders rows in time, when one exists."""
    test_columns = set(test_header)
    candidates = [
        column
        for column in train_header
        if column not in excluded and column in test_columns and _time_token_rank(column) < len(_TIME_COLUMN_TOKENS)
    ]
    if not candidates:
        return None

    samples = _sample_columns(train_path, candidates, max_rows=_TIME_SAMPLE_ROWS)
    temporal = [
        column
        for column in candidates
        if _is_temporal_sample(
            samples.get(column, []), allow_compact=_time_token_rank(column) <= _COMPACT_DATE_MAX_RANK
        )
    ]
    if not temporal:
        return None

    return min(temporal, key=_time_token_rank)


def _time_token_rank(column: str) -> int:
    """Rank a column name by how specific its time token is (lower wins, no token ranks last)."""
    name = column.strip().lower()
    for rank, token in enumerate(_TIME_COLUMN_TOKENS):
        if token in name:
            return rank
    return len(_TIME_COLUMN_TOKENS)


def _sample_columns(path: Path, columns: Sequence[str], *, max_rows: int) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {column: [] for column in columns}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            for column in columns:
                value = row.get(column)
                if value:
                    samples[column].append(value)
    return samples


def _is_temporal_sample(values: Sequence[str], *, allow_compact: bool) -> bool:
    """Report whether sampled values look like an ordered time index."""
    if len(values) < 2 or len({value.strip() for value in values}) < 2:
        return False
    parsed = sum(1 for value in values if _is_datetime_like(value, allow_compact=allow_compact))
    return parsed / len(values) >= _TIME_PARSE_RATIO


def _is_datetime_like(value: str, *, allow_compact: bool) -> bool:
    """Report whether a single value parses as a date or timestamp.

    Values without a date separator are rejected unless ``allow_compact`` is set
    and the value is an eight-digit ``YYYYMMDD`` date, so integer features such
    as ``day_of_week`` or ``year_built`` are never mistaken for a time index.
    """
    text = value.strip()
    if len(text) < _TIME_MIN_LENGTH:
        return False
    separated = any(separator in text for separator in _TIME_SEPARATORS)
    if not separated and not (allow_compact and len(text) == _COMPACT_DATE_LENGTH and text.isdigit()):
        return False
    try:
        datetime.fromisoformat(text)
    except ValueError:
        pass
    else:
        return True
    for time_format in _TIME_FORMATS:
        try:
            datetime.strptime(text, time_format)
        except ValueError:
            continue
        return True
    return False


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
