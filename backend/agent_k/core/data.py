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
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ("CompetitionSchema", "infer_competition_schema", "locate_data_files", "stage_competition_data")

_TEMPORAL_NAME_TOKENS: Final[frozenset[str]] = frozenset(
    {"date", "datetime", "timestamp", "time", "dt", "ts", "day", "week", "month", "quarter", "year", "period", "epoch"}
)
"""Generic column-name tokens that suggest a temporal ordering column."""

_TEMPORAL_NAME_PRIORITY: Final[tuple[str, ...]] = ("datetime", "timestamp", "date", "period", "epoch", "time")
"""Preferred temporal tokens, most specific first, used to rank equally valid candidates."""

_TEMPORAL_FALLBACK_FORMATS: Final[tuple[str, ...]] = ("%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%Y/%m", "%d-%m-%Y")
"""Non-ISO date layouts accepted when detecting a temporal column."""

_TEMPORAL_SAMPLE_ROWS: Final[int] = 200
"""Number of leading train rows sampled when validating temporal candidates."""

_TEMPORAL_MIN_SAMPLES: Final[int] = 3
"""Minimum non-empty sampled values required before a column can be called temporal."""

_TEMPORAL_PARSE_RATIO: Final[float] = 0.8
"""Fraction of sampled values that must parse before a column is accepted."""

_TOKEN_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
"""Splits a normalized column name into comparable tokens."""


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
        Analyzes CSV headers to determine ID, target, and temporal ordering columns.

    @dev: |
        Compares train vs test headers to identify target columns.
        Falls back to sample submission columns if no difference found.
        Detects a generic temporal ordering column by validating that a
        date/time-named feature present in both train and test either parses as
        a date or forms a non-decreasing numeric sequence.
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

    time_column = _detect_time_column(
        train_path,
        train_header=train_header,
        test_header=test_header,
        id_column=id_column,
        excluded_columns=set(target_columns) | set(train_target_columns),
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


def _detect_time_column(
    train_path: Path,
    *,
    train_header: Sequence[str],
    test_header: Sequence[str],
    id_column: str,
    excluded_columns: set[str],
) -> str | None:
    test_columns = set(test_header)
    candidates = [
        column
        for column in train_header
        if column in test_columns
        and column != id_column
        and column not in excluded_columns
        and _has_temporal_name(column)
    ]
    if not candidates:
        return None

    samples = _read_column_samples(train_path, candidates, _TEMPORAL_SAMPLE_ROWS)
    ranked: list[tuple[int, int, int, str]] = []
    for column in candidates:
        kind = _temporal_value_kind(samples.get(column, []))
        if kind is None:
            continue
        ranked.append((0 if kind == "datetime" else 1, _temporal_name_rank(column), train_header.index(column), column))

    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def _has_temporal_name(column: str) -> bool:
    tokens = {token for token in _TOKEN_SPLIT_PATTERN.split(column.lower()) if token}
    return bool(tokens & _TEMPORAL_NAME_TOKENS)


def _temporal_name_rank(column: str) -> int:
    tokens = {token for token in _TOKEN_SPLIT_PATTERN.split(column.lower()) if token}
    for rank, token in enumerate(_TEMPORAL_NAME_PRIORITY):
        if token in tokens:
            return rank
    return len(_TEMPORAL_NAME_PRIORITY)


def _read_column_samples(path: Path, columns: Sequence[str], limit: int) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {column: [] for column in columns}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            for column in columns:
                samples[column].append((row.get(column) or "").strip())
    return samples


def _temporal_value_kind(values: Sequence[str]) -> Literal["datetime", "ordinal"] | None:
    populated = [value for value in values if value]
    if len(populated) < _TEMPORAL_MIN_SAMPLES:
        return None

    parsed = sum(1 for value in populated if _parse_temporal(value) is not None)
    if parsed / len(populated) >= _TEMPORAL_PARSE_RATIO:
        return "datetime"

    numbers: list[float] = []
    for value in populated:
        try:
            numbers.append(float(value))
        except ValueError:
            return None

    if len(set(numbers)) < 2:
        return None
    if any(later < earlier for earlier, later in zip(numbers, numbers[1:], strict=False)):
        return None
    return "ordinal"


def _parse_temporal(value: str) -> datetime | None:
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for date_format in _TEMPORAL_FALLBACK_FORMATS:
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    return None


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
