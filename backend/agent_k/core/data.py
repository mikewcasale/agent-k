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
import gzip
import os
import shutil
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import logfire

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ("CompetitionSchema", "infer_competition_schema", "locate_data_files", "stage_competition_data")

_ARCHIVE_SUFFIXES: Final[frozenset[str]] = frozenset({".zip", ".gz", ".bz2", ".xz", ".tar", ".tgz", ".7z", ".rar"})
"""Suffixes treated as containers rather than readable data files."""

_TABULAR_SUFFIXES: Final[frozenset[str]] = frozenset({".csv", ".tsv", ".parquet", ".feather", ".txt"})
"""Suffixes that can carry a tabular competition dataset."""

_MAX_ARCHIVE_DEPTH: Final[int] = 3
"""Maximum nesting level expanded when unpacking competition archives."""

_ROLE_TOKENS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("train", ("train", "training")),
    ("test", ("test",)),
    ("sample", ("sample_submission", "samplesubmission", "sample-submission", "submission", "sample")),
)
"""Role name to filename tokens, most specific token first."""

_SEPARATORS: Final[tuple[str, ...]] = ("_", "-", ".")
"""Characters that delimit tokens inside dataset filenames."""

_EXACT_SCORE: Final[int] = 100
_PREFIX_SCORE: Final[int] = 70
_SUFFIX_SCORE: Final[int] = 60
_CONTAINS_SCORE: Final[int] = 40
_TOKEN_RANK_PENALTY: Final[int] = 5
_TABULAR_BONUS: Final[int] = 8
_CSV_BONUS: Final[int] = 4


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
        Expands archives (nested ZIP members and single-file gzip payloads) and
        scores every readable candidate per role, so the canonical ``train.csv``
        outranks companions such as ``train_labels.csv`` regardless of the order
        in which the filesystem yielded the paths. Archives are never returned as
        data files. Raises FileNotFoundError if a role cannot be filled.

    @effects:
        io:
            - local filesystem access
    """
    candidates = _expand_candidates(paths)
    tabular = [path for path in candidates if _data_suffix(path) in _TABULAR_SUFFIXES]

    resolved: dict[str, Path] = {}
    claimed: set[Path] = set()
    for role, tokens in _ROLE_TOKENS:
        choice = _best_candidate(tabular, tokens, claimed) or _best_candidate(candidates, tokens, claimed)
        if choice is None:
            continue
        resolved[role] = choice
        claimed.add(choice)

    missing = [role for role, _ in _ROLE_TOKENS if role not in resolved]
    if missing:
        raise FileNotFoundError(
            f"Required competition data files not found: {', '.join(missing)} "
            f"(inspected {len(candidates)} candidate files)"
        )

    logfire.debug(
        "competition_data_located",
        train=str(resolved["train"]),
        test=str(resolved["test"]),
        sample=str(resolved["sample"]),
        candidate_count=len(candidates),
    )
    return resolved["train"], resolved["test"], resolved["sample"]


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


def _expand_candidates(paths: Iterable[str | Path]) -> list[Path]:
    """Unpack archives and return every readable data file, deduplicated."""
    pending: deque[tuple[Path, int]] = deque((Path(value), 0) for value in paths)
    seen: set[Path] = set()
    candidates: list[Path] = []

    while pending:
        path, depth = pending.popleft()
        if not path.is_file():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)

        if path.suffix.lower() not in _ARCHIVE_SUFFIXES:
            candidates.append(path)
            continue
        if depth >= _MAX_ARCHIVE_DEPTH:
            continue
        pending.extend((member, depth + 1) for member in _unpack(path))

    return candidates


def _unpack(archive_path: Path) -> list[Path]:
    """Expand one archive, returning the files it produced."""
    suffix = archive_path.suffix.lower()
    try:
        if suffix == ".zip":
            return _safe_extract_zip(archive_path, archive_path.parent)
        if suffix == ".gz":
            return _extract_gzip(archive_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logfire.warning("archive_extraction_failed", archive=str(archive_path), error=str(exc))
    return []


def _best_candidate(candidates: list[Path], tokens: tuple[str, ...], claimed: set[Path]) -> Path | None:
    """Return the highest scoring unclaimed candidate for a role."""
    scored = [
        (-score, len(path.parts), len(path.name), str(path), path)
        for path in candidates
        if path not in claimed and (score := _score_candidate(path, tokens)) > 0
    ]
    if not scored:
        return None
    return min(scored)[4]


def _score_candidate(path: Path, tokens: tuple[str, ...]) -> int:
    """Score how strongly a filename identifies a role, 0 when it does not.

    Depth is deliberately excluded here and applied as a tie-break instead, so a
    nested exact match never loses to a shallow partial one.
    """
    stem = _data_stem(path)
    best = 0
    for rank, token in enumerate(tokens):
        match = _token_score(stem, token)
        if match == 0:
            continue
        best = max(best, match - rank * _TOKEN_RANK_PENALTY)

    if best == 0:
        return 0

    suffix = _data_suffix(path)
    if suffix in _TABULAR_SUFFIXES:
        best += _TABULAR_BONUS
    if suffix == ".csv":
        best += _CSV_BONUS
    return best


def _token_score(stem: str, token: str) -> int:
    if stem == token:
        return _EXACT_SCORE
    if any(stem.startswith(f"{token}{separator}") for separator in _SEPARATORS):
        return _PREFIX_SCORE
    if any(stem.endswith(f"{separator}{token}") for separator in _SEPARATORS):
        return _SUFFIX_SCORE
    if token in stem:
        return _CONTAINS_SCORE
    return 0


def _data_suffix(path: Path) -> str:
    """Return the data suffix, looking past any archive suffix."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    while suffixes and suffixes[-1] in _ARCHIVE_SUFFIXES:
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


def _data_stem(path: Path) -> str:
    """Return the filename with archive and data suffixes removed."""
    stem = path.name.lower()
    while True:
        base, separator, suffix = stem.rpartition(".")
        if not separator or f".{suffix}" not in _ARCHIVE_SUFFIXES | _TABULAR_SUFFIXES:
            return stem
        stem = base


def _extract_gzip(archive_path: Path) -> list[Path]:
    """Decompress a single-file gzip payload next to its archive."""
    if _data_suffix(archive_path) not in _TABULAR_SUFFIXES:
        return []
    target = archive_path.with_suffix("")
    if target.exists():
        return [target]
    with gzip.open(archive_path, "rb") as source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return [target]


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if any(_has_control_characters(column) for column in header):
        raise ValueError(f"Header of {path.name} is not readable text; the file is likely binary or compressed")
    return header


def _has_control_characters(value: str) -> bool:
    return any(character < " " and character not in "\t" for character in value)


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
