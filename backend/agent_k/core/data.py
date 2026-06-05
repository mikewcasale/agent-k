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
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ("CompetitionSchema", "infer_competition_schema", "locate_data_files", "stage_competition_data")

_DATA_FILE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".csv", ".tsv", ".parquet", ".feather", ".json", ".jsonl", ".txt"}
)
"""File extensions treated as candidate competition data files when matching roles."""

_STEM_WORD_SPLIT: Final[re.Pattern[str]] = re.compile(r"[_\-.\s]+")
"""Regex that splits a file stem into lowercase words on common separators."""


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
        Automatically extracts ZIP files and matches role tokens on file-stem word
        boundaries. Exact stem matches win over word-boundary matches, which in
        turn win over substring matches; competing role tokens disqualify a
        candidate (e.g. ``pretrained_features.csv`` is no longer mistaken for
        ``train.csv``). The same file is never selected for two roles.
        Raises FileNotFoundError if required files are not found.
    """
    files = _gather_candidate_files(paths)

    train_path = _pick_role(files, primary_tokens=("train",), competing_tokens=("test", "sample", "submission"))
    test_path = _pick_role(
        files,
        primary_tokens=("test",),
        competing_tokens=("train", "sample", "submission"),
        exclude=_exclude_set(train_path),
    )
    sample_path = _pick_role(
        files,
        primary_tokens=("sample_submission", "samplesubmission"),
        competing_tokens=("train", "test"),
        exclude=_exclude_set(train_path, test_path),
    ) or _pick_role(
        files,
        primary_tokens=("submission",),
        competing_tokens=("train", "test"),
        exclude=_exclude_set(train_path, test_path),
    )

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


def _gather_candidate_files(paths: Iterable[str | Path]) -> list[Path]:
    """Collect data files from the input list, extracting ZIPs once and deduplicating."""
    seen: set[Path] = set()
    files: list[Path] = []

    def _append(path: Path) -> None:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            return
        seen.add(key)
        files.append(path)

    for path_value in paths:
        path = Path(path_value)
        _append(path)
        if path.suffix.lower() == ".zip" and path.exists():
            for extracted in _safe_extract_zip(path, path.parent):
                _append(extracted)

    return files


def _exclude_set(*paths: Path | None) -> frozenset[Path]:
    """Build a frozen exclusion set from optional paths."""
    return frozenset(path for path in paths if path is not None)


def _stem_words(stem: str) -> list[str]:
    """Split a file stem into lowercase tokens on ``_``, ``-``, ``.``, or whitespace."""
    return [piece for piece in _STEM_WORD_SPLIT.split(stem.lower()) if piece]


def _pick_role(
    files: Sequence[Path],
    *,
    primary_tokens: Sequence[str],
    competing_tokens: Sequence[str] = (),
    exclude: frozenset[Path] = frozenset(),
) -> Path | None:
    """Select the file that best fits a competition data role.

    Selection priority (highest first):
        1. The stem (case-folded) exactly equals one of ``primary_tokens``.
        2. A primary token appears as a stem word AND no competing token does.
        3. A primary token appears anywhere in the stem AND no competing token
           appears as a stem word.
    Ties are broken by shorter file name, then case-folded lexicographic order
    so selection is deterministic across filesystems.
    """
    candidates = sorted(
        (file for file in files if file not in exclude and file.suffix.lower() in _DATA_FILE_EXTENSIONS),
        key=lambda path: (len(path.name), path.name.lower()),
    )
    primary_set = {token.lower() for token in primary_tokens}
    competing_set = {token.lower() for token in competing_tokens}

    for file in candidates:
        if file.stem.lower() in primary_set:
            return file

    for file in candidates:
        words = set(_stem_words(file.stem))
        if words & primary_set and not (words & competing_set):
            return file

    for file in candidates:
        stem = file.stem.lower()
        words = set(_stem_words(file.stem))
        if any(token in stem for token in primary_set) and not (words & competing_set):
            return file

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
