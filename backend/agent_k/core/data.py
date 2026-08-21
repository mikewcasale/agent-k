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

import logfire

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ("CompetitionSchema", "infer_competition_schema", "locate_data_files", "stage_competition_data")

_ARCHIVE_SUFFIXES: Final[frozenset[str]] = frozenset({".zip", ".gz", ".bz2", ".xz", ".7z", ".tar", ".tgz"})
_SUFFIX_SCORES: Final[dict[str, int]] = {
    ".csv": 30,
    ".tsv": 24,
    ".parquet": 20,
    ".feather": 20,
    ".txt": 10,
    ".json": 6,
    ".jsonl": 6,
}
_CAMEL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_AUXILIARY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "label",
        "labels",
        "meta",
        "metadata",
        "example",
        "examples",
        "dict",
        "dictionary",
        "description",
        "descriptions",
        "info",
        "extra",
        "supplement",
        "supplemental",
        "old",
        "backup",
        "raw",
    }
)
_EXACT_STEM_BONUS: Final[int] = 200
_EXACT_STEM_RANK_PENALTY: Final[int] = 10
_EXTRA_TOKEN_PENALTY: Final[int] = 12
_AUXILIARY_TOKEN_PENALTY: Final[int] = 40
_DEPTH_PENALTY: Final[int] = 2


@dataclass(frozen=True, slots=True)
class _RoleSpec:
    """Matching rules used to rank candidate files for one competition data role."""

    role: str
    exact_stems: tuple[str, ...]
    required_tokens: frozenset[str]
    allowed_tokens: frozenset[str]
    fallback_tokens: tuple[str, ...]


_ROLE_SPECS: Final[tuple[_RoleSpec, ...]] = (
    _RoleSpec(
        role="sample",
        exact_stems=("sample_submission", "submission"),
        required_tokens=frozenset({"submission"}),
        allowed_tokens=frozenset({"submission", "sample", "format", "csv"}),
        fallback_tokens=("sample_submission", "submission"),
    ),
    _RoleSpec(
        role="train",
        exact_stems=("train", "training"),
        required_tokens=frozenset({"train", "training"}),
        allowed_tokens=frozenset({"train", "training", "data", "set", "features", "values", "csv"}),
        fallback_tokens=("train",),
    ),
    _RoleSpec(
        role="test",
        exact_stems=("test", "testing"),
        required_tokens=frozenset({"test", "testing"}),
        allowed_tokens=frozenset({"test", "testing", "data", "set", "features", "values", "csv"}),
        fallback_tokens=("test",),
    ),
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
        Raises FileNotFoundError if required files are not found.
    """
    files: list[Path] = []

    for path_value in paths:
        path = Path(path_value)
        files.append(path)
        if path.suffix.lower() == ".zip" and path.exists():
            try:
                files.extend(_safe_extract_zip(path, path.parent))
            except (zipfile.BadZipFile, OSError, ValueError) as exc:
                logfire.warning("zip_extract_failed", archive=str(path), error=str(exc))

    selected: dict[str, Path] = {}
    for spec in _ROLE_SPECS:
        chosen = _select_role_file(files, spec, exclude=frozenset(selected.values()))
        if chosen is not None:
            selected[spec.role] = chosen

    missing = [spec.role for spec in _ROLE_SPECS if spec.role not in selected]
    if missing:
        raise FileNotFoundError(
            f"Required competition data files not found (missing: {', '.join(sorted(missing))}); "
            f"candidates: {sorted({path.name for path in files})}"
        )

    return selected["train"], selected["test"], selected["sample"]


def _select_role_file(files: Iterable[Path], spec: _RoleSpec, *, exclude: frozenset[Path]) -> Path | None:
    """Pick the best-matching file for one data role.

    Ranks candidates so canonical names (``train.csv``) beat auxiliary ones
    (``train_labels.csv``) regardless of directory iteration order, then falls
    back to loose substring matching so unusual layouts still resolve.
    """
    ranked: list[tuple[int, int, int, str, Path]] = []
    for path in files:
        if path in exclude:
            continue
        score = _score_candidate(path, spec)
        if score is None:
            continue
        ranked.append((-score, len(path.parts), len(path.name), str(path), path))

    if ranked:
        return min(ranked)[4]

    for token in spec.fallback_tokens:
        for path in sorted(files, key=lambda candidate: (len(candidate.parts), str(candidate))):
            if path not in exclude and token in path.name.lower():
                return path

    return None


def _score_candidate(path: Path, spec: _RoleSpec) -> int | None:
    """Score a candidate file for a role, or return ``None`` when it does not qualify."""
    suffix_score = _SUFFIX_SCORES.get(_effective_suffix(path))
    if suffix_score is None:
        return None

    tokens = _stem_tokens(path)
    if not spec.required_tokens.intersection(tokens):
        return None

    score = suffix_score
    normalized = "_".join(tokens)
    if normalized in spec.exact_stems:
        score += _EXACT_STEM_BONUS - _EXACT_STEM_RANK_PENALTY * spec.exact_stems.index(normalized)

    extra_tokens = [token for token in tokens if token not in spec.allowed_tokens]
    score -= _EXTRA_TOKEN_PENALTY * len(extra_tokens)
    score -= _AUXILIARY_TOKEN_PENALTY * sum(1 for token in extra_tokens if token in _AUXILIARY_TOKENS)
    score -= _DEPTH_PENALTY * len(path.parts)
    return score


def _effective_suffix(path: Path) -> str:
    """Return the data suffix of a path, looking through a single archive suffix."""
    suffix = path.suffix.lower()
    if suffix in _ARCHIVE_SUFFIXES:
        return Path(path.stem).suffix.lower()
    return suffix


def _stem_tokens(path: Path) -> tuple[str, ...]:
    """Split a filename stem into lowercase tokens, honouring camelCase boundaries."""
    stem = path.stem
    if _effective_suffix(path) and path.suffix.lower() in _ARCHIVE_SUFFIXES:
        stem = Path(stem).stem
    spaced = _CAMEL_BOUNDARY.sub("_", stem)
    return tuple(token for token in _TOKEN_SEPARATOR.split(spaced.lower()) if token)


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
    # utf-8-sig strips a leading BOM; without it the first column name keeps a
    # "﻿" prefix and every downstream lookup on the id column misses.
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        return [column.strip() for column in next(reader, [])]


def _safe_extract_zip(archive_path: Path, destination: Path) -> list[Path]:
    extracted: list[Path] = []
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename.endswith("/"):
                continue
            target_path = (destination / member.filename).resolve()
            if not target_path.is_relative_to(destination_resolved):
                raise ValueError(f"Zip entry escapes destination: {member.filename}")
            # Re-extracting on every mission phase costs minutes on large archives.
            if not (target_path.exists() and target_path.stat().st_size == member.file_size):
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
