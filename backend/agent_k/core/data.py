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

_DATA_FILE_SUFFIXES: Final[frozenset[str]] = frozenset({".csv", ".tsv", ".parquet", ".feather", ".json", ".jsonl"})
_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


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
        Automatically extracts ZIP files and matches role-specific filenames
        using a token-aware scoring scheme so that, for example, ``train_test.csv``
        will not be selected as the test file when ``test.csv`` is also present,
        and a single file is never returned for two different roles.

        Raises FileNotFoundError if required files cannot be uniquely resolved.
    """
    files: list[Path] = []

    for path_value in paths:
        path = Path(path_value)
        files.append(path)
        if path.suffix.lower() == ".zip" and path.exists():
            files.extend(_safe_extract_zip(path, path.parent))

    used: set[Path] = set()
    sample_path = _pick_role(files, used, ("sample_submission", "samplesubmission"), fallback=("submission",))
    train_path = _pick_role(files, used, ("train", "training"))
    test_path = _pick_role(files, used, ("test", "testing", "eval", "evaluation"))

    if not train_path or not test_path or not sample_path:
        raise FileNotFoundError("Required competition data files not found")

    return train_path, test_path, sample_path


def _pick_role(
    files: list[Path], used: set[Path], primary: Sequence[str], *, fallback: Sequence[str] = ()
) -> Path | None:
    """Pick the best file for a role using token-aware ranking.

    Ranks unused files by how well their stem matches the role's tokens
    (exact > token-at-start > token-at-end > token-as-word). ``primary``
    tokens always outrank ``fallback`` tokens. The chosen file is added to
    ``used`` so subsequent role picks cannot re-select it.
    """
    candidates: list[tuple[tuple[int, int], Path]] = []
    for path in files:
        if path in used:
            continue
        if path.suffix.lower() not in _DATA_FILE_SUFFIXES:
            continue
        stem = path.stem.lower()
        score = _score_role_match(stem, primary)
        if score == 0 and fallback:
            score = max(0, _score_role_match(stem, fallback) - 1)
        if score <= 0:
            continue
        candidates.append(((score, -len(path.name)), path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = candidates[0][1]
    used.add(chosen)
    return chosen


def _score_role_match(stem: str, tokens: Sequence[str]) -> int:
    """Score a filename stem against role tokens.

    Each token is itself tokenized so multi-word patterns like
    ``sample_submission`` match the joined stem and not just individual parts.

    Returns 0 when no token matches as a discrete word in the stem. Substring
    matches inside larger tokens (e.g., "test" inside "testimony") are ignored
    so that role detection cannot be hijacked by unrelated filenames.
    """
    parts = [part for part in _TOKEN_SPLIT.split(stem) if part]
    if not parts:
        return 0
    best = 0
    for token in tokens:
        token_parts = [part for part in _TOKEN_SPLIT.split(token) if part]
        if not token_parts:
            continue
        n = len(token_parts)
        if parts == token_parts:
            return 4
        if len(parts) <= n:
            continue
        if parts[:n] == token_parts:
            best = max(best, 3)
            continue
        if parts[-n:] == token_parts:
            best = max(best, 2)
            continue
        for i in range(1, len(parts) - n):
            if parts[i : i + n] == token_parts:
                best = max(best, 1)
                break
    return best


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
