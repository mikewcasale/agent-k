"""Competition data utilities for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
import csv
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "CompetitionSchema",
    "infer_competition_schema",
    "locate_data_files",
    "stage_competition_data",
)


# =============================================================================
# Section 9: Dataclasses
# =============================================================================
@dataclass(frozen=True, slots=True)
class CompetitionSchema:
    """Schema details inferred from competition data files."""

    id_column: str
    target_columns: list[str]
    train_target_columns: list[str]


# =============================================================================
# Section 12: Functions
# =============================================================================
def infer_competition_schema(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
) -> CompetitionSchema:
    """Infer competition schema from train/test/sample submission files."""
    train_header = _read_header(train_path)
    test_header = _read_header(test_path)
    sample_header = _read_header(sample_path)

    if len(sample_header) < 2:
        raise ValueError("Sample submission missing required columns")

    id_column = sample_header[0]
    target_columns = sample_header[1:]

    train_target_columns = [
        column
        for column in train_header
        if column not in test_header and column != id_column
    ]
    if not train_target_columns:
        train_target_columns = list(target_columns)

    return CompetitionSchema(
        id_column=id_column,
        target_columns=list(target_columns),
        train_target_columns=train_target_columns,
    )


def locate_data_files(paths: Iterable[str | Path]) -> tuple[Path, Path, Path]:
    """Locate train/test/sample files from downloaded data."""
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
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    destination: Path,
) -> dict[str, Path]:
    """Stage competition data into canonical filenames."""
    destination.mkdir(parents=True, exist_ok=True)

    staged = {
        "train": destination / "train.csv",
        "test": destination / "test.csv",
        "sample": destination / "sample_submission.csv",
    }

    _link_or_copy(train_path, staged["train"])
    _link_or_copy(test_path, staged["test"])
    _link_or_copy(sample_path, staged["sample"])

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
