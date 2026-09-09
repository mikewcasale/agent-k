"""Feature modality inference for competition datasets.

@notice: |
    Feature modality inference for competition datasets.

@dev: |
    Samples the head of a training CSV and classifies every feature column into a
    generic modality (numeric, categorical, free text, media file reference or
    datetime). Downstream policy code uses the result to pick a problem family
    without relying on platform tags, which are frequently absent.

@graph:
    id: agent_k.core.modality
    provides:
        - agent_k.core.modality
    pattern: data-access

@agent-guidance:
    do:
        - "Use agent_k.core.modality as the canonical home for this capability."
        - "Classify feature columns with infer_data_modality instead of ad-hoc header heuristics."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Add competition-specific column names; detection must stay generic by ML problem family."

@human-review:
    last-verified: 2026-09-09
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

__all__ = ("ColumnModality", "DataModality", "FeatureModality", "MediaKind", "infer_data_modality")

DEFAULT_SAMPLE_ROWS: Final[int] = 200
"""Number of data rows sampled from the head of the training file."""

_MEDIA_EXTENSIONS: Final[dict[str, str]] = {
    ".bmp": "image",
    ".gif": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".png": "image",
    ".tif": "image",
    ".tiff": "image",
    ".webp": "image",
    ".dcm": "image",
    ".flac": "audio",
    ".m4a": "audio",
    ".mp3": "audio",
    ".ogg": "audio",
    ".wav": "audio",
    ".avi": "video",
    ".mov": "video",
    ".mp4": "video",
    ".npy": "array",
    ".npz": "array",
}
_DATETIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}([ T]\d{1,2}:\d{2})?$"
)
_MEDIA_MIN_RATIO: Final[float] = 0.8
_NUMERIC_MIN_RATIO: Final[float] = 0.95
_DATETIME_MIN_RATIO: Final[float] = 0.9
_TEXT_MIN_MEAN_TOKENS: Final[float] = 6.0
_TEXT_MIN_DISTINCT_RATIO: Final[float] = 0.5

type MediaKind = str
"""Media family attached to a file-reference column (image, audio, video, array)."""


class FeatureModality(StrEnum):
    """Generic modality of a single feature column.

    @notice: |
        Generic modality of a single feature column.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum for the feature modality taxonomy."
            violations: "String literals drift across detection and policy logic."
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    FILE_REFERENCE = "file_reference"
    DATETIME = "datetime"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class ColumnModality:
    """Modality verdict for one feature column.

    @notice: |
        Modality verdict for one feature column.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: profile-model
            rationale: "Bundles the detection evidence alongside the verdict."
            violations: "Bare enums hide why a column was classified."
    """

    column: str
    modality: FeatureModality
    distinct_ratio: float
    mean_token_count: float
    media_kind: MediaKind | None


@dataclass(frozen=True, slots=True)
class DataModality:
    """Modality verdicts for every sampled feature column.

    @notice: |
        Modality verdicts for every sampled feature column.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: profile-model
            rationale: "Aggregates per-column verdicts for policy selection."
            violations: "Ad-hoc dicts drift from the detection taxonomy."
    """

    columns: tuple[ColumnModality, ...]
    sampled_rows: int

    def columns_of(self, modality: FeatureModality) -> tuple[str, ...]:
        """Return the column names classified as the given modality."""
        return tuple(entry.column for entry in self.columns if entry.modality is modality)

    @property
    def text_columns(self) -> tuple[str, ...]:
        """Return the free-text feature columns."""
        return self.columns_of(FeatureModality.TEXT)

    @property
    def file_reference_columns(self) -> tuple[str, ...]:
        """Return the feature columns holding media file references."""
        return self.columns_of(FeatureModality.FILE_REFERENCE)

    @property
    def media_kinds(self) -> frozenset[MediaKind]:
        """Return the media families referenced by file-reference columns."""
        return frozenset(entry.media_kind for entry in self.columns if entry.media_kind is not None)

    @property
    def has_tabular_features(self) -> bool:
        """Return whether any numeric, categorical or datetime feature survives."""
        tabular = {FeatureModality.NUMERIC, FeatureModality.CATEGORICAL, FeatureModality.DATETIME}
        return any(entry.modality in tabular for entry in self.columns)


def infer_data_modality(
    train_path: Path, *, exclude_columns: Iterable[str] = (), sample_rows: int = DEFAULT_SAMPLE_ROWS
) -> DataModality:
    """Classify the feature columns of a training file into generic modalities.

    @notice: |
        Samples the head of a CSV and returns a modality verdict per feature column.

    @dev: |
        Reads at most sample_rows data rows so the cost stays bounded for large
        competition files. Identifier and target columns are excluded so only
        model inputs are classified. Missing or headerless files yield an empty
        DataModality rather than raising, because detection is advisory.
    """
    excluded = {str(column) for column in exclude_columns}
    rows: list[dict[str, str]] = []
    header: Sequence[str] = ()

    try:
        with train_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or ()
            for index, row in enumerate(reader):
                if index >= sample_rows:
                    break
                rows.append(row)
    except OSError:
        return DataModality(columns=(), sampled_rows=0)

    feature_columns = [column for column in header if column and column not in excluded]
    verdicts = tuple(_classify_column(column, [row.get(column) or "" for row in rows]) for column in feature_columns)
    return DataModality(columns=verdicts, sampled_rows=len(rows))


def _classify_column(column: str, raw_values: list[str]) -> ColumnModality:
    values = [value.strip() for value in raw_values if value and value.strip()]
    if not values:
        return ColumnModality(
            column=column, modality=FeatureModality.EMPTY, distinct_ratio=0.0, mean_token_count=0.0, media_kind=None
        )

    total = len(values)
    distinct_ratio = len(set(values)) / total
    mean_token_count = sum(len(value.split()) for value in values) / total

    media_kind = _dominant_media_kind(values)
    if media_kind is not None:
        return ColumnModality(
            column=column,
            modality=FeatureModality.FILE_REFERENCE,
            distinct_ratio=distinct_ratio,
            mean_token_count=mean_token_count,
            media_kind=media_kind,
        )

    if _ratio(values, _is_numeric) >= _NUMERIC_MIN_RATIO:
        modality = FeatureModality.NUMERIC
    elif _ratio(values, _is_datetime) >= _DATETIME_MIN_RATIO:
        modality = FeatureModality.DATETIME
    elif mean_token_count >= _TEXT_MIN_MEAN_TOKENS and distinct_ratio >= _TEXT_MIN_DISTINCT_RATIO:
        modality = FeatureModality.TEXT
    else:
        modality = FeatureModality.CATEGORICAL

    return ColumnModality(
        column=column,
        modality=modality,
        distinct_ratio=distinct_ratio,
        mean_token_count=mean_token_count,
        media_kind=None,
    )


def _dominant_media_kind(values: list[str]) -> MediaKind | None:
    kinds: dict[MediaKind, int] = {}
    for value in values:
        suffix = Path(value).suffix.lower()
        kind = _MEDIA_EXTENSIONS.get(suffix)
        if kind is not None:
            kinds[kind] = kinds.get(kind, 0) + 1

    if not kinds:
        return None

    kind, count = max(kinds.items(), key=lambda item: item[1])
    if count / len(values) < _MEDIA_MIN_RATIO:
        return None
    return kind


def _ratio(values: list[str], predicate: Callable[[str], bool]) -> float:
    return sum(1 for value in values if predicate(value)) / len(values)


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_datetime(value: str) -> bool:
    return _DATETIME_PATTERN.match(value) is not None
