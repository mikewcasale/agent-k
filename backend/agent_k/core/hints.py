"""Adaptive preprocessing hint system for AGENT-K.

@notice: |
    Adaptive preprocessing hint system for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.hints
    provides:
        - agent_k.core.hints
    pattern: analysis-hints

@agent-guidance:
    do:
        - "Use agent_k.core.hints as the canonical home for this capability."
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
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import logfire
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_k.core.tracking import HintEffectivenessTracker

__all__ = (
    "ColumnProfile",
    "ColumnType",
    "DatasetProfile",
    "DistributionStats",
    "HintCategory",
    "HintResult",
    "MissingPattern",
    "PreprocessingHint",
    "build_dataset_profile",
    "compute_hint_priority",
    "detect_applied_hints",
    "generate_preprocessing_hints",
)

MAX_PROFILE_ROWS: Final[int] = 20000
LOW_CARDINALITY_THRESHOLD: Final[int] = 10
TEXT_LONG_THRESHOLD: Final[int] = 100
TEXT_SHORT_MIN_LENGTH: Final[int] = 20
TEXT_UNIQUE_RATIO_THRESHOLD: Final[float] = 0.2
MISSING_VALUE_TOKENS: Final[tuple[str, ...]] = ("", "na", "nan", "null", "none")
MISSING_RATE_THRESHOLD: Final[float] = 0.05
SKEWNESS_THRESHOLD: Final[float] = 1.0
MISSING_PATTERN_CORR_THRESHOLD: Final[float] = 0.2

_GEO_LAT_TOKENS: Final[frozenset[str]] = frozenset({"lat", "latitude"})
_GEO_LON_TOKENS: Final[frozenset[str]] = frozenset({"lon", "lng", "longitude"})
_PRICE_TOKENS: Final[frozenset[str]] = frozenset(
    {"price", "cost", "amount", "fare", "salary", "income", "usd", "value"}
)
_ORDINAL_TOKENS: Final[frozenset[str]] = frozenset({"rank", "grade", "level", "order", "ordinal", "rating", "stage"})
_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
_CAMEL_CASE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"([a-z0-9])([A-Z])|([A-Z])([A-Z][a-z])")
_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_GEO_NAME_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "latitude": re.compile(r"(?:^|[_\W])lat(?:itude)?(?:$|[_\W])", re.IGNORECASE),
    "longitude": re.compile(r"(?:^|[_\W])(?:lon|lng|longitude)(?:$|[_\W])", re.IGNORECASE),
    "neighborhood": re.compile(r"(?:^|[_\W])(?:neighborhood|district|ward|zone|region)(?:$|[_\W])", re.IGNORECASE),
    "zipcode": re.compile(r"(?:^|[_\W])(?:zip|zipcode|postal|postcode)(?:$|[_\W])", re.IGNORECASE),
    "address": re.compile(r"(?:^|[_\W])(?:address|street|location)(?:$|[_\W])", re.IGNORECASE),
    "city": re.compile(r"(?:^|[_\W])(?:city|town|municipality)(?:$|[_\W])", re.IGNORECASE),
    "state": re.compile(r"(?:^|[_\W])(?:state|province|county|prefecture)(?:$|[_\W])", re.IGNORECASE),
}
_GEO_CATEGORICAL_KEYS: Final[tuple[str, ...]] = ("neighborhood", "zipcode", "address", "city", "state")
_HINT_COMMENT_PATTERN: Final[re.Pattern[str]] = re.compile(r"#\s*Applied hint:\s*([\w\-]+)", re.IGNORECASE)


class ColumnType(StrEnum):
    """Detected column semantic type.

    @notice: |
        Detected column semantic type.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum for semantic column type classification."
    """

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_LOW_CARDINALITY = "categorical_low"
    CATEGORICAL_HIGH_CARDINALITY = "categorical_high"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TEXT_SHORT = "text_short"
    TEXT_LONG = "text_long"
    GEOGRAPHIC_LAT = "geo_lat"
    GEOGRAPHIC_LON = "geo_lon"
    PRICE_MONETARY = "price"
    ID_COLUMN = "id"
    BINARY = "binary"


class MissingPattern(StrEnum):
    """Missingness pattern classification.

    @notice: |
        Missingness pattern classification.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum for MCAR/MAR/MNAR classification."
    """

    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"


class HintCategory(StrEnum):
    """Hint category tags for prompt grouping.

    @notice: |
        Hint category tags for prompt grouping.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum for grouping preprocessing hints by category."
    """

    ENCODING = "encoding"
    SCALING = "scaling"
    IMPUTATION = "imputation"
    TRANSFORM = "transform"
    TARGET_TRANSFORM = "target_transform"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    MODEL_OPTIMIZATION = "model_optimization"
    DATA_ENRICHMENT = "data_enrichment"
    TIME_SERIES = "time_series"
    VISION = "vision"
    NLP = "nlp"
    TEXT = "text"
    GEOGRAPHIC = "geographic"
    VALIDATION = "validation"


class HintResult(StrEnum):
    """Result classification for a hint attempt.

    @notice: |
        Result classification for a hint attempt.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum for hint outcome tracking."
    """

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Profile statistics for a single column.

    @notice: |
        Profile statistics for a single column.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: value-object
            rationale: "Frozen dataclass for immutable column statistics."
    """

    name: str
    dtype: str
    column_type: ColumnType
    missing_rate: float
    unique_count: int
    unique_ratio: float
    mean: float | None = None
    std: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    skewness: float | None = None
    average_length: float | None = None
    sample_values: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "column_type": self.column_type.value,
            "missing_rate": self.missing_rate,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "skewness": self.skewness,
            "average_length": self.average_length,
            "sample_values": list(self.sample_values),
        }


@dataclass(frozen=True, slots=True)
class DistributionStats:
    """Distribution statistics for a numeric column.

    @notice: |
        Distribution statistics for a numeric column.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: value-object
            rationale: "Frozen dataclass for immutable distribution stats."
    """

    column_name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    median: float
    skewness: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "column_name": self.column_name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "median": self.median,
            "skewness": self.skewness,
        }


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Comprehensive dataset analysis for hint generation.

    @notice: |
        Comprehensive dataset analysis for hint generation.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: value-object
            rationale: "Frozen dataclass for immutable dataset profile."
    """

    columns: dict[str, ColumnProfile]
    row_count: int
    missing_pattern: MissingPattern
    has_temporal_features: bool
    has_geographic_features: bool
    has_text_features: bool
    has_price_features: bool
    target_distribution: DistributionStats | None
    feature_correlations: dict[str, float]
    target_columns: tuple[str, ...] = ()
    id_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "columns": {name: profile.to_dict() for name, profile in self.columns.items()},
            "row_count": self.row_count,
            "missing_pattern": self.missing_pattern.value,
            "has_temporal_features": self.has_temporal_features,
            "has_geographic_features": self.has_geographic_features,
            "has_text_features": self.has_text_features,
            "has_price_features": self.has_price_features,
            "target_distribution": (self.target_distribution.to_dict() if self.target_distribution else None),
            "feature_correlations": dict(self.feature_correlations),
            "target_columns": list(self.target_columns),
            "id_column": self.id_column,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetProfile:
        """Hydrate from a serialized payload."""
        columns_payload = payload.get("columns", {}) or {}
        columns: dict[str, ColumnProfile] = {}
        for name, item in columns_payload.items():
            if not isinstance(item, dict):
                continue
            column_type_raw = item.get("column_type", ColumnType.CATEGORICAL_LOW_CARDINALITY.value)
            try:
                column_type = ColumnType(column_type_raw)
            except ValueError:
                column_type = ColumnType.CATEGORICAL_LOW_CARDINALITY
            columns[name] = ColumnProfile(
                name=item.get("name", name),
                dtype=item.get("dtype", ""),
                column_type=column_type,
                missing_rate=float(item.get("missing_rate", 0.0)),
                unique_count=int(item.get("unique_count", 0)),
                unique_ratio=float(item.get("unique_ratio", 0.0)),
                mean=_coerce_float(item.get("mean")),
                std=_coerce_float(item.get("std")),
                min_value=_coerce_float(item.get("min_value")),
                max_value=_coerce_float(item.get("max_value")),
                skewness=_coerce_float(item.get("skewness")),
                average_length=_coerce_float(item.get("average_length")),
                sample_values=tuple(item.get("sample_values") or ()),
            )

        target_payload = payload.get("target_distribution")
        target_distribution = None
        if isinstance(target_payload, dict):
            target_distribution = DistributionStats(
                column_name=target_payload.get("column_name", ""),
                count=int(target_payload.get("count", 0)),
                mean=float(target_payload.get("mean", 0.0)),
                std=float(target_payload.get("std", 0.0)),
                min_value=float(target_payload.get("min_value", 0.0)),
                max_value=float(target_payload.get("max_value", 0.0)),
                median=float(target_payload.get("median", 0.0)),
                skewness=float(target_payload.get("skewness", 0.0)),
            )

        missing_pattern_raw = payload.get("missing_pattern", MissingPattern.MCAR.value)
        try:
            missing_pattern = MissingPattern(missing_pattern_raw)
        except ValueError:
            missing_pattern = MissingPattern.MCAR

        return cls(
            columns=columns,
            row_count=int(payload.get("row_count", 0)),
            missing_pattern=missing_pattern,
            has_temporal_features=bool(payload.get("has_temporal_features", False)),
            has_geographic_features=bool(payload.get("has_geographic_features", False)),
            has_text_features=bool(payload.get("has_text_features", False)),
            has_price_features=bool(payload.get("has_price_features", False)),
            target_distribution=target_distribution,
            feature_correlations={
                str(key): float(value) for key, value in (payload.get("feature_correlations") or {}).items()
            },
            target_columns=tuple(payload.get("target_columns") or ()),
            id_column=payload.get("id_column"),
        )


@dataclass(slots=True)
class PreprocessingHint:
    """Single preprocessing recommendation.

    @notice: |
        Single preprocessing recommendation.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: value-object
            rationale: "Mutable dataclass for hint tracking with state."
    """

    id: str
    category: HintCategory
    priority: float
    applicable_columns: list[str]
    description: str
    code_snippet: str
    success_rate: float
    last_attempted: datetime | None
    last_result: HintResult | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "priority": self.priority,
            "applicable_columns": list(self.applicable_columns),
            "description": self.description,
            "code_snippet": self.code_snippet,
            "success_rate": self.success_rate,
            "last_attempted": self.last_attempted.isoformat() if self.last_attempted else None,
            "last_result": self.last_result.value if self.last_result else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessingHint:
        """Hydrate from a serialized payload."""
        last_attempted = None
        last_attempted_raw = payload.get("last_attempted")
        if isinstance(last_attempted_raw, str):
            try:
                last_attempted = datetime.fromisoformat(last_attempted_raw)
            except ValueError:
                last_attempted = None
        last_result_raw = payload.get("last_result")
        if last_result_raw:
            try:
                last_result = HintResult(last_result_raw)
            except ValueError:
                last_result = None
        else:
            last_result = None
        category_raw = payload.get("category", HintCategory.FEATURE_ENGINEERING.value)
        try:
            category = HintCategory(category_raw)
        except ValueError:
            category = HintCategory.FEATURE_ENGINEERING
        return cls(
            id=str(payload.get("id", "")),
            category=category,
            priority=float(payload.get("priority", 0.0)),
            applicable_columns=list(payload.get("applicable_columns") or []),
            description=str(payload.get("description", "")),
            code_snippet=str(payload.get("code_snippet", "")),
            success_rate=float(payload.get("success_rate", 0.0)),
            last_attempted=last_attempted,
            last_result=last_result,
        )


def build_dataset_profile(
    train_path: Path, test_path: Path | None = None, sample_path: Path | None = None
) -> DatasetProfile:
    """Profile dataset columns for preprocessing hints.

    @notice: |
        Builds a comprehensive profile of dataset columns for hint generation.

    @dev: |
        Samples up to MAX_PROFILE_ROWS, infers column types, detects missing
        patterns, and computes target correlations.
    """
    target_columns, id_column = _infer_target_columns(
        train_path=train_path, test_path=test_path, sample_path=sample_path
    )
    row_count = _count_rows(train_path)
    max_rows = MAX_PROFILE_ROWS if row_count <= 0 or row_count > MAX_PROFILE_ROWS else row_count
    df = _read_profile_dataframe(train_path, max_rows=max_rows)
    df = _normalize_missing_values(df)

    column_profiles: dict[str, ColumnProfile] = {}
    for name in df.columns:
        series = df[name]
        profile = _profile_column(name, series, target_columns=target_columns)
        column_profiles[name] = profile

    missing_pattern = _infer_missing_pattern(df, target_columns)
    has_temporal = any(profile.column_type == ColumnType.DATETIME for profile in column_profiles.values())
    has_geo = _has_geo_pairs(column_profiles)
    has_text = any(
        profile.column_type in {ColumnType.TEXT_SHORT, ColumnType.TEXT_LONG} for profile in column_profiles.values()
    )
    has_price = any(profile.column_type == ColumnType.PRICE_MONETARY for profile in column_profiles.values())

    target_distribution = _build_target_distribution(df, target_columns)
    feature_correlations = _compute_feature_correlations(df, target_columns)

    return DatasetProfile(
        columns=column_profiles,
        row_count=row_count,
        missing_pattern=missing_pattern,
        has_temporal_features=has_temporal,
        has_geographic_features=has_geo,
        has_text_features=has_text,
        has_price_features=has_price,
        target_distribution=target_distribution,
        feature_correlations=feature_correlations,
        target_columns=tuple(target_columns),
        id_column=id_column,
    )


def generate_preprocessing_hints(profile: DatasetProfile, competition_id: str | None = None) -> list[PreprocessingHint]:
    """Generate preprocessing hints based on a dataset profile.

    @notice: |
        Returns a list of actionable preprocessing hints based on column profiles.

    @dev: |
        Analyzes column types, missing patterns, and data characteristics to
        generate encoding, imputation, transformation, and model selection hints.
    """
    _ = competition_id
    hints: list[PreprocessingHint] = []
    target_columns = set(profile.target_columns)
    feature_profiles = {name: column for name, column in profile.columns.items() if name not in target_columns}

    low_card = [
        name for name, col in feature_profiles.items() if col.column_type == ColumnType.CATEGORICAL_LOW_CARDINALITY
    ]
    high_card = [
        name for name, col in feature_profiles.items() if col.column_type == ColumnType.CATEGORICAL_HIGH_CARDINALITY
    ]
    ordinal = [name for name, col in feature_profiles.items() if col.column_type == ColumnType.CATEGORICAL_ORDINAL]

    if low_card:
        hints.append(
            PreprocessingHint(
                id="onehot_low_cardinality",
                category=HintCategory.ENCODING,
                priority=0.65,
                applicable_columns=low_card,
                description=(
                    "One-hot encode low-cardinality categorical features "
                    f"({_format_columns_with_uniques(profile, low_card)})."
                ),
                code_snippet=(
                    "from sklearn.preprocessing import OneHotEncoder\n"
                    'encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)\n'
                    "encoded = encoder.fit_transform(df[cols])"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    if high_card:
        hints.append(
            PreprocessingHint(
                id="target_encode_high_cardinality",
                category=HintCategory.ENCODING,
                priority=0.7,
                applicable_columns=high_card,
                description=(
                    "Use target encoding for high-cardinality categories "
                    f"({_format_columns_with_uniques(profile, high_card)})."
                ),
                code_snippet=(
                    "from category_encoders import TargetEncoder\n"
                    "encoder = TargetEncoder(cols=cols)\n"
                    "df[cols] = encoder.fit_transform(df[cols], y)"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        hints.append(
            PreprocessingHint(
                id="frequency_encode_high_cardinality",
                category=HintCategory.ENCODING,
                priority=0.6,
                applicable_columns=high_card,
                description=(
                    "Apply frequency encoding for high-cardinality categories "
                    f"({_format_columns_with_uniques(profile, high_card)})."
                ),
                code_snippet=(
                    "for col in cols:\n    freq = df[col].value_counts(normalize=True)\n    df[col] = df[col].map(freq)"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    if ordinal:
        hints.append(
            PreprocessingHint(
                id="ordinal_encode",
                category=HintCategory.ENCODING,
                priority=0.6,
                applicable_columns=ordinal,
                description=(
                    "Apply ordinal encoding for ordered categorical features "
                    f"({_format_columns_with_uniques(profile, ordinal)})."
                ),
                code_snippet=(
                    "from sklearn.preprocessing import OrdinalEncoder\n"
                    'encoder = OrdinalEncoder(handle_unknown="use_encoded_value", '
                    "unknown_value=-1)\n"
                    "df[cols] = encoder.fit_transform(df[cols])"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    geo_pairs = _match_geo_pairs(feature_profiles)
    for lat_col, lon_col in geo_pairs:
        hints.append(
            PreprocessingHint(
                id="geo_haversine",
                category=HintCategory.GEOGRAPHIC,
                priority=0.7,
                applicable_columns=[lat_col, lon_col],
                description=(
                    f"Engineer geographic distance features for '{lat_col}' and '{lon_col}' "
                    "using haversine or clustering."
                ),
                code_snippet=(
                    "import numpy as np\n"
                    "def haversine(lat1, lon1, lat2, lon2):\n"
                    "    r = 6371.0\n"
                    "    phi1, phi2 = np.radians(lat1), np.radians(lat2)\n"
                    "    dphi = np.radians(lat2 - lat1)\n"
                    "    dlambda = np.radians(lon2 - lon1)\n"
                    "    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * "
                    "np.sin(dlambda / 2) ** 2\n"
                    "    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    price_columns = [name for name, col in feature_profiles.items() if col.column_type == ColumnType.PRICE_MONETARY]
    if price_columns:
        hints.append(
            PreprocessingHint(
                id="price_log_transform",
                category=HintCategory.TRANSFORM,
                priority=0.65,
                applicable_columns=price_columns,
                description=(
                    "Apply log1p transforms for skewed price/monetary features "
                    f"({_format_columns_with_uniques(profile, price_columns)})."
                ),
                code_snippet="df[cols] = np.log1p(df[cols])",
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    datetime_columns = [name for name, col in feature_profiles.items() if col.column_type == ColumnType.DATETIME]
    if datetime_columns:
        hints.append(
            PreprocessingHint(
                id="datetime_cyclical",
                category=HintCategory.FEATURE_ENGINEERING,
                priority=0.65,
                applicable_columns=datetime_columns,
                description=(
                    "Extract cyclical datetime features (sin/cos) and rolling stats from "
                    f"{', '.join(datetime_columns)}."
                ),
                code_snippet=(
                    'df["month"] = df[date_col].dt.month\n'
                    'df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)\n'
                    'df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    missing_columns = [name for name, col in feature_profiles.items() if col.missing_rate >= MISSING_RATE_THRESHOLD]
    if missing_columns:
        hints.append(
            PreprocessingHint(
                id=f"missing_impute_{profile.missing_pattern.value}",
                category=HintCategory.IMPUTATION,
                priority=0.75,
                applicable_columns=missing_columns,
                description=_missing_hint_description(profile, missing_columns),
                code_snippet=_missing_hint_snippet(profile),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    skewed_numeric = [
        name
        for name, col in feature_profiles.items()
        if col.column_type in {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE}
        and (col.skewness or 0.0) >= SKEWNESS_THRESHOLD
    ]
    if skewed_numeric:
        hints.append(
            PreprocessingHint(
                id="numeric_skew_transform",
                category=HintCategory.TRANSFORM,
                priority=0.7,
                applicable_columns=skewed_numeric,
                description=(
                    "Apply power transforms (log1p, Yeo-Johnson) to reduce skew "
                    f"({_format_columns_with_uniques(profile, skewed_numeric)})."
                ),
                code_snippet=(
                    "from sklearn.preprocessing import PowerTransformer\n"
                    'transformer = PowerTransformer(method="yeo-johnson")\n'
                    "df[cols] = transformer.fit_transform(df[cols])"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    text_columns = [
        name
        for name, col in feature_profiles.items()
        if col.column_type in {ColumnType.TEXT_SHORT, ColumnType.TEXT_LONG}
    ]
    if text_columns:
        hints.append(
            PreprocessingHint(
                id="text_tfidf",
                category=HintCategory.TEXT,
                priority=0.6,
                applicable_columns=text_columns,
                description=(f"Vectorize text features with TF-IDF for sparse models ({', '.join(text_columns)})."),
                code_snippet=(
                    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                    "vectorizer = TfidfVectorizer(max_features=5000)\n"
                    "text_features = vectorizer.fit_transform(df[text_col])"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        hints.append(
            PreprocessingHint(
                id="text_embeddings",
                category=HintCategory.TEXT,
                priority=0.55,
                applicable_columns=text_columns,
                description="Consider sentence embeddings for richer text representations.",
                code_snippet=(
                    "from sentence_transformers import SentenceTransformer\n"
                    'model = SentenceTransformer("all-MiniLM-L6-v2")\n'
                    'embeddings = model.encode(df[text_col].fillna(""))'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        if _has_non_ascii_samples(feature_profiles, text_columns):
            hints.append(
                PreprocessingHint(
                    id="nlp_multilingual",
                    category=HintCategory.NLP,
                    priority=0.55,
                    applicable_columns=text_columns,
                    description=("Detect language and consider multilingual embeddings for mixed-language text."),
                    code_snippet=("from langdetect import detect\nlang = detect(text)"),
                    success_rate=0.0,
                    last_attempted=None,
                    last_result=None,
                )
            )

    if profile.has_temporal_features:
        hints.append(
            PreprocessingHint(
                id="timeseries_stationarity",
                category=HintCategory.TIME_SERIES,
                priority=0.7,
                applicable_columns=datetime_columns,
                description="Test for stationarity (ADF) and apply differencing if needed.",
                code_snippet=(
                    "from statsmodels.tsa.stattools import adfuller\nadf_stat, p_value, *_ = adfuller(series.dropna())"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        hints.append(
            PreprocessingHint(
                id="timeseries_seasonality",
                category=HintCategory.TIME_SERIES,
                priority=0.65,
                applicable_columns=datetime_columns,
                description="Model seasonality with decomposition or Fourier features.",
                code_snippet=(
                    "from statsmodels.tsa.seasonal import seasonal_decompose\n"
                    'decomp = seasonal_decompose(series, model="additive")'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        hints.append(
            PreprocessingHint(
                id="timeseries_lag_features",
                category=HintCategory.TIME_SERIES,
                priority=0.7,
                applicable_columns=datetime_columns,
                description="Create lag and rolling features based on autocorrelation.",
                code_snippet=('df["lag_1"] = series.shift(1)\ndf["rolling_mean"] = series.rolling(7).mean()'),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        hints.append(
            PreprocessingHint(
                id="timeseries_cv",
                category=HintCategory.VALIDATION,
                priority=0.8,
                applicable_columns=datetime_columns,
                description="Use TimeSeriesSplit or purged K-fold for temporal validation.",
                code_snippet=(
                    "from sklearn.model_selection import TimeSeriesSplit\ntscv = TimeSeriesSplit(n_splits=5)"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    image_columns = _detect_image_path_columns(feature_profiles)
    if image_columns:
        hints.append(
            PreprocessingHint(
                id="vision_augmentation",
                category=HintCategory.VISION,
                priority=0.7,
                applicable_columns=image_columns,
                description=("Apply image augmentation (flip, rotate, color jitter) during training."),
                code_snippet=(
                    "import torchvision.transforms as T\n"
                    "train_tfms = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10)])"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        hints.append(
            PreprocessingHint(
                id="vision_normalization",
                category=HintCategory.VISION,
                priority=0.6,
                applicable_columns=image_columns,
                description="Normalize images with ImageNet statistics for pretrained backbones.",
                code_snippet=("normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        if profile.row_count and profile.row_count < 10000:
            hints.append(
                PreprocessingHint(
                    id="vision_transfer_learning",
                    category=HintCategory.VISION,
                    priority=0.75,
                    applicable_columns=image_columns,
                    description=("Use transfer learning or feature extraction for small image datasets."),
                    code_snippet=('from torchvision.models import resnet18\nmodel = resnet18(weights="DEFAULT")'),
                    success_rate=0.0,
                    last_attempted=None,
                    last_result=None,
                )
            )

    if text_columns:
        hints.append(
            PreprocessingHint(
                id="nlp_tokenization",
                category=HintCategory.NLP,
                priority=0.6,
                applicable_columns=text_columns,
                description="Tune tokenization and max sequence length for NLP models.",
                code_snippet="tokens = tokenizer(text, truncation=True, max_length=256)",
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    hints.extend(_generate_geospatial_hints(profile, feature_profiles))
    hints.extend(_generate_knn_feature_hints(profile, feature_profiles))
    hints.extend(_generate_model_selection_hints(profile))
    hints.extend(_generate_feature_selection_hints(profile))
    hints.extend(_generate_data_enrichment_hints(profile, feature_profiles))

    target_distribution = profile.target_distribution
    target_profile = profile.columns.get(target_distribution.column_name) if target_distribution else None
    if (
        target_distribution
        and target_profile
        and target_profile.column_type == ColumnType.NUMERIC_CONTINUOUS
        and target_distribution.skewness >= SKEWNESS_THRESHOLD
    ):
        hints.append(
            PreprocessingHint(
                id="target_log_transform",
                category=HintCategory.TARGET_TRANSFORM,
                priority=0.75,
                applicable_columns=[target_distribution.column_name],
                description=(
                    "Apply log1p or Box-Cox on skewed targets and invert before submission "
                    f"({target_distribution.column_name})."
                ),
                code_snippet="y = np.log1p(y); preds = np.expm1(preds)",
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    return hints


def compute_hint_priority(
    hint: PreprocessingHint, tracker: HintEffectivenessTracker, competition_id: str, generation: int
) -> float:
    """Compute dynamic priority based on historical effectiveness.

    @notice: |
        Adjusts hint priority using success rates and recency from the tracker.

    @dev: |
        Applies success boost, recency penalty, and amplification factors.
        Returns 0.0 if hint is suppressed.
    """
    base_priority = hint.priority
    success_rate = tracker.get_success_rate(hint.id, competition_id)
    success_boost = success_rate * 0.3
    recency_penalty = 0.0
    last_attempt = tracker.get_last_attempt(hint.id, competition_id)
    if last_attempt is not None:
        gens_since = max(0, generation - last_attempt.generation)
        if gens_since < 3:
            recency_penalty = 0.2 * (3 - gens_since) / 3
    if tracker.is_suppressed(hint.id, competition_id):
        return 0.0
    amplified_boost = 0.1 if tracker.is_amplified(hint.id, competition_id) else 0.0
    return min(1.0, max(0.0, base_priority + success_boost + amplified_boost - recency_penalty))


def detect_applied_hints(code: str, hints: Iterable[PreprocessingHint]) -> set[str]:
    """Detect which hints were applied based on solution code patterns.

    @notice: |
        Returns hint IDs that appear to be applied in the given code.

    @dev: |
        Matches explicit hint comments, code snippet signatures, and
        pattern-based detection for each hint type.
    """
    if not isinstance(code, str):
        return set()
    applied: set[str] = set()
    hint_map = {hint.id: hint for hint in hints}

    for match in _HINT_COMMENT_PATTERN.finditer(code):
        hint_id = match.group(1)
        if hint_id in hint_map:
            applied.add(hint_id)

    for hint in hint_map.values():
        signatures = _extract_signatures(hint.code_snippet)
        for signature in signatures:
            if re.search(rf"\\b{re.escape(signature)}\\b", code):
                applied.add(hint.id)
                break

        patterns = _hint_patterns(hint)
        if not patterns:
            continue
        if any(pattern.search(code) for pattern in patterns):
            applied.add(hint.id)

    return applied


def _profile_column(name: str, series: pd.Series, *, target_columns: Iterable[str]) -> ColumnProfile:
    series = series.copy()
    missing_rate = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = float(unique_count / max(len(series), 1))
    sample_values = _sample_values(series)
    dtype = str(series.dtype)

    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_bool = pd.api.types.is_bool_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    if not is_datetime and series.dtype == object:
        is_datetime = _looks_like_datetime(series)

    avg_length = _average_length(series)

    column_type = _detect_column_type(
        name=name,
        is_numeric=is_numeric,
        is_bool=is_bool,
        is_datetime=is_datetime,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        avg_length=avg_length,
        sample_values=sample_values,
        is_target=name in target_columns,
    )

    numeric_stats = _numeric_stats(series) if is_numeric else {}
    return ColumnProfile(
        name=name,
        dtype=dtype,
        column_type=column_type,
        missing_rate=missing_rate,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        mean=numeric_stats.get("mean"),
        std=numeric_stats.get("std"),
        min_value=numeric_stats.get("min_value"),
        max_value=numeric_stats.get("max_value"),
        skewness=numeric_stats.get("skewness"),
        average_length=avg_length,
        sample_values=sample_values,
    )


def _detect_column_type(
    *,
    name: str,
    is_numeric: bool,
    is_bool: bool,
    is_datetime: bool,
    unique_count: int,
    unique_ratio: float,
    avg_length: float | None,
    sample_values: tuple[str, ...],
    is_target: bool,
) -> ColumnType:
    if _is_id_column(name, unique_ratio=unique_ratio, is_target=is_target):
        return ColumnType.ID_COLUMN
    if _is_geo_lat(name):
        return ColumnType.GEOGRAPHIC_LAT
    if _is_geo_lon(name):
        return ColumnType.GEOGRAPHIC_LON
    if _is_price_column(name, sample_values):
        return ColumnType.PRICE_MONETARY
    if is_bool or (is_numeric and unique_count <= 2):
        return ColumnType.BINARY
    if is_datetime:
        return ColumnType.DATETIME
    if is_numeric:
        if _is_ordinal_name(name) and unique_count <= 20:
            return ColumnType.CATEGORICAL_ORDINAL
        if unique_count <= 15:
            return ColumnType.NUMERIC_DISCRETE
        return ColumnType.NUMERIC_CONTINUOUS
    if avg_length is not None:
        if avg_length >= TEXT_LONG_THRESHOLD:
            return ColumnType.TEXT_LONG
        if avg_length >= TEXT_SHORT_MIN_LENGTH and unique_ratio >= TEXT_UNIQUE_RATIO_THRESHOLD:
            return ColumnType.TEXT_SHORT
    if unique_count < LOW_CARDINALITY_THRESHOLD:
        return ColumnType.CATEGORICAL_LOW_CARDINALITY
    return ColumnType.CATEGORICAL_HIGH_CARDINALITY


def _infer_target_columns(
    *, train_path: Path, test_path: Path | None, sample_path: Path | None
) -> tuple[list[str], str | None]:
    train_header = _read_header(train_path)
    test_header = _read_header(test_path) if test_path else []
    sample_header = _read_header(sample_path) if sample_path else []

    if sample_header and len(sample_header) >= 2:
        return list(sample_header[1:]), sample_header[0]

    id_column = train_header[0] if train_header else None
    target_columns = [col for col in train_header if col not in test_header and col != id_column]
    if not target_columns and len(train_header) >= 2:
        target_columns = [train_header[-1]]
    return target_columns, id_column


def _read_profile_dataframe(path: Path, *, max_rows: int) -> pd.DataFrame:
    with logfire.span("hints.read_profile_dataframe", path=str(path), max_rows=max_rows):
        try:
            return pd.read_csv(path, nrows=max_rows, low_memory=False)
        except Exception as exc:
            logfire.warning("hints_read_failed", error=str(exc), path=str(path))
            return pd.DataFrame()


def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.replace(list(MISSING_VALUE_TOKENS), np.nan)


def _read_header(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except FileNotFoundError:
        return []


def _count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except FileNotFoundError:
        return 0


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    if series.empty:
        return {}
    skew_val = series.skew()
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min_value": float(series.min()),
        "max_value": float(series.max()),
        "skewness": float(skew_val) if isinstance(skew_val, (int, float)) else 0.0,
    }


def _average_length(series: pd.Series) -> float | None:
    if series.empty:
        return None
    if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
        return None
    lengths = series.dropna().astype(str).str.len()
    if lengths.empty:
        return None
    return float(lengths.mean())


def _sample_values(series: pd.Series, *, max_samples: int = 5) -> tuple[str, ...]:
    if series.empty:
        return ()
    values = series.dropna().astype(str).unique().tolist()
    return tuple(values[:max_samples])


def _looks_like_datetime(series: pd.Series, *, threshold: float = 0.8) -> bool:
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return float(parsed.notna().mean()) >= threshold


def _name_tokens(name: str) -> frozenset[str]:
    """Split a column name into lowercase tokens across camelCase and non-alphanumeric boundaries.

    Splits on both underscore/hyphen/space style separators and camelCase transitions so tokens
    can be matched exactly instead of as substrings, avoiding false positives like "template"
    matching "lat" or "border" matching "order".
    """
    snake = _CAMEL_CASE_BOUNDARY.sub(r"\1\3_\2\4", name)
    return frozenset(token for token in _TOKEN_SPLIT.split(snake.lower()) if token)


def _is_geo_lat(name: str) -> bool:
    return bool(_name_tokens(name) & _GEO_LAT_TOKENS)


def _is_geo_lon(name: str) -> bool:
    return bool(_name_tokens(name) & _GEO_LON_TOKENS)


def _is_price_column(name: str, sample_values: tuple[str, ...]) -> bool:
    if _name_tokens(name) & _PRICE_TOKENS:
        return True
    for value in sample_values:
        if any(symbol in value for symbol in ("$", "€", "£", "¥")):
            return True
    return False


def _is_id_column(name: str, *, unique_ratio: float, is_target: bool) -> bool:
    if is_target:
        return False
    lowered = name.lower()
    if lowered == "id" or lowered.endswith("_id"):
        return True
    return unique_ratio >= 0.98


def _is_ordinal_name(name: str) -> bool:
    return bool(_name_tokens(name) & _ORDINAL_TOKENS)


def _infer_missing_pattern(df: pd.DataFrame, target_columns: Iterable[str]) -> MissingPattern:
    if df.empty:
        return MissingPattern.MCAR

    target_name = next(iter(target_columns), None)
    target_series = df[target_name] if target_name and target_name in df.columns else None
    numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    has_mar = False
    has_mnar = False
    for column in df.columns:
        missing_rate = float(df[column].isna().mean())
        if missing_rate < MISSING_RATE_THRESHOLD:
            continue
        indicator = df[column].isna().astype(int)
        if indicator.nunique() < 2:
            continue
        if target_series is not None and pd.api.types.is_numeric_dtype(target_series):
            corr = float(indicator.corr(pd.to_numeric(target_series, errors="coerce")))
            if abs(corr) >= MISSING_PATTERN_CORR_THRESHOLD:
                has_mnar = True
                continue
        for feature in numeric_features:
            if feature == column:
                continue
            corr = float(indicator.corr(pd.to_numeric(df[feature], errors="coerce")))
            if abs(corr) >= MISSING_PATTERN_CORR_THRESHOLD:
                has_mar = True
                break

    if has_mnar:
        return MissingPattern.MNAR
    if has_mar:
        return MissingPattern.MAR
    return MissingPattern.MCAR


def _build_target_distribution(df: pd.DataFrame, target_columns: Iterable[str]) -> DistributionStats | None:
    target_name = next(iter(target_columns), None)
    if target_name is None or target_name not in df.columns:
        return None
    series = pd.to_numeric(df[target_name], errors="coerce").dropna()
    if series.empty:
        return None
    skew_val = series.skew()
    return DistributionStats(
        column_name=target_name,
        count=int(series.count()),
        mean=float(series.mean()),
        std=float(series.std(ddof=0)),
        min_value=float(series.min()),
        max_value=float(series.max()),
        median=float(series.median()),
        skewness=float(skew_val) if isinstance(skew_val, (int, float)) else 0.0,
    )


def _compute_feature_correlations(df: pd.DataFrame, target_columns: Iterable[str]) -> dict[str, float]:
    target_name = next(iter(target_columns), None)
    if target_name is None or target_name not in df.columns:
        return {}
    target = pd.to_numeric(df[target_name], errors="coerce")
    if target.dropna().empty:
        return {}
    numeric_cols = [col for col in df.columns if col != target_name and pd.api.types.is_numeric_dtype(df[col])]
    correlations: dict[str, float] = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        corr = float(series.corr(target))
        if math.isnan(corr):
            continue
        correlations[col] = corr
    return correlations


def _has_geo_pairs(columns: dict[str, ColumnProfile]) -> bool:
    return bool(_match_geo_pairs(columns))


def _match_geo_pairs(columns: dict[str, ColumnProfile]) -> list[tuple[str, str]]:
    lat_candidates = [name for name, col in columns.items() if col.column_type == ColumnType.GEOGRAPHIC_LAT]
    lon_candidates = [name for name, col in columns.items() if col.column_type == ColumnType.GEOGRAPHIC_LON]
    if not lat_candidates or not lon_candidates:
        return []
    pairs: list[tuple[str, str]] = []
    for lat_name in lat_candidates:
        base = _strip_geo_token(lat_name)
        for lon_name in lon_candidates:
            if _strip_geo_token(lon_name) == base:
                pairs.append((lat_name, lon_name))
                break
    if not pairs and lat_candidates and lon_candidates:
        pairs.append((lat_candidates[0], lon_candidates[0]))
    return pairs


def _strip_geo_token(name: str) -> str:
    lowered = name.lower()
    # Longer tokens first so that "latitude" is stripped before "lat" and leaves a clean base.
    for token in sorted(_GEO_LAT_TOKENS | _GEO_LON_TOKENS, key=len, reverse=True):
        lowered = lowered.replace(token, "")
    return re.sub(r"[_\W]+", "", lowered)


def _detect_image_path_columns(columns: dict[str, ColumnProfile]) -> list[str]:
    image_columns: list[str] = []
    for name, profile in columns.items():
        if profile.column_type in {
            ColumnType.TEXT_SHORT,
            ColumnType.TEXT_LONG,
            ColumnType.CATEGORICAL_HIGH_CARDINALITY,
        }:
            for value in profile.sample_values:
                if any(value.lower().endswith(ext) for ext in _IMAGE_EXTENSIONS):
                    image_columns.append(name)
                    break
    return image_columns


def _has_non_ascii_samples(columns: dict[str, ColumnProfile], text_columns: list[str]) -> bool:
    for name in text_columns:
        profile = columns.get(name)
        if profile is None:
            continue
        for value in profile.sample_values:
            if any(ord(char) > 127 for char in value):
                return True
    return False


def _infer_problem_kind(profile: DatasetProfile) -> str:
    target_name = next(iter(profile.target_columns), None)
    if not target_name:
        return "regression"
    target_profile = profile.columns.get(target_name)
    if target_profile is None:
        return "regression"
    if target_profile.column_type in {
        ColumnType.BINARY,
        ColumnType.CATEGORICAL_LOW_CARDINALITY,
        ColumnType.CATEGORICAL_HIGH_CARDINALITY,
        ColumnType.CATEGORICAL_ORDINAL,
    }:
        return "classification"
    if target_profile.column_type == ColumnType.NUMERIC_DISCRETE and target_profile.unique_count <= 20:
        return "classification"
    return "regression"


def _detect_geo_named_columns(columns: dict[str, ColumnProfile]) -> dict[str, list[str]]:
    detected: dict[str, set[str]] = {key: set() for key in _GEO_NAME_PATTERNS}
    for name, profile in columns.items():
        for key, pattern in _GEO_NAME_PATTERNS.items():
            if pattern.search(name):
                detected[key].add(name)
        if profile.column_type == ColumnType.GEOGRAPHIC_LAT:
            detected["latitude"].add(name)
        elif profile.column_type == ColumnType.GEOGRAPHIC_LON:
            detected["longitude"].add(name)
    return {key: sorted(values) for key, values in detected.items() if values}


def _geo_categorical_columns(columns: dict[str, ColumnProfile]) -> list[str]:
    detected = _detect_geo_named_columns(columns)
    candidates: set[str] = set()
    for key in _GEO_CATEGORICAL_KEYS:
        candidates.update(detected.get(key, []))
    if not candidates:
        return []
    allowed = {
        ColumnType.CATEGORICAL_LOW_CARDINALITY,
        ColumnType.CATEGORICAL_HIGH_CARDINALITY,
        ColumnType.CATEGORICAL_ORDINAL,
        ColumnType.TEXT_SHORT,
        ColumnType.TEXT_LONG,
    }
    return sorted([name for name in candidates if columns.get(name) and columns[name].column_type in allowed])


def _generate_geospatial_hints(
    profile: DatasetProfile, feature_profiles: dict[str, ColumnProfile]
) -> list[PreprocessingHint]:
    hints: list[PreprocessingHint] = []
    geo_categories = _geo_categorical_columns(feature_profiles)
    if geo_categories:
        hints.append(
            PreprocessingHint(
                id="geospatial_target_encoding",
                category=HintCategory.FEATURE_ENGINEERING,
                priority=0.75,
                applicable_columns=geo_categories,
                description=(
                    f"Apply target encoding at multiple geographic granularities ({', '.join(geo_categories)})."
                ),
                code_snippet=(
                    "from sklearn.model_selection import KFold\n"
                    "def geo_target_encode(df, geo_col, target_col, n_splits=5):\n"
                    "    encoded = pd.Series(index=df.index, dtype=float)\n"
                    "    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\n"
                    "    for train_idx, val_idx in kf.split(df):\n"
                    "        means = df.iloc[train_idx].groupby(geo_col)[target_col].mean()\n"
                    "        encoded.iloc[val_idx] = df.iloc[val_idx][geo_col].map(means)\n"
                    "    encoded.fillna(df[target_col].mean(), inplace=True)\n"
                    "    return encoded"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
        if len(geo_categories) >= 2:
            hints.append(
                PreprocessingHint(
                    id="hierarchical_geo_encoding",
                    category=HintCategory.FEATURE_ENGINEERING,
                    priority=0.7,
                    applicable_columns=geo_categories,
                    description="Encode target statistics at multiple geographic hierarchies.",
                    code_snippet=(
                        "def hierarchical_geo_agg(df, geo_cols, target_col, agg_funcs=None):\n"
                        '    agg_funcs = agg_funcs or ["mean", "median", "std", "count"]\n'
                        "    for geo_col in geo_cols:\n"
                        "        for func in agg_funcs:\n"
                        "            agg = df.groupby(geo_col)[target_col].transform(func)\n"
                        '            df[f"{geo_col}_{target_col}_{func}"] = agg\n'
                        "    return df"
                    ),
                    success_rate=0.0,
                    last_attempted=None,
                    last_result=None,
                )
            )

    geo_pairs = _match_geo_pairs(feature_profiles)
    if geo_pairs:
        lat_cols = sorted({lat for lat, _ in geo_pairs})
        lon_cols = sorted({lon for _, lon in geo_pairs})
        hints.append(
            PreprocessingHint(
                id="geo_knn_features",
                category=HintCategory.FEATURE_ENGINEERING,
                priority=0.75,
                applicable_columns=lat_cols + lon_cols,
                description="Create KNN-based geospatial aggregates from nearby samples.",
                code_snippet=(
                    "from sklearn.neighbors import BallTree\n"
                    "def knn_geo_features(df, lat_col, lon_col, target_col, k_values=None):\n"
                    "    k_values = k_values or [5, 10, 25]\n"
                    "    coords = np.radians(df[[lat_col, lon_col]].values)\n"
                    '    tree = BallTree(coords, metric="haversine")\n'
                    "    for k in k_values:\n"
                    "        distances, indices = tree.query(coords, k=k + 1)\n"
                    "        neighbor_targets = df[target_col].values[indices[:, 1:]]\n"
                    '        df[f"{target_col}_knn{k}_mean"] = neighbor_targets.mean(axis=1)\n'
                    '        df[f"{target_col}_knn{k}_std"] = neighbor_targets.std(axis=1)\n'
                    '        df[f"geo_dist_knn{k}_mean"] = distances[:, 1:].mean(axis=1) * 6371\n'
                    "    return df"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    return hints


def _generate_knn_feature_hints(
    profile: DatasetProfile, feature_profiles: dict[str, ColumnProfile]
) -> list[PreprocessingHint]:
    hints: list[PreprocessingHint] = []
    problem_kind = _infer_problem_kind(profile)
    numeric_columns = [
        name
        for name, col in feature_profiles.items()
        if col.column_type in {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE}
    ]
    if problem_kind != "regression" or len(numeric_columns) < 5:
        return hints

    hints.append(
        PreprocessingHint(
            id="knn_feature_generation",
            category=HintCategory.FEATURE_ENGINEERING,
            priority=0.7,
            applicable_columns=numeric_columns,
            description="Generate KNN distance features from numeric inputs.",
            code_snippet=(
                "from sklearn.neighbors import NearestNeighbors\n"
                "def add_knn_features(train_df, test_df, feature_cols, n_neighbors=5):\n"
                "    nn = NearestNeighbors(n_neighbors=n_neighbors)\n"
                "    nn.fit(train_df[feature_cols].fillna(0))\n"
                "    train_dist, _ = nn.kneighbors(train_df[feature_cols].fillna(0))\n"
                "    test_dist, _ = nn.kneighbors(test_df[feature_cols].fillna(0))\n"
                '    train_df["knn_mean_dist"] = train_dist.mean(axis=1)\n'
                '    train_df["knn_min_dist"] = train_dist.min(axis=1)\n'
                '    test_df["knn_mean_dist"] = test_dist.mean(axis=1)\n'
                '    test_df["knn_min_dist"] = test_dist.min(axis=1)\n'
                "    return train_df, test_df"
            ),
            success_rate=0.0,
            last_attempted=None,
            last_result=None,
        )
    )

    return hints


def _generate_model_selection_hints(profile: DatasetProfile) -> list[PreprocessingHint]:
    hints: list[PreprocessingHint] = []
    has_tabular = any(
        col.column_type
        in {
            ColumnType.NUMERIC_CONTINUOUS,
            ColumnType.NUMERIC_DISCRETE,
            ColumnType.CATEGORICAL_LOW_CARDINALITY,
            ColumnType.CATEGORICAL_HIGH_CARDINALITY,
            ColumnType.CATEGORICAL_ORDINAL,
        }
        for col in profile.columns.values()
    )
    if not has_tabular:
        return hints
    problem_kind = _infer_problem_kind(profile)
    n_samples = profile.row_count
    model_class = "LGBMRegressor" if problem_kind == "regression" else "LGBMClassifier"
    knn_class = "KNeighborsRegressor" if problem_kind == "regression" else "KNeighborsClassifier"
    stacking_class = "StackingRegressor" if problem_kind == "regression" else "StackingClassifier"
    meta_class = "Ridge" if problem_kind == "regression" else "LogisticRegression"
    meta_snippet = f"{meta_class}(alpha=1.0)" if meta_class == "Ridge" else f"{meta_class}(max_iter=1000)"

    hints.append(
        PreprocessingHint(
            id="model_gradient_boosting",
            category=HintCategory.MODEL_SELECTION,
            priority=0.75,
            applicable_columns=[],
            description="Gradient boosting models perform well on tabular data with mixed feature types.",
            code_snippet=(
                f"from lightgbm import {model_class}\n"
                f"model = {model_class}(\n"
                "    n_estimators=1000,\n"
                "    learning_rate=0.05,\n"
                "    max_depth=7,\n"
                "    num_leaves=31,\n"
                "    min_child_samples=20,\n"
                "    subsample=0.8,\n"
                "    colsample_bytree=0.8,\n"
                "    reg_alpha=0.1,\n"
                "    reg_lambda=0.1,\n"
                "    random_state=42,\n"
                "    n_jobs=-1,\n"
                ")"
            ),
            success_rate=0.0,
            last_attempted=None,
            last_result=None,
        )
    )

    target_distribution = profile.target_distribution
    target_skewed = bool(target_distribution and target_distribution.skewness >= SKEWNESS_THRESHOLD)
    if problem_kind == "regression":
        priority = 0.95 if target_skewed else 0.85
        description = (
            "Use LightGBM with a custom objective; evolve loss parameters for better fit."
            if not target_skewed
            else "Use a custom RMSLE-style LightGBM objective for skewed regression targets."
        )
        hints.append(
            PreprocessingHint(
                id="lightgbm_custom_rmsle",
                category=HintCategory.MODEL_OPTIMIZATION,
                priority=priority,
                applicable_columns=[],
                description=description,
                code_snippet=(
                    "import lightgbm as lgb\n"
                    "import numpy as np\n"
                    "def rmsle_objective(y_pred, train_data):\n"
                    "    y_true = train_data.get_label()\n"
                    "    preds = np.maximum(y_pred, 0)\n"
                    "    grad = (np.log1p(preds) - np.log1p(y_true)) / (preds + 1)\n"
                    "    hess = (1 - np.log1p(preds) + np.log1p(y_true)) / (preds + 1) ** 2\n"
                    "    return grad, hess\n"
                    "train_data = lgb.Dataset(X_train, label=y_train)\n"
                    "params = {\n"
                    '    "objective": "regression",\n'
                    '    "learning_rate": 0.01,\n'
                    '    "num_leaves": 31,\n'
                    '    "feature_fraction": 0.8,\n'
                    '    "bagging_fraction": 0.8,\n'
                    '    "bagging_freq": 5,\n'
                    "}\n"
                    "model = lgb.train(params, train_data, num_boost_round=2000, fobj=rmsle_objective)"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    if n_samples <= 0 or n_samples < 50000:
        hints.append(
            PreprocessingHint(
                id="model_knn_ensemble",
                category=HintCategory.MODEL_SELECTION,
                priority=0.45,
                applicable_columns=[],
                description="Consider KNN as a local-pattern component in an ensemble.",
                code_snippet=(
                    f"from sklearn.neighbors import {knn_class}\n"
                    "from sklearn.preprocessing import StandardScaler\n"
                    "scaler = StandardScaler()\n"
                    "X_scaled = scaler.fit_transform(X)\n"
                    f'knn = {knn_class}(n_neighbors=15, weights="distance", n_jobs=-1)'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    if problem_kind == "regression" and (n_samples <= 0 or n_samples >= 500):
        hints.append(
            PreprocessingHint(
                id="knn_lightgbm_stack",
                category=HintCategory.MODEL_OPTIMIZATION,
                priority=0.9,
                applicable_columns=[],
                description="Stack KNN and LightGBM with a ridge meta-learner.",
                code_snippet=(
                    "from sklearn.ensemble import StackingRegressor\n"
                    "from sklearn.neighbors import KNeighborsRegressor\n"
                    "from sklearn.linear_model import Ridge\n"
                    "import lightgbm as lgb\n"
                    "stacked_model = StackingRegressor(\n"
                    "    estimators=[\n"
                    '        ("lgbm", lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, verbose=-1)),\n'
                    '        ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance")),\n'
                    "    ],\n"
                    "    final_estimator=Ridge(alpha=1.0),\n"
                    "    cv=5,\n"
                    "    n_jobs=-1,\n"
                    ")"
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    hints.append(
        PreprocessingHint(
            id="model_stacking",
            category=HintCategory.MODEL_SELECTION,
            priority=0.6,
            applicable_columns=[],
            description="Combine diverse base models with a stacking meta-learner.",
            code_snippet=(
                f"from sklearn.ensemble import {stacking_class}\n"
                f"from sklearn.linear_model import {meta_class}\n"
                f"stacking = {stacking_class}(\n"
                '    estimators=[("lgbm", lgbm_model), ("rf", rf_model)],\n'
                f"    final_estimator={meta_snippet},\n"
                "    cv=5,\n"
                "    n_jobs=-1,\n"
                ")"
            ),
            success_rate=0.0,
            last_attempted=None,
            last_result=None,
        )
    )

    return hints


def _generate_feature_selection_hints(profile: DatasetProfile) -> list[PreprocessingHint]:
    hints: list[PreprocessingHint] = []
    problem_kind = _infer_problem_kind(profile)
    model_class = "LGBMRegressor" if problem_kind == "regression" else "LGBMClassifier"
    target_names = set(profile.target_columns)
    numeric_columns = [
        name
        for name, col in profile.columns.items()
        if name not in target_names and col.column_type in {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE}
    ]
    if not numeric_columns:
        return hints

    hints.append(
        PreprocessingHint(
            id="feature_importance_selection",
            category=HintCategory.FEATURE_SELECTION,
            priority=0.6,
            applicable_columns=numeric_columns,
            description="Select top features using model-based importances.",
            code_snippet=(
                f"from lightgbm import {model_class}\n"
                "def select_by_importance(X, y, top_k=50):\n"
                f"    model = {model_class}(n_estimators=200, random_state=42)\n"
                "    model.fit(X, y)\n"
                "    importances = model.feature_importances_\n"
                "    top_indices = np.argsort(importances)[-top_k:]\n"
                "    return X.columns[top_indices].tolist()"
            ),
            success_rate=0.0,
            last_attempted=None,
            last_result=None,
        )
    )
    hints.append(
        PreprocessingHint(
            id="remove_collinear",
            category=HintCategory.FEATURE_SELECTION,
            priority=0.4,
            applicable_columns=numeric_columns,
            description="Drop highly correlated numeric features to reduce redundancy.",
            code_snippet=(
                "def remove_collinear(df, threshold=0.95):\n"
                "    corr_matrix = df.corr().abs()\n"
                "    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))\n"
                "    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]\n"
                "    return df.drop(columns=to_drop)"
            ),
            success_rate=0.0,
            last_attempted=None,
            last_result=None,
        )
    )
    return hints


def _generate_data_enrichment_hints(
    profile: DatasetProfile, feature_profiles: dict[str, ColumnProfile]
) -> list[PreprocessingHint]:
    hints: list[PreprocessingHint] = []
    geo_columns = _geo_categorical_columns(feature_profiles)
    datetime_columns = [name for name, col in feature_profiles.items() if col.column_type == ColumnType.DATETIME]
    price_columns = [name for name, col in feature_profiles.items() if col.column_type == ColumnType.PRICE_MONETARY]

    if geo_columns or profile.has_geographic_features:
        hints.append(
            PreprocessingHint(
                id="data_enrichment_geo",
                category=HintCategory.DATA_ENRICHMENT,
                priority=0.35,
                applicable_columns=geo_columns,
                description=(
                    "If external data is allowed, enrich with geographic context "
                    "(census, boundaries, points of interest)."
                ),
                code_snippet=(
                    "def merge_external_data(df, external_df, key_col):\n"
                    '    return df.merge(external_df, on=key_col, how="left")'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
    if datetime_columns:
        hints.append(
            PreprocessingHint(
                id="data_enrichment_temporal",
                category=HintCategory.DATA_ENRICHMENT,
                priority=0.35,
                applicable_columns=datetime_columns,
                description=(
                    "If external data is allowed, add temporal context such as "
                    "economic indicators or holiday calendars."
                ),
                code_snippet=(
                    'external = pd.read_csv("external_time_series.csv")\n'
                    'df = df.merge(external, on=date_col, how="left")'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )
    if price_columns:
        hints.append(
            PreprocessingHint(
                id="data_enrichment_price",
                category=HintCategory.DATA_ENRICHMENT,
                priority=0.35,
                applicable_columns=price_columns,
                description="If external data is allowed, add inflation or market index features.",
                code_snippet=(
                    'indices = pd.read_csv("inflation_index.csv")\ndf = df.merge(indices, on=date_col, how="left")'
                ),
                success_rate=0.0,
                last_attempted=None,
                last_result=None,
            )
        )

    return hints


def _missing_hint_description(profile: DatasetProfile, columns: list[str]) -> str:
    column_text = _format_columns_with_uniques(profile, columns)
    match profile.missing_pattern:
        case MissingPattern.MCAR:
            return f"Impute missing values with mean/median (numeric) or most-frequent (categorical) for {column_text}."
        case MissingPattern.MAR:
            return f"Use model-based imputation (IterativeImputer) for {column_text}."
        case MissingPattern.MNAR:
            return f"Add missingness indicators and impute values for {column_text} to capture MNAR signals."
    return f"Handle missing values for {column_text}."


def _missing_hint_snippet(profile: DatasetProfile) -> str:
    match profile.missing_pattern:
        case MissingPattern.MCAR:
            return (
                "from sklearn.impute import SimpleImputer\n"
                'imputer = SimpleImputer(strategy="median")\n'
                "df[num_cols] = imputer.fit_transform(df[num_cols])"
            )
        case MissingPattern.MAR:
            return (
                "from sklearn.experimental import enable_iterative_imputer\n"
                "from sklearn.impute import IterativeImputer\n"
                "imputer = IterativeImputer(random_state=42)\n"
                "df[num_cols] = imputer.fit_transform(df[num_cols])"
            )
        case MissingPattern.MNAR:
            return (
                "for col in cols:\n"
                '    df[f"{col}_missing"] = df[col].isna().astype(int)\n'
                "df[cols] = df[cols].fillna(df[cols].median())"
            )
    return "df[cols] = df[cols].fillna(df[cols].median())"


def _format_columns_with_uniques(profile: DatasetProfile, columns: list[str]) -> str:
    parts = []
    for name in columns:
        col = profile.columns.get(name)
        if col is None:
            parts.append(name)
        else:
            parts.append(f"'{name}' ({col.unique_count} unique)")
    return ", ".join(parts)


def _extract_signatures(snippet: str) -> list[str]:
    signatures: list[str] = []
    for match in re.finditer(r"def\s+(\w+)\s*\(", snippet):
        signatures.append(match.group(1))
    for match in re.finditer(r"(\w+(?:Regressor|Classifier|Encoder|Scaler|Transformer))\s*\(", snippet):
        signatures.append(match.group(1))
    for match in re.finditer(r"\.(\w+_encode|\w+_transform)\s*\(", snippet):
        signatures.append(match.group(1))
    seen: set[str] = set()
    ordered: list[str] = []
    for signature in signatures:
        if signature in seen:
            continue
        seen.add(signature)
        ordered.append(signature)
    return ordered


def _hint_patterns(hint: PreprocessingHint) -> list[re.Pattern[str]]:
    mapping: dict[str, list[str]] = {
        "onehot_low_cardinality": [r"OneHotEncoder", r"get_dummies"],
        "target_encode_high_cardinality": [r"TargetEncoder", r"CatBoostEncoder"],
        "frequency_encode_high_cardinality": [r"value_counts", r"FrequencyEncoder"],
        "ordinal_encode": [r"OrdinalEncoder"],
        "price_log_transform": [r"log1p", r"np\.log"],
        "numeric_skew_transform": [r"PowerTransformer", r"QuantileTransformer", r"log1p"],
        "missing_impute_mcar": [r"SimpleImputer", r"fillna"],
        "missing_impute_mar": [r"IterativeImputer", r"fillna"],
        "missing_impute_mnar": [r"_missing", r"fillna"],
        "geo_haversine": [r"haversine", r"geohash"],
        "geospatial_target_encoding": [r"geo_target_encode", r"TargetEncoder"],
        "geo_knn_features": [r"knn_geo_features", r"BallTree"],
        "hierarchical_geo_encoding": [r"hierarchical_geo_agg"],
        "knn_feature_generation": [r"NearestNeighbors", r"knn_mean_dist", r"knn_min_dist"],
        "datetime_cyclical": [r"dt\.", r"sin", r"cos"],
        "text_tfidf": [r"TfidfVectorizer"],
        "text_embeddings": [r"SentenceTransformer", r"embeddings"],
        "timeseries_cv": [r"TimeSeriesSplit", r"PurgedKFold"],
        "timeseries_stationarity": [r"adfuller"],
        "timeseries_seasonality": [r"seasonal_decompose", r"fourier"],
        "timeseries_lag_features": [r"shift\\(", r"rolling"],
        "vision_augmentation": [r"RandomRotation", r"RandomHorizontalFlip"],
        "vision_normalization": [r"Normalize"],
        "vision_transfer_learning": [r"pretrained", r"resnet", r"weights="],
        "nlp_tokenization": [r"tokenizer", r"AutoTokenizer"],
        "nlp_multilingual": [r"langdetect", r"multilingual"],
        "model_gradient_boosting": [r"LGBMRegressor", r"LGBMClassifier", r"lightgbm"],
        "lightgbm_custom_rmsle": [r"rmsle_objective", r"objective\\s*=\\s*rmsle"],
        "model_knn_ensemble": [r"KNeighborsRegressor", r"KNeighborsClassifier"],
        "model_stacking": [r"StackingRegressor", r"StackingClassifier"],
        "knn_lightgbm_stack": [r"StackingRegressor", r"KNeighborsRegressor", r"LGBMRegressor"],
        "feature_importance_selection": [r"feature_importances_", r"select_by_importance"],
        "remove_collinear": [r"remove_collinear", r"np\\.triu"],
        "data_enrichment_geo": [r"merge_external_data"],
        "data_enrichment_temporal": [r"external_time_series"],
        "data_enrichment_price": [r"inflation_index"],
        "target_log_transform": [r"log1p", r"expm1"],
    }
    patterns = mapping.get(hint.id)
    if not patterns:
        return []
    return [re.compile(pattern) for pattern in patterns]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
