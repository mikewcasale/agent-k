"""Dataset discovery and immutable dataset contracts for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import hashlib
import math
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = (
    "DataAsset",
    "DataModality",
    "DatasetManifest",
    "ProblemType",
    "TaskKind",
    "discover_dataset",
)

_TABLE_SUFFIXES: Final[frozenset[str]] = frozenset({".csv", ".parquet"})
_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_TRAIN_NAMES: Final[tuple[str, ...]] = ("train", "training", "development", "dev")
_PREDICT_NAMES: Final[tuple[str, ...]] = ("test", "testing", "predict", "inference")
_SAMPLE_NAMES: Final[tuple[str, ...]] = ("sample_submission", "submission", "sample")
_ID_NAMES: Final[frozenset[str]] = frozenset(
    {"id", "index", "row_id", "image_id", "text_id", "record_id", "sample_id"}
)
_TEXT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "body",
        "comment",
        "content",
        "description",
        "document",
        "message",
        "review",
        "sentence",
        "text",
        "title",
    }
)


class DataModality(StrEnum):
    """Supported input modalities."""

    TABULAR = "tabular"
    VISION = "vision"
    TEXT = "text"


class TaskKind(StrEnum):
    """Supported supervised-learning task kinds."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ProblemType(StrEnum):
    """Complete supported problem taxonomy."""

    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    VISION_CLASSIFICATION = "vision_classification"
    VISION_REGRESSION = "vision_regression"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_REGRESSION = "text_regression"

    @property
    def modality(self) -> DataModality:
        """Return the input modality encoded by the problem type."""
        return DataModality(self.value.split("_", maxsplit=1)[0])

    @property
    def task(self) -> TaskKind:
        """Return the task kind encoded by the problem type."""
        return TaskKind(self.value.rsplit("_", maxsplit=1)[1])

    @classmethod
    def from_parts(cls, modality: DataModality, task: TaskKind) -> ProblemType:
        """Build a problem type from its orthogonal dimensions."""
        return cls(f"{modality.value}_{task.value}")


class DataAsset(BaseModel):
    """A discovered immutable input asset."""

    model_config = ConfigDict(frozen=True, strict=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    columns: tuple[str, ...] = ()


class DatasetManifest(BaseModel):
    """Authoritative description inferred from competition data."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: str = "1.0"
    problem_type: ProblemType
    train: DataAsset
    predict: DataAsset
    sample_submission: DataAsset | None = None
    id_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    image_column: str | None = None
    train_image_root: str | None = None
    predict_image_root: str | None = None
    text_column: str | None = None
    class_labels: tuple[str, ...] = ()
    metric_id: str | None = None
    discovery_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> DatasetManifest:
        """Reject ambiguous or internally inconsistent manifests."""
        if not self.id_columns:
            raise ValueError("at least one ID column is required")
        if not self.target_columns:
            raise ValueError("at least one target column is required")
        if not self.feature_columns:
            raise ValueError("at least one feature column is required")
        roles = [set(self.id_columns), set(self.target_columns), set(self.feature_columns)]
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise ValueError("ID, target, and feature columns must be disjoint")
        if self.problem_type.modality is DataModality.VISION and self.image_column is None:
            raise ValueError("vision manifests require image_column")
        image_roots = (self.train_image_root, self.predict_image_root)
        if any(root is not None and Path(root).is_absolute() for root in image_roots):
            raise ValueError("image roots must be relative to the dataset root")
        if self.problem_type.modality is DataModality.TEXT and self.text_column is None:
            raise ValueError("text manifests require text_column")
        if self.problem_type.task is TaskKind.CLASSIFICATION and not self.class_labels:
            raise ValueError("classification manifests require class_labels")
        return self


def discover_dataset(root: Path, *, metric_id: str | None = None) -> DatasetManifest:
    """Discover train, prediction, target, ID, modality, and task metadata.

    Args:
        root: Directory containing downloaded competition data.
        metric_id: Optional trusted competition metric identifier.

    Returns:
        A validated manifest with paths relative to ``root``.

    Raises:
        FileNotFoundError: If the root or required data files do not exist.
        ValueError: If the data roles cannot be inferred without ambiguity.
    """
    dataset_root = root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    table_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _TABLE_SUFFIXES
    )
    if not table_paths:
        return _discover_image_directories(dataset_root, metric_id=metric_id)

    frames = {path: _read_table(path) for path in table_paths}
    train_path, predict_path, sample_path = _select_table_roles(frames)
    train = frames[train_path]
    predict = frames[predict_path]
    sample = frames[sample_path] if sample_path is not None else None

    target_columns = _infer_targets(train, predict, sample)
    id_columns = _infer_ids(train, predict, sample, target_columns)
    feature_columns = tuple(
        column for column in predict.columns if column not in {*id_columns, *target_columns}
    )
    if not feature_columns:
        raise ValueError("no feature columns remain after identifying IDs and targets")

    image_column, train_image_root, predict_image_root = _infer_image_layout(
        dataset_root,
        train,
        predict,
        feature_columns,
    )
    text_column = None if image_column is not None else _infer_text_column(train, feature_columns)
    modality = (
        DataModality.VISION
        if image_column is not None
        else DataModality.TEXT
        if text_column is not None
        else DataModality.TABULAR
    )
    task = _infer_task(train.loc[:, list(target_columns)], metric_id)
    labels = (
        _class_labels(train.loc[:, list(target_columns)]) if task is TaskKind.CLASSIFICATION else ()
    )
    problem_type = ProblemType.from_parts(modality, task)
    notes = (
        f"inferred train table from {train_path.name}",
        f"inferred prediction table from {predict_path.name}",
        f"inferred {problem_type.value} from data values",
    )
    return DatasetManifest(
        problem_type=problem_type,
        train=_asset(dataset_root, train_path, train),
        predict=_asset(dataset_root, predict_path, predict),
        sample_submission=(
            _asset(dataset_root, sample_path, sample)
            if sample_path is not None and sample is not None
            else None
        ),
        id_columns=id_columns,
        target_columns=target_columns,
        feature_columns=feature_columns,
        image_column=image_column,
        train_image_root=train_image_root,
        predict_image_root=predict_image_root,
        text_column=text_column,
        class_labels=labels,
        metric_id=metric_id,
        discovery_notes=notes,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def _select_table_roles(
    frames: dict[Path, pd.DataFrame],
) -> tuple[Path, Path, Path | None]:
    sample_path = _best_named_path(frames, _SAMPLE_NAMES)
    train_path = _best_named_path(frames, _TRAIN_NAMES)
    predict_path = _best_named_path(frames, _PREDICT_NAMES)
    if train_path is not None and predict_path is not None and train_path != predict_path:
        return train_path, predict_path, sample_path

    eligible = [path for path in frames if path != sample_path]
    pairs: list[tuple[int, int, Path, Path]] = []
    for left in eligible:
        for right in eligible:
            if left == right:
                continue
            left_columns = set(frames[left].columns)
            right_columns = set(frames[right].columns)
            extra = len(left_columns - right_columns)
            shared = len(left_columns & right_columns)
            if extra > 0 and shared > 0:
                pairs.append((extra, shared, left, right))
    if not pairs:
        raise ValueError("could not distinguish training and prediction tables")
    _, _, train_path, predict_path = max(
        pairs,
        key=lambda item: (item[1], -item[0], len(frames[item[2]])),
    )
    return train_path, predict_path, sample_path


def _best_named_path(frames: dict[Path, pd.DataFrame], names: tuple[str, ...]) -> Path | None:
    ranked: list[tuple[int, int, Path]] = []
    for path in frames:
        stem = path.stem.lower().replace("-", "_")
        for position, name in enumerate(names):
            if stem == name or stem.startswith(f"{name}_") or stem.endswith(f"_{name}"):
                ranked.append((position, len(path.parts), path))
                break
    return min(ranked)[2] if ranked else None


def _infer_targets(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    sample: pd.DataFrame | None,
) -> tuple[str, ...]:
    train_only = [column for column in train.columns if column not in predict.columns]
    if sample is not None:
        sample_targets = [
            column
            for column in sample.columns
            if column in train.columns and column not in predict.columns
        ]
        if sample_targets:
            return tuple(sample_targets)
    named = [
        column
        for column in train_only
        if column.lower() in {"target", "label", "labels", "class", "score", "y"}
    ]
    if named:
        return tuple(named)
    if len(train_only) == 1:
        return (train_only[0],)
    if not train_only:
        raise ValueError("training data has no target column absent from prediction data")
    raise ValueError(f"ambiguous target columns: {train_only}")


def _infer_ids(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    sample: pd.DataFrame | None,
    targets: tuple[str, ...],
) -> tuple[str, ...]:
    common = [
        column for column in predict.columns if column in train.columns and column not in targets
    ]
    if sample is not None:
        sample_ids = [
            column for column in common if column in sample.columns and column not in targets
        ]
        named_sample_ids = [column for column in sample_ids if _looks_like_id(column)]
        if named_sample_ids:
            return tuple(named_sample_ids)
        unique_sample_ids = [column for column in sample_ids if sample[column].is_unique]
        if len(unique_sample_ids) == 1:
            return (unique_sample_ids[0],)
    named = [column for column in common if _looks_like_id(column)]
    if named:
        return tuple(named)
    unique = [column for column in common if train[column].is_unique and predict[column].is_unique]
    if len(unique) == 1:
        return (unique[0],)
    raise ValueError("could not infer a stable ID column")


def _looks_like_id(column: str) -> bool:
    normalized = column.lower().replace("-", "_").replace(" ", "_")
    return normalized in _ID_NAMES or normalized.endswith("_id")


def _infer_image_layout(
    root: Path,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[str | None, str | None, str | None]:
    for column in features:
        train_root = _infer_image_root(root, train[column])
        predict_root = _infer_image_root(root, predict[column])
        if train_root is not None and predict_root is not None:
            return column, train_root, predict_root
    return None, None, None


def _infer_image_root(root: Path, series: pd.Series) -> str | None:
    values = series.dropna().astype(str).head(32)
    if values.empty or not all(value.lower().endswith(tuple(_IMAGE_SUFFIXES)) for value in values):
        return None
    candidates = [Path("")]
    image_directories = {
        path.parent.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }
    candidates.extend(sorted(image_directories))
    matches = [
        directory
        for directory in candidates
        if sum((root / directory / value).is_file() for value in values) / len(values) >= 0.8
    ]
    if len(matches) != 1:
        return None
    return matches[0].as_posix() if matches[0] != Path("") else ""


def _infer_text_column(train: pd.DataFrame, features: tuple[str, ...]) -> str | None:
    candidates: list[tuple[float, float, str]] = []
    for column in features:
        series = train[column].dropna()
        if series.empty or not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            continue
        text = series.astype(str)
        median_length = float(text.str.len().median())
        median_words = float(text.str.split().str.len().median())
        unique_ratio = float(text.nunique(dropna=True) / len(text))
        normalized = column.lower().replace("-", "_").replace(" ", "_")
        named_text = normalized in _TEXT_NAMES or any(
            normalized.endswith(f"_{suffix}") for suffix in _TEXT_NAMES
        )
        if median_length >= 20.0 and unique_ratio >= 0.2 and (named_text or median_words >= 5.0):
            candidates.append((median_length, unique_ratio, column))
    return max(candidates)[2] if candidates else None


def _infer_task(targets: pd.DataFrame, metric_id: str | None) -> TaskKind:
    normalized_metric = (metric_id or "").lower().replace("_", "").replace("-", "")
    if normalized_metric in {"accuracy", "auc", "f1", "logloss", "rocauc"}:
        return TaskKind.CLASSIFICATION
    if normalized_metric in {"mae", "rmse", "rmsle", "mse"}:
        return TaskKind.REGRESSION
    if targets.shape[1] != 1:
        numeric = all(pd.api.types.is_numeric_dtype(targets[column]) for column in targets)
        return TaskKind.REGRESSION if numeric else TaskKind.CLASSIFICATION
    target = targets.iloc[:, 0].dropna()
    if not pd.api.types.is_numeric_dtype(target):
        return TaskKind.CLASSIFICATION
    unique_count = int(target.nunique(dropna=True))
    classification_limit = max(20, math.ceil(math.sqrt(max(len(target), 1))))
    if unique_count <= classification_limit and unique_count / max(len(target), 1) <= 0.2:
        return TaskKind.CLASSIFICATION
    return TaskKind.REGRESSION


def _class_labels(targets: pd.DataFrame) -> tuple[str, ...]:
    values = pd.unique(targets.to_numpy().ravel())
    labels = sorted(str(value) for value in values if not pd.isna(value))
    return tuple(labels)


def _asset(root: Path, path: Path, frame: pd.DataFrame) -> DataAsset:
    return DataAsset(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        row_count=len(frame),
        columns=tuple(str(column) for column in frame.columns),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_image_directories(root: Path, *, metric_id: str | None) -> DatasetManifest:
    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError("no CSV, Parquet, or image data found")
    raise ValueError(
        "image directory datasets require a label table; class-folder labels are not accepted "
        "because no unlabeled prediction-row ID contract can be proven"
    )
