"""Trusted split preparation and host-side candidate evaluation for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import hashlib
import shutil
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from sklearn.model_selection import train_test_split

from .data import DatasetBundle, load_dataset, read_table
from .datasets import DataAsset, DataModality, DatasetManifest, ProblemType, TaskKind
from .metrics import (
    MetricDirection,
    MetricId,
    PredictionKind,
    metric_spec,
    prediction_columns,
    score_predictions,
    validate_prediction_values,
)

__all__ = (
    "CandidateEngine",
    "CandidateResult",
    "CandidateSpec",
    "CandidateStatus",
    "EvaluationRequest",
    "EvaluationResult",
    "PreparedSplit",
    "PreparedSplitSet",
    "SplitRole",
    "SubmissionArtifact",
    "build_candidate_spec",
    "build_submission_spec",
    "evaluate_candidate",
    "load_candidate_dataset",
    "materialize_trusted_splits",
    "write_submission",
    "write_candidate_predictions",
)


class CandidateEngine(StrEnum):
    """Supported concrete candidate implementations."""

    LIGHTGBM = "lightgbm"
    KUMO = "kumo"
    RESNET18 = "resnet18"
    TFIDF_LINEAR = "tfidf_linear"
    TRANSFORMER = "transformer"


class CandidateStatus(StrEnum):
    """Terminal candidate execution states."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SplitRole(StrEnum):
    """Trusted holdout purposes."""

    VALIDATION = "validation"
    ACCEPTANCE = "acceptance"


class CandidateSpec(BaseModel):
    """Complete sandbox input contract, intentionally excluding hidden labels."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: str = "1.0"
    candidate_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$")
    problem_type: ProblemType
    engine: CandidateEngine
    metric_id: MetricId
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    train_path: str
    predict_path: str
    train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predict_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_row_count: int = Field(gt=0)
    predict_row_count: int = Field(gt=0)
    id_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    class_labels: tuple[str, ...] = ()
    image_column: str | None = None
    train_image_root: str | None = None
    predict_image_root: str | None = None
    text_column: str | None = None
    hyperparameters: dict[str, JsonValue] = Field(default_factory=dict)
    offline_assets: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_engine_and_problem(self) -> CandidateSpec:
        """Reject engine, modality, and task combinations that cannot execute."""
        engines = {
            DataModality.TABULAR: {CandidateEngine.LIGHTGBM, CandidateEngine.KUMO},
            DataModality.VISION: {CandidateEngine.RESNET18},
            DataModality.TEXT: {
                CandidateEngine.TFIDF_LINEAR,
                CandidateEngine.TRANSFORMER,
            },
        }
        if self.engine not in engines[self.problem_type.modality]:
            raise ValueError(
                f"engine {self.engine.value} does not support {self.problem_type.value}"
            )
        if self.problem_type.task is TaskKind.CLASSIFICATION and not self.class_labels:
            raise ValueError("classification candidates require class_labels")
        if self.engine is CandidateEngine.KUMO and len(self.target_columns) != 1:
            raise ValueError("Kumo supports exactly one target column")
        if Path(self.train_path).is_absolute() or Path(self.predict_path).is_absolute():
            raise ValueError("candidate data paths must be relative to the input root")
        image_roots = (self.train_image_root, self.predict_image_root)
        if any(root is not None and Path(root).is_absolute() for root in image_roots):
            raise ValueError("candidate image roots must be relative to the input root")
        if any(Path(path).is_absolute() for path in self.offline_assets.values()):
            raise ValueError("offline asset paths must be relative to the input root")
        return self


class CandidateResult(BaseModel):
    """Sandbox result metadata; prediction content remains a separate artifact."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: str = "1.0"
    candidate_id: str
    status: CandidateStatus
    predictions_path: str | None = None
    predictions_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    row_count: int | None = Field(default=None, ge=0)
    prediction_columns: tuple[str, ...] = ()
    duration_seconds: float = Field(ge=0.0)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> CandidateResult:
        """Require complete artifacts on success and an explanation on failure."""
        artifacts = (
            self.predictions_path,
            self.predictions_sha256,
            self.row_count,
        )
        if self.status is CandidateStatus.SUCCEEDED and any(value is None for value in artifacts):
            raise ValueError("successful candidates require a complete prediction artifact")
        if self.status is CandidateStatus.FAILED and not self.error:
            raise ValueError("failed candidates require an error message")
        return self


class EvaluationRequest(BaseModel):
    """Host-only trusted evaluation request."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: str = "1.0"
    candidate_result: CandidateResult
    metric_id: MetricId
    prediction_root: str
    trusted_labels_path: str
    id_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    class_labels: tuple[str, ...] = ()


class EvaluationResult(BaseModel):
    """Validated trusted score and its immutable evidence."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: str = "1.0"
    candidate_id: str
    metric_id: MetricId
    direction: MetricDirection
    score: float
    row_count: int = Field(gt=0)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validated_columns: tuple[str, ...]
    duration_seconds: float = Field(ge=0.0)


class PreparedSplit(BaseModel):
    """Materialized candidate input paired with a host-only label location."""

    model_config = ConfigDict(frozen=True, strict=True)

    role: SplitRole
    candidate_input_root: str
    trusted_labels_path: str
    manifest: DatasetManifest


class PreparedSplitSet(BaseModel):
    """Disjoint validation and acceptance split contracts."""

    model_config = ConfigDict(frozen=True, strict=True)

    validation: PreparedSplit
    acceptance: PreparedSplit


class SubmissionArtifact(BaseModel):
    """Immutable evidence for a sample-template-compatible submission."""

    model_config = ConfigDict(frozen=True, strict=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    columns: tuple[str, ...]


def build_candidate_spec(
    split: PreparedSplit,
    *,
    candidate_id: str,
    engine: CandidateEngine,
    metric_id: MetricId,
    seed: int = 42,
    hyperparameters: dict[str, JsonValue] | None = None,
    offline_assets: dict[str, str] | None = None,
) -> CandidateSpec:
    """Build a sandbox candidate contract from one prepared split."""
    manifest = split.manifest
    return CandidateSpec(
        candidate_id=candidate_id,
        problem_type=manifest.problem_type,
        engine=engine,
        metric_id=metric_id,
        seed=seed,
        train_path=manifest.train.path,
        predict_path=manifest.predict.path,
        train_sha256=manifest.train.sha256,
        predict_sha256=manifest.predict.sha256,
        train_row_count=manifest.train.row_count or 0,
        predict_row_count=manifest.predict.row_count or 0,
        id_columns=manifest.id_columns,
        target_columns=manifest.target_columns,
        feature_columns=manifest.feature_columns,
        class_labels=manifest.class_labels,
        image_column=manifest.image_column,
        train_image_root=manifest.train_image_root,
        predict_image_root=manifest.predict_image_root,
        text_column=manifest.text_column,
        hyperparameters=hyperparameters or {},
        offline_assets=offline_assets or {},
    )


def build_submission_spec(
    manifest: DatasetManifest,
    *,
    candidate_id: str,
    engine: CandidateEngine,
    metric_id: MetricId,
    seed: int = 42,
    hyperparameters: dict[str, JsonValue] | None = None,
    offline_assets: dict[str, str] | None = None,
) -> CandidateSpec:
    """Build a final full-training/unlabeled-test candidate contract."""
    if manifest.train.row_count is None or manifest.predict.row_count is None:
        raise ValueError("submission assets require known row counts")
    return CandidateSpec(
        candidate_id=candidate_id,
        problem_type=manifest.problem_type,
        engine=engine,
        metric_id=metric_id,
        seed=seed,
        train_path=manifest.train.path,
        predict_path=manifest.predict.path,
        train_sha256=manifest.train.sha256,
        predict_sha256=manifest.predict.sha256,
        train_row_count=manifest.train.row_count,
        predict_row_count=manifest.predict.row_count,
        id_columns=manifest.id_columns,
        target_columns=manifest.target_columns,
        feature_columns=manifest.feature_columns,
        class_labels=manifest.class_labels,
        image_column=manifest.image_column,
        train_image_root=manifest.train_image_root,
        predict_image_root=manifest.predict_image_root,
        text_column=manifest.text_column,
        hyperparameters=hyperparameters or {},
        offline_assets=offline_assets or {},
    )


def load_candidate_dataset(spec: CandidateSpec, input_root: Path) -> DatasetBundle:
    """Load candidate-visible data and verify it against the sandbox contract."""
    train_path = _safe_artifact_path(input_root, spec.train_path)
    predict_path = _safe_artifact_path(input_root, spec.predict_path)
    manifest = DatasetManifest(
        problem_type=spec.problem_type,
        train=DataAsset(
            path=spec.train_path,
            sha256=spec.train_sha256,
            size_bytes=train_path.stat().st_size,
            row_count=spec.train_row_count,
            columns=(*spec.id_columns, *spec.feature_columns, *spec.target_columns),
        ),
        predict=DataAsset(
            path=spec.predict_path,
            sha256=spec.predict_sha256,
            size_bytes=predict_path.stat().st_size,
            row_count=spec.predict_row_count,
            columns=(*spec.id_columns, *spec.feature_columns),
        ),
        id_columns=spec.id_columns,
        target_columns=spec.target_columns,
        feature_columns=spec.feature_columns,
        image_column=spec.image_column,
        train_image_root=spec.train_image_root,
        predict_image_root=spec.predict_image_root,
        text_column=spec.text_column,
        class_labels=spec.class_labels,
        metric_id=spec.metric_id.value,
    )
    bundle = load_dataset(manifest, input_root)
    if len(bundle.train) != spec.train_row_count or len(bundle.predict) != spec.predict_row_count:
        raise ValueError("candidate data row counts do not match the signed specification")
    return bundle


def materialize_trusted_splits(
    manifest: DatasetManifest,
    dataset_root: Path,
    candidate_root: Path,
    trusted_root: Path,
    *,
    validation_fraction: float = 0.2,
    acceptance_fraction: float = 0.2,
    seed: int = 42,
) -> PreparedSplitSet:
    """Create disjoint tuning and acceptance inputs with host-only labels.

    The validation candidate trains on the fit partition. The acceptance
    candidate trains on all non-acceptance rows, including validation rows,
    but neither candidate input contains its own holdout labels.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if not 0.0 < acceptance_fraction < 1.0:
        raise ValueError("acceptance_fraction must be between zero and one")
    if validation_fraction + acceptance_fraction >= 1.0:
        raise ValueError("validation and acceptance fractions must sum to less than one")
    _validate_separate_roots(candidate_root, trusted_root)
    source = read_table(_safe_artifact_path(dataset_root, manifest.train.path))
    row_indices = np.arange(len(source))
    stratify = _stratification_values(source, manifest)
    development, acceptance = train_test_split(
        row_indices,
        test_size=acceptance_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    validation_share = validation_fraction / (1.0 - acceptance_fraction)
    development_stratify = stratify[development] if stratify is not None else None
    fit, validation = train_test_split(
        development,
        test_size=validation_share,
        random_state=seed + 1,
        shuffle=True,
        stratify=development_stratify,
    )
    candidate_root.mkdir(parents=True, exist_ok=True)
    trusted_root.mkdir(parents=True, exist_ok=True)
    validation_split = _materialize_split(
        SplitRole.VALIDATION,
        source.iloc[fit],
        source.iloc[validation],
        manifest,
        dataset_root,
        candidate_root,
        trusted_root,
    )
    acceptance_split = _materialize_split(
        SplitRole.ACCEPTANCE,
        source.iloc[development],
        source.iloc[acceptance],
        manifest,
        dataset_root,
        candidate_root,
        trusted_root,
    )
    return PreparedSplitSet(validation=validation_split, acceptance=acceptance_split)


def evaluate_candidate(request: EvaluationRequest) -> EvaluationResult:
    """Validate and score a candidate artifact against inaccessible labels."""
    started = time.monotonic()
    result = request.candidate_result
    if result.status is not CandidateStatus.SUCCEEDED:
        raise ValueError(f"candidate did not succeed: {result.error}")
    assert result.predictions_path is not None
    assert result.predictions_sha256 is not None
    predictions_path = _safe_artifact_path(
        Path(request.prediction_root),
        result.predictions_path,
    )
    actual_digest = _sha256(predictions_path)
    if actual_digest != result.predictions_sha256:
        raise ValueError("prediction artifact digest does not match candidate result")

    expected_predictions = prediction_columns(
        request.metric_id,
        request.target_columns,
        request.class_labels,
    )
    expected_columns = (*request.id_columns, *expected_predictions)
    prediction_frame = _read_evaluation_csv(
        predictions_path,
        request.id_columns,
        expected_predictions,
        request.metric_id,
    )
    if result.row_count != len(prediction_frame):
        raise ValueError("prediction row count differs from candidate result metadata")
    if result.prediction_columns != expected_predictions:
        raise ValueError("prediction columns differ from candidate result metadata")
    trusted_frame = pd.read_csv(
        request.trusted_labels_path,
        dtype={column: str for column in request.id_columns},
    )
    trusted_columns = (*request.id_columns, *request.target_columns)
    _require_exact_columns(prediction_frame, expected_columns, "prediction")
    _require_exact_columns(trusted_frame, trusted_columns, "trusted label")
    if prediction_frame.loc[:, list(expected_predictions)].isna().any(axis=None):
        raise ValueError("prediction values contain missing values")
    if trusted_frame.loc[:, list(request.target_columns)].isna().any(axis=None):
        raise ValueError("trusted labels contain missing values")
    if request.class_labels:
        trusted_classes = set(trusted_frame.loc[:, request.target_columns[0]].astype(str))
        unknown_classes = trusted_classes - set(request.class_labels)
        if unknown_classes:
            raise ValueError(
                f"trusted labels are outside declared classes: {sorted(unknown_classes)}"
            )
    if metric_spec(request.metric_id).prediction_kind is PredictionKind.LABEL:
        observed = set(prediction_frame.loc[:, expected_predictions[0]].astype(str))
        unknown = observed - set(request.class_labels)
        if unknown:
            raise ValueError(f"prediction labels are outside declared classes: {sorted(unknown)}")
    _validate_evaluation_ids(prediction_frame, trusted_frame, request.id_columns)
    aligned = prediction_frame.set_index(list(request.id_columns)).loc[
        trusted_frame.set_index(list(request.id_columns)).index
    ]
    score = score_predictions(
        request.metric_id,
        trusted_frame.loc[:, list(request.target_columns)].to_numpy(),
        aligned.loc[:, list(expected_predictions)].to_numpy(),
        class_labels=request.class_labels,
    )
    if not np.isfinite(score):
        raise ValueError("trusted metric produced a non-finite score")
    return EvaluationResult(
        candidate_id=result.candidate_id,
        metric_id=request.metric_id,
        direction=metric_spec(request.metric_id).direction,
        score=score,
        row_count=len(trusted_frame),
        predictions_sha256=actual_digest,
        validated_columns=expected_columns,
        duration_seconds=time.monotonic() - started,
    )


def write_candidate_predictions(
    spec: CandidateSpec,
    predict_ids: pd.DataFrame,
    predictions: pd.DataFrame,
    output_root: Path,
    *,
    started: float,
    diagnostics: dict[str, JsonValue] | None = None,
) -> CandidateResult:
    """Validate and atomically describe a candidate prediction artifact."""
    expected_values = prediction_columns(
        spec.metric_id,
        spec.target_columns,
        spec.class_labels,
    )
    _require_exact_columns(predict_ids, spec.id_columns, "candidate ID")
    _require_exact_columns(predictions, expected_values, "candidate prediction")
    if len(predict_ids) != spec.predict_row_count or len(predictions) != len(predict_ids):
        raise ValueError("candidate prediction row count does not match the specification")
    if predict_ids.isna().any(axis=None) or predict_ids.duplicated().any():
        raise ValueError("candidate prediction IDs must be non-null and unique")
    values = predictions.to_numpy()
    validate_prediction_values(
        values,
        spec.metric_id,
        class_count=len(spec.class_labels) or None,
    )
    if metric_spec(spec.metric_id).prediction_kind is PredictionKind.LABEL:
        observed = set(predictions.iloc[:, 0].astype(str))
        unknown = observed - set(spec.class_labels)
        if unknown:
            raise ValueError(f"prediction labels are outside declared classes: {sorted(unknown)}")
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "predictions.csv"
    if output_path.exists():
        raise FileExistsError(f"prediction artifact already exists: {output_path}")
    artifact = pd.concat(
        (predict_ids.reset_index(drop=True), predictions.reset_index(drop=True)),
        axis=1,
    )
    temporary = output_root / ".predictions.csv.tmp"
    artifact.to_csv(temporary, index=False)
    temporary.replace(output_path)
    digest = _sha256(output_path)
    return CandidateResult(
        candidate_id=spec.candidate_id,
        status=CandidateStatus.SUCCEEDED,
        predictions_path=output_path.relative_to(output_root).as_posix(),
        predictions_sha256=digest,
        row_count=len(artifact),
        prediction_columns=expected_values,
        duration_seconds=time.monotonic() - started,
        diagnostics=diagnostics or {},
    )


def write_submission(
    prediction_path: Path,
    manifest: DatasetManifest,
    dataset_root: Path,
    destination: Path,
    *,
    metric_id: MetricId | None = None,
) -> SubmissionArtifact:
    """Map internal predictions into the exact sample-submission schema.

    IDs are aligned to the sample template rather than trusted by row position.
    Probability columns are mapped only when class semantics can be proven;
    ambiguous same-width schemas are rejected.
    """
    if manifest.sample_submission is None:
        raise ValueError("a sample submission is required for unambiguous final mapping")
    if destination.exists():
        raise FileExistsError(f"submission destination already exists: {destination}")
    resolved_metric = metric_id or (
        MetricId(manifest.metric_id) if manifest.metric_id is not None else None
    )
    if resolved_metric is None:
        raise ValueError("metric_id is required when the dataset manifest does not define one")
    internal_columns = prediction_columns(
        resolved_metric,
        manifest.target_columns,
        manifest.class_labels,
    )
    predictions = _read_evaluation_csv(
        prediction_path,
        manifest.id_columns,
        internal_columns,
        resolved_metric,
    )
    _require_exact_columns(
        predictions,
        (*manifest.id_columns, *internal_columns),
        "final prediction",
    )
    validate_prediction_values(
        predictions.loc[:, list(internal_columns)].to_numpy(),
        resolved_metric,
        class_count=len(manifest.class_labels) or None,
    )
    if metric_spec(resolved_metric).prediction_kind is PredictionKind.LABEL:
        observed = set(predictions.loc[:, internal_columns[0]].astype(str))
        unknown = observed - set(manifest.class_labels)
        if unknown:
            raise ValueError(
                f"final prediction labels are outside declared classes: {sorted(unknown)}"
            )
    sample_path = _safe_artifact_path(dataset_root, manifest.sample_submission.path)
    if _sha256(sample_path) != manifest.sample_submission.sha256:
        raise ValueError("sample submission digest does not match the dataset manifest")
    sample = pd.read_csv(
        sample_path,
        dtype={column: str for column in manifest.id_columns},
    )
    sample_outputs = tuple(
        str(column) for column in sample.columns if column not in manifest.id_columns
    )
    if not sample_outputs:
        raise ValueError("sample submission has no output columns")
    _validate_evaluation_ids(predictions, sample, manifest.id_columns)
    aligned = predictions.set_index(list(manifest.id_columns)).loc[
        sample.set_index(list(manifest.id_columns)).index
    ]
    mapped = _map_submission_outputs(
        aligned,
        internal_columns,
        sample_outputs,
        manifest,
        resolved_metric,
    )
    submission = pd.concat(
        (sample.loc[:, list(manifest.id_columns)].reset_index(drop=True), mapped),
        axis=1,
    )
    _require_exact_columns(submission, tuple(map(str, sample.columns)), "submission")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(destination)
    return SubmissionArtifact(
        path=str(destination.resolve()),
        sha256=_sha256(destination),
        row_count=len(submission),
        columns=tuple(map(str, submission.columns)),
    )


def _materialize_split(
    role: SplitRole,
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    source_manifest: DatasetManifest,
    dataset_root: Path,
    candidate_root: Path,
    trusted_root: Path,
) -> PreparedSplit:
    split_root = candidate_root / role.value
    split_root.mkdir(parents=True, exist_ok=False)
    train = training.reset_index(drop=True).copy()
    predict = holdout.drop(columns=list(source_manifest.target_columns)).reset_index(drop=True)
    if source_manifest.image_column is not None:
        _copy_image_assets(train, predict, source_manifest, dataset_root, split_root)
    train_path = split_root / "train.csv"
    predict_path = split_root / "predict.csv"
    labels_path = trusted_root / f"{role.value}_labels.csv"
    if labels_path.exists():
        raise FileExistsError(f"trusted labels already exist: {labels_path}")
    train.to_csv(train_path, index=False)
    predict.to_csv(predict_path, index=False)
    holdout.loc[:, [*source_manifest.id_columns, *source_manifest.target_columns]].to_csv(
        labels_path,
        index=False,
    )
    split_manifest = DatasetManifest(
        problem_type=source_manifest.problem_type,
        train=_frame_asset(split_root, train_path, train),
        predict=_frame_asset(split_root, predict_path, predict),
        id_columns=source_manifest.id_columns,
        target_columns=source_manifest.target_columns,
        feature_columns=source_manifest.feature_columns,
        image_column=source_manifest.image_column,
        train_image_root=source_manifest.train_image_root,
        predict_image_root=source_manifest.train_image_root,
        text_column=source_manifest.text_column,
        class_labels=source_manifest.class_labels,
        metric_id=source_manifest.metric_id,
        discovery_notes=(f"trusted {role.value} split",),
    )
    return PreparedSplit(
        role=role,
        candidate_input_root=str(split_root),
        trusted_labels_path=str(labels_path.resolve()),
        manifest=split_manifest,
    )


def _copy_image_assets(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    manifest: DatasetManifest,
    dataset_root: Path,
    split_root: Path,
) -> None:
    assert manifest.image_column is not None
    relative_paths = pd.concat(
        (train[manifest.image_column], predict[manifest.image_column]),
        ignore_index=True,
    ).drop_duplicates()
    image_root = manifest.train_image_root or ""
    for relative in relative_paths.astype(str):
        rooted_relative = str(Path(image_root) / relative)
        source = _safe_artifact_path(dataset_root, rooted_relative)
        destination = (split_root / rooted_relative).resolve()
        try:
            destination.relative_to(split_root.resolve())
        except ValueError as exc:
            raise ValueError(f"image path escapes split root: {relative}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


def _read_evaluation_csv(
    path: Path,
    id_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    metric_id: MetricId,
) -> pd.DataFrame:
    del id_columns, value_columns, metric_id
    return pd.read_csv(path, dtype=str)


def _map_submission_outputs(
    aligned: pd.DataFrame,
    internal_columns: tuple[str, ...],
    sample_outputs: tuple[str, ...],
    manifest: DatasetManifest,
    metric_id: MetricId,
) -> pd.DataFrame:
    if sample_outputs == internal_columns:
        return aligned.loc[:, list(internal_columns)].reset_index(drop=True)
    if len(sample_outputs) == len(internal_columns) == 1:
        return (
            aligned.loc[:, [internal_columns[0]]]
            .rename(columns={internal_columns[0]: sample_outputs[0]})
            .reset_index(drop=True)
        )
    kind = metric_spec(metric_id).prediction_kind
    if kind is not PredictionKind.PROBABILITY:
        if sample_outputs != manifest.target_columns:
            raise ValueError("sample output columns do not exactly match non-probability targets")
        return (
            aligned.loc[:, list(internal_columns)]
            .set_axis(sample_outputs, axis=1)
            .reset_index(drop=True)
        )
    if len(manifest.target_columns) != 1 or len(manifest.class_labels) <= 2:
        raise ValueError("probability submission schema is ambiguous")
    target = manifest.target_columns[0]
    mapping: dict[str, str] = {}
    for label, internal in zip(manifest.class_labels, internal_columns, strict=True):
        recognized = {
            label,
            internal,
            f"{target}_{label}",
            f"{target}__{label}",
        }
        matches = [column for column in sample_outputs if column in recognized]
        if len(matches) != 1:
            raise ValueError(f"cannot prove sample column mapping for class {label!r}: {matches}")
        mapping[internal] = matches[0]
    if set(mapping.values()) != set(sample_outputs):
        raise ValueError("sample probability columns contain unmapped outputs")
    return (
        aligned.loc[:, list(internal_columns)]
        .rename(columns=mapping)
        .loc[:, list(sample_outputs)]
        .reset_index(drop=True)
    )


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    name: str,
) -> None:
    actual = tuple(str(column) for column in frame.columns)
    if actual != expected:
        raise ValueError(f"{name} columns differ: expected {expected}, got {actual}")


def _validate_evaluation_ids(
    predictions: pd.DataFrame,
    trusted: pd.DataFrame,
    id_columns: tuple[str, ...],
) -> None:
    for name, frame in (("prediction", predictions), ("trusted label", trusted)):
        ids = frame.loc[:, list(id_columns)]
        if ids.isna().any(axis=None):
            raise ValueError(f"{name} IDs contain missing values")
        if ids.duplicated().any():
            raise ValueError(f"{name} IDs contain duplicates")
    prediction_ids = pd.MultiIndex.from_frame(predictions.loc[:, list(id_columns)])
    trusted_ids = pd.MultiIndex.from_frame(trusted.loc[:, list(id_columns)])
    missing = trusted_ids.difference(prediction_ids)
    extra = prediction_ids.difference(trusted_ids)
    if len(missing) or len(extra):
        raise ValueError(
            f"prediction IDs are not an exact match: {len(missing)} missing, {len(extra)} extra"
        )


def _stratification_values(
    source: pd.DataFrame,
    manifest: DatasetManifest,
) -> np.ndarray[Any, Any] | None:
    if manifest.problem_type.task is not TaskKind.CLASSIFICATION:
        return None
    if len(manifest.target_columns) != 1:
        raise ValueError("stratified splitting requires exactly one classification target")
    values = source[manifest.target_columns[0]].to_numpy()
    counts = pd.Series(values).value_counts()
    if (counts < 3).any():
        raise ValueError("every class needs at least three rows for trusted three-way splitting")
    return values


def _validate_separate_roots(candidate_root: Path, trusted_root: Path) -> None:
    candidate = candidate_root.resolve()
    trusted = trusted_root.resolve()
    if candidate == trusted or candidate in trusted.parents or trusted in candidate.parents:
        raise ValueError("candidate and trusted roots must be disjoint directory trees")


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    candidate = (root.resolve() / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes its root: {relative_path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact does not exist: {relative_path}")
    return candidate


def _frame_asset(root: Path, path: Path, frame: pd.DataFrame) -> DataAsset:
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
