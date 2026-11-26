"""Production tabular candidates using LightGBM and NVIDIA Kumo Tabular.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Final, Literal, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sklearn.model_selection import train_test_split

from .data import DatasetBundle
from .datasets import TaskKind
from .evaluator import (
    CandidateEngine,
    CandidateResult,
    CandidateSpec,
    load_candidate_dataset,
    write_candidate_predictions,
)
from .losses import LossId, LossSpec, build_lightgbm_objective
from .metrics import MetricId, PredictionKind, metric_spec, prediction_columns

__all__ = (
    "KumoAssetManifest",
    "KumoFile",
    "main",
    "provision_kumo_assets",
    "run_candidate",
    "run_kumo_candidate",
    "run_lightgbm_candidate",
)

_KUMO_REPOSITORY: Final[str] = "nvidia/Kumo-Tabular"
_KUMO_REVISION: Final[str] = "v1.0.1"

_LIGHTGBM_PARAMETERS: Final[frozenset[str]] = frozenset(
    {
        "boosting_type",
        "colsample_bytree",
        "learning_rate",
        "max_bin",
        "max_depth",
        "min_child_samples",
        "min_split_gain",
        "n_estimators",
        "num_leaves",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "subsample_freq",
    }
)


class KumoFile(BaseModel):
    """Digest evidence for one provisioned Kumo Hub file."""

    model_config = ConfigDict(frozen=True, strict=True)

    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class KumoAssetManifest(BaseModel):
    """Pinned offline Kumo checkpoint set accepted by candidate workers."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: str = "1.0"
    repository: str = _KUMO_REPOSITORY
    revision: str = _KUMO_REVISION
    size: str
    files: tuple[KumoFile, ...]


class _KumoConfig(BaseModel):
    """Validated Kumo inference controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    context_strategy: Literal["auto", "uniform", "stratified"] = "auto"
    cpu_threads: int = Field(default=1, ge=1, le=64)
    device_policy: Literal["cuda_required", "cpu_allowed"] = "cuda_required"
    lightgbm_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    lightgbm_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_context_rows: int = Field(default=60_000, ge=2, le=60_000)
    num_estimators: int = Field(default=8, ge=1, le=64)
    size: Literal["small", "medium", "large"] = "small"


def run_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Run the selected tabular engine inside an offline candidate sandbox."""
    if spec.engine is CandidateEngine.LIGHTGBM:
        return run_lightgbm_candidate(spec, input_root, output_root)
    if spec.engine is CandidateEngine.KUMO:
        return run_kumo_candidate(spec, input_root, output_root)
    raise ValueError(f"unsupported tabular engine: {spec.engine.value}")


def run_lightgbm_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Train LightGBM models and emit metric-appropriate predictions."""
    started = time.monotonic()
    bundle = load_candidate_dataset(spec, input_root)
    prediction_frame, categorical_count, loss = _lightgbm_prediction_frame(spec, bundle)
    diagnostics: dict[str, JsonValue] = {
        "engine": CandidateEngine.LIGHTGBM.value,
        "custom_loss": loss.loss_id.value if loss is not None else None,
        "categorical_feature_count": categorical_count,
    }
    return write_candidate_predictions(
        spec,
        bundle.predict_ids,
        prediction_frame,
        output_root,
        started=started,
        diagnostics=diagnostics,
    )


def _lightgbm_prediction_frame(
    spec: CandidateSpec,
    bundle: DatasetBundle,
) -> tuple[pd.DataFrame, int, LossSpec | None]:
    train_features, predict_features, categorical = _prepare_lightgbm_features(
        bundle.train_features,
        bundle.predict_features,
    )
    parameters, loss = _lightgbm_parameters(spec)
    outputs: dict[str, np.ndarray[Any, Any]] = {}
    if spec.problem_type.task is TaskKind.CLASSIFICATION:
        if len(spec.target_columns) != 1:
            raise ValueError("classification candidates require exactly one target")
        target = bundle.train_targets.iloc[:, 0].astype(str)
        outputs = _fit_lightgbm_classifier(
            spec,
            train_features,
            target,
            predict_features,
            categorical,
            parameters,
            loss,
        )
    else:
        outputs = _fit_lightgbm_regressors(
            spec,
            train_features,
            bundle.train_targets,
            predict_features,
            categorical,
            parameters,
            loss,
        )
    columns = prediction_columns(spec.metric_id, spec.target_columns, spec.class_labels)
    prediction_frame = pd.DataFrame({column: outputs[column] for column in columns})
    return prediction_frame, len(categorical), loss


def run_kumo_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Run offline pretrained Kumo Tabular inference with eight estimators by default."""
    started = time.monotonic()
    bundle = load_candidate_dataset(spec, input_root)
    if len(spec.target_columns) != 1:
        raise ValueError("Kumo Tabular supports exactly one target")
    config = _KumoConfig.model_validate(spec.hyperparameters)
    _configure_offline_huggingface(spec, input_root, config)
    if len(spec.feature_columns) > 500:
        raise ValueError("Kumo candidates support at most 500 feature columns")

    try:
        import sdm
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Kumo requires the 'kumo' worker dependencies structured-data-models and torch"
        ) from exc

    if config.device_policy == "cuda_required" and not torch.cuda.is_available():
        raise RuntimeError("Kumo production candidates require an available CUDA device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_threads)

    target = spec.target_columns[0]
    context = _sample_kumo_context(bundle.train, spec, config)
    context_frame = context.loc[:, [*spec.feature_columns, target]].copy()
    query_frame = bundle.predict.loc[:, list(spec.feature_columns)].copy()
    if spec.problem_type.task is TaskKind.CLASSIFICATION:
        context_frame[target] = context_frame[target].astype("string")
        query_frame[target] = pd.Series(
            pd.NA,
            index=query_frame.index,
            dtype="string",
        )
    else:
        query_frame[target] = np.nan
    combined = pd.concat(
        (context_frame, query_frame),
        ignore_index=True,
    )
    target_stype = (
        "categorical" if spec.problem_type.task is TaskKind.CLASSIFICATION else "numerical"
    )
    table = sdm.TableTensor.from_pandas(
        df=combined,
        stypes=sdm.infer_stypes(combined, overrides={target: target_stype}),
        device=device,
    )
    context_rows = len(context)
    model = sdm.models.KumoTabular(
        task=spec.problem_type.task.value,
        size=config.size,
        pretrained=True,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    with torch.autocast(
        device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        result = model(
            x_context=table[:context_rows].drop_columns(target),
            y_context=table[:context_rows, target],
            x_query=table[context_rows:].drop_columns(target),
            num_estimators=config.num_estimators,
            generator=generator,
        )
    raw = result.to_pandas().reset_index(drop=True)
    inference_spec = spec
    if (
        config.lightgbm_weight > 0.0
        and spec.problem_type.task is TaskKind.CLASSIFICATION
        and metric_spec(spec.metric_id).prediction_kind is PredictionKind.LABEL
    ):
        inference_spec = spec.model_copy(update={"metric_id": MetricId.LOG_LOSS})
    prediction_frame, diagnostics = _kumo_predictions(inference_spec, raw, output_root)
    if config.lightgbm_weight > 0.0:
        lightgbm_spec = inference_spec.model_copy(
            update={
                "engine": CandidateEngine.LIGHTGBM,
                "hyperparameters": _kumo_lightgbm_parameters(spec, config),
            }
        )
        lightgbm_frame, _, _ = _lightgbm_prediction_frame(lightgbm_spec, bundle)
        prediction_frame = _blend_prediction_frames(
            prediction_frame,
            lightgbm_frame,
            lightgbm_weight=config.lightgbm_weight,
        )
    if inference_spec.metric_id is not spec.metric_id:
        prediction_frame = _probabilities_to_labels(prediction_frame, spec)
    diagnostics.update(
        {
            "engine": CandidateEngine.KUMO.value,
            "device": device.type,
            "model_size": config.size,
            "num_estimators": config.num_estimators,
            "context_rows": context_rows,
            "context_strategy": config.context_strategy,
            "lightgbm_weight": config.lightgbm_weight,
        }
    )
    return write_candidate_predictions(
        spec,
        bundle.predict_ids,
        prediction_frame,
        output_root,
        started=started,
        diagnostics=diagnostics,
    )


def _prepare_lightgbm_features(
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    training = train.copy()
    prediction = predict.copy()
    categorical: list[str] = []
    for column in training.columns:
        combined = pd.concat((training[column], prediction[column]), ignore_index=True)
        if (
            pd.api.types.is_object_dtype(combined)
            or pd.api.types.is_string_dtype(combined)
            or isinstance(combined.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(combined)
        ):
            categories = pd.Index(pd.unique(combined.dropna().astype(str)))
            training[column] = pd.Categorical(
                training[column].astype("string"),
                categories=categories,
            )
            prediction[column] = pd.Categorical(
                prediction[column].astype("string"),
                categories=categories,
            )
            categorical.append(str(column))
            continue
        if not pd.api.types.is_numeric_dtype(combined):
            raise ValueError(f"LightGBM feature has unsupported dtype: {column}")
        training[column] = pd.to_numeric(training[column], errors="raise").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        prediction[column] = pd.to_numeric(prediction[column], errors="raise").replace(
            [np.inf, -np.inf],
            np.nan,
        )
    return training, prediction, categorical


def _sample_kumo_context(
    training: pd.DataFrame,
    spec: CandidateSpec,
    config: _KumoConfig,
) -> pd.DataFrame:
    desired = min(
        config.max_context_rows,
        len(training),
        max(2, int(np.ceil(len(training) * config.context_fraction))),
    )
    if desired >= len(training):
        return training.reset_index(drop=True)
    strategy = config.context_strategy
    if strategy == "auto":
        strategy = "stratified" if spec.problem_type.task is TaskKind.CLASSIFICATION else "uniform"
    indices = np.arange(len(training))
    if strategy == "stratified":
        if spec.problem_type.task is not TaskKind.CLASSIFICATION:
            raise ValueError("stratified Kumo context sampling requires classification")
        if desired < len(spec.class_labels):
            raise ValueError("Kumo context is too small to include every class")
        selected, _ = train_test_split(
            indices,
            train_size=desired,
            random_state=spec.seed,
            shuffle=True,
            stratify=training[spec.target_columns[0]].astype(str),
        )
    else:
        generator = np.random.default_rng(spec.seed)
        selected = generator.choice(indices, size=desired, replace=False)
    return cast(pd.DataFrame, training.iloc[np.sort(selected)].reset_index(drop=True))


def _kumo_lightgbm_parameters(
    spec: CandidateSpec,
    config: _KumoConfig,
) -> dict[str, JsonValue]:
    parameters = dict(config.lightgbm_parameters)
    if "loss" not in parameters:
        if spec.problem_type.task is TaskKind.REGRESSION:
            loss_id = LossId.SQUARED_ERROR
        elif len(spec.class_labels) == 2:
            loss_id = LossId.BINARY_LOG_LOSS
        else:
            loss_id = LossId.MULTICLASS_LOG_LOSS
        parameters["loss"] = {"loss_id": loss_id.value}
    return parameters


def _blend_prediction_frames(
    kumo: pd.DataFrame,
    lightgbm: pd.DataFrame,
    *,
    lightgbm_weight: float,
) -> pd.DataFrame:
    if tuple(kumo.columns) != tuple(lightgbm.columns) or len(kumo) != len(lightgbm):
        raise ValueError("Kumo and LightGBM predictions are not schema-compatible")
    kumo_values = kumo.to_numpy(dtype=np.float64)
    lightgbm_values = lightgbm.to_numpy(dtype=np.float64)
    blended = (1.0 - lightgbm_weight) * kumo_values + lightgbm_weight * lightgbm_values
    return pd.DataFrame(blended, columns=kumo.columns)


def _probabilities_to_labels(
    probabilities: pd.DataFrame,
    spec: CandidateSpec,
) -> pd.DataFrame:
    values = probabilities.to_numpy(dtype=np.float64)
    if len(spec.class_labels) == 2:
        indices = (values.reshape(-1) >= 0.5).astype(np.int64)
    else:
        indices = np.argmax(values, axis=1)
    return pd.DataFrame({spec.target_columns[0]: [spec.class_labels[index] for index in indices]})


def _lightgbm_parameters(
    spec: CandidateSpec,
) -> tuple[dict[str, Any], LossSpec | None]:
    unknown = set(spec.hyperparameters) - {*_LIGHTGBM_PARAMETERS, "loss"}
    if unknown:
        raise ValueError(f"unsupported LightGBM hyperparameters: {sorted(unknown)}")
    parameters = {key: value for key, value in spec.hyperparameters.items() if key != "loss"}
    defaults: dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 1,
        "random_state": spec.seed,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    defaults.update(parameters)
    raw_loss = spec.hyperparameters.get("loss")
    loss = None
    if raw_loss is not None:
        if not isinstance(raw_loss, dict):
            raise ValueError("LightGBM loss must be an object")
        loss = LossSpec.model_validate(raw_loss, strict=False)
    return defaults, loss


def _fit_lightgbm_classifier(
    spec: CandidateSpec,
    train: pd.DataFrame,
    target: pd.Series,
    predict: pd.DataFrame,
    categorical: list[str],
    parameters: dict[str, Any],
    loss: LossSpec | None,
) -> dict[str, np.ndarray[Any, Any]]:
    labels = spec.class_labels
    observed = set(target)
    if observed != set(labels):
        raise ValueError(
            f"training classes differ from candidate classes: {sorted(observed)} != {labels}"
        )
    encoding = {label: index for index, label in enumerate(labels)}
    encoded = target.map(encoding).to_numpy(dtype=np.int64)
    custom = loss is not None
    if custom:
        assert loss is not None
        expected_loss = LossId.BINARY_LOG_LOSS if len(labels) == 2 else LossId.MULTICLASS_LOG_LOSS
        if loss.loss_id is not expected_loss:
            raise ValueError(f"{len(labels)}-class classification requires {expected_loss.value}")
        parameters["objective"] = build_lightgbm_objective(loss)
        if len(labels) > 2:
            parameters["num_class"] = len(labels)
    model = lgb.LGBMClassifier(**parameters)
    model.fit(train, encoded, categorical_feature=categorical)
    kind = metric_spec(spec.metric_id).prediction_kind
    columns = prediction_columns(spec.metric_id, spec.target_columns, labels)
    if custom:
        margins = np.asarray(model.predict(predict, raw_score=True), dtype=np.float64)
        if len(labels) == 2:
            positive = 1.0 / (1.0 + np.exp(-np.clip(margins, -40.0, 40.0)))
            probabilities = np.column_stack((1.0 - positive, positive))
        else:
            shifted = margins - margins.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    else:
        probabilities = np.asarray(model.predict_proba(predict), dtype=np.float64)
    if kind is PredictionKind.PROBABILITY:
        if len(labels) == 2:
            return {columns[0]: probabilities[:, 1]}
        return {column: probabilities[:, index] for index, column in enumerate(columns)}
    hard = np.argmax(probabilities, axis=1)
    return {columns[0]: np.asarray([labels[index] for index in hard], dtype=object)}


def _fit_lightgbm_regressors(
    spec: CandidateSpec,
    train: pd.DataFrame,
    targets: pd.DataFrame,
    predict: pd.DataFrame,
    categorical: list[str],
    parameters: dict[str, Any],
    loss: LossSpec | None,
) -> dict[str, np.ndarray[Any, Any]]:
    if loss is not None:
        parameters["objective"] = build_lightgbm_objective(loss)
    outputs: dict[str, np.ndarray[Any, Any]] = {}
    for target in spec.target_columns:
        values = pd.to_numeric(targets[target], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"training target contains non-finite values: {target}")
        model = lgb.LGBMRegressor(**parameters)
        model.fit(train, values, categorical_feature=categorical)
        prediction = np.asarray(model.predict(predict), dtype=np.float64)
        if spec.metric_id is MetricId.RMSLE:
            prediction = np.maximum(prediction, 0.0)
        outputs[target] = prediction
    return outputs


def _configure_offline_huggingface(
    spec: CandidateSpec,
    input_root: Path,
    config: _KumoConfig,
) -> None:
    cache_relative = spec.offline_assets.get("hf_cache")
    manifest_relative = spec.offline_assets.get("kumo_manifest")
    if cache_relative is None:
        raise ValueError("Kumo requires offline_assets['hf_cache']")
    if manifest_relative is None:
        raise ValueError("Kumo requires offline_assets['kumo_manifest']")
    cache = (input_root.resolve() / cache_relative).resolve()
    try:
        cache.relative_to(input_root.resolve())
    except ValueError as exc:
        raise ValueError("Kumo Hugging Face cache escapes the input root") from exc
    if not cache.is_dir():
        raise FileNotFoundError(f"mounted Kumo cache does not exist: {cache_relative}")
    manifest_path = (input_root.resolve() / manifest_relative).resolve()
    try:
        manifest_path.relative_to(input_root.resolve())
    except ValueError as exc:
        raise ValueError("Kumo asset manifest escapes the input root") from exc
    if not manifest_path.is_file():
        raise FileNotFoundError(f"mounted Kumo asset manifest does not exist: {manifest_relative}")
    manifest = KumoAssetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.repository != _KUMO_REPOSITORY or manifest.revision != _KUMO_REVISION:
        raise ValueError("Kumo assets are not pinned to nvidia/Kumo-Tabular v1.0.1")
    if manifest.size != config.size:
        raise ValueError(
            f"Kumo asset size {manifest.size!r} does not match candidate size {config.size!r}"
        )
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import constants as huggingface_constants

    huggingface_constants.HF_HUB_CACHE = str(cache)
    huggingface_constants.HUGGINGFACE_HUB_CACHE = str(cache)
    _verify_kumo_files(cache, manifest)


def _kumo_predictions(
    spec: CandidateSpec,
    raw: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = prediction_columns(spec.metric_id, spec.target_columns, spec.class_labels)
    if spec.problem_type.task is TaskKind.CLASSIFICATION:
        aligned = raw.set_axis(tuple(str(column) for column in raw.columns), axis=1)
        missing = set(spec.class_labels) - set(aligned.columns)
        if missing:
            raise ValueError(f"Kumo output omitted classes: {sorted(missing)}")
        probabilities = aligned.loc[:, list(spec.class_labels)].to_numpy(dtype=np.float64)
        if metric_spec(spec.metric_id).prediction_kind is PredictionKind.PROBABILITY:
            if len(spec.class_labels) == 2:
                frame = pd.DataFrame({columns[0]: probabilities[:, 1]})
            else:
                frame = pd.DataFrame(
                    {column: probabilities[:, index] for index, column in enumerate(columns)}
                )
        else:
            hard = np.argmax(probabilities, axis=1)
            frame = pd.DataFrame({columns[0]: [spec.class_labels[index] for index in hard]})
        return frame, {"raw_output_columns": list(map(str, raw.columns))}

    expected_quantiles = [f"q{index:03d}" for index in range(1, 1000)]
    if list(map(str, raw.columns)) != expected_quantiles:
        raise ValueError("Kumo regression output did not contain q001 through q999")
    quantiles = raw.to_numpy(dtype=np.float64)
    if not np.isfinite(quantiles).all():
        raise ValueError("Kumo returned non-finite quantiles")
    if spec.metric_id is MetricId.MAE:
        point = quantiles[:, 499]
    elif spec.metric_id is MetricId.RMSLE:
        point = np.expm1(np.log1p(np.maximum(quantiles, 0.0)).mean(axis=1))
    else:
        point = quantiles.mean(axis=1)
    output_root.mkdir(parents=True, exist_ok=True)
    quantile_path = output_root / "quantiles.csv"
    if quantile_path.exists():
        raise FileExistsError(f"Kumo quantile artifact already exists: {quantile_path}")
    raw.to_csv(quantile_path, index=False)
    return (
        pd.DataFrame({columns[0]: point}),
        {
            "quantiles_path": quantile_path.relative_to(output_root).as_posix(),
            "quantiles_sha256": _sha256(quantile_path),
            "point_reduction": (
                "median"
                if spec.metric_id is MetricId.MAE
                else "log_quantile_mean"
                if spec.metric_id is MetricId.RMSLE
                else "quantile_mean"
            ),
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def provision_kumo_assets(output: Path, *, size: str = "small") -> KumoAssetManifest:
    """Download the pinned Kumo files for a manually approved offline worker gate.

    This provisioning function is deliberately separate from candidate execution.
    Candidate sandboxes remain network-disabled and accept only the resulting
    cache plus its digest manifest.
    """
    if size not in {"small", "medium", "large"}:
        raise ValueError("Kumo size must be small, medium, or large")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to provision Kumo assets") from exc
    destination = output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "hf-cache"
    cache.mkdir(exist_ok=True)
    filenames = ("config.json", f"{size}/classifier.pt", f"{size}/regressor.pt")
    files: list[KumoFile] = []
    for filename in filenames:
        resolved = Path(
            hf_hub_download(
                repo_id=_KUMO_REPOSITORY,
                filename=filename,
                revision=_KUMO_REVISION,
                cache_dir=cache,
                local_files_only=False,
            )
        ).resolve()
        files.append(
            KumoFile(
                filename=filename,
                sha256=_sha256(resolved),
                size_bytes=resolved.stat().st_size,
            )
        )
    manifest = KumoAssetManifest(size=size, files=tuple(files))
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing = KumoAssetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError("existing Kumo manifest does not match provisioned files")
    else:
        temporary = destination / ".manifest.json.tmp"
        temporary.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    return manifest


def _verify_kumo_files(cache: Path, manifest: KumoAssetManifest) -> None:
    from huggingface_hub import hf_hub_download

    for expected in manifest.files:
        resolved = Path(
            hf_hub_download(
                repo_id=manifest.repository,
                filename=expected.filename,
                revision=manifest.revision,
                cache_dir=cache,
                local_files_only=True,
            )
        ).resolve()
        if resolved.stat().st_size != expected.size_bytes or _sha256(resolved) != expected.sha256:
            raise ValueError(f"Kumo asset failed digest verification: {expected.filename}")


def main(argv: list[str] | None = None) -> int:
    """Run explicit asset provisioning commands; candidate execution uses the runner."""
    parser = argparse.ArgumentParser(description="AGENT-K tabular worker utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    provision = subparsers.add_parser(
        "provision-kumo",
        help="download pinned Kumo assets for later offline candidate execution",
    )
    provision.add_argument("--output", type=Path, required=True)
    provision.add_argument(
        "--size",
        choices=("small", "medium", "large"),
        default="small",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "provision-kumo":
        manifest = provision_kumo_assets(arguments.output, size=arguments.size)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2))
        return 0
    parser.error(f"unsupported command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
