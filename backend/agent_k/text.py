"""TF-IDF linear and offline transformer candidates for AGENT-K text tasks.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os
import time
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from .datasets import TaskKind
from .evaluator import (
    CandidateEngine,
    CandidateResult,
    CandidateSpec,
    load_candidate_dataset,
    write_candidate_predictions,
)
from .metrics import MetricId, PredictionKind, metric_spec, prediction_columns

__all__ = ("run_candidate", "run_tfidf_candidate", "run_transformer_candidate")

_TFIDF_PARAMETERS: Final[frozenset[str]] = frozenset(
    {
        "alpha",
        "c",
        "class_weight",
        "max_features",
        "max_iter",
        "min_df",
        "ngram_max",
    }
)
_TRANSFORMER_PARAMETERS: Final[frozenset[str]] = frozenset(
    {
        "batch_size",
        "cpu_threads",
        "device_policy",
        "epochs",
        "learning_rate",
        "max_length",
        "weight_decay",
    }
)


class _TfidfConfig(BaseModel):
    """Validated TF-IDF and linear estimator controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha: float = Field(default=1.0, gt=0.0, le=1e6)
    c: float = Field(default=1.0, gt=0.0, le=1e6)
    class_weight: Literal["balanced"] | None = None
    max_features: int = Field(default=100_000, ge=100, le=1_000_000)
    max_iter: int = Field(default=1_000, ge=1, le=100_000)
    min_df: int = Field(default=2, ge=1, le=1_000)
    ngram_max: int = Field(default=2, ge=1, le=5)


class _TransformerConfig(BaseModel):
    """Validated offline transformer fine-tuning controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=16, ge=1, le=256)
    cpu_threads: int = Field(default=1, ge=1, le=64)
    device_policy: str = "auto"
    epochs: int = Field(default=3, ge=1, le=20)
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1e-2)
    max_length: int = Field(default=256, ge=8, le=4_096)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)


def run_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Run the selected production text engine."""
    if spec.engine is CandidateEngine.TFIDF_LINEAR:
        return run_tfidf_candidate(spec, input_root, output_root)
    if spec.engine is CandidateEngine.TRANSFORMER:
        return run_transformer_candidate(spec, input_root, output_root)
    raise ValueError(f"unsupported text engine: {spec.engine.value}")


def run_tfidf_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Fit a word n-gram TF-IDF representation and a regularized linear model."""
    started = time.monotonic()
    _validate_text_spec(spec, _TFIDF_PARAMETERS)
    config = _TfidfConfig.model_validate(spec.hyperparameters)
    bundle = load_candidate_dataset(spec, input_root)
    assert spec.text_column is not None
    train_text = bundle.train[spec.text_column].fillna("").astype(str)
    predict_text = bundle.predict[spec.text_column].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, config.ngram_max),
        min_df=config.min_df,
        max_features=config.max_features,
        sublinear_tf=True,
        dtype=np.float64,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    if train_matrix.shape[1] == 0:
        raise ValueError("TF-IDF produced no usable text features")
    predict_matrix = vectorizer.transform(predict_text)
    target = bundle.train_targets.iloc[:, 0]
    columns = prediction_columns(spec.metric_id, spec.target_columns, spec.class_labels)
    if spec.problem_type.task is TaskKind.CLASSIFICATION:
        labels = target.astype(str)
        if set(labels) != set(spec.class_labels):
            raise ValueError("text training classes differ from CandidateSpec.class_labels")
        model = LogisticRegression(
            C=config.c,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            random_state=spec.seed,
        )
        model.fit(train_matrix, labels)
        model_classes = tuple(str(label) for label in model.classes_)
        if model_classes != spec.class_labels:
            order = [model_classes.index(label) for label in spec.class_labels]
        else:
            order = list(range(len(model_classes)))
        probabilities = np.asarray(model.predict_proba(predict_matrix), dtype=np.float64)[:, order]
        predictions = _classification_frame(spec, columns, probabilities)
    else:
        values = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.float64)
        model = Ridge(alpha=config.alpha)
        model.fit(train_matrix, values)
        prediction = np.asarray(model.predict(predict_matrix), dtype=np.float64)
        if spec.metric_id is MetricId.RMSLE:
            prediction = np.maximum(prediction, 0.0)
        predictions = pd.DataFrame({columns[0]: prediction})
    return write_candidate_predictions(
        spec,
        bundle.predict_ids,
        predictions,
        output_root,
        started=started,
        diagnostics={
            "engine": CandidateEngine.TFIDF_LINEAR.value,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
    )


def run_transformer_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Fine-tune a locally mounted Hugging Face sequence model without network access."""
    started = time.monotonic()
    _validate_text_spec(spec, _TRANSFORMER_PARAMETERS)
    config = _TransformerConfig.model_validate(spec.hyperparameters)
    bundle = load_candidate_dataset(spec, input_root)
    model_relative = spec.offline_assets.get("transformer_model")
    if model_relative is None:
        raise ValueError("transformer candidates require offline_assets['transformer_model']")
    model_path = _safe_asset_directory(input_root, model_relative)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformer worker dependencies are not installed") from exc

    assert spec.text_column is not None
    train_text = bundle.train[spec.text_column].fillna("").astype(str).tolist()
    predict_text = bundle.predict[spec.text_column].fillna("").astype(str).tolist()
    device = _select_device(config.device_policy)
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    output_count = (
        len(spec.class_labels) if spec.problem_type.task is TaskKind.CLASSIFICATION else 1
    )
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=output_count,
        ignore_mismatched_sizes=True,
        problem_type=(
            "single_label_classification"
            if spec.problem_type.task is TaskKind.CLASSIFICATION
            else "regression"
        ),
    ).to(device)
    train_encoding = tokenizer(
        train_text,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    predict_encoding = tokenizer(
        predict_text,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    labels = _transformer_labels(spec, bundle.train_targets.iloc[:, 0])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    order_generator = torch.Generator().manual_seed(spec.seed)
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(train_text), generator=order_generator)
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = {name: tensor[indices].to(device) for name, tensor in train_encoding.items()}
            batch["labels"] = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError("transformer training produced a non-finite loss")
            output.loss.backward()
            optimizer.step()

    logits: list[np.ndarray[Any, Any]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(predict_text), config.batch_size):
            stop = start + config.batch_size
            batch = {
                name: tensor[start:stop].to(device) for name, tensor in predict_encoding.items()
            }
            logits.append(model(**batch).logits.cpu().numpy())
    raw = np.concatenate(logits, axis=0)
    columns = prediction_columns(spec.metric_id, spec.target_columns, spec.class_labels)
    if spec.problem_type.task is TaskKind.CLASSIFICATION:
        shifted = raw - raw.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        predictions = _classification_frame(spec, columns, probabilities)
    else:
        prediction = raw.reshape(-1).astype(np.float64)
        if spec.metric_id is MetricId.RMSLE:
            prediction = np.maximum(prediction, 0.0)
        predictions = pd.DataFrame({columns[0]: prediction})
    return write_candidate_predictions(
        spec,
        bundle.predict_ids,
        predictions,
        output_root,
        started=started,
        diagnostics={
            "engine": CandidateEngine.TRANSFORMER.value,
            "device": device.type,
            "epochs": config.epochs,
            "max_length": config.max_length,
        },
    )


def _classification_frame(
    spec: CandidateSpec,
    columns: tuple[str, ...],
    probabilities: np.ndarray[Any, Any],
) -> pd.DataFrame:
    if metric_spec(spec.metric_id).prediction_kind is PredictionKind.PROBABILITY:
        if len(spec.class_labels) == 2:
            return pd.DataFrame({columns[0]: probabilities[:, 1]})
        return pd.DataFrame(
            {column: probabilities[:, index] for index, column in enumerate(columns)}
        )
    hard = np.argmax(probabilities, axis=1)
    return pd.DataFrame({columns[0]: [spec.class_labels[index] for index in hard]})


def _transformer_labels(spec: CandidateSpec, target: pd.Series) -> torch.Tensor:
    if spec.problem_type.task is TaskKind.REGRESSION:
        values = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("text regression target contains non-finite values")
        return torch.from_numpy(values)
    labels = target.astype(str)
    if set(labels) != set(spec.class_labels):
        raise ValueError("text training classes differ from CandidateSpec.class_labels")
    encoding = {label: index for index, label in enumerate(spec.class_labels)}
    return torch.from_numpy(labels.map(encoding).to_numpy(dtype=np.int64))


def _validate_text_spec(spec: CandidateSpec, allowed: frozenset[str]) -> None:
    if spec.text_column is None:
        raise ValueError("text candidates require text_column")
    if len(spec.target_columns) != 1:
        raise ValueError("text candidates require exactly one target")
    unknown = set(spec.hyperparameters) - allowed
    if unknown:
        raise ValueError(f"unsupported text hyperparameters: {sorted(unknown)}")


def _select_device(policy: str) -> torch.device:
    if policy not in {"auto", "cuda_required", "cpu"}:
        raise ValueError("device_policy must be auto, cuda_required, or cpu")
    if policy == "cuda_required" and not torch.cuda.is_available():
        raise RuntimeError("transformer requires CUDA but no CUDA device is available")
    if policy == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_asset_directory(root: Path, relative_path: str) -> Path:
    path = (root.resolve() / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"transformer asset escapes input root: {relative_path}") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"transformer asset directory does not exist: {relative_path}")
    return path
