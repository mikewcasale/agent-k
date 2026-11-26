"""Trusted metric definitions and prediction validation for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from enum import StrEnum
from urllib.parse import quote

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    ndcg_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

__all__ = (
    "MetricDirection",
    "MetricId",
    "MetricSpec",
    "PredictionKind",
    "metric_spec",
    "prediction_columns",
    "score_predictions",
    "validate_prediction_values",
)


class MetricId(StrEnum):
    """Exact supported competition metric identifiers."""

    ACCURACY = "accuracy"
    AUC = "auc"
    LOG_LOSS = "logLoss"
    F1 = "f1"
    RMSE = "rmse"
    MAE = "mae"
    RMSLE = "rmsle"
    MAP = "map"
    NDCG = "ndcg"


class MetricDirection(StrEnum):
    """Optimization direction for a trusted metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class PredictionKind(StrEnum):
    """Semantic form required from a candidate."""

    LABEL = "label"
    PROBABILITY = "probability"
    CONTINUOUS = "continuous"
    RANKING_SCORE = "ranking_score"


class MetricSpec(BaseModel):
    """Immutable trusted metric behavior."""

    model_config = ConfigDict(frozen=True, strict=True)

    metric_id: MetricId
    direction: MetricDirection
    prediction_kind: PredictionKind


_METRICS: dict[MetricId, MetricSpec] = {
    MetricId.ACCURACY: MetricSpec(
        metric_id=MetricId.ACCURACY,
        direction=MetricDirection.MAXIMIZE,
        prediction_kind=PredictionKind.LABEL,
    ),
    MetricId.AUC: MetricSpec(
        metric_id=MetricId.AUC,
        direction=MetricDirection.MAXIMIZE,
        prediction_kind=PredictionKind.PROBABILITY,
    ),
    MetricId.LOG_LOSS: MetricSpec(
        metric_id=MetricId.LOG_LOSS,
        direction=MetricDirection.MINIMIZE,
        prediction_kind=PredictionKind.PROBABILITY,
    ),
    MetricId.F1: MetricSpec(
        metric_id=MetricId.F1,
        direction=MetricDirection.MAXIMIZE,
        prediction_kind=PredictionKind.LABEL,
    ),
    MetricId.RMSE: MetricSpec(
        metric_id=MetricId.RMSE,
        direction=MetricDirection.MINIMIZE,
        prediction_kind=PredictionKind.CONTINUOUS,
    ),
    MetricId.MAE: MetricSpec(
        metric_id=MetricId.MAE,
        direction=MetricDirection.MINIMIZE,
        prediction_kind=PredictionKind.CONTINUOUS,
    ),
    MetricId.RMSLE: MetricSpec(
        metric_id=MetricId.RMSLE,
        direction=MetricDirection.MINIMIZE,
        prediction_kind=PredictionKind.CONTINUOUS,
    ),
    MetricId.MAP: MetricSpec(
        metric_id=MetricId.MAP,
        direction=MetricDirection.MAXIMIZE,
        prediction_kind=PredictionKind.PROBABILITY,
    ),
    MetricId.NDCG: MetricSpec(
        metric_id=MetricId.NDCG,
        direction=MetricDirection.MAXIMIZE,
        prediction_kind=PredictionKind.RANKING_SCORE,
    ),
}


def metric_spec(metric_id: MetricId) -> MetricSpec:
    """Return trusted behavior for an exact metric identifier."""
    return _METRICS[metric_id]


def prediction_columns(
    metric_id: MetricId,
    target_columns: tuple[str, ...],
    class_labels: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the exact candidate output columns for a metric."""
    kind = metric_spec(metric_id).prediction_kind
    if kind is not PredictionKind.PROBABILITY:
        return target_columns
    if len(target_columns) != 1:
        raise ValueError("probability metrics require exactly one target column")
    target = target_columns[0]
    if len(class_labels) <= 2:
        return (f"{target}__probability",)
    return tuple(f"{target}__probability__{quote(label, safe='')}" for label in class_labels)


def validate_prediction_values(
    predictions: npt.ArrayLike,
    metric_id: MetricId,
    *,
    class_count: int | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.object_]:
    """Validate shape, finiteness, and probability invariants.

    Label metrics intentionally permit string arrays. All other metrics require
    finite numeric values. Probability matrices must have one column per class
    and every row must sum to one.
    """
    kind = metric_spec(metric_id).prediction_kind
    raw = np.asarray(predictions)
    if raw.ndim not in {1, 2}:
        raise ValueError("predictions must be a one- or two-dimensional array")
    if raw.shape[0] == 0:
        raise ValueError("predictions cannot be empty")
    if kind is PredictionKind.LABEL:
        labels = np.asarray(raw, dtype=object).reshape(-1)
        if any(
            value is None
            or (isinstance(value, (float, np.floating)) and np.isnan(value))
            or (isinstance(value, str) and not value.strip())
            for value in labels
        ):
            raise ValueError("label predictions contain missing values")
        return np.asarray(raw, dtype=object)

    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("predictions must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("predictions contain non-finite values")
    if kind is PredictionKind.PROBABILITY:
        if ((numeric < 0.0) | (numeric > 1.0)).any():
            raise ValueError("probabilities must be within [0, 1]")
        if class_count is not None and class_count > 2:
            if numeric.ndim != 2 or numeric.shape[1] != class_count:
                raise ValueError("multiclass probability column count does not match classes")
            if not np.allclose(numeric.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("multiclass probabilities must sum to one per row")
        elif numeric.ndim == 2 and numeric.shape[1] != 1:
            raise ValueError("binary probabilities require exactly one column")
    return numeric


def score_predictions(
    metric_id: MetricId,
    labels: npt.ArrayLike,
    predictions: npt.ArrayLike,
    *,
    class_labels: tuple[str, ...] = (),
) -> float:
    """Score trusted hidden labels against already ID-aligned predictions."""
    class_count = len(class_labels) or None
    checked = validate_prediction_values(
        predictions,
        metric_id,
        class_count=class_count,
    )
    truth = np.asarray(labels)
    if truth.ndim == 2 and truth.shape[1] == 1:
        truth = truth[:, 0]
    if len(truth) != checked.shape[0]:
        raise ValueError("label and prediction row counts differ")

    if metric_id is MetricId.ACCURACY:
        return float(accuracy_score(_label_strings(truth), _label_strings(checked)))
    if metric_id is MetricId.F1:
        average = "binary" if len(np.unique(truth)) == 2 else "macro"
        truth_strings = _label_strings(truth)
        prediction_strings = _label_strings(checked)
        positive = class_labels[-1] if class_labels else sorted(set(truth_strings))[-1]
        return float(
            f1_score(
                truth_strings,
                prediction_strings,
                average=average,
                pos_label=positive,
            )
        )
    if metric_id is MetricId.AUC:
        return _score_auc(truth, np.asarray(checked, dtype=np.float64), class_labels)
    if metric_id is MetricId.LOG_LOSS:
        return _score_log_loss(truth, np.asarray(checked, dtype=np.float64), class_labels)
    if metric_id is MetricId.RMSE:
        return float(mean_squared_error(truth, checked) ** 0.5)
    if metric_id is MetricId.MAE:
        return float(mean_absolute_error(truth, checked))
    if metric_id is MetricId.RMSLE:
        if (np.asarray(truth, dtype=np.float64) < 0.0).any() or (checked < 0.0).any():
            raise ValueError("RMSLE requires non-negative labels and predictions")
        return float(mean_squared_log_error(truth, checked) ** 0.5)
    if metric_id is MetricId.MAP:
        return _score_average_precision(truth, np.asarray(checked, dtype=np.float64), class_labels)
    if metric_id is MetricId.NDCG:
        return float(
            ndcg_score(
                np.asarray(truth, dtype=np.float64).reshape(1, -1),
                np.asarray(checked, dtype=np.float64).reshape(1, -1),
            )
        )
    raise AssertionError(f"unhandled metric: {metric_id}")


def _label_strings(values: npt.ArrayLike) -> npt.NDArray[np.str_]:
    return np.asarray(values).reshape(-1).astype(str)


def _score_auc(
    truth: npt.NDArray[np.generic],
    predictions: npt.NDArray[np.float64],
    class_labels: tuple[str, ...],
) -> float:
    labels = class_labels or tuple(sorted(set(_label_strings(truth))))
    truth_strings = _label_strings(truth)
    if len(labels) == 2:
        return float(roc_auc_score(truth_strings == labels[-1], predictions.reshape(-1)))
    return float(
        roc_auc_score(
            truth_strings,
            predictions,
            labels=list(labels),
            multi_class="ovr",
            average="macro",
        )
    )


def _score_log_loss(
    truth: npt.NDArray[np.generic],
    predictions: npt.NDArray[np.float64],
    class_labels: tuple[str, ...],
) -> float:
    labels = class_labels or tuple(sorted(set(_label_strings(truth))))
    truth_strings = _label_strings(truth)
    if len(labels) == 2 and (predictions.ndim == 1 or predictions.shape[1] == 1):
        positive = predictions.reshape(-1)
        predictions = np.column_stack((1.0 - positive, positive))
    return float(log_loss(truth_strings, predictions, labels=list(labels)))


def _score_average_precision(
    truth: npt.NDArray[np.generic],
    predictions: npt.NDArray[np.float64],
    class_labels: tuple[str, ...],
) -> float:
    labels = class_labels or tuple(sorted(set(_label_strings(truth))))
    truth_strings = _label_strings(truth)
    if len(labels) == 2:
        return float(average_precision_score(truth_strings == labels[-1], predictions.reshape(-1)))
    binary = label_binarize(truth_strings, classes=list(labels))
    return float(average_precision_score(binary, predictions, average="macro"))
