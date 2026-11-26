"""Differentiable custom LightGBM objectives for AGENT-K evolution.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = (
    "LightGBMObjective",
    "LossId",
    "LossSpec",
    "asymmetric_squared_objective",
    "binary_log_loss_objective",
    "build_lightgbm_objective",
    "multiclass_log_loss_objective",
    "pseudo_huber_objective",
    "smooth_l1_objective",
    "smooth_quantile_objective",
    "squared_error_objective",
)

LightGBMObjective = Callable[
    [npt.ArrayLike, npt.ArrayLike],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]


class LossId(StrEnum):
    """Evolvable objective families implemented by this module."""

    SQUARED_ERROR = "squared_error"
    ASYMMETRIC_SQUARED = "asymmetric_squared"
    PSEUDO_HUBER = "pseudo_huber"
    SMOOTH_L1 = "smooth_l1"
    SMOOTH_QUANTILE = "smooth_quantile"
    BINARY_LOG_LOSS = "binary_log_loss"
    MULTICLASS_LOG_LOSS = "multiclass_log_loss"


class LossSpec(BaseModel):
    """Validated parameters for a custom LightGBM objective."""

    model_config = ConfigDict(frozen=True, strict=True)

    loss_id: LossId
    underprediction_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    overprediction_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    delta: float = Field(default=1.0, gt=1e-9, le=1e6)
    squared_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    epsilon: float = Field(default=1e-3, gt=1e-12, le=1.0)
    quantile: float = Field(default=0.5, gt=0.0, lt=1.0)
    smoothing: float = Field(default=0.05, gt=1e-9, le=100.0)
    min_hessian: float = Field(default=1e-6, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_relevant_parameters(self) -> LossSpec:
        """Ensure an objective cannot be configured with unusable curvature."""
        if self.loss_id is LossId.SMOOTH_L1 and self.squared_weight == 0.0:
            if self.epsilon < self.min_hessian:
                raise ValueError("smooth L1 epsilon must not be smaller than min_hessian")
        return self


def squared_error_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return exact gradient and Hessian for half squared error."""
    truth, predictions = _coerce_pair(labels, raw_predictions)
    gradient = predictions - truth
    return gradient, np.ones_like(gradient)


def asymmetric_squared_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
    *,
    underprediction_weight: float = 1.0,
    overprediction_weight: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Penalize under- and overprediction with independently evolvable weights."""
    truth, predictions = _coerce_pair(labels, raw_predictions)
    residual = predictions - truth
    weights = np.where(residual < 0.0, underprediction_weight, overprediction_weight)
    return weights * residual, weights.astype(np.float64, copy=False)


def pseudo_huber_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
    *,
    delta: float = 1.0,
    min_hessian: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return robust pseudo-Huber derivatives with strictly positive curvature."""
    truth, predictions = _coerce_pair(labels, raw_predictions)
    scaled = (predictions - truth) / delta
    base = 1.0 + np.square(scaled)
    gradient = (predictions - truth) / np.sqrt(base)
    hessian = np.maximum(np.power(base, -1.5), min_hessian)
    return gradient, hessian


def smooth_l1_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
    *,
    squared_weight: float = 0.5,
    epsilon: float = 1e-3,
    min_hessian: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Blend squared error with a differentiable absolute-error approximation."""
    truth, predictions = _coerce_pair(labels, raw_predictions)
    residual = predictions - truth
    smooth_radius = np.sqrt(np.square(residual) + epsilon**2)
    absolute_weight = 1.0 - squared_weight
    gradient = squared_weight * residual + absolute_weight * residual / smooth_radius
    hessian = squared_weight + absolute_weight * epsilon**2 / np.power(smooth_radius, 3)
    return gradient, np.maximum(hessian, min_hessian)


def smooth_quantile_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
    *,
    quantile: float = 0.5,
    smoothing: float = 0.05,
    min_hessian: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return derivatives for a smooth approximation to quantile pinball loss."""
    truth, predictions = _coerce_pair(labels, raw_predictions)
    scaled = np.clip((predictions - truth) / smoothing, -40.0, 40.0)
    sigmoid = 1.0 / (1.0 + np.exp(-scaled))
    gradient = sigmoid - quantile
    hessian = np.maximum(sigmoid * (1.0 - sigmoid) / smoothing, min_hessian)
    return gradient, hessian


def binary_log_loss_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
    *,
    min_hessian: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return binary cross-entropy derivatives from raw LightGBM margins."""
    truth, predictions = _coerce_pair(labels, raw_predictions)
    unique = np.unique(truth)
    if not set(unique).issubset({0.0, 1.0}):
        raise ValueError("binary log loss labels must be encoded as 0 or 1")
    margins = np.clip(predictions, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-margins))
    gradient = probability - truth
    hessian = np.maximum(probability * (1.0 - probability), min_hessian)
    return gradient, hessian


def multiclass_log_loss_objective(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
    *,
    min_hessian: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return softmax cross-entropy gradient and diagonal Hessian."""
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(raw_predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[0] != len(truth):
        raise ValueError("multiclass raw predictions must have shape (rows, classes)")
    if not np.isfinite(predictions).all():
        raise ValueError("multiclass predictions must be finite")
    class_count = predictions.shape[1]
    if class_count < 3:
        raise ValueError("multiclass log loss requires at least three classes")
    if (truth < 0).any() or (truth >= class_count).any():
        raise ValueError("multiclass labels are outside the prediction class range")
    shifted = predictions - predictions.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probability = exponentials / exponentials.sum(axis=1, keepdims=True)
    gradient = probability.copy()
    gradient[np.arange(len(truth)), truth] -= 1.0
    hessian = np.maximum(probability * (1.0 - probability), min_hessian)
    return gradient, hessian


def build_lightgbm_objective(spec: LossSpec) -> LightGBMObjective:
    """Build a top-level-compatible objective closure from a validated spec."""
    if spec.loss_id is LossId.SQUARED_ERROR:
        return squared_error_objective
    if spec.loss_id is LossId.ASYMMETRIC_SQUARED:
        return lambda labels, predictions: asymmetric_squared_objective(
            labels,
            predictions,
            underprediction_weight=spec.underprediction_weight,
            overprediction_weight=spec.overprediction_weight,
        )
    if spec.loss_id is LossId.PSEUDO_HUBER:
        return lambda labels, predictions: pseudo_huber_objective(
            labels,
            predictions,
            delta=spec.delta,
            min_hessian=spec.min_hessian,
        )
    if spec.loss_id is LossId.SMOOTH_L1:
        return lambda labels, predictions: smooth_l1_objective(
            labels,
            predictions,
            squared_weight=spec.squared_weight,
            epsilon=spec.epsilon,
            min_hessian=spec.min_hessian,
        )
    if spec.loss_id is LossId.SMOOTH_QUANTILE:
        return lambda labels, predictions: smooth_quantile_objective(
            labels,
            predictions,
            quantile=spec.quantile,
            smoothing=spec.smoothing,
            min_hessian=spec.min_hessian,
        )
    if spec.loss_id is LossId.BINARY_LOG_LOSS:
        return lambda labels, predictions: binary_log_loss_objective(
            labels,
            predictions,
            min_hessian=spec.min_hessian,
        )
    if spec.loss_id is LossId.MULTICLASS_LOG_LOSS:
        return lambda labels, predictions: multiclass_log_loss_objective(
            labels,
            predictions,
            min_hessian=spec.min_hessian,
        )
    raise AssertionError(f"unhandled loss: {spec.loss_id}")


def _coerce_pair(
    labels: npt.ArrayLike,
    raw_predictions: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    truth = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(raw_predictions, dtype=np.float64)
    if truth.shape != predictions.shape:
        raise ValueError(
            f"label and prediction shapes differ: {truth.shape} != {predictions.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("labels and predictions must be finite")
    return truth, predictions
