"""Numerical and integration tests for custom LightGBM objectives.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_diabetes

from agent_k.losses import (
    LossId,
    LossSpec,
    build_lightgbm_objective,
    multiclass_log_loss_objective,
    pseudo_huber_objective,
)

__all__ = ()


def test_pseudo_huber_derivatives_match_finite_differences() -> None:
    labels = np.array([0.2, -1.0, 3.0])
    predictions = np.array([0.7, -0.2, 2.5])
    delta = 1.3
    gradient, hessian = pseudo_huber_objective(
        labels,
        predictions,
        delta=delta,
    )
    step = 1e-5

    def loss(values: np.ndarray) -> np.ndarray:
        residual = (values - labels) / delta
        return delta**2 * (np.sqrt(1.0 + residual**2) - 1.0)

    numerical_gradient = (loss(predictions + step) - loss(predictions - step)) / (2 * step)
    gradient_plus, _ = pseudo_huber_objective(labels, predictions + step, delta=delta)
    gradient_minus, _ = pseudo_huber_objective(labels, predictions - step, delta=delta)
    numerical_hessian = (gradient_plus - gradient_minus) / (2 * step)

    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-5, atol=1e-6)


def test_multiclass_softmax_objective_has_finite_positive_curvature() -> None:
    labels = np.array([0, 2, 1])
    margins = np.array([[2.0, 0.0, -1.0], [0.1, -0.2, 0.4], [0.0, 1.0, 0.0]])

    gradient, hessian = multiclass_log_loss_objective(labels, margins)

    np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=1e-12)
    assert np.isfinite(gradient).all()
    assert (hessian > 0.0).all()


def test_custom_objective_trains_real_lightgbm_on_diabetes() -> None:
    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    target = dataset.target
    objective = build_lightgbm_objective(LossSpec(loss_id=LossId.SQUARED_ERROR))
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=12,
        learning_rate=0.1,
        num_leaves=15,
        verbosity=-1,
        n_jobs=1,
        random_state=7,
    )

    model.fit(features[:350], target[:350])
    predictions = model.predict(features[350:])

    assert predictions.shape == (len(features) - 350,)
    assert np.isfinite(predictions).all()
