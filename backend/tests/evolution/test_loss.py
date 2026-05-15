"""Tests for evolved LightGBM custom objective functions.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import lightgbm as lgb
import numpy as np

from agent_k.evolution.loss import LossGenome, build_lightgbm_objective, build_lightgbm_objective_params

__all__ = ()


def test_objective_params_emit_only_valid_keys() -> None:
    regression = build_lightgbm_objective_params(LossGenome(objective="regression"))
    assert regression == {"objective": "regression"}

    huber = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=2.5))
    assert huber == {"objective": "huber", "alpha": 2.5}

    quantile = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=0.9))
    assert quantile == {"objective": "quantile", "alpha": 0.9}


def test_pure_squared_error_blend_matches_residual() -> None:
    objective = build_lightgbm_objective(LossGenome(objective="regression", mae_rmse_blend=1.0, asymmetric_weight=1.0))
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.0, 3.0, 6.0])

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, y_pred - y_true)
    assert np.allclose(hess, 1.0)


def test_huber_blend_clips_gradient_at_delta() -> None:
    objective = build_lightgbm_objective(
        LossGenome(objective="huber", huber_delta=1.0, mae_rmse_blend=0.0, asymmetric_weight=1.0)
    )
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.5, 5.0, -8.0])

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, [0.5, 1.0, -1.0])
    # Inside the delta band the hessian is 1.0; outside it is floored above zero.
    assert hess[0] == 1.0
    assert np.all(hess > 0.0)


def test_asymmetric_weight_penalizes_over_prediction() -> None:
    objective = build_lightgbm_objective(LossGenome(objective="regression", mae_rmse_blend=1.0, asymmetric_weight=3.0))
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([2.0, -2.0])

    grad, hess = objective(y_true, y_pred)

    # Over-prediction (residual > 0) gradient and hessian scale by the weight.
    assert grad[0] == 6.0
    assert grad[1] == -2.0
    assert hess[0] == 3.0
    assert hess[1] == 1.0


def test_quantile_objective_uses_pinball_gradient() -> None:
    objective = build_lightgbm_objective(LossGenome(objective="quantile", quantile_alpha=0.8, asymmetric_weight=1.0))
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, -1.0])

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, [1.0 - 0.8, -0.8])
    assert np.allclose(hess, 1.0)


def test_objective_trains_a_lightgbm_model() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(256, 4))
    target = features @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(scale=0.1, size=256)

    objective = build_lightgbm_objective(
        LossGenome(objective="huber", huber_delta=1.2, mae_rmse_blend=0.4, asymmetric_weight=1.5)
    )
    train_set = lgb.Dataset(features, label=target)
    booster = lgb.train(
        params={"objective": objective, "num_leaves": 7, "verbosity": -1, "min_data_in_leaf": 5},
        train_set=train_set,
        num_boost_round=10,
    )

    predictions = booster.predict(features)
    assert predictions.shape == (256,)
    assert np.all(np.isfinite(predictions))
