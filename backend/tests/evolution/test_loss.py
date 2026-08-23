"""Tests for evolved LightGBM loss objectives.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from agent_k.evolution.loss import (
    LOSS_OBJECTIVES,
    LossGenome,
    build_custom_objective,
    build_lightgbm_objective_params,
    render_lightgbm_objective_source,
)

__all__ = ()

_LIGHTGBM_REGRESSION_PARAMS: frozenset[str] = frozenset({"objective", "alpha"})


def _sample_targets() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([1.0, -2.0, 3.5, 0.0, 10.0, 4.25])
    y_pred = np.array([1.5, -3.0, 3.5, 2.0, 4.0, 4.25])
    return y_true, y_pred


def _load_rendered(genome: LossGenome) -> Any:
    namespace: dict[str, Any] = {"np": np}
    exec(render_lightgbm_objective_source(genome), namespace)
    return namespace["custom_objective"]


def test_objective_params_only_emit_lightgbm_keys() -> None:
    for objective in LOSS_OBJECTIVES:
        genome = LossGenome(objective=objective, huber_delta=2.5, quantile_alpha=0.8, asymmetric_weight=1.7)
        params = build_lightgbm_objective_params(genome)

        assert set(params) <= _LIGHTGBM_REGRESSION_PARAMS
        assert params["objective"] == objective

    assert build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=2.5))["alpha"] == 2.5
    assert build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=0.8))["alpha"] == 0.8
    assert "alpha" not in build_lightgbm_objective_params(LossGenome(objective="regression"))


def test_squared_error_blend_matches_analytic_gradient() -> None:
    objective = build_custom_objective(LossGenome(mae_rmse_blend=0.0))
    y_true, y_pred = _sample_targets()

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, y_pred - y_true)
    assert np.allclose(hess, np.ones_like(y_true))


def test_pseudo_huber_gradient_saturates_at_delta() -> None:
    objective = build_custom_objective(LossGenome(objective="huber", huber_delta=2.0))
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1000.0, -1000.0])

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, [2.0, -2.0], atol=1e-3)
    assert np.all(hess > 0.0)


def test_quantile_gradient_matches_pinball_slopes() -> None:
    objective = build_custom_objective(LossGenome(objective="quantile", quantile_alpha=0.9, huber_delta=0.5))
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([500.0, -500.0])

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, [0.1, -0.9], atol=1e-4)
    assert np.all(hess > 0.0)


def test_asymmetric_weight_penalizes_over_prediction() -> None:
    objective = build_custom_objective(LossGenome(asymmetric_weight=3.0, mae_rmse_blend=0.0))
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, -1.0])

    grad, hess = objective(y_true, y_pred)

    assert np.allclose(grad, [3.0, -1.0])
    assert np.allclose(hess, [3.0, 1.0])


def test_hessian_is_strictly_positive_across_objectives() -> None:
    y_true, y_pred = _sample_targets()
    for objective_name in LOSS_OBJECTIVES:
        genome = LossGenome(objective=objective_name, asymmetric_weight=0.0, huber_delta=0.0, quantile_alpha=0.0)
        grad, hess = build_custom_objective(genome)(y_true, y_pred)

        assert np.all(np.isfinite(grad))
        assert np.all(hess > 0.0)


def test_rendered_source_matches_runtime_objective() -> None:
    y_true, y_pred = _sample_targets()
    for objective_name in LOSS_OBJECTIVES:
        genome = LossGenome(
            objective=objective_name, asymmetric_weight=1.6, huber_delta=1.3, mae_rmse_blend=0.35, quantile_alpha=0.65
        )
        expected_grad, expected_hess = build_custom_objective(genome)(y_true, y_pred)
        grad, hess = _load_rendered(genome)(y_true, y_pred)

        assert np.array_equal(grad, expected_grad)
        assert np.array_equal(hess, expected_hess)


def test_objective_accepts_dataset_argument_order() -> None:
    y_true, y_pred = _sample_targets()
    genome = LossGenome(objective="huber", asymmetric_weight=2.0, huber_delta=1.5)
    dataset = lgb.Dataset(np.zeros((y_true.size, 1)), label=y_true, free_raw_data=False).construct()

    expected_grad, expected_hess = build_custom_objective(genome)(y_true, y_pred)
    grad, hess = build_custom_objective(genome)(y_pred, dataset)

    assert np.array_equal(grad, expected_grad)
    assert np.array_equal(hess, expected_hess)


def test_lightgbm_trains_with_custom_objective() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(400, 3))
    labels = features @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=400)
    genome = LossGenome(objective="regression", asymmetric_weight=2.5, huber_delta=1.2, mae_rmse_blend=0.4)
    train_set = lgb.Dataset(features, label=labels - labels.mean())

    booster = lgb.train(
        {"objective": build_custom_objective(genome), "verbose": -1, "num_leaves": 7, "learning_rate": 0.15},
        train_set,
        num_boost_round=25,
    )
    predictions = booster.predict(features) + labels.mean()

    assert np.all(np.isfinite(predictions))
    assert float(np.mean(np.abs(predictions - labels))) < float(np.mean(np.abs(labels - labels.mean())))


def test_over_prediction_penalty_shifts_fit_downward() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(400, 2))
    labels = features @ np.array([1.0, -1.0]) + rng.normal(scale=0.5, size=400)
    centered = labels - labels.mean()
    params: dict[str, Any] = {"verbose": -1, "num_leaves": 7, "learning_rate": 0.15}

    symmetric = lgb.train(
        {**params, "objective": build_custom_objective(LossGenome(asymmetric_weight=1.0))},
        lgb.Dataset(features, label=centered),
        num_boost_round=30,
    ).predict(features)
    penalized = lgb.train(
        {**params, "objective": build_custom_objective(LossGenome(asymmetric_weight=6.0))},
        lgb.Dataset(features, label=centered),
        num_boost_round=30,
    ).predict(features)

    assert float(np.mean(penalized)) < float(np.mean(symmetric))
