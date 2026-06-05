"""Tests for the LightGBM loss function evolver.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

import lightgbm as lgb
import numpy as np
import pytest

from agent_k.evolution.loss import (
    LossFunctionEvolver,
    LossGenome,
    build_custom_objective_callable,
    build_lightgbm_objective_params,
)

__all__ = ()


def _toy_regression() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 4))
    coef = np.array([1.5, -0.7, 0.3, 2.0])
    y = X @ coef + rng.standard_normal(128) * 0.1
    return X, y


def test_build_lightgbm_objective_params_emits_only_native_keys() -> None:
    """`asymmetric_weight`/`mae_rmse_blend` are custom knobs, not LightGBM params."""
    genome = LossGenome(
        objective="regression", asymmetric_weight=1.5, mae_rmse_blend=0.3, huber_delta=1.0, quantile_alpha=0.5
    )
    params = build_lightgbm_objective_params(genome)

    assert params == {"objective": "regression"}
    assert "asymmetric_weight" not in params
    assert "mae_rmse_blend" not in params


def test_build_lightgbm_objective_params_huber_uses_alpha() -> None:
    """LightGBM exposes the Huber transition as `alpha`, not `huber_delta`."""
    genome = LossGenome(objective="huber", huber_delta=2.5)
    params = build_lightgbm_objective_params(genome)

    assert params["objective"] == "huber"
    assert params["alpha"] == 2.5
    assert "huber_delta" not in params


def test_build_lightgbm_objective_params_quantile_uses_alpha() -> None:
    genome = LossGenome(objective="quantile", quantile_alpha=0.85)
    params = build_lightgbm_objective_params(genome)

    assert params == {"objective": "quantile", "alpha": 0.85}


@pytest.mark.parametrize("objective", ["regression", "regression_l1", "huber", "quantile"])
def test_build_lightgbm_objective_params_trains_with_real_lightgbm(objective: str) -> None:
    """Returned params must be accepted by `lgb.train` without raising."""
    genome = LossGenome(objective=objective)
    params = build_lightgbm_objective_params(genome)
    params["verbose"] = -1

    X, y = _toy_regression()
    booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=5)

    preds = np.asarray(booster.predict(X))
    assert preds.shape == (X.shape[0],)
    assert np.isfinite(preds).all()


def test_build_custom_objective_callable_trains_under_lightgbm_4x() -> None:
    """The custom callable must be accepted via params["objective"] (the LightGBM 4.x API)."""
    genome = LossGenome(asymmetric_weight=1.0, mae_rmse_blend=1.0)
    callable_objective = build_custom_objective_callable(genome)

    X, y = _toy_regression()
    booster = lgb.train(
        {"objective": callable_objective, "verbose": -1, "learning_rate": 0.1},
        lgb.Dataset(X, label=y),
        num_boost_round=20,
    )

    preds = booster.predict(X)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    # With L2 settings the booster should clearly beat the constant mean predictor.
    baseline = float(np.sqrt(np.mean((y.mean() - y) ** 2)))
    assert rmse < baseline


def test_build_custom_objective_callable_asymmetric_penalty_shifts_predictions_down() -> None:
    """An asymmetric weight > 1 on positive residuals should bias predictions downward."""
    X, y = _toy_regression()

    fair = build_custom_objective_callable(LossGenome(asymmetric_weight=1.0, mae_rmse_blend=1.0))
    over_penalty = build_custom_objective_callable(LossGenome(asymmetric_weight=5.0, mae_rmse_blend=1.0))

    train_kwargs = {"verbose": -1, "learning_rate": 0.1}
    fair_booster = lgb.train({**train_kwargs, "objective": fair}, lgb.Dataset(X, label=y), num_boost_round=30)
    biased_booster = lgb.train({**train_kwargs, "objective": over_penalty}, lgb.Dataset(X, label=y), num_boost_round=30)

    fair_preds = np.asarray(fair_booster.predict(X))
    biased_preds = np.asarray(biased_booster.predict(X))
    assert biased_preds.mean() < fair_preds.mean()


def test_build_custom_objective_callable_blend_extremes_match_l2_and_l1_gradients() -> None:
    """blend=1.0 reduces to L2 gradients; blend=0.0 reduces to sign-based L1 gradients."""

    class _Dataset:
        def __init__(self, labels: np.ndarray) -> None:
            self._labels = labels

        def get_label(self) -> np.ndarray:
            return self._labels

    y_true = np.array([1.0, 2.0, -1.0, 0.5])
    y_pred = np.array([1.5, 1.0, -1.0, 1.0])
    dataset = _Dataset(y_true)

    l2 = build_custom_objective_callable(LossGenome(asymmetric_weight=1.0, mae_rmse_blend=1.0))
    l1 = build_custom_objective_callable(LossGenome(asymmetric_weight=1.0, mae_rmse_blend=0.0))

    grad_l2, hess_l2 = l2(y_pred, dataset)
    np.testing.assert_allclose(grad_l2, y_pred - y_true)
    np.testing.assert_allclose(hess_l2, np.ones_like(y_pred))

    grad_l1, hess_l1 = l1(y_pred, dataset)
    np.testing.assert_allclose(grad_l1, np.sign(y_pred - y_true))
    assert (hess_l1 > 0).all()


def test_loss_function_evolver_runs_real_lightgbm_fitness() -> None:
    """End-to-end smoke: the evolver should improve fitness on a real LightGBM-backed loop."""
    X, y = _toy_regression()
    dataset = lgb.Dataset(X, label=y)

    def fitness(genome: LossGenome) -> float:
        params = build_lightgbm_objective_params(genome)
        params["verbose"] = -1
        booster = lgb.train(params, dataset, num_boost_round=10)
        preds = booster.predict(X)
        return -float(np.sqrt(np.mean((preds - y) ** 2)))

    evolver = LossFunctionEvolver(fitness, population_size=4, rng=random.Random(0))
    result = evolver.evolve(generations=2)

    assert result["best_genome"] is not None
    assert result["best_fitness"] is not None
    assert result["best_fitness"] > -1e6
