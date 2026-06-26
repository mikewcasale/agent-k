"""Tests for LightGBM loss-function evolution helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

import lightgbm as lgb
import numpy as np
import pytest

from agent_k.evolution.loss import (
    BUILTIN_OBJECTIVES,
    CUSTOM_OBJECTIVES,
    LossFunctionEvolver,
    LossGenome,
    build_lightgbm_custom_objective,
    build_lightgbm_objective_params,
)

__all__ = ()


_VALID_REGRESSION_KEYS: frozenset[str] = frozenset(
    {"objective", "alpha", "metric", "boosting", "boosting_type", "verbosity"}
)


@pytest.mark.parametrize("objective", BUILTIN_OBJECTIVES)
def test_build_objective_params_only_emits_valid_keys(objective: str) -> None:
    """build_lightgbm_objective_params must never emit unknown LightGBM keys."""
    genome = LossGenome(objective=objective, huber_delta=1.5, quantile_alpha=0.3)
    params = build_lightgbm_objective_params(genome)
    unknown = set(params) - _VALID_REGRESSION_KEYS
    assert not unknown, f"unexpected params for {objective}: {unknown}"


def test_build_objective_params_quantile_uses_alpha_not_quantile_alpha() -> None:
    genome = LossGenome(objective="quantile", quantile_alpha=0.27)
    params = build_lightgbm_objective_params(genome)
    assert params["objective"] == "quantile"
    assert params["alpha"] == pytest.approx(0.27)
    assert "quantile_alpha" not in params


def test_build_objective_params_huber_uses_alpha_not_huber_delta() -> None:
    """LightGBM's huber objective takes ``alpha``; ``huber_delta`` is unrecognised."""
    genome = LossGenome(objective="huber", huber_delta=2.5)
    params = build_lightgbm_objective_params(genome)
    assert params["objective"] == "huber"
    assert params["alpha"] == pytest.approx(2.5)
    assert "huber_delta" not in params


def test_build_objective_params_clips_quantile_alpha() -> None:
    genome = LossGenome(objective="quantile", quantile_alpha=2.0)
    params = build_lightgbm_objective_params(genome)
    assert 0.0 < params["alpha"] < 1.0


@pytest.mark.parametrize("objective", CUSTOM_OBJECTIVES)
def test_build_objective_params_custom_omits_builtin_objective(objective: str) -> None:
    """Custom objectives are attached as callables; params must not carry a string objective."""
    genome = LossGenome(objective=objective)
    params = build_lightgbm_objective_params(genome)
    assert "objective" not in params
    assert "metric" in params


def test_build_custom_objective_returns_none_for_pure_builtins() -> None:
    for objective in ("regression", "regression_l1"):
        genome = LossGenome(objective=objective)
        assert build_lightgbm_custom_objective(genome) is None


@pytest.mark.parametrize("objective", ["huber", "quantile", "asymmetric", "mae_rmse_blend"])
def test_custom_objective_returns_grad_hess_arrays(objective: str) -> None:
    genome = LossGenome(objective=objective)
    fobj = build_lightgbm_custom_objective(genome)
    assert fobj is not None
    y_pred = np.linspace(-1.0, 1.0, 11)
    y_true = np.zeros_like(y_pred)
    grad, hess = fobj(y_pred, y_true)
    assert grad.shape == y_pred.shape
    assert hess.shape == y_pred.shape
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))
    assert np.all(hess > 0.0), "Hessian must be strictly positive for LightGBM stability"


def test_asymmetric_objective_penalises_overprediction_more() -> None:
    genome = LossGenome(objective="asymmetric", asymmetric_weight=4.0)
    fobj = build_lightgbm_custom_objective(genome)
    assert fobj is not None
    y_pred = np.array([1.0, -1.0])
    y_true = np.array([0.0, 0.0])
    grad, _ = fobj(y_pred, y_true)
    assert grad[0] == pytest.approx(4.0)
    assert grad[1] == pytest.approx(-1.0)


def test_quantile_custom_objective_matches_pinball_gradient() -> None:
    genome = LossGenome(objective="quantile", quantile_alpha=0.2)
    fobj = build_lightgbm_custom_objective(genome)
    assert fobj is not None
    y_pred = np.array([0.0, 0.0])
    y_true = np.array([1.0, -1.0])
    grad, _ = fobj(y_pred, y_true)
    assert grad[0] == pytest.approx(-0.2)
    assert grad[1] == pytest.approx(0.8)


def test_huber_custom_objective_caps_gradient_at_delta() -> None:
    genome = LossGenome(objective="huber", huber_delta=0.5)
    fobj = build_lightgbm_custom_objective(genome)
    assert fobj is not None
    y_pred = np.array([0.1, 5.0, -5.0])
    y_true = np.zeros_like(y_pred)
    grad, _ = fobj(y_pred, y_true)
    assert grad[0] == pytest.approx(0.1)
    assert grad[1] == pytest.approx(0.5)
    assert grad[2] == pytest.approx(-0.5)


def test_blend_objective_is_pure_mae_at_blend_one() -> None:
    genome = LossGenome(objective="mae_rmse_blend", mae_rmse_blend=1.0)
    fobj = build_lightgbm_custom_objective(genome)
    assert fobj is not None
    y_pred = np.array([2.0, -3.0, 0.0])
    y_true = np.zeros_like(y_pred)
    grad, _ = fobj(y_pred, y_true)
    assert np.array_equal(grad, np.sign(y_pred))


def test_lightgbm_trains_with_custom_objective() -> None:
    """End-to-end smoke test: LightGBM accepts the custom callable in params."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((128, 4))
    target = features.sum(axis=1) + 0.1 * rng.standard_normal(128)
    dataset = lgb.Dataset(features, label=target)

    genome = LossGenome(objective="asymmetric", asymmetric_weight=1.5)
    params = build_lightgbm_objective_params(genome)
    fobj = build_lightgbm_custom_objective(genome)
    assert fobj is not None

    train_params = {**params, "objective": fobj, "verbosity": -1, "min_data_in_leaf": 5}
    model = lgb.train(train_params, dataset, num_boost_round=8)
    predictions = model.predict(features)
    assert predictions.shape == (128,)
    assert np.all(np.isfinite(predictions))


def test_loss_evolver_explores_custom_objectives() -> None:
    rng = random.Random(7)
    evolver = LossFunctionEvolver(fitness_fn=lambda _: 0.0, population_size=32, rng=rng)
    objectives = {individual.genome.objective for individual in evolver._population.individuals}
    assert objectives.issubset(set(BUILTIN_OBJECTIVES + CUSTOM_OBJECTIVES))
    assert objectives & set(CUSTOM_OBJECTIVES), "Evolver must explore custom objectives"
