"""Tests for LightGBM loss function evolution.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import numpy as np
import pytest

from agent_k.evolution.loss import (
    LossFunctionEvolver,
    LossGenome,
    build_lightgbm_objective_params,
    make_asymmetric_objective,
    make_blended_objective,
)

__all__ = ()


@dataclass(slots=True)
class _LabelHolder:
    """Minimal stand-in for lightgbm.Dataset.get_label() in gradient math tests."""

    labels: np.ndarray

    def get_label(self) -> np.ndarray:
        return self.labels


def _regression_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((300, 3))
    y = x @ np.array([1.5, -0.8, 0.4]) + 0.05 * rng.standard_normal(300)
    return x, y


def test_build_params_regression_returns_string_objective() -> None:
    params = build_lightgbm_objective_params(LossGenome(objective="regression"))
    assert params == {"objective": "regression"}


def test_build_params_regression_l1_returns_string_objective() -> None:
    params = build_lightgbm_objective_params(LossGenome(objective="regression_l1"))
    assert params == {"objective": "regression_l1"}


def test_build_params_huber_maps_to_native_alpha() -> None:
    params = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=2.5))
    # LightGBM's Huber transition point parameter is named `alpha`, not
    # `huber_delta` (which LightGBM silently ignores).
    assert params == {"objective": "huber", "alpha": 2.5}


def test_build_params_quantile_maps_to_native_alpha() -> None:
    params = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=0.9))
    assert params == {"objective": "quantile", "alpha": 0.9}


def test_build_params_asymmetric_returns_callable() -> None:
    params = build_lightgbm_objective_params(LossGenome(objective="asymmetric", asymmetric_weight=2.5))
    objective = params["objective"]
    assert callable(objective)
    assert set(params) == {"objective"}


def test_build_params_blended_returns_callable() -> None:
    params = build_lightgbm_objective_params(LossGenome(objective="blended", mae_rmse_blend=0.4))
    objective = params["objective"]
    assert callable(objective)
    assert set(params) == {"objective"}


def test_build_params_unknown_objective_raises() -> None:
    with pytest.raises(ValueError, match="Unknown loss objective"):
        build_lightgbm_objective_params(LossGenome(objective="does_not_exist"))


def test_asymmetric_objective_penalizes_under_prediction() -> None:
    fn = make_asymmetric_objective(weight=3.0)
    _, y = _regression_dataset()
    dataset = _LabelHolder(y)

    # Under-prediction (pred < label) should carry weighted gradient.
    preds_under = y - 1.0
    grad_under, hess_under = fn(preds_under, cast(lgb.Dataset, dataset))
    # residual = -1, weight = 3 → grad = 2 * -1 * 3 = -6, hess = 2 * 3 = 6
    np.testing.assert_allclose(grad_under, np.full_like(y, -6.0))
    np.testing.assert_allclose(hess_under, np.full_like(y, 6.0))

    # Over-prediction (pred > label) uses unit weight.
    preds_over = y + 1.0
    grad_over, hess_over = fn(preds_over, cast(lgb.Dataset, dataset))
    np.testing.assert_allclose(grad_over, np.full_like(y, 2.0))
    np.testing.assert_allclose(hess_over, np.full_like(y, 2.0))


def test_asymmetric_objective_symmetric_at_unit_weight() -> None:
    fn = make_asymmetric_objective(weight=1.0)
    residual = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    labels = np.zeros_like(residual)
    dataset = _LabelHolder(labels)
    grad, hess = fn(residual, cast(lgb.Dataset, dataset))
    np.testing.assert_allclose(grad, 2.0 * residual)
    np.testing.assert_allclose(hess, np.full_like(residual, 2.0))


def test_asymmetric_objective_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="asymmetric weight must be positive"):
        make_asymmetric_objective(weight=0.0)
    with pytest.raises(ValueError, match="asymmetric weight must be positive"):
        make_asymmetric_objective(weight=-1.0)


def test_blended_objective_recovers_mse_at_blend_zero() -> None:
    fn = make_blended_objective(blend=0.0)
    residual = np.array([0.5, -0.5, 1.5, -1.5])
    labels = np.zeros_like(residual)
    dataset = _LabelHolder(labels)
    grad, hess = fn(residual, cast(lgb.Dataset, dataset))
    np.testing.assert_allclose(grad, 2.0 * residual)
    np.testing.assert_allclose(hess, np.full_like(residual, 2.0))


def test_blended_objective_recovers_mae_at_blend_one() -> None:
    fn = make_blended_objective(blend=1.0)
    residual = np.array([-3.0, -0.1, 0.4, 5.0])
    labels = np.zeros_like(residual)
    dataset = _LabelHolder(labels)
    grad, hess = fn(residual, cast(lgb.Dataset, dataset))
    np.testing.assert_allclose(grad, np.sign(residual))
    np.testing.assert_allclose(hess, np.full_like(residual, 1.0))


def test_blended_objective_interpolates_between_l1_l2() -> None:
    fn = make_blended_objective(blend=0.25)
    residual = np.array([-2.0, -0.5, 0.5, 2.0])
    labels = np.zeros_like(residual)
    dataset = _LabelHolder(labels)
    grad, hess = fn(residual, cast(lgb.Dataset, dataset))
    expected_grad = 0.25 * np.sign(residual) + 0.75 * 2.0 * residual
    np.testing.assert_allclose(grad, expected_grad)
    # Hess = blend + 2 * (1 - blend) = 0.25 + 1.5 = 1.75
    np.testing.assert_allclose(hess, np.full_like(residual, 1.75))


def test_blended_objective_rejects_out_of_range_blend() -> None:
    with pytest.raises(ValueError, match=r"blend must be in"):
        make_blended_objective(blend=-0.1)
    with pytest.raises(ValueError, match=r"blend must be in"):
        make_blended_objective(blend=1.1)


def test_huber_alpha_actually_affects_training() -> None:
    # Regression guard: previously the genome emitted `huber_delta`, which
    # LightGBM silently discards — two different values produced identical
    # boosters. With the fix, distinct alphas must produce distinct trees.
    x, y = _regression_dataset()
    train_set = lgb.Dataset(x, label=y)

    params_small = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=0.1))
    params_large = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=5.0))

    common = {"verbose": -1, "num_leaves": 5, "seed": 1, "deterministic": True}
    booster_small = lgb.train({**common, **params_small}, train_set, num_boost_round=10)
    booster_large = lgb.train({**common, **params_large}, train_set, num_boost_round=10)

    pred_small = np.asarray(booster_small.predict(x))
    pred_large = np.asarray(booster_large.predict(x))
    assert not np.allclose(pred_small, pred_large), (
        "huber_delta must map to LightGBM's `alpha` and change the model, otherwise the genome parameter has no effect."
    )


def test_asymmetric_objective_trains_end_to_end() -> None:
    x, y = _regression_dataset(seed=1)
    train_set = lgb.Dataset(x, label=y)
    params = build_lightgbm_objective_params(LossGenome(objective="asymmetric", asymmetric_weight=3.0))
    booster = lgb.train({**params, "verbose": -1, "num_leaves": 5, "seed": 1}, train_set, num_boost_round=15)
    preds = np.asarray(booster.predict(x))
    assert preds.shape == y.shape
    # Weighting under-predictions more heavily should bias the fit upward
    # relative to symmetric training.
    baseline = np.asarray(
        lgb.train(
            {"objective": "regression", "verbose": -1, "num_leaves": 5, "seed": 1}, train_set, num_boost_round=15
        ).predict(x)
    )
    assert preds.mean() > baseline.mean()


def test_blended_objective_trains_end_to_end() -> None:
    x, y = _regression_dataset(seed=2)
    train_set = lgb.Dataset(x, label=y)
    params = build_lightgbm_objective_params(LossGenome(objective="blended", mae_rmse_blend=0.5))
    booster = lgb.train({**params, "verbose": -1, "num_leaves": 5, "seed": 1}, train_set, num_boost_round=15)
    preds = np.asarray(booster.predict(x))
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_evolver_search_space_includes_custom_objectives() -> None:
    assert set(LossFunctionEvolver._objectives) >= {"asymmetric", "blended"}


def test_evolver_runs_across_all_objectives() -> None:
    # Fitness prefers whichever genome yields the smallest residual on a
    # tiny dataset — exercises the full generation loop over every
    # objective the evolver can sample, guarding against callable-vs-str
    # dispatch regressions inside build_lightgbm_objective_params.
    x, y = _regression_dataset(seed=3)
    train_set = lgb.Dataset(x, label=y)

    def fitness(genome: LossGenome) -> float:
        params = build_lightgbm_objective_params(genome)
        booster = lgb.train({**params, "verbose": -1, "num_leaves": 5, "seed": 1}, train_set, num_boost_round=5)
        return -float(np.mean(np.square(np.asarray(booster.predict(x)) - y)))

    evolver = LossFunctionEvolver(fitness_fn=fitness, population_size=4, rng=random.Random(0))
    result = evolver.evolve(generations=1)
    best = result["best_genome"]
    assert isinstance(best, LossGenome)
    assert best.objective in LossFunctionEvolver._objectives
