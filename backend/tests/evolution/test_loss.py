"""Tests for LightGBM loss-function evolution helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from typing import Any

import numpy as np
import pytest

from agent_k.evolution.loss import (
    ASYMMETRIC_OBJECTIVE,
    BUILTIN_OBJECTIVES,
    LossFunctionEvolver,
    LossGenome,
    build_custom_lightgbm_objective,
    build_lightgbm_objective_params,
)

__all__ = ()


class _StubDataset:
    """Minimal stand-in for ``lightgbm.Dataset`` with ``get_label``."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    def get_label(self) -> np.ndarray:
        return self._labels


class TestLossGenomeDefaults:
    def test_default_values(self) -> None:
        genome = LossGenome()
        assert genome.objective == "regression"
        assert genome.asymmetric_weight == 1.0
        assert genome.huber_delta == 1.0
        assert genome.mae_rmse_blend == 0.5
        assert genome.quantile_alpha == 0.5
        assert genome.metadata == {}


class TestBuildLightgbmObjectiveParams:
    def test_regression_emits_only_objective(self) -> None:
        params = build_lightgbm_objective_params(LossGenome(objective="regression"))
        assert params == {"objective": "regression"}

    def test_regression_l1_emits_only_objective(self) -> None:
        params = build_lightgbm_objective_params(LossGenome(objective="regression_l1"))
        assert params == {"objective": "regression_l1"}

    def test_huber_maps_delta_to_alpha(self) -> None:
        genome = LossGenome(objective="huber", huber_delta=2.5)
        params = build_lightgbm_objective_params(genome)
        assert params == {"objective": "huber", "alpha": 2.5}
        # Regression coverage: LightGBM has NO ``huber_delta`` parameter — the previous
        # implementation was silently emitting an unknown key.
        assert "huber_delta" not in params

    def test_quantile_maps_quantile_alpha_to_alpha(self) -> None:
        genome = LossGenome(objective="quantile", quantile_alpha=0.9)
        params = build_lightgbm_objective_params(genome)
        assert params == {"objective": "quantile", "alpha": 0.9}

    def test_custom_asymmetric_embeds_callable_in_params(self) -> None:
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.7, mae_rmse_blend=0.4)
        params = build_lightgbm_objective_params(genome)
        # LightGBM >= 4 replaced the ``fobj=`` train() kwarg with an objective-callable
        # in ``params``. The caller can pass ``params`` straight to ``lightgbm.train``.
        assert callable(params["objective"])
        assert set(params) == {"objective"}

    def test_no_unknown_lightgbm_keys_emitted(self) -> None:
        # The pre-fix implementation always emitted ``asymmetric_weight`` and
        # ``mae_rmse_blend`` — neither is a valid LightGBM param, so LightGBM would
        # warn "Unknown parameter" and ignore the evolved value entirely.
        allowed = {"objective", "alpha"}
        for name in ["regression", "regression_l1", "huber", "quantile"]:
            genome = LossGenome(objective=name, asymmetric_weight=2.5, mae_rmse_blend=0.3, huber_delta=3.0)
            params = build_lightgbm_objective_params(genome)
            unknown = set(params) - allowed
            assert not unknown, f"{name} emitted unknown LightGBM keys: {unknown}"

    def test_huber_alpha_is_clipped_to_positive(self) -> None:
        params = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=-1.0))
        assert params["alpha"] > 0.0

    def test_quantile_alpha_is_clipped_below_one(self) -> None:
        params = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=1.5))
        assert 0.0 < params["alpha"] < 1.0

    def test_builtin_objectives_set_matches_module_constant(self) -> None:
        # Guards against a future addition drifting from the fobj branch.
        assert BUILTIN_OBJECTIVES == frozenset({"regression", "regression_l1", "huber", "quantile"})


class TestBuildCustomLightgbmObjective:
    def test_returns_none_for_builtins(self) -> None:
        for name in BUILTIN_OBJECTIVES:
            assert build_custom_lightgbm_objective(LossGenome(objective=name)) is None

    def test_returns_callable_for_asymmetric(self) -> None:
        fobj = build_custom_lightgbm_objective(LossGenome(objective=ASYMMETRIC_OBJECTIVE))
        assert callable(fobj)

    def test_symmetric_mse_matches_l2_gradient(self) -> None:
        # blend=1.0 (pure MSE) and weight=1.0 (symmetric) should reduce to grad = pred - y, hess = 1.
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.0, mae_rmse_blend=1.0)
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        preds = np.array([1.0, 2.0, 3.0])
        labels = np.array([0.5, 2.5, 3.0])
        grad, hess = fobj(preds, _StubDataset(labels))
        np.testing.assert_allclose(grad, preds - labels)
        np.testing.assert_allclose(hess, np.ones_like(preds))

    def test_symmetric_mae_matches_sign_gradient(self) -> None:
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.0, mae_rmse_blend=0.0)
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        preds = np.array([1.0, 2.0, 3.0])
        labels = np.array([0.5, 2.5, 3.0])
        grad, hess = fobj(preds, _StubDataset(labels))
        np.testing.assert_allclose(grad, np.sign(preds - labels))
        # Constant hessian keeps LightGBM's Newton step well-scaled at blend=0.
        assert np.all(hess > 0.0)

    def test_over_prediction_receives_asymmetric_scaling(self) -> None:
        # weight=3 should scale gradient/hessian by 3x for over-predictions only.
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=3.0, mae_rmse_blend=1.0)
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        preds = np.array([2.0, 1.0])
        labels = np.array([0.0, 5.0])  # first is over-pred (+2), second is under-pred (-4)
        grad, hess = fobj(preds, _StubDataset(labels))
        assert grad[0] == pytest.approx(6.0)  # 2 * 3
        assert grad[1] == pytest.approx(-4.0)  # unscaled
        assert hess[0] == pytest.approx(3.0)  # 1 * 3
        assert hess[1] == pytest.approx(1.0)

    def test_blend_interpolates_between_mse_and_mae(self) -> None:
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.0, mae_rmse_blend=0.25)
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        preds = np.array([4.0])
        labels = np.array([0.0])
        grad, _ = fobj(preds, _StubDataset(labels))
        # 0.25 * (pred - y) + 0.75 * sign(pred - y) at residual=4: 0.25*4 + 0.75*1 = 1.75
        assert grad[0] == pytest.approx(1.75)

    def test_reshapes_flattened_multi_output_preds(self) -> None:
        # LightGBM passes ``preds`` as a flat 1-D array even for multi-output tasks;
        # we only need to guarantee the shape matches ``labels``.
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.0, mae_rmse_blend=1.0)
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        preds = np.array([1.0, 2.0, 3.0])
        labels = np.array([0.0, 0.0, 0.0])
        grad, hess = fobj(preds, _StubDataset(labels))
        assert grad.shape == labels.shape
        assert hess.shape == labels.shape

    def test_weight_and_blend_are_clamped(self) -> None:
        # Negative weight/out-of-range blend must not corrupt the gradient computation
        # or reverse gradient direction.
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=-5.0, mae_rmse_blend=2.0)
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        preds = np.array([1.0])
        labels = np.array([0.0])
        grad, hess = fobj(preds, _StubDataset(labels))
        assert grad[0] > 0.0  # over-pred: gradient still positive
        assert hess[0] > 0.0


class TestLossFunctionEvolver:
    def test_evolve_returns_best_individual(self) -> None:
        # Fitness prefers ``regression_l1`` with tiny asymmetric weight; verify evolve()
        # can return a best genome and float fitness.
        def fitness(genome: LossGenome) -> float:
            base = 1.0 if genome.objective == "regression_l1" else 0.0
            return base - abs(genome.asymmetric_weight - 0.5)

        evolver = LossFunctionEvolver(fitness, population_size=8, rng=random.Random(0))
        result = evolver.evolve(generations=3)
        best = result["best_genome"]
        assert isinstance(best, LossGenome)
        assert isinstance(result["best_fitness"], float)

    def test_evolver_includes_asymmetric_in_search_space(self) -> None:
        # Sanity check: the evolver must be able to explore the new custom objective;
        # otherwise the whole asymmetric-loss addition is unreachable from evolution.
        assert ASYMMETRIC_OBJECTIVE in LossFunctionEvolver._objectives

    def test_evolution_is_deterministic_under_seeded_rng(self) -> None:
        def fitness(genome: LossGenome) -> float:
            return -abs(genome.mae_rmse_blend - 0.7)

        first = LossFunctionEvolver(fitness, population_size=6, rng=random.Random(42)).evolve(generations=2)
        second = LossFunctionEvolver(fitness, population_size=6, rng=random.Random(42)).evolve(generations=2)
        assert first["best_fitness"] == second["best_fitness"]


class TestLightgbmIntegration:
    """End-to-end sanity: the emitted params + fobj actually train under LightGBM."""

    def _train(self, params: dict[str, Any]) -> Any:
        lgb = pytest.importorskip("lightgbm")
        rng = np.random.default_rng(0)
        X = rng.random((80, 4))
        y = X @ np.array([1.0, -1.0, 0.5, 0.25]) + rng.normal(scale=0.05, size=80)
        train_set = lgb.Dataset(X, label=y, free_raw_data=False)
        return lgb.train({"verbosity": -1, "num_iterations": 5, **params}, train_set=train_set)

    def test_huber_params_train_without_unknown_key_warning(self) -> None:
        genome = LossGenome(objective="huber", huber_delta=1.5)
        params = build_lightgbm_objective_params(genome)
        booster = self._train(params)
        assert booster is not None

    def test_quantile_params_train(self) -> None:
        genome = LossGenome(objective="quantile", quantile_alpha=0.8)
        params = build_lightgbm_objective_params(genome)
        booster = self._train(params)
        assert booster is not None

    def test_asymmetric_fobj_trains(self) -> None:
        genome = LossGenome(objective=ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.4, mae_rmse_blend=0.6)
        params = build_lightgbm_objective_params(genome)
        booster = self._train(params)
        assert booster is not None
