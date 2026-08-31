"""Tests for LightGBM objective families and custom loss construction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from agent_k.evolution.loss import (
    LIGHTGBM_OBJECTIVE_FAMILIES,
    LossFunctionEvolver,
    LossGenome,
    alternative_objectives,
    build_custom_objective,
    build_lightgbm_objective_params,
    canonical_objective,
    family_objectives,
    objective_family,
)

__all__ = ()

lgb = pytest.importorskip("lightgbm")


class TestObjectiveFamilies:
    """Tests for objective family resolution."""

    @pytest.mark.parametrize(
        ("objective", "family"),
        [
            ("regression", "regression"),
            ("l2", "regression"),
            ("MAE", "regression"),
            ('"huber"', "regression"),
            ("binary", "binary"),
            ("xentropy", "binary"),
            ("multiclass", "multiclass"),
            ("ovr", "multiclass"),
            ("lambdarank", "ranking"),
        ],
    )
    def test_objective_family_resolves_aliases(self, objective: str, family: str) -> None:
        """Aliases, quoting, and casing should resolve to the documented family."""
        assert objective_family(objective) == family

    @pytest.mark.parametrize("objective", ["poisson", "tweedie", "mape", "custom_objective", ""])
    def test_unknown_objectives_have_no_family(self, objective: str) -> None:
        """Narrow-domain and callable objectives stay unclassified."""
        assert objective_family(objective) is None
        assert canonical_objective(objective) is None
        assert alternative_objectives(objective) == ()

    def test_alternatives_stay_inside_family(self) -> None:
        """Every alternative must belong to the same family as the original objective."""
        for family, objectives in LIGHTGBM_OBJECTIVE_FAMILIES.items():
            for objective in objectives:
                alternatives = alternative_objectives(objective)
                assert objective not in alternatives
                assert all(objective_family(candidate) == family for candidate in alternatives)

    def test_alternatives_exclude_canonical_form_of_alias(self) -> None:
        """An alias must not be swapped for its own canonical spelling."""
        assert "regression_l1" not in alternative_objectives("mae")
        assert "regression" not in alternative_objectives("l2")

    def test_family_objectives_unknown_family_is_empty(self) -> None:
        """Unknown families degrade to an empty pool instead of raising."""
        assert family_objectives("survival") == ()


class TestObjectiveParams:
    """Tests for LightGBM parameter construction."""

    @pytest.mark.parametrize(
        ("genome", "expected"),
        [
            (LossGenome(objective="regression"), {"objective": "regression"}),
            (LossGenome(objective="quantile", quantile_alpha=0.3), {"objective": "quantile", "alpha": 0.3}),
            (LossGenome(objective="huber", huber_delta=2.5), {"objective": "huber", "alpha": 2.5}),
            (LossGenome(objective="fair", fair_c=1.5), {"objective": "fair", "fair_c": 1.5}),
            (
                LossGenome(objective="multiclass", family="multiclass", num_class=4),
                {"objective": "multiclass", "num_class": 4},
            ),
            (LossGenome(objective="mse"), {"objective": "regression"}),
        ],
    )
    def test_params_are_lightgbm_native(self, genome: LossGenome, expected: dict[str, object]) -> None:
        """Only parameters LightGBM actually reads should be emitted."""
        assert build_lightgbm_objective_params(genome) == expected

    def test_huber_alpha_changes_the_fitted_model(self) -> None:
        """The emitted huber parameter must have a measurable effect on training."""
        features, target = load_diabetes(return_X_y=True, as_frame=True)
        dataset = lgb.Dataset(features, label=target)

        def fit(delta: float) -> np.ndarray:
            params = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=delta))
            booster = lgb.train({**params, "verbosity": -1, "seed": 0}, dataset, num_boost_round=10)
            return np.asarray(booster.predict(features))

        assert not np.allclose(fit(0.5), fit(50.0))


class TestCustomObjective:
    """Tests for the evolved custom objective callables."""

    def test_regression_gradient_matches_finite_differences(self) -> None:
        """Analytic gradient and hessian must agree with numeric derivatives of the loss."""
        genome = LossGenome(objective="regression", mae_rmse_blend=0.4, huber_delta=1.5, asymmetric_weight=1.8)
        objective = build_custom_objective(genome)
        y_true = np.array([-2.0, 0.5, 1.0, 4.0])
        y_pred = np.array([1.0, -0.5, 1.25, 2.0])
        grad, hess = objective(y_pred, _FakeDataset(y_true))

        step = 1e-5
        for index in range(len(y_true)):
            shifted_up = y_pred.copy()
            shifted_down = y_pred.copy()
            shifted_up[index] += step
            shifted_down[index] -= step
            numeric_grad = (
                _regression_loss(shifted_up, y_true, genome) - _regression_loss(shifted_down, y_true, genome)
            ) / (2 * step)
            numeric_hess = (
                _regression_loss(shifted_up, y_true, genome)
                - 2 * _regression_loss(y_pred, y_true, genome)
                + _regression_loss(shifted_down, y_true, genome)
            ) / step**2
            assert grad[index] == pytest.approx(numeric_grad, abs=1e-5)
            assert hess[index] == pytest.approx(numeric_hess, abs=1e-2)

    def test_regression_asymmetry_penalises_under_prediction(self) -> None:
        """A weight above one must push predictions upward relative to the symmetric loss."""
        features, target = load_diabetes(return_X_y=True, as_frame=True)
        dataset = lgb.Dataset(features, label=target)

        def fit(weight: float) -> np.ndarray:
            objective = build_custom_objective(LossGenome(asymmetric_weight=weight, mae_rmse_blend=0.0))
            booster = lgb.train(
                {"objective": objective, "verbosity": -1, "seed": 0, "learning_rate": 0.1}, dataset, num_boost_round=40
            )
            return np.asarray(booster.predict(features))

        assert fit(3.0).mean() > fit(1.0).mean()

    def test_regression_objective_trains_a_usable_model(self) -> None:
        """A blended custom objective must fit real data better than the constant baseline."""
        features, target = load_diabetes(return_X_y=True, as_frame=True)
        x_train, x_valid, y_train, y_valid = train_test_split(features, target, test_size=0.25, random_state=0)
        offset = float(np.mean(y_train))
        objective = build_custom_objective(LossGenome(mae_rmse_blend=0.6, huber_delta=20.0))
        booster = lgb.train(
            {"objective": objective, "verbosity": -1, "seed": 0, "learning_rate": 0.1},
            lgb.Dataset(x_train, label=y_train, init_score=np.full(len(y_train), offset)),
            num_boost_round=120,
        )
        predictions = np.asarray(booster.predict(x_valid)) + offset
        baseline = np.full(len(y_valid), offset)
        assert mean_absolute_error(y_valid, predictions) < mean_absolute_error(y_valid, baseline)

    def test_binary_objective_trains_a_usable_model(self) -> None:
        """The binary custom objective must separate classes on real data."""
        features, target = load_breast_cancer(return_X_y=True, as_frame=True)
        x_train, x_valid, y_train, y_valid = train_test_split(
            features, target, test_size=0.25, random_state=0, stratify=target
        )
        objective = build_custom_objective(LossGenome(objective="binary", family="binary", asymmetric_weight=1.5))
        booster = lgb.train(
            {"objective": objective, "verbosity": -1, "seed": 0, "learning_rate": 0.1},
            lgb.Dataset(x_train, label=y_train),
            num_boost_round=60,
        )
        assert roc_auc_score(y_valid, booster.predict(x_valid)) > 0.95

    def test_multiclass_objective_trains_a_usable_model(self) -> None:
        """The multiclass custom objective must produce per-class scores that classify correctly."""
        features, target = load_wine(return_X_y=True, as_frame=True)
        x_train, x_valid, y_train, y_valid = train_test_split(
            features, target, test_size=0.3, random_state=0, stratify=target
        )
        genome = LossGenome(objective="multiclass", family="multiclass", num_class=3, asymmetric_weight=1.5)
        booster = lgb.train(
            {
                "objective": build_custom_objective(genome),
                "num_class": 3,
                "verbosity": -1,
                "seed": 0,
                "learning_rate": 0.1,
            },
            lgb.Dataset(x_train, label=y_train),
            num_boost_round=60,
        )
        scores = np.asarray(booster.predict(x_valid))
        assert scores.shape == (len(y_valid), 3)
        assert accuracy_score(y_valid, scores.argmax(axis=1)) > 0.9

    def test_ranking_family_has_no_custom_formulation(self) -> None:
        """Families without a custom formulation must fail loudly, not silently."""
        with pytest.raises(ValueError, match="No custom objective formulation"):
            build_custom_objective(LossGenome(objective="lambdarank", family="ranking"))


class TestLossFunctionEvolver:
    """Tests for family-scoped loss evolution."""

    @pytest.mark.parametrize("family", ["regression", "binary", "multiclass", "ranking"])
    def test_population_stays_inside_family(self, family: str) -> None:
        """Every evolved genome must keep an objective valid for its problem family."""
        evolver = LossFunctionEvolver(
            lambda genome: float(genome.mae_rmse_blend),
            family=family,  # type: ignore[arg-type]
            population_size=8,
            rng=random.Random(7),
        )
        result = evolver.evolve(generations=3)
        population = result["population"]
        assert population.individuals
        for individual in population.individuals:
            assert individual.genome.family == family
            assert individual.genome.objective in family_objectives(family)

    def test_unknown_family_is_rejected(self) -> None:
        """An unsupported family must raise instead of silently evolving regression losses."""
        with pytest.raises(ValueError, match="Unknown LightGBM objective family"):
            LossFunctionEvolver(lambda genome: 0.0, family="survival")  # type: ignore[arg-type]


class _FakeDataset:
    """Minimal stand-in exposing the ``get_label`` method LightGBM objectives call."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    def get_label(self) -> np.ndarray:
        """Return the stored labels."""
        return self._labels


def _regression_loss(y_pred: np.ndarray, y_true: np.ndarray, genome: LossGenome) -> float:
    residual = y_pred - y_true
    delta = genome.huber_delta
    squared = 0.5 * residual**2
    pseudo_huber = delta**2 * (np.sqrt(1.0 + (residual / delta) ** 2) - 1.0)
    blended = (1.0 - genome.mae_rmse_blend) * squared + genome.mae_rmse_blend * pseudo_huber
    weights = np.where(residual < 0.0, genome.asymmetric_weight, 1.0)
    return float(np.sum(blended * weights))
