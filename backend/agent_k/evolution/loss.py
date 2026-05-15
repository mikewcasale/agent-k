"""Loss function evolution for LightGBM objectives.

@notice: |
    Loss function evolution for LightGBM objectives.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evolution.loss
    provides:
        - agent_k.evolution.loss
    pattern: loss-functions

@agent-guidance:
    do:
        - "Use agent_k.evolution.loss as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from agent_k.evolution.framework import FitnessFn, Individual, Population

type LossObjective = str
"""LightGBM objective identifiers."""

type GradHess = tuple[NDArray[np.float64], NDArray[np.float64]]
"""Gradient and hessian arrays returned by a LightGBM custom objective."""

type LightGBMObjective = Callable[[Any, Any], GradHess]
"""Custom objective callable accepted by ``lightgbm.train`` via ``params['objective']``."""

__all__ = (
    "LightGBMObjective",
    "LossFunctionEvolver",
    "LossGenome",
    "build_lightgbm_objective",
    "build_lightgbm_objective_params",
)

_MIN_HESSIAN: Final[float] = 1e-6
"""Lower bound on the hessian so LightGBM never divides by zero."""

_QUANTILE_HESSIAN: Final[float] = 1.0
"""Constant hessian for the pinball loss, mirroring LightGBM's built-in quantile objective."""


@dataclass(slots=True)
class LossGenome:
    """Genome for evolving custom objective functions.

    @notice: |
        Genome for evolving custom objective functions.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: genome-model
            rationale: "Dataclass for LightGBM loss function parameters."
    """

    objective: LossObjective = "regression"
    asymmetric_weight: float = 1.0
    huber_delta: float = 1.0
    mae_rmse_blend: float = 0.5
    quantile_alpha: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class LossFunctionEvolver:
    """Evolve LightGBM loss function parameters via genetic search.

    @notice: |
        Evolve LightGBM loss function parameters via genetic search.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: evolver
            rationale: "Coordinates evolutionary search over loss function genomes."
    """

    _objectives: tuple[str, ...] = ("regression", "regression_l1", "huber", "quantile")

    def __init__(
        self, fitness_fn: FitnessFn[LossGenome], *, population_size: int = 16, rng: random.Random | None = None
    ) -> None:
        self._fitness_fn = fitness_fn
        self._population_size = population_size
        self._rng = rng or random.Random()
        self._population = self._initialize_population()

    def evolve(self, *, generations: int = 5) -> dict[str, Any]:
        """Run evolution for a fixed number of generations."""
        for _ in range(generations):
            self._population.evolve_generation(
                fitness_fn=self._fitness_fn, mutation_fn=self._mutate, crossover_fn=self._crossover, mutation_rate=0.3
            )
        best = self._population.best()
        return {
            "best_genome": best.genome if best else None,
            "best_fitness": best.fitness if best else None,
            "population": self._population,
        }

    def _initialize_population(self) -> Population[LossGenome]:
        individuals = [Individual(self._random_genome()) for _ in range(self._population_size)]
        return Population(individuals, rng=self._rng)

    def _random_genome(self) -> LossGenome:
        return LossGenome(
            objective=self._rng.choice(self._objectives),
            asymmetric_weight=self._rng.uniform(0.5, 2.0),
            huber_delta=self._rng.uniform(0.5, 5.0),
            mae_rmse_blend=self._rng.uniform(0.0, 1.0),
            quantile_alpha=self._rng.uniform(0.05, 0.95),
        )

    def _mutate(self, genome: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=rng.choice(self._objectives) if rng.random() < 0.2 else genome.objective,
            asymmetric_weight=_clip(genome.asymmetric_weight + rng.uniform(-0.2, 0.2), 0.1, 3.0),
            huber_delta=_clip(genome.huber_delta + rng.uniform(-0.3, 0.3), 0.1, 10.0),
            mae_rmse_blend=_clip(genome.mae_rmse_blend + rng.uniform(-0.15, 0.15), 0.0, 1.0),
            quantile_alpha=_clip(genome.quantile_alpha + rng.uniform(-0.1, 0.1), 0.01, 0.99),
            metadata=dict(genome.metadata),
        )

    def _crossover(self, parent_a: LossGenome, parent_b: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=parent_a.objective if rng.random() < 0.5 else parent_b.objective,
            asymmetric_weight=_pick(parent_a.asymmetric_weight, parent_b.asymmetric_weight, rng),
            huber_delta=_pick(parent_a.huber_delta, parent_b.huber_delta, rng),
            mae_rmse_blend=_pick(parent_a.mae_rmse_blend, parent_b.mae_rmse_blend, rng),
            quantile_alpha=_pick(parent_a.quantile_alpha, parent_b.quantile_alpha, rng),
            metadata={},
        )


def build_lightgbm_objective_params(genome: LossGenome) -> dict[str, Any]:
    """Create LightGBM params for a genome's built-in objective.

    @notice: |
        Create LightGBM params for a genome's built-in objective.

    @dev: |
        Emits only parameters LightGBM recognizes: ``objective`` plus ``alpha``
        for the ``huber`` (transition point) and ``quantile`` (target quantile)
        objectives. Genome-only fields such as ``asymmetric_weight`` are not
        valid LightGBM params; use :func:`build_lightgbm_objective` to apply
        them through a custom objective callable instead.
    """
    params: dict[str, Any] = {"objective": genome.objective}
    if genome.objective == "quantile":
        params["alpha"] = genome.quantile_alpha
    elif genome.objective == "huber":
        params["alpha"] = genome.huber_delta
    return params


def build_lightgbm_objective(genome: LossGenome) -> LightGBMObjective:
    """Build a LightGBM custom objective callable from an evolved genome.

    @notice: |
        Build a LightGBM custom objective callable from an evolved genome.

    @dev: |
        The returned callable returns ``(grad, hess)`` arrays and accepts both
        invocation conventions: ``lightgbm.train`` passes ``(y_pred, Dataset)``
        while the scikit-learn API passes ``(y_true, y_pred)``. ``quantile``
        genomes yield an asymmetric pinball objective; every other objective
        yields an asymmetric Huber objective whose ``mae_rmse_blend``
        interpolates between Huber (robust, MAE-leaning) and squared-error
        (RMSE) gradients. ``asymmetric_weight`` scales the gradient and hessian
        for over-predictions so the search can favor under- or over-shooting.
    """
    if genome.objective == "quantile":
        return _build_quantile_objective(alpha=genome.quantile_alpha, over_weight=genome.asymmetric_weight)
    return _build_asymmetric_huber_objective(
        delta=genome.huber_delta, mae_rmse_blend=genome.mae_rmse_blend, over_weight=genome.asymmetric_weight
    )


def _residual(first: Any, second: Any) -> NDArray[np.float64]:
    """Return ``y_pred - y_true`` for either LightGBM calling convention."""
    get_label = getattr(second, "get_label", None)
    if callable(get_label):
        # lightgbm.train convention: (y_pred, Dataset).
        pred = np.asarray(first, dtype=np.float64)
        true = np.asarray(get_label(), dtype=np.float64)
    else:
        # scikit-learn convention: (y_true, y_pred).
        true = np.asarray(first, dtype=np.float64)
        pred = np.asarray(second, dtype=np.float64)
    return pred - true


def _build_asymmetric_huber_objective(*, delta: float, mae_rmse_blend: float, over_weight: float) -> LightGBMObjective:
    safe_delta = max(float(delta), _MIN_HESSIAN)
    blend = _clip(float(mae_rmse_blend), 0.0, 1.0)
    safe_weight = max(float(over_weight), _MIN_HESSIAN)

    def objective(first: Any, second: Any) -> GradHess:
        residual = _residual(first, second)
        clipped = np.clip(residual, -safe_delta, safe_delta)
        huber_hess = (np.abs(residual) <= safe_delta).astype(np.float64)
        grad = blend * residual + (1.0 - blend) * clipped
        hess = blend + (1.0 - blend) * huber_hess
        weight = np.where(residual > 0.0, safe_weight, 1.0)
        grad = grad * weight
        hess = np.maximum(hess * weight, _MIN_HESSIAN)
        return grad, hess

    return objective


def _build_quantile_objective(*, alpha: float, over_weight: float) -> LightGBMObjective:
    safe_alpha = _clip(float(alpha), _MIN_HESSIAN, 1.0 - _MIN_HESSIAN)
    safe_weight = max(float(over_weight), _MIN_HESSIAN)

    def objective(first: Any, second: Any) -> GradHess:
        residual = _residual(first, second)
        weight = np.where(residual > 0.0, safe_weight, 1.0)
        grad = np.where(residual > 0.0, 1.0 - safe_alpha, -safe_alpha) * weight
        hess = np.full_like(residual, _QUANTILE_HESSIAN) * weight
        return grad, hess

    return objective


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
