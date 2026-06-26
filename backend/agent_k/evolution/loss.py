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
    last-verified: 2026-06-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_k.evolution.framework import FitnessFn, Individual, Population

if TYPE_CHECKING:
    import numpy as np

type LossObjective = str
"""LightGBM objective identifiers."""

type LightGBMCustomObjective = Callable[[Any, Any], tuple[Any, Any]]
"""Callable signature for LightGBM custom objectives returning (grad, hess) arrays."""

__all__ = (
    "BUILTIN_OBJECTIVES",
    "CUSTOM_OBJECTIVES",
    "LightGBMCustomObjective",
    "LossFunctionEvolver",
    "LossGenome",
    "build_lightgbm_custom_objective",
    "build_lightgbm_objective_params",
)

BUILTIN_OBJECTIVES: tuple[str, ...] = ("regression", "regression_l1", "huber", "quantile")
"""LightGBM built-in regression objectives passed through `params['objective']`."""

CUSTOM_OBJECTIVES: tuple[str, ...] = ("asymmetric", "mae_rmse_blend")
"""Custom objectives implemented as Python callables for `params['objective']`."""

_EPS: float = 1e-6
_HUBER_HESS_FLOOR: float = 1e-3


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

    _objectives: tuple[str, ...] = BUILTIN_OBJECTIVES + CUSTOM_OBJECTIVES

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
    """Create LightGBM-valid objective params from a LossGenome.

    @notice: |
        Emit only parameters that LightGBM recognizes for the chosen objective.

    @dev: |
        Custom objectives (asymmetric, mae_rmse_blend) are exposed via
        `build_lightgbm_custom_objective`; callers should attach the returned
        callable to `params['objective']` (LightGBM 4.x) or `lgb.train(..., fobj=)`.
        The returned dict for custom objectives carries a baseline metric so
        LightGBM logs remain meaningful.
    """
    objective = genome.objective

    if objective == "quantile":
        return {"objective": "quantile", "alpha": _clip(genome.quantile_alpha, _EPS, 1.0 - _EPS)}
    if objective == "huber":
        return {"objective": "huber", "alpha": max(genome.huber_delta, _EPS)}
    if objective in CUSTOM_OBJECTIVES:
        metric = "mae" if objective == "asymmetric" or genome.mae_rmse_blend >= 0.5 else "rmse"
        return {"metric": metric}
    return {"objective": objective}


def build_lightgbm_custom_objective(genome: LossGenome) -> LightGBMCustomObjective | None:
    """Build a LightGBM custom objective callable from a LossGenome.

    @notice: |
        Returns a (grad, hess) callable for `params['objective']` / `fobj`,
        or ``None`` when the genome targets a LightGBM built-in objective.

    @dev: |
        Implements four custom objectives:

        * ``huber`` — smooth L1/L2 transition at ``huber_delta``.
        * ``quantile`` — pinball loss at ``quantile_alpha``.
        * ``asymmetric`` — asymmetric MSE penalising positive residuals by
          ``asymmetric_weight`` (over-prediction) vs. negative residuals.
        * ``mae_rmse_blend`` — convex combination of MAE and RMSE gradients
          with a numerically stable Hessian floor.

        Custom variants exist for ``huber``/``quantile`` so the evolver can
        compare LightGBM's native implementation against a controllable
        gradient form (e.g. for asymmetric penalties on the same family).
    """
    import numpy as np

    objective = genome.objective

    if objective == "huber":
        delta = max(genome.huber_delta, _EPS)

        def huber_fobj(y_pred: np.ndarray, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
            y_true = _coerce_label(train_data)
            residual = y_pred - y_true
            abs_r = np.abs(residual)
            grad = np.where(abs_r <= delta, residual, delta * np.sign(residual))
            hess = np.where(abs_r <= delta, 1.0, _HUBER_HESS_FLOOR)
            return grad, hess

        return huber_fobj

    if objective == "quantile":
        alpha = _clip(genome.quantile_alpha, _EPS, 1.0 - _EPS)

        def quantile_fobj(y_pred: np.ndarray, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
            y_true = _coerce_label(train_data)
            residual = y_true - y_pred
            grad = np.where(residual >= 0, -alpha, 1.0 - alpha)
            hess = np.ones_like(residual)
            return grad, hess

        return quantile_fobj

    if objective == "asymmetric":
        weight = max(genome.asymmetric_weight, _EPS)

        def asymmetric_fobj(y_pred: np.ndarray, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
            y_true = _coerce_label(train_data)
            residual = y_pred - y_true
            grad = np.where(residual >= 0, weight * residual, residual)
            hess = np.where(residual >= 0, weight, 1.0)
            return grad, hess

        return asymmetric_fobj

    if objective == "mae_rmse_blend":
        blend = _clip(genome.mae_rmse_blend, 0.0, 1.0)
        hess_value = max(1.0 - blend, _HUBER_HESS_FLOOR)

        def blend_fobj(y_pred: np.ndarray, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
            y_true = _coerce_label(train_data)
            residual = y_pred - y_true
            mae_grad = np.sign(residual)
            mse_grad = residual
            grad = blend * mae_grad + (1.0 - blend) * mse_grad
            hess = np.full_like(residual, hess_value)
            return grad, hess

        return blend_fobj

    return None


def _coerce_label(train_data: Any) -> np.ndarray:
    """Extract the label array from a LightGBM Dataset or numpy-like input."""
    import numpy as np

    label_getter = getattr(train_data, "get_label", None)
    if callable(label_getter):
        return np.asarray(label_getter(), dtype=float)
    return np.asarray(train_data, dtype=float)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
