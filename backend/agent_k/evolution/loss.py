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
from typing import TYPE_CHECKING, Any

from agent_k.evolution.framework import FitnessFn, Individual, Population

if TYPE_CHECKING:
    import numpy as np

type LossObjective = str
"""LightGBM objective identifiers."""

CustomObjectiveFn = Callable[["np.ndarray", Any], tuple["np.ndarray", "np.ndarray"]]
"""Signature for LightGBM custom objective callables: (y_pred, dataset) -> (grad, hess)."""

__all__ = (
    "CustomObjectiveFn",
    "LossFunctionEvolver",
    "LossGenome",
    "build_custom_objective_callable",
    "build_lightgbm_objective_params",
)


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
    """Create LightGBM objective params from a LossGenome.

    @notice: |
        Maps a genome to keyword arguments accepted by ``lightgbm.train``.

    @dev: |
        Only emits keys LightGBM actually recognises. The ``asymmetric_weight``
        and ``mae_rmse_blend`` knobs are surfaced via
        ``build_custom_objective_callable`` because LightGBM silently ignores
        unknown params instead of erroring.
    """
    params: dict[str, Any] = {"objective": genome.objective}
    if genome.objective == "quantile":
        params["alpha"] = genome.quantile_alpha
    elif genome.objective == "huber":
        # LightGBM uses ``alpha`` for the Huber transition; ``huber_delta`` is
        # silently ignored otherwise.
        params["alpha"] = genome.huber_delta
    return params


def build_custom_objective_callable(genome: LossGenome) -> CustomObjectiveFn:
    """Compile a LightGBM-compatible custom objective from the genome.

    @notice: |
        Returns a ``(y_pred, dataset) -> (grad, hess)`` callable that blends
        MAE/RMSE gradients and applies an asymmetric residual weight.

    @dev: |
        Pass the returned callable directly via ``params["objective"]`` for
        ``lightgbm.train`` (the legacy ``fobj=`` keyword was removed in
        LightGBM 4.0). The blend follows ``mae_rmse_blend`` (0 = MAE,
        1 = RMSE) and over-/under-predictions are weighted by
        ``asymmetric_weight`` so positive residuals (over-prediction) can be
        penalised more heavily than negative residuals.
    """
    import numpy as np

    blend = float(_clip(genome.mae_rmse_blend, 0.0, 1.0))
    asym = float(max(genome.asymmetric_weight, 0.0))
    mae_hess_floor = 1e-3

    def custom_objective(y_pred: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(dataset.get_label(), dtype=np.float64)
        preds = np.asarray(y_pred, dtype=np.float64)
        residual = preds - y_true

        l2_grad = residual
        l2_hess = np.ones_like(residual)
        l1_grad = np.sign(residual)
        l1_hess = np.full_like(residual, mae_hess_floor)

        grad = blend * l2_grad + (1.0 - blend) * l1_grad
        hess = blend * l2_hess + (1.0 - blend) * l1_hess

        if asym != 1.0:
            weights = np.where(residual > 0.0, asym, 1.0)
            grad = grad * weights
            hess = hess * weights

        return grad, hess

    return custom_objective


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
