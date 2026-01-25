"""Loss function evolution for LightGBM objectives.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import random
from dataclasses import dataclass, field

# Third-party (alphabetical)
from typing import Any, TypeAliasType

# Local imports (core first, then alphabetical)
from agent_k.evolution.framework import FitnessFn, Individual, Population

LossObjective = TypeAliasType("LossObjective", str)
"""LightGBM objective identifiers."""

__all__ = ("LossFunctionEvolver", "LossGenome", "build_lightgbm_objective_params")


@dataclass(slots=True)
class LossGenome:
    """Genome for evolving custom objective functions."""

    objective: LossObjective = "regression"
    asymmetric_weight: float = 1.0
    huber_delta: float = 1.0
    mae_rmse_blend: float = 0.5
    quantile_alpha: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class LossFunctionEvolver:
    """Evolve LightGBM loss function parameters via genetic search."""

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
    """Create LightGBM objective params from a LossGenome."""
    params: dict[str, Any] = {
        "objective": genome.objective,
        "asymmetric_weight": genome.asymmetric_weight,
        "mae_rmse_blend": genome.mae_rmse_blend,
    }
    if genome.objective == "quantile":
        params["alpha"] = genome.quantile_alpha
    if genome.objective == "huber":
        params["huber_delta"] = genome.huber_delta
    return params


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
