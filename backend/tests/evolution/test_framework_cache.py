"""Tests for genome fitness memoization in the evolutionary framework.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agent_k.evolution.features import FeatureGenome, TransformGene
from agent_k.evolution.framework import (
    HyperparamEvolver,
    HyperparamGenome,
    HyperparamSpace,
    Individual,
    Population,
    genome_fingerprint,
)
from agent_k.evolution.loss import LossFunctionEvolver, LossGenome

__all__ = ()


@dataclass(slots=True)
class _Genome:
    """Minimal genome with an unhashable field, mirroring real genomes."""

    value: int
    metadata: dict[str, Any] = field(default_factory=dict)


class _CountingFitness:
    """Fitness function that records every genome it is actually called with."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, genome: _Genome) -> float:
        self.calls.append(genome.value)
        return float(genome.value)


def test_fingerprint_matches_structurally_equal_genomes() -> None:
    left = _Genome(value=3, metadata={"a": 1, "b": [2, 3]})
    right = _Genome(value=3, metadata={"b": [2, 3], "a": 1})

    assert genome_fingerprint(left) == genome_fingerprint(right)
    assert genome_fingerprint(left) != genome_fingerprint(_Genome(value=4))


def test_fingerprint_handles_nested_dataclass_genomes() -> None:
    left = FeatureGenome(transforms=[TransformGene(feature="x", transform="log")], selected_features=["x", "y"])
    right = FeatureGenome(transforms=[TransformGene(feature="x", transform="log")], selected_features=["x", "y"])
    other = FeatureGenome(transforms=[TransformGene(feature="x", transform="sqrt")], selected_features=["x", "y"])

    assert genome_fingerprint(left) == genome_fingerprint(right)
    assert genome_fingerprint(left) != genome_fingerprint(other)


def test_fingerprint_normalizes_non_finite_floats() -> None:
    assert genome_fingerprint(LossGenome(huber_delta=float("nan"))) == genome_fingerprint(
        LossGenome(huber_delta=float("nan"))
    )
    assert genome_fingerprint(LossGenome(huber_delta=float("nan"))) != genome_fingerprint(
        LossGenome(huber_delta=float("inf"))
    )


def test_evaluate_reuses_cached_fitness_for_duplicate_genomes() -> None:
    fitness = _CountingFitness()
    population = Population([Individual(_Genome(value=1)), Individual(_Genome(value=1)), Individual(_Genome(value=2))])

    population.evaluate(fitness)

    assert fitness.calls == [1, 2]
    assert [individual.fitness for individual in population.individuals] == [1.0, 1.0, 2.0]
    assert population.fitness_evaluations == 2
    assert population.fitness_cache_hits == 1


def test_evaluate_reuses_cached_fitness_across_generations() -> None:
    fitness = _CountingFitness()
    population = Population([Individual(_Genome(value=7))])
    population.evaluate(fitness)

    population.individuals = [Individual(_Genome(value=7))]
    population.evaluate(fitness)

    assert fitness.calls == [7]
    assert population.fitness_cache_hits == 1


def test_memoize_fitness_disabled_reevaluates_duplicates() -> None:
    fitness = _CountingFitness()
    population = Population([Individual(_Genome(value=5)), Individual(_Genome(value=5))], memoize_fitness=False)

    population.evaluate(fitness)

    assert fitness.calls == [5, 5]
    assert population.fitness_evaluations == 2
    assert population.fitness_cache_hits == 0


def test_fitness_cache_respects_entry_bound() -> None:
    fitness = _CountingFitness()
    population = Population([Individual(_Genome(value=1)), Individual(_Genome(value=2))], max_fitness_cache_entries=1)
    population.evaluate(fitness)

    # Only the newest genome stayed cached, so genome 1 has to be scored again.
    population.individuals = [Individual(_Genome(value=2)), Individual(_Genome(value=1))]
    population.evaluate(fitness)

    assert fitness.calls == [1, 2, 1]


def test_zero_entry_cache_disables_memoization() -> None:
    fitness = _CountingFitness()
    population = Population([Individual(_Genome(value=4)), Individual(_Genome(value=4))], max_fitness_cache_entries=0)

    population.evaluate(fitness)

    assert fitness.calls == [4, 4]
    assert population.fitness_evaluations == 2
    assert population.fitness_cache_hits == 0


def test_unfingerprintable_genome_still_evaluates() -> None:
    calls: list[object] = []

    def fitness(genome: object) -> float:
        calls.append(genome)
        return 1.0

    def broken_fingerprint(_genome: object) -> Any:
        raise TypeError("unfingerprintable")

    population: Population[object] = Population(
        [Individual(object()), Individual(object())], fingerprint_fn=broken_fingerprint
    )
    population.evaluate(fitness)

    assert len(calls) == 2
    assert population.fitness_cache_hits == 0


def test_evolve_generation_skips_duplicate_offspring_evaluations() -> None:
    """Clone offspring must not consume fitness evaluations."""
    fitness = _CountingFitness()
    rng = random.Random(1234)
    population = Population([Individual(_Genome(value=index)) for index in range(8)], rng=rng)

    def mutate(genome: _Genome, _rng: random.Random) -> _Genome:
        return _Genome(value=genome.value + 1)

    def crossover(left: _Genome, right: _Genome, _rng: random.Random) -> _Genome:
        return _Genome(value=max(left.value, right.value))

    for _ in range(6):
        population.evolve_generation(fitness_fn=fitness, mutation_fn=mutate, crossover_fn=crossover)

    # Every distinct genome is scored exactly once, and the clones the
    # selection loop produces are served from the cache instead.
    assert population.fitness_evaluations == len(set(fitness.calls))
    assert len(fitness.calls) == len(set(fitness.calls))
    assert population.fitness_cache_hits > 0


def test_loss_evolver_reports_cache_savings() -> None:
    evolver = LossFunctionEvolver(
        fitness_fn=lambda genome: genome.asymmetric_weight, population_size=6, rng=random.Random(7)
    )

    result = evolver.evolve(generations=4)

    assert result["evaluations"] > 0
    assert result["cache_hits"] > 0
    assert result["best_fitness"] is not None


def test_hyperparam_evolver_reports_cache_savings() -> None:
    space = HyperparamSpace(integer={"num_leaves": (8, 12)}, categorical={"boosting": ("gbdt", "dart")})
    calls: list[tuple[Any, ...]] = []

    def fitness(genome: HyperparamGenome) -> float:
        calls.append(tuple(sorted(genome.parameters.items(), key=repr)))
        return float(genome.parameters["num_leaves"])

    evolver = HyperparamEvolver(space, population_size=6, fitness_fn=fitness, rng=random.Random(11))
    result = evolver.evolve(generations=5)

    assert result["status"] == "ok"
    assert result["cache_hits"] > 0
    assert result["evaluations"] == len(calls)
    assert len(set(calls)) == len(calls)
