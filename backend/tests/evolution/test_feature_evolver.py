"""Tests for the generic FeatureEvolver.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.features import DomainFeatureGene, FeatureEvolver, FeatureGenome

__all__ = ()


def _build_evolver(*, domain_features: tuple[DomainFeatureGene, ...] = (), population_size: int = 4) -> FeatureEvolver:
    return FeatureEvolver(
        feature_names=["f1", "f2", "f3", "f4", "f5", "f6"],
        rng=random.Random(0),
        population_size=population_size,
        domain_features=domain_features,
    )


def test_default_population_has_no_competition_specific_features() -> None:
    """Default construction must not inject any domain features.

    Generic-by-problem-type policy: callers supply competition-specific genes;
    the evolver itself owns no defaults.
    """
    evolver = _build_evolver()
    population = evolver._population

    assert population.individuals, "population should be initialized"
    for individual in population.individuals:
        genome = individual.genome
        assert isinstance(genome, FeatureGenome)
        assert genome.domain_features == []


def test_supplied_domain_features_propagate_to_genomes() -> None:
    """User-provided domain features should appear in random genomes (capped at 2)."""
    domain_features = (
        DomainFeatureGene("ratio_a", ("f1", "f2")),
        DomainFeatureGene("sum_b", ("f3", "f4")),
        DomainFeatureGene("delta_c", ("f5", "f6")),
    )
    evolver = _build_evolver(domain_features=domain_features)

    genomes = [individual.genome for individual in evolver._population.individuals]
    assert genomes, "population should not be empty"
    for genome in genomes:
        # Constructor caps domain_features sample at 2 to keep mutations bounded.
        assert genome.domain_features == list(domain_features[:2])


def test_evolve_runs_with_custom_fitness_and_progresses_generations() -> None:
    """Evolution should advance generations and pick the best genome by fitness."""
    evolver = _build_evolver(population_size=6)

    def fitness_fn(genome: FeatureGenome) -> float:
        # Reward selecting more transforms - simple monotonic signal.
        return float(len(genome.transforms))

    result = evolver.evolve(generations=2, fitness_fn=fitness_fn)

    population = result["population"]
    best = result["best"]

    assert population.generation == 2
    assert best is not None
    assert best.fitness is not None
    assert best.fitness >= 0.0
