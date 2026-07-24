"""Tests for the generic FeatureEvolver hyperparameter search.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

import pytest

from agent_k.evolution.features import DomainFeatureGene, FeatureEvolver, FeatureGenome

__all__ = ()


def _fitness(genome: FeatureGenome) -> float:
    return float(len(genome.selected_features))


def test_default_evolver_seeds_no_domain_features() -> None:
    evolver = FeatureEvolver(feature_names=["alpha", "beta", "gamma", "delta"], rng=random.Random(0))
    result = evolver.evolve(generations=1, fitness_fn=_fitness)
    population = result["population"].individuals

    assert population, "population must not be empty"
    for individual in population:
        assert individual.genome.domain_features == [], (
            "FeatureEvolver must not seed dataset-specific domain features by default"
        )


def test_domain_templates_only_seed_when_inputs_present() -> None:
    templates = (
        DomainFeatureGene(name="Total", inputs=("alpha", "beta")),
        DomainFeatureGene(name="Ratio", inputs=("gamma", "missing_column")),
    )
    evolver = FeatureEvolver(feature_names=["alpha", "beta", "gamma"], rng=random.Random(1), domain_templates=templates)
    seeded = {gene.name for individual in evolver._population.individuals for gene in individual.genome.domain_features}

    assert "Total" in seeded
    assert "Ratio" not in seeded


def test_random_genome_handles_small_feature_lists() -> None:
    evolver = FeatureEvolver(feature_names=["only_feature"], rng=random.Random(2))
    genome = evolver._random_genome()

    assert genome.selected_features == ["only_feature"]
    assert genome.ratios == []
    assert genome.interactions == []


def test_random_genome_handles_empty_feature_list() -> None:
    evolver = FeatureEvolver(feature_names=[], rng=random.Random(3))
    genome = evolver._random_genome()

    assert genome.selected_features == []
    assert genome.transforms == []
    assert genome.ratios == []
    assert genome.interactions == []


def test_evolver_completes_generations_without_error() -> None:
    evolver = FeatureEvolver(feature_names=[f"f{i}" for i in range(10)], rng=random.Random(4), population_size=6)
    result = evolver.evolve(generations=2, fitness_fn=_fitness)

    assert result["best"] is not None
    assert result["population"].generation == 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_evolver_never_creates_self_ratios(seed: int) -> None:
    evolver = FeatureEvolver(feature_names=["a", "b", "c", "d"], rng=random.Random(seed))
    for individual in evolver._population.individuals:
        for ratio in individual.genome.ratios:
            assert ratio.numerator != ratio.denominator, "ratio feature must never divide a column by itself"
