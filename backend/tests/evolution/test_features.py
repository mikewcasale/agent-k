"""Tests for the feature engineering evolver.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.features import DomainFeatureGene, FeatureEvolver, FeatureGenome

__all__ = ()


def _evolver(
    feature_names: list[str], *, domain_templates: tuple[DomainFeatureGene, ...] | None = None, population_size: int = 4
) -> FeatureEvolver:
    return FeatureEvolver(
        feature_names, rng=random.Random(0), population_size=population_size, domain_templates=domain_templates
    )


def test_default_evolver_does_not_seed_any_domain_features() -> None:
    """Default behaviour must stay competition-agnostic - no hardcoded gene templates."""
    evolver = _evolver(["a", "b", "c", "d", "e"])

    for individual in evolver._population.individuals:
        genome = individual.genome
        assert isinstance(genome, FeatureGenome)
        assert genome.domain_features == []


def test_domain_templates_are_filtered_to_those_matching_feature_names() -> None:
    """Templates whose inputs aren't all present in the dataset must be dropped."""
    feature_names = ["price", "size", "rooms"]
    templates = (
        DomainFeatureGene(name="PricePerSize", inputs=("price", "size")),
        DomainFeatureGene(name="UnavailableRatio", inputs=("foo", "bar")),
    )

    evolver = _evolver(feature_names, domain_templates=templates)

    assert evolver._domain_templates == (DomainFeatureGene(name="PricePerSize", inputs=("price", "size")),)
    for individual in evolver._population.individuals:
        names = [gene.name for gene in individual.genome.domain_features]
        assert "UnavailableRatio" not in names
        assert names == ["PricePerSize"] or names == []


def test_domain_templates_seed_random_genomes_when_inputs_are_present() -> None:
    """Templates matching the dataset should appear in seeded genomes (first two only)."""
    feature_names = ["x", "y", "z", "w"]
    templates = (
        DomainFeatureGene(name="XY", inputs=("x", "y")),
        DomainFeatureGene(name="ZW", inputs=("z", "w")),
        DomainFeatureGene(name="ALL", inputs=("x", "y", "z", "w")),
    )

    evolver = _evolver(feature_names, domain_templates=templates)

    assert evolver._domain_templates == templates
    for individual in evolver._population.individuals:
        names = [gene.name for gene in individual.genome.domain_features]
        # _random_genome seeds at most the first two templates.
        assert names == ["XY", "ZW"]


def test_evolve_runs_with_empty_domain_templates_and_fitness_callback() -> None:
    """End-to-end smoke: the evolver must still progress without any domain templates."""
    feature_names = ["a", "b", "c", "d", "e"]
    evolver = _evolver(feature_names, population_size=4)
    seen: list[int] = []

    def fitness(genome: FeatureGenome) -> float:
        seen.append(len(genome.selected_features))
        return float(len(genome.selected_features))

    result = evolver.evolve(generations=2, fitness_fn=fitness)

    assert seen, "Fitness function should be invoked during evolution"
    assert result["best"] is not None
    assert result["best"].fitness is not None


def test_domain_templates_do_not_leak_house_prices_specific_feature_names() -> None:
    """Regression: the evolver previously seeded Kaggle House Prices column names by default."""
    forbidden = {"TotalSF", "TotalBath", "YearBuilt", "YrSold", "OverallQual", "OverallCond", "YearRemodAdd"}
    evolver = _evolver(["a", "b", "c"])

    for individual in evolver._population.individuals:
        for gene in individual.genome.domain_features:
            assert gene.name not in forbidden
            assert not set(gene.inputs) & forbidden
