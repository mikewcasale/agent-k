"""Tests for the generic FeatureEvolver domain-feature injection.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.features import DomainFeatureGene, FeatureEvolver, FeatureGenome

__all__ = ()


def _seeded_evolver(**kwargs: object) -> FeatureEvolver:
    return FeatureEvolver(
        ["age", "income", "score", "tenure", "balance", "products"],
        rng=random.Random(0),
        population_size=6,
        **kwargs,  # type: ignore[arg-type]
    )


def test_random_genome_has_no_domain_features_by_default() -> None:
    """FeatureEvolver ships no hardcoded domain features anymore."""
    evolver = _seeded_evolver()

    for individual in evolver._population.individuals:
        assert individual.genome.domain_features == []


def test_domain_features_seed_only_when_inputs_are_present() -> None:
    """Genes whose inputs match the dataset get seeded; others are dropped."""
    applicable = DomainFeatureGene("wealth_ratio", ("income", "balance"))
    inapplicable = DomainFeatureGene("housing_score", ("SqFt", "YearBuilt"))
    evolver = _seeded_evolver(domain_features=[applicable, inapplicable])

    seeded_domains = {
        gene for individual in evolver._population.individuals for gene in individual.genome.domain_features
    }
    assert seeded_domains == {applicable}


def test_domain_seed_count_limits_injected_genes() -> None:
    """domain_seed_count caps how many genes seed each random genome."""
    genes = [
        DomainFeatureGene("g1", ("age", "income")),
        DomainFeatureGene("g2", ("score", "tenure")),
        DomainFeatureGene("g3", ("balance", "products")),
    ]
    evolver = _seeded_evolver(domain_features=genes, domain_seed_count=1)

    for individual in evolver._population.individuals:
        assert len(individual.genome.domain_features) == 1


def test_domain_seed_count_zero_disables_seeding() -> None:
    """domain_seed_count=0 leaves domain_features empty even when genes exist."""
    genes = [DomainFeatureGene("wealth", ("income", "balance"))]
    evolver = _seeded_evolver(domain_features=genes, domain_seed_count=0)

    for individual in evolver._population.individuals:
        assert individual.genome.domain_features == []


def test_domain_gene_with_empty_inputs_is_ignored() -> None:
    """A degenerate gene with no inputs is filtered out at construction time."""
    genes = [DomainFeatureGene("noop", ()), DomainFeatureGene("real", ("age", "income"))]
    evolver = _seeded_evolver(domain_features=genes)

    seeded_domains = {
        gene for individual in evolver._population.individuals for gene in individual.genome.domain_features
    }
    assert seeded_domains == {genes[1]}


def test_domain_gene_applicable_helper() -> None:
    """The applicability helper matches every input against the dataset columns."""
    evolver = _seeded_evolver()

    assert evolver._domain_gene_applicable(DomainFeatureGene("wealth", ("income", "balance")))
    assert not evolver._domain_gene_applicable(DomainFeatureGene("wealth", ("income", "SqFt")))
    assert not evolver._domain_gene_applicable(DomainFeatureGene("empty", ()))


def test_evolver_handles_empty_feature_names() -> None:
    """An empty feature set produces empty-but-valid genomes and evolves safely."""
    evolver = FeatureEvolver([], rng=random.Random(0), population_size=4)

    for individual in evolver._population.individuals:
        assert individual.genome.selected_features == []
        assert individual.genome.transforms == []
        assert individual.genome.ratios == []
        assert individual.genome.domain_features == []

    result = evolver.evolve(fitness_fn=lambda _genome: 0.0, generations=1)
    assert isinstance(result["population"].best().genome, FeatureGenome)
