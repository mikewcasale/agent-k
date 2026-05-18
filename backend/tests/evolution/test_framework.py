"""Tests for the evolutionary framework population primitives.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.framework import Individual, Population

__all__ = ()


def _genome_fitness(genome: int) -> float:
    """Deterministic fitness equal to the integer genome value."""
    return float(genome)


def test_with_genome_carries_inherited_fitness() -> None:
    parent = Individual(genome=7, fitness=12.5)
    clone = parent.with_genome(parent.genome, fitness=parent.fitness)

    assert clone.fitness == 12.5
    assert clone.genome == 7
    assert clone.lineage == [parent.identifier]


def test_with_genome_defaults_to_unevaluated() -> None:
    parent = Individual(genome=3, fitness=9.0)
    child = parent.with_genome(genome=4)

    assert child.fitness is None


def test_evolve_generation_skips_reeval_of_unmodified_clones() -> None:
    calls: list[int] = []

    def counting_fitness(genome: int) -> float:
        calls.append(genome)
        return float(genome)

    individuals = [Individual(genome=value) for value in range(8)]
    population = Population(individuals, rng=random.Random(0))

    population.evolve_generation(
        fitness_fn=counting_fitness,
        mutation_fn=lambda genome, rng: genome + 1,
        crossover_fn=lambda a, b, rng: a + b,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
    )

    # First generation evaluates every founder exactly once.
    assert sorted(calls) == list(range(8))
    calls.clear()

    # With crossover and mutation disabled, every child is an unmodified clone
    # whose fitness is inherited, so the next generation triggers no new work.
    population.evolve_generation(
        fitness_fn=counting_fitness,
        mutation_fn=lambda genome, rng: genome + 1,
        crossover_fn=lambda a, b, rng: a + b,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
    )

    assert calls == []
    assert all(individual.fitness is not None for individual in population.individuals)


def test_evolve_generation_evaluates_modified_children() -> None:
    calls: list[int] = []

    def counting_fitness(genome: int) -> float:
        calls.append(genome)
        return float(genome)

    individuals = [Individual(genome=value) for value in range(6)]
    population = Population(individuals, rng=random.Random(1))

    population.evolve_generation(
        fitness_fn=counting_fitness,
        mutation_fn=lambda genome, rng: genome + 100,
        crossover_fn=lambda a, b, rng: a + b,
        mutation_rate=1.0,
        crossover_rate=0.0,
        elitism=1,
    )
    calls.clear()

    population.evolve_generation(
        fitness_fn=counting_fitness,
        mutation_fn=lambda genome, rng: genome + 100,
        crossover_fn=lambda a, b, rng: a + b,
        mutation_rate=1.0,
        crossover_rate=0.0,
        elitism=1,
    )

    # Every non-elite child is mutated, so its fitness must be recomputed.
    assert len(calls) == len(population.individuals) - 1
