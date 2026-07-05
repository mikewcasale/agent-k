"""Tests for the evolutionary framework primitives.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
import random

from agent_k.evolution.framework import Individual, MapElitesArchive, Population, _fitness_key

__all__ = ()


def test_fitness_key_preserves_zero() -> None:
    assert _fitness_key(0.0) == 0.0
    assert _fitness_key(-0.5) == -0.5
    assert _fitness_key(1.25) == 1.25
    assert _fitness_key(None) == -math.inf


def test_population_best_prefers_zero_over_negative_fitness() -> None:
    zero = Individual(genome="zero", fitness=0.0)
    negative = Individual(genome="neg", fitness=-0.5)
    population: Population[str] = Population([negative, zero])

    best = population.best()

    assert best is zero, "fitness=0.0 must outrank fitness<0.0 rather than being coerced to -inf"


def test_population_best_handles_unevaluated_individuals() -> None:
    evaluated = Individual(genome="a", fitness=-0.25)
    unevaluated = Individual(genome="b")
    population: Population[str] = Population([unevaluated, evaluated])

    best = population.best()

    assert best is evaluated


def test_population_best_returns_none_when_empty() -> None:
    population: Population[str] = Population([])
    assert population.best() is None


def test_population_tournament_prefers_zero_over_negative_fitness() -> None:
    zero = Individual(genome="zero", fitness=0.0)
    negative_a = Individual(genome="a", fitness=-0.75)
    negative_b = Individual(genome="b", fitness=-1.5)
    population: Population[str] = Population([negative_a, zero, negative_b], rng=random.Random(0))

    winner = population.tournament(size=3)

    assert winner is zero


def test_population_evolve_generation_elitism_preserves_zero_fitness_best() -> None:
    zero = Individual(genome=0, fitness=0.0)
    lower_a = Individual(genome=1, fitness=-0.25)
    lower_b = Individual(genome=2, fitness=-0.5)
    population: Population[int] = Population([lower_a, zero, lower_b], rng=random.Random(0))

    def _fitness(genome: int) -> float:
        return {0: 0.0, 1: -0.25, 2: -0.5}[genome]

    next_gen = population.evolve_generation(
        fitness_fn=_fitness,
        mutation_fn=lambda genome, _rng: genome,
        crossover_fn=lambda a, _b, _rng: a,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism=1,
    )

    assert next_gen[0].genome == 0
    assert next_gen[0].fitness == 0.0


def test_map_elites_add_replaces_when_zero_fitness_beats_negative() -> None:
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda _genome: (0,))
    archive.add(Individual(genome="loser", fitness=-1.0))
    archive.add(Individual(genome="winner", fitness=0.0))

    cells = archive.cells

    assert len(cells) == 1
    assert cells[(0,)].genome == "winner"


def test_map_elites_add_keeps_zero_over_lower_new_candidate() -> None:
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda _genome: (0,))
    archive.add(Individual(genome="incumbent", fitness=0.0))
    archive.add(Individual(genome="worse", fitness=-2.0))

    assert archive.cells[(0,)].genome == "incumbent"


def test_map_elites_add_skips_unevaluated() -> None:
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda _genome: (0,))
    archive.add(Individual(genome="unscored"))

    assert archive.cells == {}


def test_map_elites_sample_orders_zero_fitness_above_negative() -> None:
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda genome: (hash(genome) % 2,))
    archive.add(Individual(genome="zero", fitness=0.0))
    archive.add(Individual(genome="neg", fitness=-1.0))

    sampled = archive.sample(top=1, diverse=1)

    assert sampled[0].genome == "zero"


def test_map_elites_trim_keeps_higher_fitness_cells() -> None:
    archive: MapElitesArchive[int] = MapElitesArchive(descriptor_fn=lambda genome: (genome,), max_cells=2)
    archive.add(Individual(genome=0, fitness=0.0))
    archive.add(Individual(genome=1, fitness=-0.5))
    archive.add(Individual(genome=2, fitness=-2.0))

    genomes = {cell.genome for cell in archive.cells.values()}

    assert genomes == {0, 1}
