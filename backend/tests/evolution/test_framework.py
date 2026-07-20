"""Tests for the evolutionary framework fitness handling.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.framework import Individual, MapElitesArchive, Population, _fitness_or

__all__ = ()


def _deterministic_rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def test_population_best_returns_zero_fitness_over_unevaluated() -> None:
    """`Population.best` must not treat fitness=0.0 as unevaluated (-inf)."""
    zero = Individual(genome="zero", fitness=0.0)
    unevaluated = Individual(genome="unset", fitness=None)
    population = Population([unevaluated, zero], rng=_deterministic_rng())

    best = population.best()

    assert best is zero, "fitness=0.0 must beat fitness=None, not the reverse"


def test_population_best_returns_negative_zero_over_more_negative() -> None:
    """`-0.0` produced by minimize-direction fitness is not falsy for ranking."""
    negative_zero = Individual(genome="perfect", fitness=-0.0)
    worse = Individual(genome="bad", fitness=-0.5)
    population = Population([worse, negative_zero], rng=_deterministic_rng())

    best = population.best()

    assert best is negative_zero, "fitness=-0.0 must outrank fitness=-0.5"


def test_population_best_prefers_positive_over_zero() -> None:
    """Sanity: positive fitness still wins over zero."""
    zero = Individual(genome="zero", fitness=0.0)
    positive = Individual(genome="positive", fitness=0.25)
    population = Population([zero, positive], rng=_deterministic_rng())

    assert population.best() is positive


def test_fitness_or_helper_distinguishes_zero_from_unset() -> None:
    """`_fitness_or` must return 0.0 when fitness is 0.0, and the sentinel only for None.

    This is the private sort-key helper used by `Population.best`,
    `Population.tournament`, `Population.evolve_generation`, and
    `MapElitesArchive.{add, sample, _trim_archive}` — the truthy `or`
    idiom it replaces collapses `0.0` and `-0.0` to the sentinel.
    """
    import math

    assert _fitness_or(Individual(genome="x", fitness=0.0), -math.inf) == 0.0
    assert _fitness_or(Individual(genome="x", fitness=-0.0), -math.inf) == 0.0
    assert _fitness_or(Individual(genome="x", fitness=-1.5), -math.inf) == -1.5
    assert _fitness_or(Individual(genome="x", fitness=None), -math.inf) == -math.inf
    assert _fitness_or(Individual(genome="x", fitness=None), 0.0) == 0.0


def test_population_evolve_generation_elitism_ranks_zero_above_negative() -> None:
    """Elitism ranking must place fitness=0.0 above a strictly worse negative fitness."""
    zero = Individual(genome="zero", fitness=0.0)
    negative = Individual(genome="neg", fitness=-1.0)
    population = Population([negative, zero], rng=_deterministic_rng())

    def fitness_fn(genome: str) -> float:
        return {"zero": 0.0, "neg": -1.0}[genome]

    def mutation_fn(genome: str, _rng: random.Random) -> str:
        return genome

    def crossover_fn(parent_a: str, _parent_b: str, _rng: random.Random) -> str:
        return parent_a

    next_generation = population.evolve_generation(
        fitness_fn=fitness_fn, mutation_fn=mutation_fn, crossover_fn=crossover_fn, elitism=1
    )

    assert next_generation[0] is zero, "elitism must keep fitness=0.0, not the fitness=-1.0 individual"


def test_map_elites_archive_keeps_zero_fitness_over_worse_negative() -> None:
    """Archive must retain fitness=0.0 when a worse (more-negative) candidate arrives."""
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda genome: (0,))
    archive.add(Individual(genome="zero", fitness=0.0))
    archive.add(Individual(genome="worse", fitness=-1.0))

    cell = archive.cells[(0,)]

    assert cell.genome == "zero"
    assert cell.fitness == 0.0


def test_map_elites_archive_replaces_when_new_is_strictly_better() -> None:
    """Sanity: better fitness wins in the same cell."""
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda genome: (0,))
    archive.add(Individual(genome="baseline", fitness=-0.5))
    archive.add(Individual(genome="improved", fitness=0.0))

    cell = archive.cells[(0,)]

    assert cell.genome == "improved"
    assert cell.fitness == 0.0


def test_map_elites_archive_trim_keeps_zero_fitness() -> None:
    """`_trim_archive` must not discard a zero-fitness elite in favor of a negative one."""
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda genome: (len(genome),), max_cells=2)
    archive.add(Individual(genome="a", fitness=0.0))
    archive.add(Individual(genome="bb", fitness=-2.0))
    archive.add(Individual(genome="ccc", fitness=-3.0))

    genomes = {ind.genome for ind in archive.cells.values()}

    assert "a" in genomes, "fitness=0.0 must survive trimming against more-negative peers"


def test_map_elites_archive_sample_orders_zero_above_negative() -> None:
    """`sample(top=1)` must return the fitness=0.0 individual, not a negative-fitness one."""
    archive: MapElitesArchive[str] = MapElitesArchive(descriptor_fn=lambda genome: (len(genome),))
    negative_one = Individual(genome="a", fitness=-1.0)
    zero = Individual(genome="bb", fitness=0.0)
    archive.add(negative_one)
    archive.add(zero)

    top = archive.sample(top=1, diverse=0)

    assert top == [zero]


def test_population_best_all_unevaluated_returns_first() -> None:
    """When every candidate is unevaluated, `best` still returns one deterministically."""
    a = Individual(genome="a", fitness=None)
    b = Individual(genome="b", fitness=None)
    population = Population([a, b], rng=_deterministic_rng())

    assert population.best() is a
