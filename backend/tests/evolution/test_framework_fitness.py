"""Tests for zero-fitness ranking in the evolutionary framework.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.framework import Individual, MapElitesArchive, Population

__all__ = ()


def test_best_prefers_zero_over_negative_fitness() -> None:
    population: Population[float] = Population(
        [
            Individual(genome=-2.0, fitness=-2.0),
            Individual(genome=0.0, fitness=0.0),
            Individual(genome=-0.5, fitness=-0.5),
        ]
    )

    best = population.best()

    assert best is not None
    assert best.fitness == 0.0


def test_best_treats_unevaluated_as_worst() -> None:
    population: Population[float] = Population(
        [Individual(genome=0.0, fitness=0.0), Individual(genome=1.0, fitness=None)]
    )

    best = population.best()

    assert best is not None
    assert best.fitness == 0.0


def test_tournament_selects_zero_over_negative_fitness() -> None:
    population: Population[float] = Population(
        [
            Individual(genome=-1.0, fitness=-1.0),
            Individual(genome=0.0, fitness=0.0),
            Individual(genome=-3.0, fitness=-3.0),
        ]
    )

    winner = population.tournament(size=3)

    assert winner.fitness == 0.0


def test_evolve_generation_keeps_zero_fitness_elite() -> None:
    population: Population[float] = Population(
        [Individual(genome=-1.0), Individual(genome=0.0), Individual(genome=-3.0)], rng=random.Random(0)
    )

    next_generation = population.evolve_generation(
        fitness_fn=lambda genome: genome,
        mutation_fn=lambda genome, _rng: genome,
        crossover_fn=lambda left, _right, _rng: left,
        elitism=1,
    )

    assert next_generation[0].genome == 0.0
    assert next_generation[0].fitness == 0.0


def test_archive_zero_fitness_outranks_negative() -> None:
    archive: MapElitesArchive[float] = MapElitesArchive(lambda _genome: (0,))

    archive.add(Individual(genome=-1.0, fitness=-1.0))
    archive.add(Individual(genome=0.0, fitness=0.0))

    cell = archive.cells[(0,)]
    assert cell.fitness == 0.0

    # A worse (negative) individual must not displace the zero-fitness elite.
    archive.add(Individual(genome=-2.0, fitness=-2.0))
    assert archive.cells[(0,)].fitness == 0.0
