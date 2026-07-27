"""Tests for evolution-framework generation semantics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.framework import HyperparamEvolver, HyperparamSpace, Individual, MapElitesArchive, Population
from agent_k.evolution.loss import LossFunctionEvolver, LossGenome

__all__ = ()


def _identity_fitness(g: dict[str, float]) -> float:
    return g["x"]


def _clone_genome(genome: dict[str, float]) -> dict[str, float]:
    return dict(genome)


def test_evolve_generation_scores_children_after_advance() -> None:
    rng = random.Random(0)
    individuals = [Individual({"x": rng.random()}) for _ in range(6)]
    population = Population(individuals, rng=rng)

    population.evolve_generation(
        fitness_fn=_identity_fitness,
        mutation_fn=lambda g, r: {"x": r.random()},
        crossover_fn=lambda a, b, r: _clone_genome(a if r.random() < 0.5 else b),
    )

    assert all(ind.fitness is not None for ind in population.individuals), (
        "Every individual in the current population should carry a fitness score."
    )


def test_hyperparam_evolver_reports_true_best_of_final_population() -> None:
    space = HyperparamSpace(continuous={"lr": (0.01, 0.1)})

    def fitness(g: object) -> float:
        return getattr(g, "parameters")["lr"]  # type: ignore[no-any-return]

    for seed in range(20):
        evolver = HyperparamEvolver(space, population_size=8, rng=random.Random(seed))
        result = evolver.evolve(fitness_fn=fitness, generations=5)
        population = result["population"]
        best = result["best"]

        assert best.fitness is not None
        assert all(ind.fitness is not None for ind in population.individuals)
        true_best = max(ind.fitness for ind in population.individuals)
        assert best.fitness >= true_best - 1e-12, (
            f"seed={seed}: reported best {best.fitness} lags true best {true_best}"
        )


def test_loss_function_evolver_evaluates_last_generation_children() -> None:
    call_count = {"n": 0}

    def fitness(genome: LossGenome) -> float:
        call_count["n"] += 1
        return genome.asymmetric_weight

    evolver = LossFunctionEvolver(fitness_fn=fitness, population_size=8, rng=random.Random(3))
    result = evolver.evolve(generations=3)

    population = result["population"]
    assert all(ind.fitness is not None for ind in population.individuals)
    assert result["best_fitness"] is not None


def test_map_elites_archive_receives_last_generation_children() -> None:
    rng = random.Random(7)
    seen_keys: set[tuple[int, ...]] = set()

    def descriptor(genome: dict[str, float]) -> tuple[int, ...]:
        key = (int(genome["x"] * 100),)
        seen_keys.add(key)
        return key

    archive: MapElitesArchive[dict[str, float]] = MapElitesArchive(descriptor)
    individuals = [Individual({"x": rng.random()}) for _ in range(4)]
    population = Population(individuals, rng=rng, archive=archive)

    population.evolve_generation(
        fitness_fn=_identity_fitness,
        mutation_fn=lambda g, r: {"x": r.random()},
        crossover_fn=lambda a, b, r: _clone_genome(a if r.random() < 0.5 else b),
    )

    archived_fitness = {cell.fitness for cell in archive.cells.values()}
    population_fitness = {ind.fitness for ind in population.individuals}
    assert population_fitness.issubset(archived_fitness), (
        "Archive should contain every current-generation individual's fitness (including freshly-created children)."
    )
