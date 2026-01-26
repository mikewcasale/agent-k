"""Evolutionary framework primitives for AGENT-K.

@notice: |
    Evolutionary framework primitives for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evolution.framework
    provides:
        - agent_k.evolution.framework
    pattern: evolution-framework

@agent-guidance:
    do:
        - "Use agent_k.evolution.framework as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAliasType, TypeVar

GenomeT = TypeVar("GenomeT")
"""Type variable for genomes."""

FitnessFn = TypeAliasType("FitnessFn", Callable[[GenomeT], float], type_params=(GenomeT,))
"""Callable signature for fitness evaluation."""

MutationFn = TypeAliasType("MutationFn", Callable[[GenomeT, random.Random], GenomeT], type_params=(GenomeT,))
"""Callable signature for genome mutation."""

CrossoverFn = TypeAliasType("CrossoverFn", Callable[[GenomeT, GenomeT, random.Random], GenomeT], type_params=(GenomeT,))
"""Callable signature for genome crossover."""

DescriptorFn = TypeAliasType("DescriptorFn", Callable[[GenomeT], tuple[int, ...]], type_params=(GenomeT,))
"""Callable signature for MAP-Elites descriptors."""

__all__ = (
    "EvolutionaryFramework",
    "HyperparamEvolver",
    "HyperparamGenome",
    "HyperparamSpace",
    "Individual",
    "MapElitesArchive",
    "Population",
)


@dataclass(slots=True)
class Individual(Generic[GenomeT]):
    """Evolutionary individual with lineage tracking.

    @pattern:
        name: individual-model
        rationale: "Generic dataclass for evolutionary individuals with lineage."
    """

    genome: GenomeT
    fitness: float | None = None
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)
    parents: tuple[str, str] | None = None
    lineage: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_genome(self, genome: GenomeT, *, parents: tuple[str, str] | None = None) -> Individual[GenomeT]:
        """Create a child individual while updating lineage."""
        lineage = [*self.lineage, self.identifier]
        return Individual(genome=genome, parents=parents, lineage=lineage)


class MapElitesArchive(Generic[GenomeT]):
    """MAP-Elites archive for diversity-aware preservation.

    @pattern:
        name: archive-model
        rationale: "Generic archive for MAP-Elites diversity preservation."
    """

    def __init__(self, descriptor_fn: DescriptorFn[GenomeT], *, max_cells: int | None = None) -> None:
        self._descriptor_fn = descriptor_fn
        self._cells: dict[tuple[int, ...], Individual[GenomeT]] = {}
        self._max_cells = max_cells

    @property
    def cells(self) -> dict[tuple[int, ...], Individual[GenomeT]]:
        """Return the current archive cells."""
        return dict(self._cells)

    def add(self, individual: Individual[GenomeT]) -> None:
        """Add an individual if it improves its descriptor cell."""
        if individual.fitness is None:
            return
        key = self._descriptor_fn(individual.genome)
        current = self._cells.get(key)
        if current is None or (current.fitness or 0.0) < (individual.fitness or 0.0):
            self._cells[key] = individual
            if self._max_cells and len(self._cells) > self._max_cells:
                self._trim_archive()

    def sample(self, *, top: int = 1, diverse: int = 1) -> list[Individual[GenomeT]]:
        """Sample top and diverse individuals from the archive."""
        entries = list(self._cells.values())
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda item: item.fitness or 0.0, reverse=True)
        selected = list(sorted_entries[:top])
        used_keys = {self._descriptor_fn(item.genome) for item in selected}
        for entry in sorted_entries:
            if len(selected) >= top + diverse:
                break
            key = self._descriptor_fn(entry.genome)
            if key in used_keys:
                continue
            selected.append(entry)
            used_keys.add(key)
        return selected

    def _trim_archive(self) -> None:
        if not self._max_cells:
            return
        entries = sorted(self._cells.items(), key=lambda item: item[1].fitness or 0.0, reverse=True)
        self._cells = dict(entries[: self._max_cells])


class Population(Generic[GenomeT]):
    """Population container with selection, crossover, and mutation helpers.

    @pattern:
        name: population-model
        rationale: "Generic container for evolutionary population management."
    """

    def __init__(
        self,
        individuals: list[Individual[GenomeT]] | None = None,
        *,
        rng: random.Random | None = None,
        archive: MapElitesArchive[GenomeT] | None = None,
    ) -> None:
        self.individuals = individuals or []
        self.generation = 0
        self.rng = rng or random.Random()
        self.archive = archive

    def evaluate(self, fitness_fn: FitnessFn[GenomeT]) -> None:
        """Evaluate fitness for all individuals and update archive."""
        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = fitness_fn(individual.genome)
            if self.archive is not None:
                self.archive.add(individual)

    def best(self) -> Individual[GenomeT] | None:
        """Return the current best individual."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda item: item.fitness or -math.inf)

    def tournament(self, *, size: int = 3) -> Individual[GenomeT]:
        """Select an individual via tournament selection."""
        candidates = self.rng.sample(self.individuals, k=min(size, len(self.individuals)))
        return max(candidates, key=lambda item: item.fitness or -math.inf)

    def evolve_generation(
        self,
        *,
        fitness_fn: FitnessFn[GenomeT],
        mutation_fn: MutationFn[GenomeT],
        crossover_fn: CrossoverFn[GenomeT],
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elitism: int = 1,
        tournament_size: int = 3,
    ) -> list[Individual[GenomeT]]:
        """Advance the population by one generation."""
        self.evaluate(fitness_fn)
        ranked = sorted(self.individuals, key=lambda item: item.fitness or -math.inf, reverse=True)
        next_generation: list[Individual[GenomeT]] = list(ranked[: max(0, elitism)])

        while len(next_generation) < len(self.individuals):
            parent_a = self.tournament(size=tournament_size)
            parent_b = self.tournament(size=tournament_size)
            if self.rng.random() < crossover_rate:
                genome = crossover_fn(parent_a.genome, parent_b.genome, self.rng)
                parents = (parent_a.identifier, parent_b.identifier)
            else:
                genome = parent_a.genome
                parents = None
            if self.rng.random() < mutation_rate:
                genome = mutation_fn(genome, self.rng)
            child = parent_a.with_genome(genome, parents=parents)
            next_generation.append(child)

        self.individuals = next_generation
        self.generation += 1
        return next_generation


@dataclass(slots=True)
class HyperparamSpace:
    """Search space definition for hyperparameter evolution.

    @pattern:
        name: search-space
        rationale: "Dataclass defining continuous, integer, categorical bounds."
    """

    continuous: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer: dict[str, tuple[int, int]] = field(default_factory=dict)
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class HyperparamGenome:
    """Genome wrapper for hyperparameter configurations.

    @pattern:
        name: genome-model
        rationale: "Dataclass wrapping hyperparameter key-value pairs."
    """

    parameters: dict[str, Any] = field(default_factory=dict)


class HyperparamEvolver:
    """Evolutionary hyperparameter optimizer with adaptive mutation rates.

    @pattern:
        name: evolver
        rationale: "Coordinates hyperparameter search with adaptive mutation."
    """

    def __init__(
        self,
        space: HyperparamSpace,
        *,
        population_size: int = 20,
        fitness_fn: FitnessFn[HyperparamGenome] | None = None,
        rng: random.Random | None = None,
        mutation_rate: float = 0.25,
        mutation_rate_bounds: tuple[float, float] = (0.1, 0.6),
        stagnation_generations: int = 3,
    ) -> None:
        self._space = space
        self._population_size = population_size
        self._fitness_fn = fitness_fn
        self._rng = rng or random.Random()
        self._mutation_rate = mutation_rate
        self._mutation_bounds = mutation_rate_bounds
        self._stagnation_generations = stagnation_generations
        self._stagnation = 0
        self._best_fitness: float | None = None

    def initialize_population(self) -> Population[HyperparamGenome]:
        """Create an initial hyperparameter population."""
        individuals = [Individual(self._random_genome()) for _ in range(self._population_size)]
        return Population(individuals, rng=self._rng)

    def evolve(self, *, fitness_fn: FitnessFn[HyperparamGenome] | None = None, generations: int = 10) -> dict[str, Any]:
        """Run hyperparameter evolution for a fixed number of generations."""
        population = self.initialize_population()
        resolved_fitness = fitness_fn or self._fitness_fn
        if resolved_fitness is None:
            return {"population": population, "best": None, "status": "fitness_fn_missing"}
        for _ in range(generations):
            population.evolve_generation(
                fitness_fn=resolved_fitness,
                mutation_fn=self._mutate,
                crossover_fn=self._crossover,
                mutation_rate=self._mutation_rate,
            )
            self._adjust_mutation_rate(population.best())
        return {"population": population, "best": population.best(), "status": "ok"}

    def _random_genome(self) -> HyperparamGenome:
        params: dict[str, Any] = {}
        for name, bounds in self._space.continuous.items():
            params[name] = self._rng.uniform(bounds[0], bounds[1])
        for name, bounds in self._space.integer.items():
            params[name] = self._rng.randint(bounds[0], bounds[1])
        for name, options in self._space.categorical.items():
            params[name] = self._rng.choice(options)
        return HyperparamGenome(parameters=params)

    def _mutate(self, genome: HyperparamGenome, rng: random.Random) -> HyperparamGenome:
        params = dict(genome.parameters)
        for name, bounds in self._space.continuous.items():
            if rng.random() < 0.5:
                span = bounds[1] - bounds[0]
                params[name] = min(
                    bounds[1], max(bounds[0], params.get(name, bounds[0]) + rng.uniform(-0.1, 0.1) * span)
                )
        for name, bounds in self._space.integer.items():
            if rng.random() < 0.4:
                params[name] = rng.randint(bounds[0], bounds[1])
        for name, options in self._space.categorical.items():
            if rng.random() < 0.3:
                params[name] = rng.choice(options)
        return HyperparamGenome(parameters=params)

    def _crossover(
        self, parent_a: HyperparamGenome, parent_b: HyperparamGenome, rng: random.Random
    ) -> HyperparamGenome:
        params: dict[str, Any] = {}
        keys = set(parent_a.parameters) | set(parent_b.parameters)
        for key in keys:
            params[key] = parent_a.parameters.get(key) if rng.random() < 0.5 else parent_b.parameters.get(key)
        return HyperparamGenome(parameters=params)

    def _adjust_mutation_rate(self, best: Individual[HyperparamGenome] | None) -> None:
        if best is None or best.fitness is None:
            return
        if self._best_fitness is None or best.fitness > self._best_fitness:
            self._best_fitness = best.fitness
            self._stagnation = 0
            return
        self._stagnation += 1
        if self._stagnation >= self._stagnation_generations:
            lower, upper = self._mutation_bounds
            self._mutation_rate = min(upper, self._mutation_rate * 1.1)
            self._stagnation = 0
        else:
            lower, _upper = self._mutation_bounds
            self._mutation_rate = max(lower, self._mutation_rate * 0.98)


class EvolutionaryFramework:
    """Coordinator for composing evolution components into a single loop.

    @pattern:
        name: framework
        rationale: "Composes multiple evolvers into a unified evolution loop."
    """

    def __init__(
        self,
        *,
        loss_evolver: Any | None = None,
        feature_evolver: Any | None = None,
        feature_selector: Any | None = None,
        hyperparam_evolver: HyperparamEvolver | None = None,
    ) -> None:
        self._loss_evolver = loss_evolver
        self._feature_evolver = feature_evolver
        self._feature_selector = feature_selector
        self._hyperparam_evolver = hyperparam_evolver

    def run(self, *, generations: int = 1) -> dict[str, Any]:
        """Run one coordinated evolution loop across available components."""
        results: dict[str, Any] = {"generations": generations}
        if self._loss_evolver is not None:
            results["loss"] = self._loss_evolver.evolve(generations=generations)
        if self._feature_evolver is not None:
            results["features"] = self._feature_evolver.evolve(generations=generations)
        if self._feature_selector is not None:
            results["selection"] = self._feature_selector.evolve(generations=generations)
        if self._hyperparam_evolver is not None:
            results["hyperparams"] = self._hyperparam_evolver.evolve(generations=generations)
        return results
