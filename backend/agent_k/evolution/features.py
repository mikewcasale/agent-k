"""Evolutionary feature engineering and selection primitives.

@notice: |
    Evolutionary feature engineering and selection primitives.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evolution.features
    provides:
        - agent_k.evolution.features
    pattern: feature-engineering

@agent-guidance:
    do:
        - "Use agent_k.evolution.features as the canonical home for this capability."
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

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_k.evolution.framework import FitnessFn, Individual, Population

type TransformType = Literal["log", "sqrt", "square", "boxcox", "yeojohnson"]
"""Supported transformation genes."""

type BinningStrategy = Literal["equal_width", "equal_frequency", "kmeans"]
"""Supported binning strategies."""

__all__ = (
    "BinningGene",
    "DomainFeatureGene",
    "FeatureEvolver",
    "FeatureGenome",
    "FeatureSelector",
    "FeatureSelectionIndividual",
    "InteractionGene",
    "RatioGene",
    "TransformGene",
)


@dataclass(frozen=True, slots=True)
class TransformGene:
    """Single-variable transformation gene.

    @notice: |
        Single-variable transformation gene.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: gene-model
            rationale: "Frozen dataclass representing a transformation gene."
    """

    feature: str
    transform: TransformType


@dataclass(frozen=True, slots=True)
class InteractionGene:
    """Polynomial interaction gene.

    @notice: |
        Polynomial interaction gene.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: gene-model
            rationale: "Frozen dataclass representing an interaction gene."
    """

    feature_a: str
    feature_b: str


@dataclass(frozen=True, slots=True)
class RatioGene:
    """Ratio feature gene (feature_a / feature_b).

    @notice: |
        Ratio feature gene (feature_a / feature_b).

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: gene-model
            rationale: "Frozen dataclass representing a ratio feature gene."
    """

    numerator: str
    denominator: str


@dataclass(frozen=True, slots=True)
class BinningGene:
    """Binning strategy gene.

    @notice: |
        Binning strategy gene.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: gene-model
            rationale: "Frozen dataclass representing a binning strategy gene."
    """

    feature: str
    strategy: BinningStrategy
    bins: int


@dataclass(frozen=True, slots=True)
class DomainFeatureGene:
    """Domain-specific engineered feature gene.

    @notice: |
        Domain-specific engineered feature gene.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: gene-model
            rationale: "Frozen dataclass representing a domain feature gene."
    """

    name: str
    inputs: tuple[str, ...]


@dataclass(slots=True)
class FeatureGenome:
    """Complete feature engineering genome.

    @notice: |
        Complete feature engineering genome.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: genome-model
            rationale: "Mutable dataclass aggregating feature engineering genes."
    """

    transforms: list[TransformGene] = field(default_factory=list)
    interactions: list[InteractionGene] = field(default_factory=list)
    ratios: list[RatioGene] = field(default_factory=list)
    binnings: list[BinningGene] = field(default_factory=list)
    domain_features: list[DomainFeatureGene] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureSelectionIndividual:
    """Binary chromosome for feature inclusion.

    @notice: |
        Binary chromosome for feature inclusion.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: individual-model
            rationale: "Mutable dataclass for binary feature mask with fitness."
    """

    mask: list[int]
    score: float | None = None

    def selected_count(self) -> int:
        """Return the number of selected features."""
        return sum(self.mask)


class FeatureEvolver:
    """Evolve feature engineering pipelines with mutation and crossover.

    @notice: |
        Evolve feature engineering pipelines with mutation and crossover.

    @dev: |
        Domain features are caller-supplied so the evolver stays generic across
        ML problem types. Defaults to no domain features when none are provided.

        @pattern:
            name: evolver
            rationale: "Coordinates evolutionary search over feature genomes."
    """

    def __init__(
        self,
        feature_names: list[str],
        *,
        rng: random.Random | None = None,
        population_size: int = 12,
        domain_features: tuple[DomainFeatureGene, ...] = (),
    ) -> None:
        self._feature_names = feature_names
        self._rng = rng or random.Random()
        self._population_size = population_size
        self._domain_features: tuple[DomainFeatureGene, ...] = tuple(domain_features)
        self._population = self._initialize_population()

    def evolve(self, *, generations: int = 4, fitness_fn: FitnessFn[FeatureGenome] | None = None) -> dict[str, Any]:
        """Run feature evolution for a fixed number of generations."""
        if fitness_fn is None:
            return {"population": self._population, "best": None}
        for _ in range(generations):
            self._population.evolve_generation(
                fitness_fn=fitness_fn, mutation_fn=self._mutate, crossover_fn=self._crossover, mutation_rate=0.35
            )
        best = self._population.best()
        return {"population": self._population, "best": best}

    def _initialize_population(self) -> Population[FeatureGenome]:
        individuals = [Individual(self._random_genome()) for _ in range(self._population_size)]
        return Population(individuals, rng=self._rng)

    def _random_genome(self) -> FeatureGenome:
        selected = self._rng.sample(self._feature_names, k=max(1, len(self._feature_names) // 3))
        transforms = [
            TransformGene(feature=feat, transform=self._rng.choice(["log", "sqrt", "square"])) for feat in selected[:2]
        ]
        ratios = [RatioGene(numerator=selected[0], denominator=selected[-1])] if len(selected) > 2 else []
        return FeatureGenome(
            transforms=transforms,
            interactions=[InteractionGene(selected[0], selected[-1])] if len(selected) > 2 else [],
            ratios=ratios,
            binnings=[BinningGene(selected[0], "equal_frequency", bins=10)] if selected else [],
            domain_features=list(self._domain_features[:2]),
            selected_features=selected,
        )

    def _mutate(self, genome: FeatureGenome, rng: random.Random) -> FeatureGenome:
        selected = list(genome.selected_features)
        if selected and rng.random() < 0.4:
            removed = rng.choice(selected)
            selected = [feat for feat in selected if feat != removed]
        if rng.random() < 0.5:
            selected.append(rng.choice(self._feature_names))
        transforms = list(genome.transforms)
        if selected and rng.random() < 0.5:
            transforms.append(
                TransformGene(
                    feature=rng.choice(selected),
                    transform=rng.choice(["log", "sqrt", "square", "boxcox", "yeojohnson"]),
                )
            )
        ratios = list(genome.ratios)
        if len(selected) >= 2 and rng.random() < 0.4:
            ratios.append(RatioGene(numerator=selected[0], denominator=selected[-1]))
        return FeatureGenome(
            transforms=transforms,
            interactions=list(genome.interactions),
            ratios=ratios,
            binnings=list(genome.binnings),
            domain_features=list(genome.domain_features),
            selected_features=sorted(set(selected)),
            metadata=dict(genome.metadata),
        )

    def _crossover(self, parent_a: FeatureGenome, parent_b: FeatureGenome, rng: random.Random) -> FeatureGenome:
        selected = list({*parent_a.selected_features, *parent_b.selected_features})
        rng.shuffle(selected)
        return FeatureGenome(
            transforms=(parent_a.transforms if rng.random() < 0.5 else parent_b.transforms),
            interactions=(parent_a.interactions if rng.random() < 0.5 else parent_b.interactions),
            ratios=(parent_a.ratios if rng.random() < 0.5 else parent_b.ratios),
            binnings=(parent_a.binnings if rng.random() < 0.5 else parent_b.binnings),
            domain_features=(parent_a.domain_features if rng.random() < 0.5 else parent_b.domain_features),
            selected_features=selected[: max(1, len(selected) // 2)],
        )


class FeatureSelector:
    """Maintain a Pareto front for feature selection trade-offs.

    @notice: |
        Maintain a Pareto front for feature selection trade-offs.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: selector
            rationale: "Pareto-based feature selection with population evolution."
    """

    def __init__(
        self, feature_names: list[str], *, population_size: int = 20, rng: random.Random | None = None
    ) -> None:
        self._feature_names = feature_names
        self._population_size = population_size
        self._rng = rng or random.Random()
        self._population = self._initialize_population()
        self._pareto_front: list[FeatureSelectionIndividual] = []

    def evolve(self, *, generations: int = 5, fitness_fn: Callable[[list[str]], float] | None = None) -> dict[str, Any]:
        """Evolve feature selections while maintaining Pareto front."""
        if fitness_fn is None:
            return {"pareto_front": self._pareto_front, "population": self._population}
        for _ in range(generations):
            for individual in self._population:
                if individual.score is None:
                    features = self._apply_mask(individual.mask)
                    individual.score = fitness_fn(features)
            self._update_pareto_front()
            self._population = self._breed_next_generation()
        return {"pareto_front": self._pareto_front, "population": self._population}

    def _initialize_population(self) -> list[FeatureSelectionIndividual]:
        population: list[FeatureSelectionIndividual] = []
        for _ in range(self._population_size):
            mask = [1 if self._rng.random() < 0.5 else 0 for _ in self._feature_names]
            population.append(FeatureSelectionIndividual(mask=mask))
        return population

    def _apply_mask(self, mask: list[int]) -> list[str]:
        return [name for name, flag in zip(self._feature_names, mask, strict=False) if flag]

    def _update_pareto_front(self) -> None:
        front: list[FeatureSelectionIndividual] = []
        for candidate in self._population:
            if candidate.score is None:
                continue
            dominated = False
            for other in self._population:
                if other is candidate or other.score is None:
                    continue
                if _dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        self._pareto_front = front

    def _breed_next_generation(self) -> list[FeatureSelectionIndividual]:
        next_population: list[FeatureSelectionIndividual] = []
        while len(next_population) < self._population_size:
            parent_a = self._rng.choice(self._population)
            parent_b = self._rng.choice(self._population)
            split = self._rng.randint(1, len(parent_a.mask) - 1)
            child_mask = parent_a.mask[:split] + parent_b.mask[split:]
            if self._rng.random() < 0.4:
                idx = self._rng.randrange(len(child_mask))
                child_mask[idx] = 1 - child_mask[idx]
            next_population.append(FeatureSelectionIndividual(mask=child_mask))
        return next_population


def _dominates(left: FeatureSelectionIndividual, right: FeatureSelectionIndividual) -> bool:
    if left.score is None or right.score is None:
        return False
    return left.score >= right.score and left.selected_count() <= right.selected_count()
