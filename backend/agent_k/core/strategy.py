"""Generic problem profiling, technique policy, and fitness factory for AGENT-K.

@notice: |
    Generic problem profiling, technique policy, and fitness factory for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.strategy
    provides:
        - agent_k.core.strategy
    pattern: strategy

@agent-guidance:
    do:
        - "Use agent_k.core.strategy as the canonical home for this capability."
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

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from .data import CompetitionSchema
from .models import Competition, EvaluationMetric, MissionCriteria
from .types import MetricDirection

if TYPE_CHECKING:
    pass  # FitnessInput is defined later in this module

__all__ = (
    "FitnessFunction",
    "FitnessInput",
    "FitnessPolicy",
    "ProblemProfile",
    "ProblemType",
    "TechniquePolicy",
    "apply_solution_policy",
    "build_fitness_function",
    "build_fitness_policy",
    "build_problem_profile",
    "build_technique_policy",
)

_CLASSIFICATION_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {EvaluationMetric.ACCURACY, EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS, EvaluationMetric.F1}
)
_VISION_TAGS: Final[frozenset[str]] = frozenset({"vision", "computer vision", "image", "images"})
_TEXT_TAGS: Final[frozenset[str]] = frozenset({"nlp", "text", "language"})

type FitnessFunction = Callable[["FitnessInput"], float]


class ProblemType(StrEnum):
    """Supported ML problem types for policy selection.

    @notice: |
        Supported ML problem types for policy selection.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: enumeration
            rationale: "StrEnum for ML problem taxonomy."
            violations: "String literals drift across policy logic."
    """

    TABULAR_REGRESSION = "tabular_regression"
    TABULAR_CLASSIFICATION = "tabular_classification"
    VISION_REGRESSION = "vision_regression"
    VISION_CLASSIFICATION = "vision_classification"
    TEXT_REGRESSION = "text_regression"
    TEXT_CLASSIFICATION = "text_classification"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProblemProfile:
    """Profile describing the ML task for a competition.

    @notice: |
        Profile describing the ML task for a competition.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: profile-model
            rationale: "Bundles task metadata for downstream policies."
            violations: "Implicit profiles make policy decisions inconsistent."
    """

    problem_type: ProblemType
    metric: EvaluationMetric
    metric_direction: MetricDirection
    target_columns: list[str]
    train_target_columns: list[str]
    id_column: str
    uses_proba: bool
    is_classification: bool


@dataclass(frozen=True, slots=True)
class TechniquePolicy:
    """Generic technique policy for solution enforcement.

    @notice: |
        Generic technique policy for solution enforcement.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: policy-model
            rationale: "Encodes technique constraints for evolution."
            violations: "Inline tuning fragments policy definitions."
    """

    problem_type: ProblemType
    enable_outlier_clipping: bool = False  # Disabled - handled by evolver.
    outlier_quantiles: tuple[float, float] = (0.01, 0.99)
    enable_target_transform: bool = False  # Disabled - handled by evolver.
    target_skew_threshold: float = 0.75
    min_generations: int = 5
    min_population_size: int = 6
    min_elite_archive_size: int = 3
    fitness_improvement_threshold: float = 0.001


@dataclass(frozen=True, slots=True)
class FitnessInput:
    """Input payload for fitness scoring.

    @notice: |
        Input payload for fitness scoring.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: fitness-input
            rationale: "Captures evaluation signals for scoring."
            violations: "Ad-hoc inputs drift across fitness logic."
    """

    cv_score: float
    runtime_ms: int
    complexity: int
    valid: bool
    stage: str | None
    code: str


@dataclass(frozen=True, slots=True)
class FitnessPolicy:
    """Policy controlling fitness adjustments.

    @notice: |
        Policy controlling fitness adjustments.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: policy-model
            rationale: "Encodes penalty tuning for fitness scoring."
            violations: "Inline penalties drift across runs."
    """

    metric_direction: MetricDirection
    runtime_ms_threshold: int | None = None
    complexity_threshold: int | None = None
    penalty_weight: float = 0.05
    stage_weights: dict[str, float] = field(default_factory=dict)


def build_problem_profile(competition: Competition, schema: CompetitionSchema) -> ProblemProfile:
    """Infer a generic ML task profile from competition metadata and schema.

    @notice: |
        Creates a ProblemProfile from competition metadata and data schema.

    @dev: |
        Infers problem type (tabular/vision/text, classification/regression)
        from competition tags and evaluation metric.
    """
    metric = competition.metric
    is_classification = metric in _CLASSIFICATION_METRICS
    uses_proba = metric in {EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS}

    tags = {tag.lower() for tag in competition.tags}
    if tags & _VISION_TAGS:
        problem_type = ProblemType.VISION_CLASSIFICATION if is_classification else ProblemType.VISION_REGRESSION
    elif tags & _TEXT_TAGS:
        problem_type = ProblemType.TEXT_CLASSIFICATION if is_classification else ProblemType.TEXT_REGRESSION
    else:
        problem_type = ProblemType.TABULAR_CLASSIFICATION if is_classification else ProblemType.TABULAR_REGRESSION

    return ProblemProfile(
        problem_type=problem_type,
        metric=metric,
        metric_direction=competition.metric_direction,
        target_columns=schema.target_columns,
        train_target_columns=schema.train_target_columns,
        id_column=schema.id_column,
        uses_proba=uses_proba,
        is_classification=is_classification,
    )


def build_technique_policy(profile: ProblemProfile, criteria: MissionCriteria | None = None) -> TechniquePolicy:
    """Build a technique policy from problem profile and mission criteria.

    @notice: |
        Creates a TechniquePolicy with evolution parameters for the problem type.

    @dev: |
        Adjusts population size and generation limits based on criteria.
        Target transforms and outlier clipping are disabled (handled by evolver).
    """
    enable_target_transform = profile.problem_type == ProblemType.TABULAR_REGRESSION
    min_generations = 5
    min_population_size = 6
    min_elite_archive_size = 3
    fitness_improvement_threshold = 0.001

    if criteria is not None and criteria.max_evolution_rounds >= 25:
        min_generations = 10
        min_population_size = 8
        min_elite_archive_size = 4
        fitness_improvement_threshold = 0.002

    return TechniquePolicy(
        problem_type=profile.problem_type,
        enable_outlier_clipping=profile.problem_type
        in {ProblemType.TABULAR_REGRESSION, ProblemType.TABULAR_CLASSIFICATION},
        enable_target_transform=enable_target_transform,
        min_generations=min_generations,
        min_population_size=min_population_size,
        min_elite_archive_size=min_elite_archive_size,
        fitness_improvement_threshold=fitness_improvement_threshold,
    )


def build_fitness_policy(
    profile: ProblemProfile,
    criteria: MissionCriteria | None,
    *,
    max_runtime_ms: int | None,
    complexity_threshold: int | None = None,
) -> FitnessPolicy:
    """Create a fitness policy using variable conditions and criteria.

    @notice: |
        Creates a FitnessPolicy with penalty weights and thresholds.

    @dev: |
        Penalty weight scales with min_improvements_required.
        Complexity threshold defaults based on problem type.
    """
    penalty_weight = 0.05
    if criteria is not None:
        penalty_weight = min(0.2, 0.05 + criteria.min_improvements_required * 0.02)

    if complexity_threshold is None:
        if profile.problem_type in {ProblemType.TABULAR_CLASSIFICATION, ProblemType.TABULAR_REGRESSION}:
            complexity_threshold = 800
        else:
            complexity_threshold = 600

    stage_weights = {"stage1": 0.6, "cached": 1.0}
    return FitnessPolicy(
        metric_direction=profile.metric_direction,
        runtime_ms_threshold=max_runtime_ms,
        complexity_threshold=complexity_threshold,
        penalty_weight=penalty_weight,
        stage_weights=stage_weights,
    )


def build_fitness_function(policy: FitnessPolicy) -> FitnessFunction:
    """Build a fitness function from policy.

    @notice: |
        Creates a callable fitness function from a FitnessPolicy.

    @dev: |
        Returns a closure that scores FitnessInput based on policy thresholds.
        Invalid inputs return 0.0; penalties applied for runtime/complexity.
    """

    def fitness(input_data: FitnessInput) -> float:
        if not input_data.valid:
            return 0.0

        base = _score_to_fitness(input_data.cv_score, policy.metric_direction)
        if policy.runtime_ms_threshold and input_data.runtime_ms > policy.runtime_ms_threshold:
            base *= max(0.0, 1.0 - policy.penalty_weight)
        if policy.complexity_threshold and input_data.complexity > policy.complexity_threshold:
            base *= max(0.0, 1.0 - policy.penalty_weight)
        if input_data.stage and input_data.stage in policy.stage_weights:
            base *= policy.stage_weights[input_data.stage]
        return max(0.0, base)

    return fitness


def apply_solution_policy(code: str, policy: TechniquePolicy) -> tuple[str, list[str]]:
    """Apply a technique policy to solution code when possible.

    @notice: |
        Transforms solution code based on technique policy (currently no-op).

    @dev: |
        Policy injection is disabled to allow evolutionary search to handle
        data preparation generically. Returns (code, []) unchanged.
    """
    return code, []


def _score_to_fitness(score: float, direction: MetricDirection) -> float:
    if direction == "minimize":
        return 1.0 / (1.0 + max(score, 0.0))
    return max(score, 0.0)
