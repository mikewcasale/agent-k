"""Core domain models for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Final, Self

# Third-party (alphabetical)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

# Local imports (core first, then alphabetical)
from .types import (  # noqa: TC001
    CompetitionId,
    ErrorCategory,
    FitnessScore,
    MemoryScope,
    MetricDirection,
    MissionId,
    MissionPhase,
    RecoveryStrategy,
    TaskId,
    TaskPriority,
    TaskStatus,
    ToolType,
)

__all__ = (
    # Enums
    "CompetitionType",
    "EvaluationMetric",
    # Competition models
    "Competition",
    "LeaderboardEntry",
    "Submission",
    # Mission models
    "MissionCriteria",
    # Task models
    "PlannedTask",
    "PhasePlan",
    "MissionPlan",
    # Tool models
    "ToolCall",
    "WebSearchCall",
    "KaggleMCPCall",
    "CodeExecutorCall",
    "MemoryCall",
    # Evolution models
    "GenerationMetrics",
    "EvolutionState",
    "LeaderboardSubmission",
    # Memory models
    "MemoryEntry",
    "Checkpoint",
    "MemoryState",
    # Error models
    "ErrorEvent",
    # Research models
    "LeaderboardAnalysis",
    "ResearchFindings",
)

SCHEMA_VERSION: Final[str] = "1.0.0"


class CompetitionType(str, Enum):
    """Type of Kaggle competition."""

    FEATURED = "featured"
    RESEARCH = "research"
    GETTING_STARTED = "getting_started"
    PLAYGROUND = "playground"
    COMMUNITY = "community"


class EvaluationMetric(str, Enum):
    """Standard evaluation metrics for competitions."""

    # Classification
    ACCURACY = "accuracy"
    AUC = "auc"
    LOG_LOSS = "logLoss"
    F1 = "f1"

    # Regression
    RMSE = "rmse"
    MAE = "mae"
    RMSLE = "rmsle"

    # Ranking
    MAP = "map"
    NDCG = "ndcg"


# =============================================================================
# Competition Models
# =============================================================================
class Competition(BaseModel):
    """Kaggle competition entity.

    Represents a competition with all metadata required for evaluation
    and participation decisions.
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    id: CompetitionId = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9-]+$",
        description="Unique competition identifier (slug)",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Competition display title",
    )
    description: str | None = Field(
        default=None,
        max_length=10000,
        description="Competition description",
    )
    competition_type: CompetitionType = Field(
        ...,
        description="Category of competition",
    )
    metric: EvaluationMetric = Field(
        ...,
        description="Primary evaluation metric",
    )
    metric_direction: MetricDirection = Field(
        default="maximize",
        description="Whether higher or lower metric values are better",
    )
    deadline: datetime = Field(
        ...,
        description="Competition submission deadline (UTC)",
    )
    prize_pool: int | None = Field(
        default=None,
        ge=0,
        description="Total prize pool in USD",
    )
    max_team_size: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Maximum allowed team size",
    )
    max_daily_submissions: int = Field(
        default=5,
        ge=1,
        description="Maximum submissions per day",
    )
    tags: frozenset[str] = Field(
        default_factory=frozenset,
        description="Competition tags/categories",
    )
    url: str | None = Field(
        default=None,
        description="Full URL to competition page",
    )

    @field_validator("deadline")
    @classmethod
    def validate_deadline_timezone(cls, v: datetime) -> datetime:
        """Ensure deadline has timezone information."""
        if v.tzinfo is None:
            raise ValueError("deadline must be timezone-aware")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_active(self) -> bool:
        """Whether competition is still accepting submissions."""
        return datetime.now(UTC) < self.deadline

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days_remaining(self) -> int:
        """Days until deadline (negative if passed)."""
        delta = self.deadline - datetime.now(UTC)
        return delta.days


class LeaderboardEntry(BaseModel):
    """Entry in competition leaderboard."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    rank: int = Field(..., ge=1, description="Position on leaderboard")
    team_name: str = Field(..., min_length=1, description="Team name")
    score: float = Field(..., description="Public leaderboard score")
    entries: int = Field(default=1, ge=1, description="Number of submissions")
    last_submission: datetime | None = Field(default=None, description="Last submission time")


class Submission(BaseModel):
    """Competition submission entity."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    id: str = Field(..., description="Unique submission identifier")
    competition_id: CompetitionId = Field(..., description="Target competition")
    file_name: str = Field(..., description="Submission file name")
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Submission timestamp",
    )
    public_score: float | None = Field(
        default=None,
        description="Public leaderboard score (if evaluated)",
    )
    private_score: float | None = Field(
        default=None,
        description="Private leaderboard score (after competition ends)",
    )
    status: str = Field(
        default="pending",
        pattern=r"^(pending|complete|error)$",
        description="Submission status",
    )
    error_message: str | None = Field(default=None, description="Error message if failed")


# =============================================================================
# Tool Call Models (For Frontend Event Streaming)
# =============================================================================
class ToolCall(BaseModel):
    """Base model for tool invocation tracking."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    id: str = Field(..., description="Unique tool call identifier")
    type: ToolType = Field(..., description="Type of tool")
    operation: str = Field(..., description="Operation name")
    params: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    thinking: str | None = Field(default=None, description="Agent thinking block")
    result: Any | None = Field(default=None, description="Tool result payload")
    error: str | None = Field(default=None, description="Tool error message")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Tool start time",
    )
    completed_at: datetime | None = Field(default=None, description="Tool completion time")
    duration_ms: int | None = Field(default=None, description="Duration in milliseconds")


class WebSearchCall(ToolCall):
    """Web search tool call."""

    type: ToolType = Field(default="web_search", description="Tool type")
    query: str = Field(..., description="Search query")
    result_count: int | None = Field(default=None, description="Number of results returned")
    results: list[dict[str, str]] = Field(default_factory=list, description="Search result entries")


class KaggleMCPCall(ToolCall):
    """Kaggle MCP tool call."""

    type: ToolType = Field(default="kaggle_mcp", description="Tool type")
    competition_id: CompetitionId | None = Field(default=None, description="Target competition id")


class CodeExecutorCall(ToolCall):
    """Code execution tool call."""

    type: ToolType = Field(default="code_executor", description="Tool type")
    code: str = Field(..., description="Executed code")
    language: str = Field(default="python", description="Execution language")
    stdout: str | None = Field(default=None, description="Captured stdout")
    stderr: str | None = Field(default=None, description="Captured stderr")
    execution_time_ms: int | None = Field(default=None, description="Execution time in ms")
    memory_usage_mb: float | None = Field(default=None, description="Memory usage in MB")


class MemoryCall(ToolCall):
    """Memory operation tool call."""

    type: ToolType = Field(default="memory", description="Tool type")
    key: str = Field(..., description="Memory key")
    scope: MemoryScope = Field(default="session", description="Memory scope")
    value_preview: str | None = Field(default=None, description="Preview of stored value")


# =============================================================================
# Task Planning Models
# =============================================================================
class PlannedTask(BaseModel):
    """A planned task within a phase."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    id: TaskId = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task display name")
    description: str = Field(..., description="Task description")
    agent: str = Field(..., description="Agent responsible for task")
    tools_required: list[ToolType] = Field(default_factory=list, description="Required tool types")
    estimated_duration_ms: int = Field(default=30000, description="Estimated duration in ms")
    actual_duration_ms: int | None = Field(default=None, description="Actual duration in ms")
    priority: TaskPriority = Field(default="medium", description="Task priority")
    dependencies: list[TaskId] = Field(default_factory=list, description="Dependent task ids")
    status: TaskStatus = Field(default="pending", description="Task status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
    result: Any | None = Field(default=None, description="Task result payload")
    error: str | None = Field(default=None, description="Task error message")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls executed")
    started_at: datetime | None = Field(default=None, description="Task start time")
    completed_at: datetime | None = Field(default=None, description="Task completion time")


class PhasePlan(BaseModel):
    """Plan for a mission phase."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    phase: MissionPhase = Field(..., description="Phase identifier")
    display_name: str = Field(..., description="Human-readable phase name")
    objectives: list[str] = Field(default_factory=list, description="Phase objectives")
    success_criteria: list[str] = Field(default_factory=list, description="Success criteria")
    tasks: list[PlannedTask] = Field(default_factory=list, description="Tasks in phase")
    timeout_ms: int = Field(default=300000, description="Phase timeout in ms")
    fallback_strategy: str | None = Field(default=None, description="Fallback strategy")
    status: TaskStatus = Field(default="pending", description="Phase status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Phase progress")
    started_at: datetime | None = Field(default=None, description="Phase start time")
    completed_at: datetime | None = Field(default=None, description="Phase completion time")


class MissionPlan(BaseModel):
    """Complete mission plan with all phases."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    mission_id: MissionId = Field(..., description="Unique mission identifier")
    competition_id: CompetitionId | None = Field(default=None, description="Competition id")
    phases: list[PhasePlan] = Field(default_factory=list, description="Phase plans")
    total_estimated_duration_ms: int = Field(
        default=0, description="Total estimated duration in ms"
    )
    checkpoints: list[str] = Field(default_factory=list, description="Checkpoint identifiers")


# =============================================================================
# Evolution Models
# =============================================================================
class GenerationMetrics(BaseModel):
    """Metrics for a single evolution generation."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    generation: int = Field(..., ge=0, description="Generation index")
    best_fitness: FitnessScore = Field(..., description="Best fitness score")
    mean_fitness: FitnessScore = Field(..., description="Mean fitness score")
    worst_fitness: FitnessScore = Field(..., description="Worst fitness score")
    population_size: int = Field(..., ge=1, description="Population size")
    mutations: dict[str, int] = Field(
        default_factory=lambda: {
            "point": 0,
            "structural": 0,
            "hyperparameter": 0,
            "crossover": 0,
        },
        description="Mutation counts by type",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp for generation metrics",
    )


class LeaderboardSubmission(BaseModel):
    """Record of a submission to the competition leaderboard."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    submission_id: str = Field(..., description="Submission identifier")
    generation: int = Field(..., description="Generation index")
    cv_score: FitnessScore = Field(..., description="Cross-validation score")
    public_score: FitnessScore | None = Field(default=None, description="Public leaderboard score")
    rank: int | None = Field(default=None, description="Leaderboard rank")
    total_teams: int | None = Field(default=None, description="Total teams on leaderboard")
    percentile: float | None = Field(default=None, description="Leaderboard percentile")
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Submission timestamp",
    )


class EvolutionState(BaseModel):
    """State of the evolution process."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    current_generation: int = Field(default=0, description="Current generation")
    max_generations: int = Field(default=100, description="Maximum generations")
    population_size: int = Field(default=50, description="Population size")
    best_solution: dict[str, Any] | None = Field(default=None, description="Best solution payload")
    generation_history: list[GenerationMetrics] = Field(
        default_factory=list, description="History of generations"
    )
    convergence_detected: bool = Field(default=False, description="Whether convergence detected")
    convergence_reason: str | None = Field(default=None, description="Convergence reason")
    leaderboard_submissions: list[LeaderboardSubmission] = Field(
        default_factory=list,
        description="Leaderboard submissions",
    )


# =============================================================================
# Memory Models
# =============================================================================
class MemoryEntry(BaseModel):
    """Entry in the memory store."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    key: str = Field(..., description="Memory key")
    scope: MemoryScope = Field(default="session", description="Memory scope")
    category: str = Field(..., description="Category for grouping")
    value_preview: str = Field(..., max_length=200, description="Preview of stored value")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last accessed timestamp",
    )
    access_count: int = Field(default=1, description="Access count")
    size_bytes: int = Field(default=0, description="Approximate size in bytes")


class Checkpoint(BaseModel):
    """Mission state checkpoint."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    name: str = Field(..., description="Checkpoint name")
    phase: MissionPhase = Field(..., description="Phase when checkpoint was created")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Checkpoint timestamp",
    )
    state_snapshot: str = Field(..., description="Serialized state")


class MemoryState(BaseModel):
    """Overall memory state."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    entries: list[MemoryEntry] = Field(default_factory=list, description="Memory entries")
    checkpoints: list[Checkpoint] = Field(default_factory=list, description="State checkpoints")
    total_size_bytes: int = Field(default=0, description="Total size in bytes")


# =============================================================================
# Error Models
# =============================================================================
class ErrorEvent(BaseModel):
    """Record of an error event."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    id: str = Field(..., description="Unique error identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Error timestamp",
    )
    category: ErrorCategory = Field(..., description="Error category")
    error_type: str = Field(..., description="Exception class name")
    message: str = Field(..., description="Error message")
    context: str = Field(default="", description="Error context")
    task_id: TaskId | None = Field(default=None, description="Related task id")
    phase: MissionPhase | None = Field(default=None, description="Related phase")
    recovery_strategy: RecoveryStrategy = Field(default="retry", description="Recovery strategy")
    recovery_attempts: int = Field(default=0, description="Recovery attempts")
    resolved: bool = Field(default=False, description="Whether resolved")
    resolution: str | None = Field(default=None, description="Resolution details")


# =============================================================================
# Research Models
# =============================================================================
class LeaderboardAnalysis(BaseModel):
    """Analysis of competition leaderboard."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    top_score: float = Field(..., description="Top leaderboard score")
    median_score: float = Field(..., description="Median leaderboard score")
    target_score: float = Field(..., description="Target score for goal percentile")
    target_percentile: float = Field(..., description="Target percentile")
    total_teams: int = Field(..., description="Total teams on leaderboard")
    score_distribution: list[dict[str, float]] = Field(
        default_factory=list, description="Score distribution"
    )
    common_approaches: list[str] = Field(default_factory=list, description="Common approaches")
    improvement_opportunities: list[str] = Field(
        default_factory=list, description="Improvement opportunities"
    )


class ResearchFindings(BaseModel):
    """Complete research findings for a competition."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    leaderboard_analysis: LeaderboardAnalysis | None = Field(
        default=None, description="Leaderboard analysis"
    )
    papers: list[dict[str, Any]] = Field(default_factory=list, description="Paper findings")
    approaches: list[dict[str, Any]] = Field(default_factory=list, description="Approach findings")
    eda_results: dict[str, Any] | None = Field(default=None, description="EDA results")
    strategy_recommendations: list[str] = Field(
        default_factory=list, description="Strategy recommendations"
    )


# =============================================================================
# Mission State Models
# =============================================================================
class MissionCriteria(BaseModel):
    """Criteria constraining mission execution."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    target_competition_types: frozenset[CompetitionType] = Field(
        default=frozenset({CompetitionType.FEATURED, CompetitionType.RESEARCH}),
        description="Target competition types",
    )
    min_prize_pool: int | None = Field(default=None, ge=0, description="Minimum prize pool")
    max_team_size: int | None = Field(default=None, ge=1, description="Maximum team size")
    min_days_remaining: int = Field(default=7, ge=1, description="Minimum days remaining")
    target_domains: frozenset[str] = Field(default_factory=frozenset, description="Target domains")
    exclude_domains: frozenset[str] = Field(
        default_factory=frozenset, description="Excluded domains"
    )
    max_evolution_rounds: int = Field(default=100, ge=1, description="Max evolution rounds")
    target_leaderboard_percentile: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Target top N percentile on leaderboard",
    )

    @model_validator(mode="after")
    def validate_domains_disjoint(self) -> Self:
        """Ensure target and exclude domains don't overlap."""
        overlap = self.target_domains & self.exclude_domains
        if overlap:
            raise ValueError(f"Domains cannot be both targeted and excluded: {overlap}")
        return self
