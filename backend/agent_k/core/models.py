"""Core domain models for AGENT-K.

These models represent the fundamental entities in the competition domain,
including enhanced models for frontend event streaming.

All models are immutable by default (frozen=True) to prevent accidental mutation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from .types import (
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

__all__ = [
    # Enums
    'CompetitionType',
    'EvaluationMetric',
    # Competition models
    'Competition',
    'LeaderboardEntry',
    'Submission',
    # Mission models
    'MissionCriteria',
    'MissionState',
    'MissionResult',
    # Task models
    'PlannedTask',
    'PhasePlan',
    'MissionPlan',
    # Tool models
    'ToolCall',
    'WebSearchCall',
    'KaggleMCPCall',
    'CodeExecutorCall',
    'MemoryCall',
    # Evolution models
    'GenerationMetrics',
    'EvolutionState',
    'LeaderboardSubmission',
    # Memory models
    'MemoryEntry',
    'Checkpoint',
    'MemoryState',
    # Error models
    'ErrorEvent',
    # Research models
    'LeaderboardAnalysis',
    'ResearchFindings',
]


# =============================================================================
# Enumerations
# =============================================================================
class CompetitionType(str, Enum):
    """Type of Kaggle competition."""
    
    FEATURED = 'featured'
    RESEARCH = 'research'
    GETTING_STARTED = 'getting_started'
    PLAYGROUND = 'playground'
    COMMUNITY = 'community'


class EvaluationMetric(str, Enum):
    """Standard evaluation metrics for competitions."""
    
    # Classification
    ACCURACY = 'accuracy'
    AUC = 'auc'
    LOG_LOSS = 'logLoss'
    F1 = 'f1'
    
    # Regression
    RMSE = 'rmse'
    MAE = 'mae'
    RMSLE = 'rmsle'
    
    # Ranking
    MAP = 'map'
    NDCG = 'ndcg'


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
    
    id: CompetitionId = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9-]+$',
        description='Unique competition identifier (slug)',
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description='Competition display title',
    )
    description: str | None = Field(
        default=None,
        max_length=10000,
        description='Competition description',
    )
    competition_type: CompetitionType = Field(
        ...,
        description='Category of competition',
    )
    metric: EvaluationMetric = Field(
        ...,
        description='Primary evaluation metric',
    )
    metric_direction: MetricDirection = Field(
        default='maximize',
        description='Whether higher or lower metric values are better',
    )
    deadline: datetime = Field(
        ...,
        description='Competition submission deadline (UTC)',
    )
    prize_pool: int | None = Field(
        default=None,
        ge=0,
        description='Total prize pool in USD',
    )
    max_team_size: int = Field(
        default=1,
        ge=1,
        le=100,
        description='Maximum allowed team size',
    )
    max_daily_submissions: int = Field(
        default=5,
        ge=1,
        description='Maximum submissions per day',
    )
    tags: frozenset[str] = Field(
        default_factory=frozenset,
        description='Competition tags/categories',
    )
    url: str | None = Field(
        default=None,
        description='Full URL to competition page',
    )
    
    @field_validator('deadline')
    @classmethod
    def validate_deadline_timezone(cls, v: datetime) -> datetime:
        """Ensure deadline has timezone information."""
        if v.tzinfo is None:
            raise ValueError('deadline must be timezone-aware')
        return v
    
    @computed_field
    @property
    def is_active(self) -> bool:
        """Whether competition is still accepting submissions."""
        return datetime.now(timezone.utc) < self.deadline
    
    @computed_field
    @property
    def days_remaining(self) -> int:
        """Days until deadline (negative if passed)."""
        delta = self.deadline - datetime.now(timezone.utc)
        return delta.days


class LeaderboardEntry(BaseModel):
    """Entry in competition leaderboard."""
    
    model_config = ConfigDict(frozen=True)
    
    rank: int = Field(..., ge=1, description='Position on leaderboard')
    team_name: str = Field(..., min_length=1)
    score: float = Field(..., description='Public leaderboard score')
    entries: int = Field(default=1, ge=1, description='Number of submissions')
    last_submission: datetime | None = Field(default=None)


class Submission(BaseModel):
    """Competition submission entity."""
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description='Unique submission identifier')
    competition_id: CompetitionId = Field(..., description='Target competition')
    file_name: str = Field(..., description='Submission file name')
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    public_score: float | None = Field(
        default=None,
        description='Public leaderboard score (if evaluated)',
    )
    private_score: float | None = Field(
        default=None,
        description='Private leaderboard score (after competition ends)',
    )
    status: Annotated[
        str,
        Field(pattern=r'^(pending|complete|error)$'),
    ] = Field(default='pending')
    error_message: str | None = Field(default=None)


# =============================================================================
# Tool Call Models (For Frontend Event Streaming)
# =============================================================================
class ToolCall(BaseModel):
    """Base model for tool invocation tracking."""
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description='Unique tool call identifier')
    type: ToolType = Field(..., description='Type of tool')
    operation: str = Field(..., description='Operation name')
    params: dict[str, Any] = Field(default_factory=dict)
    thinking: str | None = Field(default=None, description='Agent thinking block')
    result: Any | None = Field(default=None)
    error: str | None = Field(default=None)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)
    duration_ms: int | None = Field(default=None)


class WebSearchCall(ToolCall):
    """Web search tool call."""
    
    type: ToolType = Field(default='web_search')
    query: str = Field(..., description='Search query')
    result_count: int | None = Field(default=None)
    results: list[dict[str, str]] = Field(default_factory=list)


class KaggleMCPCall(ToolCall):
    """Kaggle MCP tool call."""
    
    type: ToolType = Field(default='kaggle_mcp')
    competition_id: CompetitionId | None = Field(default=None)


class CodeExecutorCall(ToolCall):
    """Code execution tool call."""
    
    type: ToolType = Field(default='code_executor')
    code: str = Field(..., description='Executed code')
    language: str = Field(default='python')
    stdout: str | None = Field(default=None)
    stderr: str | None = Field(default=None)
    execution_time_ms: int | None = Field(default=None)
    memory_usage_mb: float | None = Field(default=None)


class MemoryCall(ToolCall):
    """Memory operation tool call."""
    
    type: ToolType = Field(default='memory')
    key: str = Field(..., description='Memory key')
    scope: MemoryScope = Field(default='session')
    value_preview: str | None = Field(default=None)


# =============================================================================
# Task Planning Models
# =============================================================================
class PlannedTask(BaseModel):
    """A planned task within a phase."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: TaskId = Field(..., description='Unique task identifier')
    name: str = Field(..., description='Task display name')
    description: str = Field(..., description='Task description')
    agent: str = Field(..., description='Agent responsible for task')
    tools_required: list[ToolType] = Field(default_factory=list)
    estimated_duration_ms: int = Field(default=30000)
    actual_duration_ms: int | None = Field(default=None)
    priority: TaskPriority = Field(default='medium')
    dependencies: list[TaskId] = Field(default_factory=list)
    status: TaskStatus = Field(default='pending')
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    result: Any | None = Field(default=None)
    error: str | None = Field(default=None)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)


class PhasePlan(BaseModel):
    """Plan for a mission phase."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    phase: MissionPhase = Field(..., description='Phase identifier')
    display_name: str = Field(..., description='Human-readable phase name')
    objectives: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    tasks: list[PlannedTask] = Field(default_factory=list)
    timeout_ms: int = Field(default=300000)
    fallback_strategy: str | None = Field(default=None)
    status: TaskStatus = Field(default='pending')
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)


class MissionPlan(BaseModel):
    """Complete mission plan with all phases."""
    
    model_config = ConfigDict(frozen=True)
    
    mission_id: MissionId = Field(..., description='Unique mission identifier')
    competition_id: CompetitionId | None = Field(default=None)
    phases: list[PhasePlan] = Field(default_factory=list)
    total_estimated_duration_ms: int = Field(default=0)
    checkpoints: list[str] = Field(default_factory=list)


# =============================================================================
# Evolution Models
# =============================================================================
class GenerationMetrics(BaseModel):
    """Metrics for a single evolution generation."""
    
    model_config = ConfigDict(frozen=True)
    
    generation: int = Field(..., ge=0)
    best_fitness: FitnessScore = Field(...)
    mean_fitness: FitnessScore = Field(...)
    worst_fitness: FitnessScore = Field(...)
    population_size: int = Field(..., ge=1)
    mutations: dict[str, int] = Field(
        default_factory=lambda: {
            'point': 0,
            'structural': 0,
            'hyperparameter': 0,
            'crossover': 0,
        }
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LeaderboardSubmission(BaseModel):
    """Record of a submission to the competition leaderboard."""
    
    model_config = ConfigDict(frozen=True)
    
    submission_id: str = Field(...)
    generation: int = Field(...)
    cv_score: FitnessScore = Field(...)
    public_score: FitnessScore | None = Field(default=None)
    rank: int | None = Field(default=None)
    total_teams: int | None = Field(default=None)
    percentile: float | None = Field(default=None)
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvolutionState(BaseModel):
    """State of the evolution process."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    current_generation: int = Field(default=0)
    max_generations: int = Field(default=100)
    population_size: int = Field(default=50)
    best_solution: dict[str, Any] | None = Field(default=None)
    generation_history: list[GenerationMetrics] = Field(default_factory=list)
    convergence_detected: bool = Field(default=False)
    convergence_reason: str | None = Field(default=None)
    leaderboard_submissions: list[LeaderboardSubmission] = Field(default_factory=list)


# =============================================================================
# Memory Models
# =============================================================================
class MemoryEntry(BaseModel):
    """Entry in the memory store."""
    
    model_config = ConfigDict(frozen=True)
    
    key: str = Field(..., description='Memory key')
    scope: MemoryScope = Field(default='session')
    category: str = Field(..., description='Category for grouping')
    value_preview: str = Field(..., max_length=200)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(default=1)
    size_bytes: int = Field(default=0)


class Checkpoint(BaseModel):
    """Mission state checkpoint."""
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(..., description='Checkpoint name')
    phase: MissionPhase = Field(..., description='Phase when checkpoint was created')
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_snapshot: str = Field(..., description='Serialized state')


class MemoryState(BaseModel):
    """Overall memory state."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    entries: list[MemoryEntry] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    total_size_bytes: int = Field(default=0)


# =============================================================================
# Error Models
# =============================================================================
class ErrorEvent(BaseModel):
    """Record of an error event."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str = Field(..., description='Unique error identifier')
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    category: ErrorCategory = Field(...)
    error_type: str = Field(..., description='Exception class name')
    message: str = Field(...)
    context: str = Field(default='')
    task_id: TaskId | None = Field(default=None)
    phase: MissionPhase | None = Field(default=None)
    recovery_strategy: RecoveryStrategy = Field(default='retry')
    recovery_attempts: int = Field(default=0)
    resolved: bool = Field(default=False)
    resolution: str | None = Field(default=None)


# =============================================================================
# Research Models
# =============================================================================
class LeaderboardAnalysis(BaseModel):
    """Analysis of competition leaderboard."""
    
    model_config = ConfigDict(frozen=True)
    
    top_score: float = Field(...)
    median_score: float = Field(...)
    target_score: float = Field(...)
    target_percentile: float = Field(...)
    total_teams: int = Field(...)
    score_distribution: list[dict[str, float]] = Field(default_factory=list)
    common_approaches: list[str] = Field(default_factory=list)
    improvement_opportunities: list[str] = Field(default_factory=list)


class ResearchFindings(BaseModel):
    """Complete research findings for a competition."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    leaderboard_analysis: LeaderboardAnalysis | None = Field(default=None)
    papers: list[dict[str, Any]] = Field(default_factory=list)
    approaches: list[dict[str, Any]] = Field(default_factory=list)
    eda_results: dict[str, Any] | None = Field(default=None)
    strategy_recommendations: list[str] = Field(default_factory=list)


# =============================================================================
# Mission State Models
# =============================================================================
class MissionCriteria(BaseModel):
    """Criteria constraining mission execution."""
    
    model_config = ConfigDict(frozen=True)
    
    target_competition_types: frozenset[CompetitionType] = Field(
        default=frozenset({CompetitionType.FEATURED, CompetitionType.RESEARCH}),
    )
    min_prize_pool: int | None = Field(default=None, ge=0)
    max_team_size: int | None = Field(default=None, ge=1)
    min_days_remaining: int = Field(default=7, ge=1)
    target_domains: frozenset[str] = Field(default_factory=frozenset)
    exclude_domains: frozenset[str] = Field(default_factory=frozenset)
    max_evolution_rounds: int = Field(default=100, ge=1)
    target_leaderboard_percentile: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description='Target top N percentile on leaderboard',
    )
    
    @model_validator(mode='after')
    def validate_domains_disjoint(self) -> Self:
        """Ensure target and exclude domains don't overlap."""
        overlap = self.target_domains & self.exclude_domains
        if overlap:
            raise ValueError(f'Domains cannot be both targeted and excluded: {overlap}')
        return self


class MissionState(BaseModel):
    """Complete mission state for frontend streaming.
    
    This model is mutable to track progress during execution.
    """
    
    model_config = ConfigDict(validate_assignment=True)
    
    # Identity
    mission_id: MissionId = Field(...)
    competition_id: CompetitionId | None = Field(default=None)
    
    # Status
    status: str = Field(default='idle')  # idle, planning, executing, paused, completed, failed
    current_phase: MissionPhase | None = Field(default=None)
    current_task_id: TaskId | None = Field(default=None)
    
    # Progress
    overall_progress: float = Field(default=0.0, ge=0.0, le=100.0)
    started_at: datetime | None = Field(default=None)
    estimated_completion_at: datetime | None = Field(default=None)
    
    # Planning
    phases: list[PhasePlan] = Field(default_factory=list)
    
    # Competition context
    competition: Competition | None = Field(default=None)
    research: ResearchFindings | None = Field(default=None)
    
    # Evolution state (only during evolution phase)
    evolution: EvolutionState | None = Field(default=None)
    
    # Memory
    memory: MemoryState = Field(default_factory=MemoryState)
    
    # Errors
    errors: list[ErrorEvent] = Field(default_factory=list)
    
    # Final results
    result: dict[str, Any] | None = Field(default=None)


class MissionResult(BaseModel):
    """Final result of a mission."""
    
    model_config = ConfigDict(frozen=True)
    
    success: bool = Field(...)
    mission_id: MissionId = Field(...)
    competition_id: CompetitionId | None = Field(default=None)
    final_rank: int | None = Field(default=None)
    final_score: float | None = Field(default=None)
    total_submissions: int = Field(default=0)
    evolution_generations: int = Field(default=0)
    duration_ms: int = Field(default=0)
    phases_completed: list[MissionPhase] = Field(default_factory=list)
    error_message: str | None = Field(default=None)
