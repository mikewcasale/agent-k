"""Core domain models for AGENT-K.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from .config import MAX_MISSION_EVOLUTION_ROUNDS
from .types import CompetitionId, MetricDirection

__all__ = (
    "Competition",
    "CompetitionType",
    "EvaluationMetric",
    "LeaderboardEntry",
    "MissionCriteria",
    "SubjectDomain",
    "Submission",
)


SCHEMA_VERSION: Final[str] = "1.0.0"


class CompetitionType(StrEnum):
    """Type of Kaggle competition."""

    FEATURED = "featured"
    RESEARCH = "research"
    GETTING_STARTED = "getting_started"
    PLAYGROUND = "playground"
    COMMUNITY = "community"


class SubjectDomain(StrEnum):
    """Subject domains for competition classification."""

    FINANCE = "finance"
    MEDICAL = "medical"
    WEATHER = "weather"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    AUDIO = "audio"
    GEOSPATIAL = "geospatial"


class EvaluationMetric(StrEnum):
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


class Competition(BaseModel):
    """Kaggle competition entity.

    Represents a competition with all metadata required for evaluation
    and participation decisions.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    id: CompetitionId = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9-]+$",
        description="Unique competition identifier (slug)",
    )
    title: str = Field(..., min_length=1, max_length=500, description="Competition display title")
    description: str | None = Field(
        default=None, max_length=10000, description="Competition description"
    )
    competition_type: CompetitionType = Field(..., description="Category of competition")
    metric: EvaluationMetric = Field(..., description="Primary evaluation metric")
    metric_direction: MetricDirection = Field(
        default="maximize", description="Whether higher or lower metric values are better"
    )
    deadline: datetime = Field(..., description="Competition submission deadline (UTC)")
    prize_pool: int | None = Field(default=None, ge=0, description="Total prize pool in USD")
    max_team_size: int = Field(default=1, ge=1, le=100, description="Maximum allowed team size")
    max_daily_submissions: int = Field(default=5, ge=1, description="Maximum submissions per day")
    tags: frozenset[str] = Field(
        default_factory=frozenset, description="Competition tags/categories"
    )
    url: str | None = Field(default=None, description="Full URL to competition page")

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
        default_factory=lambda: datetime.now(UTC), description="Submission timestamp"
    )
    public_score: float | None = Field(
        default=None, description="Public leaderboard score (if evaluated)"
    )
    private_score: float | None = Field(
        default=None, description="Private leaderboard score (after competition ends)"
    )
    status: str = Field(
        default="pending", pattern=r"^(pending|complete|error)$", description="Submission status"
    )
    error_message: str | None = Field(default=None, description="Error message if failed")


class MissionCriteria(BaseModel):
    """Criteria constraining mission execution."""

    model_config = ConfigDict(
        frozen=True,
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )
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
    max_evolution_rounds: int = Field(
        default=100, ge=1, le=MAX_MISSION_EVOLUTION_ROUNDS, description="Max evolution rounds"
    )
    min_improvements_required: int = Field(
        default=0,
        ge=0,
        description="Minimum number of fitness improvements required before submission",
    )
    target_leaderboard_percentile: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Target top N percentile on leaderboard"
    )
    evolution_models: tuple[str, ...] = Field(
        default_factory=tuple, description="Ordered model specs to rotate during evolution"
    )
    enable_kumo: bool = True
    allow_submission: bool = False
    min_evolution_iterations: int = Field(default=1, ge=1, le=200)
    use_openevolve: bool = Field(default=True, description="Deprecated; OpenEvolve is always used")

    @model_validator(mode="after")
    def validate_domains_disjoint(self) -> Self:
        """Ensure target and exclude domains don't overlap."""
        overlap = self.target_domains & self.exclude_domains
        if overlap:
            raise ValueError(f"Domains cannot be both targeted and excluded: {overlap}")
        if self.min_evolution_iterations > self.max_evolution_rounds:
            raise ValueError("Minimum evolution iterations exceed the maximum")
        if self.min_improvements_required > self.max_evolution_rounds:
            raise ValueError("Minimum improvements exceed the available iterations")
        return self
