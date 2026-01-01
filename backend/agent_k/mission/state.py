"""Mission state for pydantic-graph.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

# Third-party (alphabetical)
from pydantic import BaseModel, ConfigDict, Field

# Local imports (core first, then alphabetical)
from agent_k.core.models import Competition, EvolutionState, MemoryState, MissionCriteria, PhasePlan, ResearchFindings
from agent_k.core.types import CompetitionId, MissionId, MissionPhase, TaskId

if TYPE_CHECKING:
    import httpx

    from agent_k.core.protocols import PlatformAdapter
    from agent_k.ui.ag_ui import EventEmitter

__all__ = ('MissionResult', 'MissionState', 'GraphContext', 'SCHEMA_VERSION')

SCHEMA_VERSION: Final[str] = '1.0.0'


class MissionState(BaseModel):
    """State flowing through the mission graph."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    mission_id: MissionId = Field(description='Unique mission identifier')
    competition_id: CompetitionId | None = Field(default=None, description='Selected competition id')
    criteria: MissionCriteria = Field(default_factory=MissionCriteria, description='Mission selection criteria')
    current_phase: MissionPhase = Field(default='discovery', description='Current mission phase')
    phases_completed: list[MissionPhase] = Field(default_factory=list, description='Completed phases')
    discovered_competitions: list[Competition] = Field(
        default_factory=list, description='Competitions found during discovery'
    )
    selected_competition: Competition | None = Field(default=None, description='Competition selected for execution')
    research_findings: ResearchFindings | None = Field(
        default=None, description='Research findings from scientist phase'
    )
    prototype_code: str | None = Field(default=None, description='Prototype solution code')
    prototype_score: float | None = Field(default=None, description='Prototype evaluation score')
    evolution_state: EvolutionState | None = Field(default=None, description='Evolution phase state')
    final_submission_id: str | None = Field(default=None, description='Final submission identifier')
    final_score: float | None = Field(default=None, description='Final leaderboard score')
    final_rank: int | None = Field(default=None, description='Final leaderboard rank')
    errors: list[dict[str, Any]] = Field(default_factory=list, description='Collected error details')
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description='Mission start time')
    phase_started_at: datetime | None = Field(default=None, description='Current phase start time')
    memory: MemoryState = Field(default_factory=MemoryState, description='Mission memory state')
    phases: list[PhasePlan] = Field(default_factory=list, description='Phase plans for UI display')
    current_task_id: TaskId | None = Field(default=None, description='Current task identifier')
    overall_progress: float = Field(default=0.0, ge=0.0, le=100.0, description='Overall mission progress (percentage)')


class MissionResult(BaseModel):
    """Final result of mission execution."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    success: bool = Field(..., description='Whether mission completed successfully')
    mission_id: MissionId = Field(..., description='Mission identifier')
    competition_id: CompetitionId | None = Field(default=None, description='Competition identifier')
    final_rank: int | None = Field(default=None, description='Final leaderboard rank')
    final_score: float | None = Field(default=None, description='Final leaderboard score')
    total_submissions: int = Field(default=0, description='Number of submissions made')
    evolution_generations: int = Field(default=0, description='Evolution generations completed')
    duration_ms: int = Field(default=0, description='Mission duration in milliseconds')
    phases_completed: list[MissionPhase] = Field(default_factory=list, description='Phases completed during mission')
    error_message: str | None = Field(default=None, description='Error message when mission fails')


@dataclass(slots=True)
class GraphContext:
    """Context passed to graph execution."""

    event_emitter: EventEmitter | None = None
    http_client: httpx.AsyncClient | None = None
    platform_adapter: PlatformAdapter | None = None
