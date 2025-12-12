"""State models for the pydantic-graph state machine.

These models represent the state that flows through the graph nodes.
Per spec Section 4.4, models are immutable where possible.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..core.models import (
    Competition,
    EvolutionState,
    MissionCriteria,
    MissionResult,
    MemoryState,
    PhasePlan,
    ResearchFindings,
)
from ..core.types import CompetitionId, MissionId, MissionPhase, TaskId

__all__ = ['MissionState', 'MissionResult', 'GraphContext']


class MissionState(BaseModel):
    """State flowing through the mission graph.
    
    This is the central state object passed between graph nodes.
    It accumulates results from each phase for downstream phases.
    """
    
    # Identity
    mission_id: MissionId
    competition_id: CompetitionId | None = None
    
    # Configuration
    criteria: MissionCriteria = Field(default_factory=MissionCriteria)
    
    # Phase tracking
    current_phase: MissionPhase = 'discovery'
    phases_completed: list[MissionPhase] = Field(default_factory=list)
    
    # Phase results (accumulated)
    discovered_competitions: list[Competition] = Field(default_factory=list)
    selected_competition: Competition | None = None
    research_findings: ResearchFindings | None = None
    prototype_code: str | None = None
    prototype_score: float | None = None
    evolution_state: EvolutionState | None = None
    final_submission_id: str | None = None
    final_score: float | None = None
    final_rank: int | None = None
    
    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase_started_at: datetime | None = None
    
    # Memory
    memory: MemoryState = Field(default_factory=MemoryState)
    
    # Planning (for frontend)
    phases: list[PhasePlan] = Field(default_factory=list)
    current_task_id: TaskId | None = None
    overall_progress: float = 0.0


class GraphContext(BaseModel):
    """Context passed to graph execution.
    
    Contains dependencies and event emitter for node operations.
    """
    
    event_emitter: Any = None  # EventEmitter
    http_client: Any = None  # httpx.AsyncClient
    platform_adapter: Any = None  # PlatformAdapter
