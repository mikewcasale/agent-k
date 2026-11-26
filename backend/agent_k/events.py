"""Define durable mission events and browser-compatible SSE serialization.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal, Protocol, cast

import logfire
from pydantic import BaseModel, ConfigDict, Field, JsonValue

__all__ = (
    "AGENT_K_EVENT_TYPES",
    "AgentKEvent",
    "BrowserEvent",
    "EventData",
    "EventEmitter",
    "EventSink",
    "EventType",
    "ModelFamily",
    "ModelSize",
    "WorkerEvent",
    "parse_last_event_id",
)

SCHEMA_VERSION: Final[str] = "1.0.0"

EventType = Literal[
    "state-snapshot",
    "state-delta",
    "phase-start",
    "phase-complete",
    "phase-error",
    "task-start",
    "task-progress",
    "task-complete",
    "task-error",
    "tool-start",
    "tool-thinking",
    "tool-result",
    "tool-error",
    "generation-start",
    "generation-complete",
    "fitness-update",
    "submission-result",
    "convergence-detected",
    "memory-store",
    "memory-retrieve",
    "checkpoint-created",
    "error-occurred",
    "recovery-attempt",
    "recovery-complete",
    "mission-complete",
]

ModelFamily = Literal["lightgbm", "kumo", "vision", "text"]
ModelSize = Literal["small", "medium", "large"]

AGENT_K_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "state-snapshot",
        "state-delta",
        "phase-start",
        "phase-complete",
        "phase-error",
        "task-start",
        "task-progress",
        "task-complete",
        "task-error",
        "tool-start",
        "tool-thinking",
        "tool-result",
        "tool-error",
        "generation-start",
        "generation-complete",
        "fitness-update",
        "submission-result",
        "convergence-detected",
        "memory-store",
        "memory-retrieve",
        "checkpoint-created",
        "error-occurred",
        "recovery-attempt",
        "recovery-complete",
        "mission-complete",
    }
)

EventData = JsonValue


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class _WireModel(BaseModel):
    """Base model for strict camelCase wire payloads."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        populate_by_name=True,
    )


class WorkerEvent(_WireModel):
    """Event submitted by a leased worker."""

    event_id: str = Field(min_length=1, max_length=200)
    sequence: int = Field(ge=1)
    type: EventType
    timestamp: datetime
    data: EventData = Field(default_factory=dict)


class AgentKEvent(WorkerEvent):
    """Durable event associated with a mission."""

    schema_version: str = SCHEMA_VERSION
    mission_id: str = Field(min_length=1, max_length=200)

    def to_browser_event(self) -> BrowserEvent:
        """Return the stable event envelope consumed by the frontend."""
        return BrowserEvent(type=self.type, timestamp=self.timestamp, data=self.data)

    def to_sse(self) -> str:
        """Serialize the event as an ordered, resumable SSE frame."""
        payload = self.to_browser_event().model_dump_json(
            by_alias=True,
            exclude_none=True,
        )
        return f"id: {self.sequence}\ndata: {payload}\n\n"


class BrowserEvent(_WireModel):
    """Stable event envelope exposed to browser clients."""

    type: EventType
    timestamp: datetime
    data: EventData = Field(default_factory=dict)


class EventSink(Protocol):
    """Append mission events to a durable ordered store."""

    async def append_event(
        self,
        mission_id: str,
        event_type: EventType,
        data: EventData,
        *,
        timestamp: datetime,
    ) -> AgentKEvent:
        """Append one event and return its assigned durable identity."""
        ...


@dataclass(slots=True)
class EventEmitter:
    """Emit camelCase mission events through a durable event sink."""

    mission_id: str
    sink: EventSink

    async def emit(
        self,
        event_type: EventType,
        data: EventData,
    ) -> AgentKEvent:
        """Append one event to the mission's ordered stream.

        Args:
            event_type: Stable browser-facing event name.
            data: JSON-compatible camelCase event payload.

        Returns:
            The durable event with its assigned sequence and identifier.
        """
        event = await self.sink.append_event(
            self.mission_id,
            event_type,
            data,
            timestamp=datetime.now(UTC),
        )
        logfire.debug(
            "mission_event_appended",
            mission_id=self.mission_id,
            event_type=event_type,
            sequence=event.sequence,
        )
        return event

    async def emit_phase_start(
        self,
        phase: str,
        objectives: list[str],
    ) -> AgentKEvent:
        """Emit a phase-start event."""
        return await self.emit(
            "phase-start",
            {"phase": phase, "objectives": cast(list[JsonValue], objectives)},
        )

    async def emit_phase_complete(
        self,
        phase: str,
        success: bool,
        duration_ms: int,
    ) -> AgentKEvent:
        """Emit a phase-complete event."""
        return await self.emit(
            "phase-complete",
            {"phase": phase, "success": success, "durationMs": duration_ms},
        )

    async def emit_phase_error(
        self,
        phase: str,
        error: str,
        recoverable: bool,
    ) -> AgentKEvent:
        """Emit a phase-error event."""
        return await self.emit(
            "phase-error",
            {"phase": phase, "error": error, "recoverable": recoverable},
        )

    async def emit_task_start(
        self,
        task_id: str,
        phase: str,
        name: str,
    ) -> AgentKEvent:
        """Emit a task-start event."""
        return await self.emit(
            "task-start",
            {"taskId": task_id, "phase": phase, "name": name},
        )

    async def emit_task_progress(
        self,
        task_id: str,
        progress: float,
        message: str | None = None,
    ) -> AgentKEvent:
        """Emit a task-progress event."""
        return await self.emit(
            "task-progress",
            {"taskId": task_id, "progress": progress, "message": message},
        )

    async def emit_task_complete(
        self,
        task_id: str,
        success: bool,
        result: EventData = None,
        duration_ms: int = 0,
    ) -> AgentKEvent:
        """Emit a task-complete event."""
        return await self.emit(
            "task-complete",
            {
                "taskId": task_id,
                "success": success,
                "result": result,
                "durationMs": duration_ms,
            },
        )

    async def emit_tool_start(
        self,
        task_id: str,
        tool_call_id: str,
        tool_type: str,
        operation: str,
        params: dict[str, EventData] | None = None,
    ) -> AgentKEvent:
        """Emit a tool-start event."""
        return await self.emit(
            "tool-start",
            {
                "taskId": task_id,
                "toolCallId": tool_call_id,
                "toolType": tool_type,
                "operation": operation,
                "params": params,
            },
        )

    async def emit_tool_thinking(
        self,
        task_id: str,
        tool_call_id: str,
        chunk: str,
    ) -> AgentKEvent:
        """Emit a tool-thinking event."""
        return await self.emit(
            "tool-thinking",
            {"taskId": task_id, "toolCallId": tool_call_id, "chunk": chunk},
        )

    async def emit_tool_result(
        self,
        task_id: str,
        tool_call_id: str,
        result: EventData,
        duration_ms: int,
    ) -> AgentKEvent:
        """Emit a tool-result event."""
        return await self.emit(
            "tool-result",
            {
                "taskId": task_id,
                "toolCallId": tool_call_id,
                "result": result,
                "durationMs": duration_ms,
            },
        )

    async def emit_tool_error(
        self,
        task_id: str,
        tool_call_id: str,
        error: str,
    ) -> AgentKEvent:
        """Emit a tool-error event."""
        return await self.emit(
            "tool-error",
            {"taskId": task_id, "toolCallId": tool_call_id, "error": error},
        )

    async def emit_generation_start(
        self,
        generation: int,
        population_size: int,
        *,
        model_family: ModelFamily | None = None,
        model_size: ModelSize | None = None,
        iteration: int | None = None,
        total_iterations: int | None = None,
    ) -> AgentKEvent:
        """Emit a generation-start event with optional runtime metadata."""
        return await self.emit(
            "generation-start",
            {
                "generation": generation,
                "populationSize": population_size,
                "modelFamily": model_family,
                "modelSize": model_size,
                "iteration": iteration,
                "totalIterations": total_iterations,
            },
        )

    async def emit_generation_complete(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        worst_fitness: float,
        population_size: int,
        mutations: dict[str, int],
        *,
        model_family: ModelFamily | None = None,
        model_size: ModelSize | None = None,
        iteration: int | None = None,
        total_iterations: int | None = None,
    ) -> AgentKEvent:
        """Emit a generation-complete event."""
        return await self.emit(
            "generation-complete",
            {
                "generation": generation,
                "bestFitness": best_fitness,
                "meanFitness": mean_fitness,
                "worstFitness": worst_fitness,
                "populationSize": population_size,
                "mutations": {
                    "point": mutations.get("point", 0),
                    "structural": mutations.get("structural", 0),
                    "hyperparameter": mutations.get("hyperparameter", 0),
                    "crossover": mutations.get("crossover", 0),
                },
                "timestamp": datetime.now(UTC).isoformat(),
                "modelFamily": model_family,
                "modelSize": model_size,
                "iteration": iteration,
                "totalIterations": total_iterations,
            },
        )

    async def emit_submission_result(
        self,
        submission_id: str,
        generation: int,
        cv_score: float,
        public_score: float | None,
        rank: int | None,
        total_teams: int | None,
    ) -> AgentKEvent:
        """Emit a submission-result event."""
        percentile = (rank / total_teams * 100) if rank and total_teams else None
        return await self.emit(
            "submission-result",
            {
                "submissionId": submission_id,
                "generation": generation,
                "cvScore": cv_score,
                "publicScore": public_score,
                "rank": rank,
                "totalTeams": total_teams,
                "percentile": percentile,
                "submittedAt": datetime.now(UTC).isoformat(),
            },
        )

    async def emit_convergence_detected(
        self,
        generation: int,
        reason: str,
    ) -> AgentKEvent:
        """Emit a convergence-detected event."""
        return await self.emit(
            "convergence-detected",
            {"generation": generation, "reason": reason},
        )

    async def emit_memory_store(
        self,
        key: str,
        scope: str,
        category: str,
    ) -> AgentKEvent:
        """Emit a memory-store event."""
        return await self.emit(
            "memory-store",
            {"key": key, "scope": scope, "category": category},
        )

    async def emit_memory_retrieve(
        self,
        key: str,
        found: bool,
    ) -> AgentKEvent:
        """Emit a memory-retrieve event."""
        return await self.emit("memory-retrieve", {"key": key, "found": found})

    async def emit_checkpoint_created(
        self,
        name: str,
        phase: str,
    ) -> AgentKEvent:
        """Emit a checkpoint-created event."""
        return await self.emit(
            "checkpoint-created",
            {"name": name, "phase": phase},
        )

    async def emit_error(
        self,
        error_id: str,
        category: str,
        error_type: str,
        message: str,
        context: str,
        recovery_strategy: str,
    ) -> AgentKEvent:
        """Emit an error-occurred event."""
        return await self.emit(
            "error-occurred",
            {
                "id": error_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "errorType": error_type,
                "message": message,
                "context": context,
                "recoveryStrategy": recovery_strategy,
                "recoveryAttempts": 0,
                "resolved": False,
            },
        )

    async def emit_recovery_attempt(
        self,
        error_id: str,
        strategy: str,
        attempt: int,
    ) -> AgentKEvent:
        """Emit a recovery-attempt event."""
        return await self.emit(
            "recovery-attempt",
            {"errorId": error_id, "strategy": strategy, "attempt": attempt},
        )

    async def emit_recovery_complete(
        self,
        error_id: str,
        success: bool,
        resolution: str | None = None,
    ) -> AgentKEvent:
        """Emit a recovery-complete event."""
        return await self.emit(
            "recovery-complete",
            {"errorId": error_id, "success": success, "resolution": resolution},
        )

    async def emit_mission_complete(
        self,
        *,
        success: bool,
        final_rank: int | None,
        final_score: float | None,
        error_message: str | None = None,
        total_submissions: int,
        evolution_generations: int,
        evolution_iterations: int | None = None,
        duration_ms: int,
        phases_completed: list[str],
        evolution_failure_summary: dict[str, int] | None = None,
    ) -> AgentKEvent:
        """Emit the canonical camelCase mission-complete event."""
        return await self.emit(
            "mission-complete",
            {
                "success": success,
                "finalRank": final_rank,
                "finalScore": final_score,
                "errorMessage": error_message,
                "totalSubmissions": total_submissions,
                "evolutionGenerations": evolution_generations,
                "evolutionIterations": evolution_iterations,
                "evolutionFailureSummary": cast(
                    dict[str, JsonValue],
                    evolution_failure_summary or {},
                ),
                "durationMs": duration_ms,
                "phasesCompleted": cast(list[JsonValue], phases_completed),
            },
        )


def parse_last_event_id(value: str | None) -> int:
    """Parse an SSE Last-Event-ID value into the last delivered sequence.

    Args:
        value: Header value supplied by the browser, or ``None``.

    Returns:
        The non-negative last delivered event sequence.

    Raises:
        ValueError: If the value is not a non-negative integer.
    """
    if value is None or not value.strip():
        return 0
    try:
        sequence = int(value)
    except ValueError as exc:
        raise ValueError("Last-Event-ID must be a non-negative integer") from exc
    if sequence < 0:
        raise ValueError("Last-Event-ID must be a non-negative integer")
    return sequence
