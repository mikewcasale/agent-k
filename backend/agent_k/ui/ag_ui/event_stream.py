"""Event streaming utilities for AG-UI protocol.

Provides Server-Sent Events (SSE) streaming to the frontend.
Per spec Section 8, includes comprehensive observability.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import logfire
from pydantic import BaseModel

__all__ = ['EventEmitter', 'AgentKEvent', 'TaskEmissionContext']


# =============================================================================
# Event Types
# =============================================================================
EventType = Literal[
    # State management
    'state-snapshot',
    'state-delta',
    # Phase lifecycle
    'phase-start',
    'phase-complete',
    'phase-error',
    # Task lifecycle
    'task-start',
    'task-progress',
    'task-complete',
    'task-error',
    # Tool usage
    'tool-start',
    'tool-thinking',
    'tool-result',
    'tool-error',
    # Evolution specific
    'generation-start',
    'generation-complete',
    'fitness-update',
    'submission-result',
    'convergence-detected',
    # Memory operations
    'memory-store',
    'memory-retrieve',
    'checkpoint-created',
    # Error handling
    'error-occurred',
    'recovery-attempt',
    'recovery-complete',
]


# =============================================================================
# Event Model
# =============================================================================
class AgentKEvent(BaseModel):
    """Event to be streamed to the frontend."""
    
    type: EventType
    timestamp: str
    data: dict[str, Any]
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        return f"data: {self.model_dump_json()}\n\n"


# =============================================================================
# Event Emitter
# =============================================================================
@dataclass
class EventEmitter:
    """Emitter for AG-UI events.
    
    Provides methods for emitting various event types during mission execution.
    Events are queued and streamed to the frontend via SSE.
    
    Per spec Section 8, all emissions are traced via Logfire.
    """
    
    _queue: asyncio.Queue[AgentKEvent] = field(default_factory=asyncio.Queue)
    _closed: bool = False
    
    async def emit(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Emit an event to the stream.
        
        Args:
            event_type: Type of event to emit.
            data: Event payload data.
        """
        if self._closed:
            return
        
        event = AgentKEvent(
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        
        # Log event emission
        logfire.debug(
            'event_emitted',
            event_type=event_type,
            data_keys=list(data.keys()),
        )
        
        await self._queue.put(event)
    
    async def stream(self) -> AsyncIterator[str]:
        """Stream events as SSE.
        
        Yields:
            SSE-formatted event strings.
        """
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=30)
                yield event.to_sse()
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield ": heartbeat\n\n"
    
    def close(self) -> None:
        """Close the event stream."""
        self._closed = True
    
    # =========================================================================
    # Convenience Methods for Phase Events
    # =========================================================================
    async def emit_phase_start(
        self,
        phase: str,
        objectives: list[str],
    ) -> None:
        """Emit phase start event."""
        await self.emit('phase-start', {
            'phase': phase,
            'objectives': objectives,
        })
    
    async def emit_phase_complete(
        self,
        phase: str,
        success: bool,
        duration_ms: int,
    ) -> None:
        """Emit phase completion event."""
        await self.emit('phase-complete', {
            'phase': phase,
            'success': success,
            'durationMs': duration_ms,
        })
    
    async def emit_phase_error(
        self,
        phase: str,
        error: str,
        recoverable: bool,
    ) -> None:
        """Emit phase error event."""
        await self.emit('phase-error', {
            'phase': phase,
            'error': error,
            'recoverable': recoverable,
        })
    
    # =========================================================================
    # Convenience Methods for Task Events
    # =========================================================================
    async def emit_task_start(
        self,
        task_id: str,
        phase: str,
        name: str,
    ) -> None:
        """Emit task start event."""
        await self.emit('task-start', {
            'taskId': task_id,
            'phase': phase,
            'name': name,
        })
    
    async def emit_task_progress(
        self,
        task_id: str,
        progress: float,
        message: str | None = None,
    ) -> None:
        """Emit task progress update."""
        await self.emit('task-progress', {
            'taskId': task_id,
            'progress': progress,
            'message': message,
        })
    
    async def emit_task_complete(
        self,
        task_id: str,
        success: bool,
        result: Any | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Emit task completion event."""
        await self.emit('task-complete', {
            'taskId': task_id,
            'success': success,
            'result': result,
            'durationMs': duration_ms,
        })
    
    # =========================================================================
    # Convenience Methods for Tool Events
    # =========================================================================
    async def emit_tool_start(
        self,
        task_id: str,
        tool_call_id: str,
        tool_type: str,
        operation: str,
    ) -> None:
        """Emit tool invocation start."""
        await self.emit('tool-start', {
            'taskId': task_id,
            'toolCallId': tool_call_id,
            'toolType': tool_type,
            'operation': operation,
        })
    
    async def emit_tool_thinking(
        self,
        task_id: str,
        tool_call_id: str,
        chunk: str,
    ) -> None:
        """Emit thinking stream chunk."""
        await self.emit('tool-thinking', {
            'taskId': task_id,
            'toolCallId': tool_call_id,
            'chunk': chunk,
        })
    
    async def emit_tool_result(
        self,
        task_id: str,
        tool_call_id: str,
        result: Any,
        duration_ms: int,
    ) -> None:
        """Emit tool result."""
        await self.emit('tool-result', {
            'taskId': task_id,
            'toolCallId': tool_call_id,
            'result': result,
            'durationMs': duration_ms,
        })
    
    async def emit_tool_error(
        self,
        task_id: str,
        tool_call_id: str,
        error: str,
    ) -> None:
        """Emit tool error."""
        await self.emit('tool-error', {
            'taskId': task_id,
            'toolCallId': tool_call_id,
            'error': error,
        })
    
    # =========================================================================
    # Convenience Methods for Evolution Events
    # =========================================================================
    async def emit_generation_start(
        self,
        generation: int,
        population_size: int,
    ) -> None:
        """Emit generation start event."""
        await self.emit('generation-start', {
            'generation': generation,
            'populationSize': population_size,
        })
    
    async def emit_generation_complete(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        worst_fitness: float,
        population_size: int,
        mutations: dict[str, int],
    ) -> None:
        """Emit evolution generation completion."""
        await self.emit('generation-complete', {
            'generation': generation,
            'bestFitness': best_fitness,
            'meanFitness': mean_fitness,
            'worstFitness': worst_fitness,
            'populationSize': population_size,
            'mutations': mutations,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
    
    async def emit_submission_result(
        self,
        submission_id: str,
        generation: int,
        cv_score: float,
        public_score: float | None,
        rank: int | None,
        total_teams: int | None,
    ) -> None:
        """Emit Kaggle submission result."""
        percentile = (rank / total_teams * 100) if rank and total_teams else None
        
        await self.emit('submission-result', {
            'submissionId': submission_id,
            'generation': generation,
            'cvScore': cv_score,
            'publicScore': public_score,
            'rank': rank,
            'totalTeams': total_teams,
            'percentile': percentile,
            'submittedAt': datetime.now(timezone.utc).isoformat(),
        })
    
    async def emit_convergence_detected(
        self,
        generation: int,
        reason: str,
    ) -> None:
        """Emit convergence detection event."""
        await self.emit('convergence-detected', {
            'generation': generation,
            'reason': reason,
        })
    
    # =========================================================================
    # Convenience Methods for Memory Events
    # =========================================================================
    async def emit_memory_store(
        self,
        key: str,
        scope: str,
        category: str,
    ) -> None:
        """Emit memory store operation."""
        await self.emit('memory-store', {
            'key': key,
            'scope': scope,
            'category': category,
        })
    
    async def emit_memory_retrieve(
        self,
        key: str,
        found: bool,
    ) -> None:
        """Emit memory retrieve operation."""
        await self.emit('memory-retrieve', {
            'key': key,
            'found': found,
        })
    
    async def emit_checkpoint_created(
        self,
        name: str,
        phase: str,
    ) -> None:
        """Emit checkpoint creation."""
        await self.emit('checkpoint-created', {
            'name': name,
            'phase': phase,
        })
    
    # =========================================================================
    # Convenience Methods for Error Events
    # =========================================================================
    async def emit_error(
        self,
        error_id: str,
        category: str,
        error_type: str,
        message: str,
        context: str,
        recovery_strategy: str,
    ) -> None:
        """Emit error event."""
        await self.emit('error-occurred', {
            'id': error_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'category': category,
            'errorType': error_type,
            'message': message,
            'context': context,
            'recoveryStrategy': recovery_strategy,
            'recoveryAttempts': 0,
            'resolved': False,
        })
    
    async def emit_recovery_attempt(
        self,
        error_id: str,
        strategy: str,
        attempt: int,
    ) -> None:
        """Emit recovery attempt event."""
        await self.emit('recovery-attempt', {
            'errorId': error_id,
            'strategy': strategy,
            'attempt': attempt,
        })
    
    async def emit_recovery_complete(
        self,
        error_id: str,
        success: bool,
        resolution: str | None = None,
    ) -> None:
        """Emit recovery completion event."""
        await self.emit('recovery-complete', {
            'errorId': error_id,
            'success': success,
            'resolution': resolution,
        })


# =============================================================================
# Context Manager for Scoped Emissions
# =============================================================================
class TaskEmissionContext:
    """Context manager for task-scoped event emission.
    
    Automatically emits task-start and task-complete events.
    
    Example:
        async with TaskEmissionContext(emitter, 'task_1', 'discovery', 'Search') as ctx:
            # Do task work
            await ctx.emit_progress(0.5, 'Halfway done')
    """
    
    def __init__(
        self,
        emitter: EventEmitter,
        task_id: str,
        phase: str,
        name: str,
    ) -> None:
        self.emitter = emitter
        self.task_id = task_id
        self.phase = phase
        self.name = name
        self._start_time: float | None = None
    
    async def __aenter__(self) -> TaskEmissionContext:
        import time
        self._start_time = time.time()
        await self.emitter.emit_task_start(self.task_id, self.phase, self.name)
        return self
    
    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        import time
        duration_ms = int((time.time() - self._start_time) * 1000) if self._start_time else 0
        
        if exc_type:
            await self.emitter.emit('task-error', {
                'taskId': self.task_id,
                'error': str(exc_val),
            })
        else:
            await self.emitter.emit_task_complete(
                self.task_id,
                success=True,
                duration_ms=duration_ms,
            )
    
    async def emit_progress(
        self,
        progress: float,
        message: str | None = None,
    ) -> None:
        """Emit progress within the task."""
        await self.emitter.emit_task_progress(self.task_id, progress, message)
