"""FastAPI application for AG-UI protocol.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias

# Third-party (alphabetical)
import logfire
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings

# Local imports (core first, then alphabetical)
from agent_k.core.constants import DEFAULT_MODEL, MISSION_PHASES
from agent_k.core.exceptions import AgentKError, AuthenticationError, CompetitionNotFoundError, classify_error
from agent_k.core.models import Competition, CompetitionType, LeaderboardSubmission, MissionCriteria
from agent_k.infra.instrumentation import configure_instrumentation
from agent_k.infra.providers import get_model
from agent_k.mission.persistence import CHECKPOINT_DIR, MissionPersistence, create_persistence

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agent_k.mission.state import MissionResult, MissionState

__all__ = (
    'AgentKEvent',
    'EventEmitter',
    'ChatHandler',
    'IntentClassifier',
    'MissionCriteriaParser',
    'MissionRequest',
    'CompetitionSearchRequest',
    'CompetitionFetchRequest',
    'MissionIntentResult',
    'TaskEmissionContext',
    'app',
    'create_app',
    'main',
    'parse_mission_intent',
    'stream_text_response',
    'transform_to_vercel_stream',
)

SCHEMA_VERSION: Final[str] = '1.0.0'
APP_VERSION: Final[str] = '0.1.0'
DEFAULT_HOST: Final[str] = '0.0.0.0'
DEFAULT_PORT: Final[int] = 9000
APP_MODULE: Final[str] = 'agent_k.ui.ag_ui:app'
MAX_COMPETITION_RESULTS: Final[int] = 25
_DOMAIN_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    'finance': ('finance', 'financial', 'trading', 'stock', 'market'),
    'medical': ('medical', 'health', 'healthcare', 'clinical', 'diagnosis'),
    'weather': ('weather', 'climate', 'forecast'),
    'computer_vision': ('computer vision', 'vision', 'image', 'cv'),
    'nlp': ('nlp', 'text', 'language', 'transformer'),
    'tabular': ('tabular', 'structured', 'csv', 'table'),
    'time_series': ('time series', 'timeseries', 'temporal', 'forecast'),
    'audio': ('audio', 'speech', 'sound', 'acoustic'),
    'geospatial': ('geospatial', 'geo', 'spatial', 'gis', 'satellite'),
}

_MISSION_KEYWORDS: Final[tuple[str, ...]] = (
    'find',
    'competition',
    'kaggle',
    'compete',
    'enter',
    'discover',
    'search',
    'challenge',
    'participate',
)
_ANTI_KEYWORDS: Final[tuple[str, ...]] = ('what is', 'explain', 'how does', 'tell me about', 'help')
_PRIZE_PATTERN: Final[re.Pattern[str]] = re.compile(r'\$([\d,]+(?:\.\d+)?)\s*(k|m|thousand|million)?\b')
_DAYS_PATTERN: Final[re.Pattern[str]] = re.compile(r'(\d+)\s*(days?|weeks?)')
_PERCENTILE_PATTERN: Final[re.Pattern[str]] = re.compile(r'top\s+(\d+)%')

INTENT_SYSTEM_PROMPT: Final[
    str
] = """You are an intent detection system for AGENT-K, a multi-agent Kaggle competition system.

Your task is to determine if the user wants to start a Kaggle competition mission, and if so, extract the mission criteria.

A mission request typically includes phrases like:
- "find a competition", "find me a Kaggle competition"
- "compete in", "enter a competition"
- "discover competitions", "search for challenges"
- "participate in Kaggle"

If the user is asking about AGENT-K itself, explaining what it does, or asking for help, that is NOT a mission request.

Extract the following criteria if mentioned:
- Competition types: featured, research, playground, getting_started, community
- Minimum prize pool (in USD)
- Minimum days remaining before deadline
- Target domains: computer vision (cv), natural language processing (nlp), tabular data, time series, etc.
- Target leaderboard percentile (e.g., "top 10%" = 0.10)
- Population size for evolution (default: 50)
- Max evolution rounds (default: 100)

If the message is NOT about starting a mission, return is_mission=False with null criteria.
If it IS a mission request, return is_mission=True with extracted criteria (use defaults for unspecified fields).
"""

# Event types that should be sent as data events (type "8")
AGENT_K_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        'state-snapshot',
        'state-delta',
        'phase-start',
        'phase-complete',
        'phase-error',
        'task-start',
        'task-progress',
        'task-complete',
        'task-error',
        'tool-start',
        'tool-thinking',
        'tool-result',
        'tool-error',
        'generation-start',
        'generation-complete',
        'fitness-update',
        'submission-result',
        'convergence-detected',
        'memory-store',
        'memory-retrieve',
        'checkpoint-created',
        'error-occurred',
        'recovery-attempt',
        'recovery-complete',
        'mission-complete',
    }
)

EventType: TypeAlias = """Literal[
    # State management
    "state-snapshot",
    "state-delta",
    # Phase lifecycle
    "phase-start",
    "phase-complete",
    "phase-error",
    # Task lifecycle
    "task-start",
    "task-progress",
    "task-complete",
    "task-error",
    # Tool usage
    "tool-start",
    "tool-thinking",
    "tool-result",
    "tool-error",
    # Evolution specific
    "generation-start",
    "generation-complete",
    "fitness-update",
    "submission-result",
    "convergence-detected",
    # Memory operations
    "memory-store",
    "memory-retrieve",
    "checkpoint-created",
    # Error handling
    "error-occurred",
    "recovery-attempt",
    "recovery-complete",
    # Mission lifecycle
    "mission-complete",
]"""


class MissionRequest(BaseModel):
    """Request to start a new mission."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    criteria: MissionCriteria = Field(..., description='Mission selection criteria')
    evolution_models: list[str] | None = Field(
        default=None, description='Ordered list of model specs to rotate during evolution'
    )
    user_prompt: str | None = Field(default=None, description='Optional user context for the mission')
    competition_id: str | None = Field(default=None, description='Optional competition id override')
    competition_url: str | None = Field(default=None, description='Optional competition URL override')


class CompetitionSearchRequest(BaseModel):
    """Request payload for competition search."""

    model_config = ConfigDict(frozen=True)
    paid_only: bool = Field(default=False, description='Only return competitions with prize pools')
    domains: list[str] = Field(default_factory=list, description='Subject domains to match')
    competition_types: list[CompetitionType] = Field(default_factory=list, description='Competition types')
    min_prize: int | None = Field(default=None, ge=0, description='Minimum prize pool in USD')
    min_days_remaining: int = Field(default=7, ge=1, description='Minimum days remaining before deadline')


class CompetitionFetchRequest(BaseModel):
    """Request payload for fetching competition details by URL."""

    model_config = ConfigDict(frozen=True)
    url: str = Field(..., min_length=10, description='Kaggle competition URL')


class AgentKEvent(BaseModel):
    """Event to be streamed to the frontend."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    type: EventType = Field(..., description='Event type identifier')
    timestamp: str = Field(..., description='ISO-8601 timestamp')
    data: dict[str, Any] = Field(default_factory=dict, description='Event payload')

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        return f'data: {self.model_dump_json()}\n\n'


class MissionIntentOutput(BaseModel):
    """Structured output for intent detection."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    is_mission: bool = Field(default=False, description='Whether the user requested a mission')
    criteria: MissionCriteria | None = Field(default=None, description='Extracted mission criteria')


# =============================================================================
# Agent Singleton (Module-Level)
# =============================================================================
intent_agent: Final[Agent[None, MissionIntentOutput]] = Agent(
    get_model(DEFAULT_MODEL),
    output_type=MissionIntentOutput,
    instructions=INTENT_SYSTEM_PROMPT,
    retries=1,
    name='intent_parser',
)


@dataclass(slots=True)
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

        event = AgentKEvent(type=event_type, timestamp=datetime.now(UTC).isoformat(), data=data)

        # Log event emission
        logfire.debug('event_emitted', event_type=event_type, data_keys=list(data.keys()))

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
            except TimeoutError:
                # Send heartbeat to keep connection alive
                yield ': heartbeat\n\n'

    def close(self) -> None:
        """Close the event stream."""
        self._closed = True

    # =========================================================================
    # Convenience Methods for Phase Events
    # =========================================================================
    async def emit_phase_start(self, phase: str, objectives: list[str]) -> None:
        """Emit phase start event."""
        await self.emit('phase-start', {'phase': phase, 'objectives': objectives})

    async def emit_phase_complete(self, phase: str, success: bool, duration_ms: int) -> None:
        """Emit phase completion event."""
        await self.emit('phase-complete', {'phase': phase, 'success': success, 'durationMs': duration_ms})

    async def emit_phase_error(self, phase: str, error: str, recoverable: bool) -> None:
        """Emit phase error event."""
        await self.emit('phase-error', {'phase': phase, 'error': error, 'recoverable': recoverable})

    # =========================================================================
    # Convenience Methods for Task Events
    # =========================================================================
    async def emit_task_start(self, task_id: str, phase: str, name: str) -> None:
        """Emit task start event."""
        await self.emit('task-start', {'taskId': task_id, 'phase': phase, 'name': name})

    async def emit_task_progress(self, task_id: str, progress: float, message: str | None = None) -> None:
        """Emit task progress update."""
        await self.emit('task-progress', {'taskId': task_id, 'progress': progress, 'message': message})

    async def emit_task_complete(
        self, task_id: str, success: bool, result: Any | None = None, duration_ms: int = 0
    ) -> None:
        """Emit task completion event."""
        await self.emit(
            'task-complete', {'taskId': task_id, 'success': success, 'result': result, 'durationMs': duration_ms}
        )

    # =========================================================================
    # Convenience Methods for Tool Events
    # =========================================================================
    async def emit_tool_start(self, task_id: str, tool_call_id: str, tool_type: str, operation: str) -> None:
        """Emit tool invocation start."""
        await self.emit(
            'tool-start', {'taskId': task_id, 'toolCallId': tool_call_id, 'toolType': tool_type, 'operation': operation}
        )

    async def emit_tool_thinking(self, task_id: str, tool_call_id: str, chunk: str) -> None:
        """Emit thinking stream chunk."""
        await self.emit('tool-thinking', {'taskId': task_id, 'toolCallId': tool_call_id, 'chunk': chunk})

    async def emit_tool_result(self, task_id: str, tool_call_id: str, result: Any, duration_ms: int) -> None:
        """Emit tool result."""
        await self.emit(
            'tool-result', {'taskId': task_id, 'toolCallId': tool_call_id, 'result': result, 'durationMs': duration_ms}
        )

    async def emit_tool_error(self, task_id: str, tool_call_id: str, error: str) -> None:
        """Emit tool error."""
        await self.emit('tool-error', {'taskId': task_id, 'toolCallId': tool_call_id, 'error': error})

    # =========================================================================
    # Convenience Methods for Evolution Events
    # =========================================================================
    async def emit_generation_start(self, generation: int, population_size: int) -> None:
        """Emit generation start event."""
        await self.emit('generation-start', {'generation': generation, 'populationSize': population_size})

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
        await self.emit(
            'generation-complete',
            {
                'generation': generation,
                'bestFitness': best_fitness,
                'meanFitness': mean_fitness,
                'worstFitness': worst_fitness,
                'populationSize': population_size,
                'mutations': mutations,
                'timestamp': datetime.now(UTC).isoformat(),
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
    ) -> None:
        """Emit Kaggle submission result."""
        percentile = (rank / total_teams * 100) if rank and total_teams else None

        await self.emit(
            'submission-result',
            {
                'submissionId': submission_id,
                'generation': generation,
                'cvScore': cv_score,
                'publicScore': public_score,
                'rank': rank,
                'totalTeams': total_teams,
                'percentile': percentile,
                'submittedAt': datetime.now(UTC).isoformat(),
            },
        )

    async def emit_convergence_detected(self, generation: int, reason: str) -> None:
        """Emit convergence detection event."""
        await self.emit('convergence-detected', {'generation': generation, 'reason': reason})

    # =========================================================================
    # Convenience Methods for Memory Events
    # =========================================================================
    async def emit_memory_store(self, key: str, scope: str, category: str) -> None:
        """Emit memory store operation."""
        await self.emit('memory-store', {'key': key, 'scope': scope, 'category': category})

    async def emit_memory_retrieve(self, key: str, found: bool) -> None:
        """Emit memory retrieve operation."""
        await self.emit('memory-retrieve', {'key': key, 'found': found})

    async def emit_checkpoint_created(self, name: str, phase: str) -> None:
        """Emit checkpoint creation."""
        await self.emit('checkpoint-created', {'name': name, 'phase': phase})

    # =========================================================================
    # Convenience Methods for Error Events
    # =========================================================================
    async def emit_error(
        self, error_id: str, category: str, error_type: str, message: str, context: str, recovery_strategy: str
    ) -> None:
        """Emit error event."""
        await self.emit(
            'error-occurred',
            {
                'id': error_id,
                'timestamp': datetime.now(UTC).isoformat(),
                'category': category,
                'errorType': error_type,
                'message': message,
                'context': context,
                'recoveryStrategy': recovery_strategy,
                'recoveryAttempts': 0,
                'resolved': False,
            },
        )

    async def emit_recovery_attempt(self, error_id: str, strategy: str, attempt: int) -> None:
        """Emit recovery attempt event."""
        await self.emit('recovery-attempt', {'errorId': error_id, 'strategy': strategy, 'attempt': attempt})

    async def emit_recovery_complete(self, error_id: str, success: bool, resolution: str | None = None) -> None:
        """Emit recovery completion event."""
        await self.emit('recovery-complete', {'errorId': error_id, 'success': success, 'resolution': resolution})


class TaskEmissionContext:
    """Context manager for task-scoped event emission.

    Automatically emits task-start and task-complete events.

    Example:
        async with TaskEmissionContext(emitter, 'task_1', 'discovery', 'Search') as ctx:
            # Do task work
            await ctx.emit_progress(0.5, 'Halfway done')
    """

    def __init__(self, emitter: EventEmitter, task_id: str, phase: str, name: str) -> None:
        self.emitter = emitter
        self.task_id = task_id
        self.phase = phase
        self.name = name
        self._start_time: float | None = None

    async def __aenter__(self) -> TaskEmissionContext:
        self._start_time = time.time()
        await self.emitter.emit_task_start(self.task_id, self.phase, self.name)
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        duration_ms = int((time.time() - self._start_time) * 1000) if self._start_time else 0

        if exc_type:
            await self.emitter.emit('task-error', {'taskId': self.task_id, 'error': str(exc_val)})
        else:
            await self.emitter.emit_task_complete(self.task_id, success=True, duration_ms=duration_ms)

    async def emit_progress(self, progress: float, message: str | None = None) -> None:
        """Emit progress within the task."""
        await self.emitter.emit_task_progress(self.task_id, progress, message)


@dataclass(slots=True)
class MissionIntentResult:
    """Result of mission intent parsing."""

    is_mission: bool
    criteria: MissionCriteria | None = None


@dataclass(slots=True)
class IntentClassifier:
    """Classify whether a message requests a mission."""

    agent: Agent[None, MissionIntentOutput] = field(default_factory=lambda: intent_agent)
    mission_keywords: tuple[str, ...] = _MISSION_KEYWORDS
    anti_keywords: tuple[str, ...] = _ANTI_KEYWORDS

    async def classify(self, message: str) -> MissionIntentResult:
        text_lower = message.lower()
        has_keyword = any(keyword in text_lower for keyword in self.mission_keywords)
        has_anti_keyword = any(keyword in text_lower for keyword in self.anti_keywords)

        if not has_keyword or has_anti_keyword:
            logfire.debug('intent_parse_rejected', message=message[:100], reason='keywords')
            return MissionIntentResult(is_mission=False)

        with logfire.span('parse_mission_intent', message_preview=message[:100]):
            prompt = f"""Analyze this user message and determine if they want to start a Kaggle competition mission:

"{message}"

Extract mission criteria if this is a mission request."""
            result = await self.agent.run(prompt)
            output = result.output

            if not output.is_mission:
                return MissionIntentResult(is_mission=False)

            if output.criteria is None:
                return MissionIntentResult(is_mission=True)

            logfire.info('mission_intent_detected', criteria=output.criteria.model_dump())
            return MissionIntentResult(is_mission=True, criteria=output.criteria)


@dataclass(slots=True)
class MissionCriteriaParser:
    """Parse mission criteria from intent output."""

    def parse(self, message: str, intent: MissionIntentResult) -> MissionCriteria | None:
        if not intent.is_mission:
            return None
        if intent.criteria is None:
            return _heuristic_parse(message.lower()) or MissionCriteria()
        return intent.criteria


@dataclass(slots=True)
class ChatHandler:
    """Handle chat requests for mission and non-mission modes."""

    classifier: IntentClassifier
    parser: MissionCriteriaParser

    async def parse_mission_criteria(self, messages: list[dict[str, Any]]) -> MissionCriteria | None:
        message_text = _extract_latest_user_message(messages)
        if not message_text:
            return None
        try:
            intent = await self.classifier.classify(message_text)
            return self.parser.parse(message_text, intent)
        except Exception as exc:
            logfire.error('intent_parse_error', error=str(exc))
            return _heuristic_parse(message_text.lower())

    async def handle(self, request: Request) -> StreamingResponse:
        """Handle chat messages from frontend."""
        try:
            body = await request.json()
            messages = body.get('messages', [])
            chat_id = body.get('id', str(uuid.uuid4()))

            logfire.info('chat_request_received', chat_id=chat_id, message_count=len(messages))

            mission_criteria = await self.parse_mission_criteria(messages)
            if mission_criteria:
                logfire.info('mission_mode_activated', chat_id=chat_id, criteria=mission_criteria.model_dump())

                emitter = EventEmitter()
                asyncio.create_task(self._run_mission(emitter=emitter, chat_id=chat_id, criteria=mission_criteria))
                return StreamingResponse(
                    transform_to_vercel_stream(emitter),
                    media_type='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                        'X-Vercel-AI-Data-Stream': 'v1',
                    },
                )

            logfire.info('chat_mode_activated', chat_id=chat_id)
            response_text = _default_chat_response()
            return StreamingResponse(
                stream_text_response(response_text),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                    'X-Vercel-AI-Data-Stream': 'v1',
                },
            )
        except Exception as exc:
            logfire.error('chat_endpoint_error', error=str(exc))
            error_text = f'An error occurred: {str(exc)}'
            return StreamingResponse(
                stream_text_response(error_text),
                media_type='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Vercel-AI-Data-Stream': 'v1'},
            )

    async def _run_mission(self, *, emitter: EventEmitter, chat_id: str, criteria: MissionCriteria) -> None:
        from agent_k.agents.lycurgus import LycurgusOrchestrator

        error_id = f'mission_{chat_id}'
        attempt = 0
        max_attempts = 2

        try:
            while True:
                try:
                    with logfire.span('mission_execution', chat_id=chat_id, attempt=attempt + 1):
                        async with LycurgusOrchestrator() as orchestrator:
                            result = await orchestrator.execute_mission(
                                competition_id=None, criteria=criteria, event_emitter=emitter
                            )
                            await emitter.emit(
                                'mission-complete',
                                {
                                    'success': result.success,
                                    'final_rank': result.final_rank,
                                    'final_score': result.final_score,
                                },
                            )
                    if attempt > 0:
                        await emitter.emit_recovery_complete(
                            error_id=error_id, success=True, resolution='mission_completed'
                        )
                    break
                except Exception as exc:
                    category, strategy = classify_error(exc)
                    recoverable = isinstance(exc, AgentKError) and exc.recoverable
                    await emitter.emit_error(
                        error_id=error_id,
                        category=category,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        context='mission_execution',
                        recovery_strategy=strategy,
                    )
                    logfire.error('mission_execution_failed', error=str(exc), chat_id=chat_id, recoverable=recoverable)
                    if not recoverable or attempt >= max_attempts - 1:
                        if attempt > 0:
                            await emitter.emit_recovery_complete(
                                error_id=error_id, success=False, resolution='exhausted'
                            )
                        break
                    attempt += 1
                    await emitter.emit_recovery_attempt(error_id=error_id, strategy=strategy, attempt=attempt)
                    logfire.warning('mission_recovery_attempt', chat_id=chat_id, attempt=attempt)
        finally:
            emitter.close()


def _extract_latest_user_message(messages: list[dict[str, Any]]) -> str | None:
    user_messages = [message for message in messages if message.get('role') == 'user']
    if not user_messages:
        return None

    latest_message = user_messages[-1]
    text_parts = [part.get('text', '') for part in latest_message.get('parts', []) if part.get('type') == 'text']
    message_text = ' '.join(text_parts).strip()
    return message_text or None


def _default_chat_response() -> str:
    return """I'm Agent-K, an autonomous multi-agent system for Kaggle competitions.

I can help you discover, research, and compete in Kaggle competitions using a team of specialized AI agents:

- **LOBBYIST**: Discovers competitions matching your criteria
- **SCIENTIST**: Researches winning approaches and analyzes data
- **EVOLVER**: Evolves solutions using evolutionary code search
- **LYCURGUS**: Orchestrates the entire mission

To start a mission, try saying something like:
- "Find me a Kaggle competition"
- "Compete in a computer vision challenge with at least $10,000 prize"
- "Discover featured competitions in NLP"

What would you like to do?"""


async def parse_mission_intent(messages: list[dict[str, Any]]) -> MissionCriteria | None:
    """Parse chat messages to detect mission intent and extract criteria."""
    handler = ChatHandler(IntentClassifier(), MissionCriteriaParser())
    return await handler.parse_mission_criteria(messages)


async def transform_to_vercel_stream(emitter: EventEmitter) -> AsyncIterator[str]:
    r"""Transform EventEmitter output to Vercel AI Data Stream format.

    Vercel AI SDK format:
    - Text delta: `0:"text content"\n`
    - Data event: `8:{"type":"event-type","data":{...}}\n`
    - Finish: `d:{"finishReason":"stop"}\n`

    Args:
        emitter: EventEmitter producing AgentKEvent instances.

    Yields:
        Vercel AI formatted event strings.
    """
    try:
        async for event_str in emitter.stream():
            # Parse the SSE formatted event
            # Event comes as "data: {json}\n\n"
            if not event_str.startswith('data: '):
                continue

            json_str = event_str[6:].strip()
            if not json_str:
                continue

            try:
                event_data = json.loads(json_str)
                event_type = event_data.get('type')

                if event_type in AGENT_K_EVENT_TYPES:
                    # Transform to Vercel AI data event format
                    data_part = json.dumps(
                        {
                            'type': event_type,
                            'data': event_data.get('data', {}),
                            'timestamp': event_data.get('timestamp'),
                        }
                    )
                    yield f'8:{data_part}\n'

            except json.JSONDecodeError:
                # Skip malformed events
                continue

    except Exception as exc:
        logfire.error('vercel_stream_transform_failed', error=str(exc))
    finally:
        # Always send finish event
        yield 'd:{"finishReason":"stop"}\n'


async def stream_text_response(text: str) -> AsyncIterator[str]:
    """Stream a simple text response in Vercel AI format.

    Used for chat mode (non-mission) responses.

    Args:
        text: Text content to stream.

    Yields:
        Vercel AI formatted text deltas.
    """
    # Stream text in chunks for better UX
    chunk_size = 50
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        # Escape quotes and backslashes for JSON
        escaped = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        yield f'0:"{escaped}"\n'

    # Send finish event
    yield 'd:{"finishReason":"stop"}\n'


def _calculate_progress_percent(state: MissionState) -> float:
    """Derive a progress percentage when the state lacks explicit progress."""
    if state.overall_progress:
        return state.overall_progress
    phases = MISSION_PHASES
    completed = float(len(state.phases_completed))
    if state.current_phase in phases and state.current_phase not in state.phases_completed:
        completed += 0.5
    return round(min((completed / len(phases)) * 100.0, 100.0), 1)


async def _load_persisted_status(mission_id: str) -> dict[str, Any] | None:
    persistence = create_persistence(mission_id)
    if not persistence.has_snapshots():
        return None

    result = await persistence.load_latest_result()
    if result is not None:
        return {
            'missionId': mission_id,
            'status': 'completed' if result.success else 'failed',
            'currentPhase': None,
            'progress': 100.0,
            'competitionId': result.competition_id,
            'errorMessage': result.error_message,
        }

    state = await persistence.load_latest_state()
    if state is None:
        return None

    error_message = state.errors[-1].get('error') if state.errors else None
    return {
        'missionId': mission_id,
        'status': 'paused',
        'currentPhase': state.current_phase,
        'progress': _calculate_progress_percent(state),
        'competitionId': state.competition_id,
        'errorMessage': error_message,
    }


def create_app() -> FastAPI:  # noqa: C901
    """Create and configure the FastAPI application."""
    configure_instrumentation()
    app = FastAPI(title='AGENT-K', description='Multi-agent Kaggle competition system', version=APP_VERSION)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Store active missions
    missions: dict[str, dict[str, Any]] = {}
    chat_handler = ChatHandler(IntentClassifier(), MissionCriteriaParser())

    async def _schedule_mission_resume(
        mission_id: str,
        *,
        persistence: MissionPersistence | None = None,
        source: str = "api",
    ) -> str | None:
        from agent_k.agents.lycurgus import LycurgusOrchestrator

        entry = missions.get(mission_id)
        if entry and entry.get("task") and not entry["task"].done():
            return "Mission already active"

        persistence = persistence or create_persistence(mission_id)
        if not persistence.has_snapshots():
            return "Mission not found"

        existing_result = await persistence.load_latest_result()
        if existing_result is not None:
            return "Mission already completed"

        emitter = EventEmitter()
        orchestrator = LycurgusOrchestrator(event_emitter=emitter)
        state = await persistence.load_latest_state()

        missions[mission_id] = {
            "emitter": emitter,
            "orchestrator": orchestrator,
            "result": None,
            "competition_id": state.competition_id if state else None,
        }

        async def run_mission() -> None:
            error_id = f"mission_{mission_id}"
            attempt = 0
            max_attempts = 2

            try:
                while True:
                    try:
                        result = await orchestrator.resume_persisted_mission(
                            mission_id, event_emitter=emitter, persistence=persistence
                        )
                        missions[mission_id]["result"] = result
                        await emitter.emit(
                            "mission-complete",
                            {
                                "success": result.success,
                                "finalRank": result.final_rank,
                                "finalScore": result.final_score,
                                "errorMessage": result.error_message,
                                "totalSubmissions": result.total_submissions,
                                "evolutionGenerations": result.evolution_generations,
                                "durationMs": result.duration_ms,
                                "phasesCompleted": list(result.phases_completed),
                            },
                        )
                        if attempt > 0:
                            await emitter.emit_recovery_complete(
                                error_id=error_id, success=True, resolution="mission_completed"
                            )
                        break
                    except Exception as exc:
                        category, strategy = classify_error(exc)
                        recoverable = isinstance(exc, AgentKError) and exc.recoverable
                        await emitter.emit_error(
                            error_id=error_id,
                            category=category,
                            error_type=type(exc).__name__,
                            message=str(exc),
                            context="mission_execution",
                            recovery_strategy=strategy,
                        )
                        logfire.error(
                            "mission_execution_failed",
                            error=str(exc),
                            mission_id=mission_id,
                            recoverable=recoverable,
                        )
                        if not recoverable or attempt >= max_attempts - 1:
                            if attempt > 0:
                                await emitter.emit_recovery_complete(
                                    error_id=error_id, success=False, resolution="exhausted"
                                )
                            break
                        attempt += 1
                        await emitter.emit_recovery_attempt(
                            error_id=error_id,
                            strategy=strategy,
                            attempt=attempt,
                        )
                        logfire.warning(
                            "mission_recovery_attempt",
                            mission_id=mission_id,
                            attempt=attempt,
                        )
            finally:
                emitter.close()

        missions[mission_id]["task"] = asyncio.create_task(run_mission())

        logfire.info("mission_resumed", mission_id=mission_id, source=source)
        return None

    @app.on_event("startup")
    async def auto_resume_missions() -> None:
        mission_ids = _list_persisted_mission_ids()
        if not mission_ids:
            return

        resumed: list[str] = []
        for mission_id in mission_ids:
            try:
                error = await _schedule_mission_resume(mission_id, source="startup")
            except Exception as exc:
                logfire.error(
                    "mission_auto_resume_failed",
                    mission_id=mission_id,
                    error=str(exc),
                )
                continue
            if error:
                logfire.info("mission_auto_resume_skipped", mission_id=mission_id, reason=error)
                continue
            resumed.append(mission_id)

        if resumed:
            logfire.info(
                "mission_auto_resume_started",
                mission_ids=resumed,
                count=len(resumed),
            )

    @app.get('/health')
    async def health_check() -> dict[str, str]:
        """Health check endpoint for Render."""
        return {'status': 'healthy'}

    @app.post('/api/mission/start')
    async def start_mission(request: MissionRequest) -> dict[str, str]:
        """Start a new mission and return mission ID."""
        from agent_k.agents.lycurgus import LycurgusOrchestrator

        mission_id = str(uuid.uuid4())
        competition_id = request.competition_id
        criteria = request.criteria
        if request.evolution_models:
            criteria = criteria.model_copy(update={'evolution_models': tuple(request.evolution_models)})

        if competition_id is None and request.competition_url:
            try:
                competition = await _fetch_competition(request.competition_url)
                competition_id = competition.id
            except AuthenticationError as exc:
                logfire.warning('mission_start_auth_failed', error=str(exc))
                return {'error': str(exc)}
            except CompetitionNotFoundError as exc:
                logfire.warning('mission_start_competition_not_found', error=str(exc))
                return {'error': str(exc)}
            except Exception as exc:
                logfire.error('mission_start_failed', error=str(exc))
                return {'error': 'Mission start failed'}

        emitter = EventEmitter()
        orchestrator = LycurgusOrchestrator(event_emitter=emitter)

        missions[mission_id] = {
            'emitter': emitter,
            'orchestrator': orchestrator,
            'result': None,
            'competition_id': competition_id,
        }

        async def run_mission() -> None:
            error_id = f'mission_{mission_id}'
            attempt = 0
            max_attempts = 2

            try:
                while True:
                    try:
                        result = await orchestrator.execute_mission(
                            competition_id, mission_id=mission_id, criteria=criteria, event_emitter=emitter
                        )
                        missions[mission_id]['result'] = result
                        await emitter.emit(
                            'mission-complete',
                            {
                                'success': result.success,
                                'finalRank': result.final_rank,
                                'finalScore': result.final_score,
                                'errorMessage': result.error_message,
                                'totalSubmissions': result.total_submissions,
                                'evolutionGenerations': result.evolution_generations,
                                'durationMs': result.duration_ms,
                                'phasesCompleted': list(result.phases_completed),
                            },
                        )
                        if attempt > 0:
                            await emitter.emit_recovery_complete(
                                error_id=error_id, success=True, resolution='mission_completed'
                            )
                        break
                    except Exception as exc:
                        category, strategy = classify_error(exc)
                        recoverable = isinstance(exc, AgentKError) and exc.recoverable
                        await emitter.emit_error(
                            error_id=error_id,
                            category=category,
                            error_type=type(exc).__name__,
                            message=str(exc),
                            context='mission_execution',
                            recovery_strategy=strategy,
                        )
                        logfire.error(
                            'mission_execution_failed', error=str(exc), mission_id=mission_id, recoverable=recoverable
                        )
                        if not recoverable or attempt >= max_attempts - 1:
                            if attempt > 0:
                                await emitter.emit_recovery_complete(
                                    error_id=error_id, success=False, resolution='exhausted'
                                )
                            break
                        attempt += 1
                        await emitter.emit_recovery_attempt(error_id=error_id, strategy=strategy, attempt=attempt)
                        logfire.warning('mission_recovery_attempt', mission_id=mission_id, attempt=attempt)
            finally:
                emitter.close()

        missions[mission_id]['task'] = asyncio.create_task(run_mission())

        logfire.info('mission_started', mission_id=mission_id)

        return {'missionId': mission_id}

    @app.post('/api/mission/{mission_id}/resume')
    async def resume_mission(mission_id: str) -> dict[str, str]:
        """Resume a persisted mission and return mission ID."""
        error = await _schedule_mission_resume(mission_id)
        if error:
            return {"error": error}
        return {"missionId": mission_id}

    @app.post('/api/competitions/search')
    async def search_competitions(request: CompetitionSearchRequest) -> dict[str, Any]:
        """Search for competitions that match the requested criteria."""
        try:
            competitions = await _search_competitions(request)
            return {'competitions': competitions, 'count': len(competitions)}
        except AuthenticationError as exc:
            logfire.warning('competition_search_auth_failed', error=str(exc))
            return {'error': str(exc)}
        except Exception as exc:
            logfire.error('competition_search_failed', error=str(exc))
            return {'error': 'Competition search failed'}

    @app.post('/api/competitions/fetch')
    async def fetch_competition(request: CompetitionFetchRequest) -> dict[str, Any]:
        """Fetch a competition by its Kaggle URL."""
        try:
            competition = await _fetch_competition(request.url)
            return {'competition': _serialize_competition(competition)}
        except AuthenticationError as exc:
            logfire.warning('competition_fetch_auth_failed', error=str(exc))
            return {'error': str(exc)}
        except CompetitionNotFoundError as exc:
            logfire.warning('competition_fetch_not_found', error=str(exc))
            return {'error': str(exc)}
        except Exception as exc:
            logfire.error('competition_fetch_failed', error=str(exc))
            return {'error': 'Competition fetch failed'}

    @app.get("/api/mission/best-results")
    async def get_best_results(limit: int | None = None) -> dict[str, Any]:
        """Return best historical results from persisted missions."""
        try:
            results = await _load_best_results(limit=limit)
            return {"results": results}
        except Exception as exc:
            logfire.error("best_results_failed", error=str(exc))
            return {"results": [], "error": "Unable to load best results"}

    @app.get('/api/mission/{mission_id}/stream')
    async def stream_mission(mission_id: str, request: Request) -> StreamingResponse:
        """Stream mission events via SSE."""
        if mission_id not in missions:
            return StreamingResponse(iter(['data: {"error": "Mission not found"}\n\n']), media_type='text/event-stream')

        emitter = missions[mission_id]['emitter']

        async def event_generator() -> AsyncIterator[str]:
            async for event in emitter.stream():
                if await request.is_disconnected():
                    break
                yield event

        return StreamingResponse(
            event_generator(),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'},
        )

    @app.get('/api/mission/{mission_id}/status')
    async def get_mission_status(mission_id: str) -> dict[str, Any]:
        """Get current mission status."""
        if mission_id not in missions:
            persisted = await _load_persisted_status(mission_id)
            if persisted is None:
                return {'error': 'Mission not found'}
            return persisted

        entry = missions[mission_id]
        orchestrator = entry.get('orchestrator')
        state = orchestrator.state if orchestrator else None
        result = entry.get('result')

        if state is not None:
            status = 'executing'
            progress = _calculate_progress_percent(state)
            current_phase = state.current_phase
            competition_id = state.competition_id
        elif result is not None:
            status = 'completed' if result.success else 'failed'
            progress = 100.0
            current_phase = None
            competition_id = result.competition_id
        else:
            status = 'planning'
            progress = 0.0
            current_phase = None
            competition_id = entry.get('competition_id')

        return {
            'missionId': mission_id,
            'status': status,
            'currentPhase': current_phase,
            'progress': progress,
            'competitionId': competition_id,
            'errorMessage': result.error_message if result is not None else None,
        }

    @app.post('/api/mission/{mission_id}/abort')
    async def abort_mission(mission_id: str) -> dict[str, str]:
        """Abort an active mission."""
        if mission_id not in missions:
            return {'error': 'Mission not found'}

        entry = missions[mission_id]
        emitter = entry['emitter']
        orchestrator = entry.get('orchestrator')
        task = entry.get('task')

        if orchestrator and orchestrator.is_active:
            try:
                await orchestrator.abort_mission('aborted_via_api')
            except Exception as exc:
                logfire.warning('mission_abort_failed', mission_id=mission_id, error=str(exc))

        if task and not task.done():
            task.cancel()

        emitter.close()

        logfire.info('mission_aborted', mission_id=mission_id)

        return {'status': 'aborted'}

    @app.post('/agentic_generative_ui/')
    async def handle_chat(request: Request) -> StreamingResponse:
        """Handle chat messages from frontend."""
        return await chat_handler.handle(request)

    return app


# =============================================================================
# Competition Search Helpers
# =============================================================================
def _build_kaggle_adapter() -> KaggleAdapter:
    username = os.getenv('KAGGLE_USERNAME')
    api_key = os.getenv('KAGGLE_KEY')
    if not username or not api_key:
        raise AuthenticationError('kaggle', 'Missing KAGGLE_USERNAME/KAGGLE_KEY')
    return KaggleAdapter(KaggleSettings(username=username, api_key=api_key))


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower().replace(' ', '_').replace('-', '_')


def _matches_domains(competition: Competition, domains: list[str]) -> bool:
    if not domains:
        return True
    tags = {tag.lower() for tag in competition.tags}
    description = competition.description or ''
    haystack = f'{competition.title} {description}'.lower()
    for domain in domains:
        key = _normalize_domain(domain)
        keywords = _DOMAIN_KEYWORDS.get(key, (key,))
        for keyword in keywords:
            if keyword in tags or keyword in haystack:
                return True
    return False


def _serialize_competition(competition: Competition) -> dict[str, Any]:
    return {
        'id': competition.id,
        'title': competition.title,
        'description': competition.description,
        'competitionType': competition.competition_type.value,
        'metric': competition.metric.value,
        'metricDirection': competition.metric_direction,
        'deadline': competition.deadline.isoformat(),
        'prizePool': competition.prize_pool,
        'maxTeamSize': competition.max_team_size,
        'maxDailySubmissions': competition.max_daily_submissions,
        'tags': sorted(competition.tags),
        'url': competition.url,
    }


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _derive_best_category(competition: Competition | None) -> str | None:
    if competition is None:
        return None
    if competition.tags:
        return sorted(competition.tags)[0]
    return competition.competition_type.value


def _select_best_submission(
    submissions: list[LeaderboardSubmission],
) -> LeaderboardSubmission | None:
    if not submissions:
        return None

    best = submissions[0]
    for current in submissions[1:]:
        if best.rank is None and current.rank is not None:
            best = current
            continue

        if best.rank is not None and current.rank is None:
            continue

        if best.rank is not None and current.rank is not None:
            if current.rank < best.rank:
                best = current
            continue

        if best.percentile is None and current.percentile is not None:
            best = current
            continue

        if best.percentile is not None and current.percentile is None:
            continue

        if best.percentile is not None and current.percentile is not None:
            if current.percentile < best.percentile:
                best = current
            continue

        if best.public_score is None and current.public_score is not None:
            best = current
            continue

        if best.public_score is not None and current.public_score is None:
            continue

        if best.public_score is not None and current.public_score is not None:
            if current.public_score > best.public_score:
                best = current
            continue

        if current.cv_score != best.cv_score and current.cv_score > best.cv_score:
            best = current
            continue

        if current.submitted_at > best.submitted_at:
            best = current

    return best


def _build_best_result(
    state: MissionState | None,
    result: MissionResult | None,
) -> dict[str, Any] | None:
    competition = state.selected_competition if state else None
    competition_id = (
        competition.id
        if competition
        else state.competition_id
        if state and state.competition_id
        else result.competition_id
        if result and result.competition_id
        else None
    )
    if not competition_id:
        return None

    competition_title = competition.title if competition else competition_id
    best_submission = (
        _select_best_submission(state.evolution_state.leaderboard_submissions)
        if state and state.evolution_state
        else None
    )
    rank = (
        best_submission.rank
        if best_submission and best_submission.rank is not None
        else result.final_rank
        if result and result.final_rank is not None
        else state.final_rank
        if state and state.final_rank is not None
        else None
    )
    score = (
        best_submission.public_score
        if best_submission and best_submission.public_score is not None
        else best_submission.cv_score
        if best_submission
        else result.final_score
        if result and result.final_score is not None
        else state.final_score
        if state and state.final_score is not None
        else None
    )
    total_teams = best_submission.total_teams if best_submission else None
    percentile = best_submission.percentile if best_submission else None
    if percentile is None and rank and total_teams:
        percentile = (rank / total_teams) * 100

    submission_id = (
        best_submission.submission_id
        if best_submission
        else state.final_submission_id
        if state and state.final_submission_id
        else None
    )
    submitted_at = best_submission.submitted_at if best_submission else None
    recorded_at = submitted_at or (state.started_at if state else None) or datetime.now(UTC)

    if not best_submission and result is None and submission_id is None:
        return None

    return {
        "competitionId": competition_id,
        "competitionTitle": competition_title,
        "competitionUrl": competition.url if competition else None,
        "submissionId": submission_id,
        "rank": rank,
        "totalTeams": total_teams,
        "percentile": percentile,
        "score": score,
        "submittedAt": _format_datetime(submitted_at),
        "category": _derive_best_category(competition),
        "recordedAt": _format_datetime(recorded_at) or datetime.now(UTC).isoformat(),
    }


def _is_better_best_result(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_rank = candidate.get("rank")
    current_rank = current.get("rank")
    if candidate_rank is not None and current_rank is None:
        return True
    if candidate_rank is None and current_rank is not None:
        return False
    if candidate_rank is not None and current_rank is not None:
        return candidate_rank < current_rank

    candidate_percentile = candidate.get("percentile")
    current_percentile = current.get("percentile")
    if candidate_percentile is not None and current_percentile is None:
        return True
    if candidate_percentile is None and current_percentile is not None:
        return False
    if candidate_percentile is not None and current_percentile is not None:
        return candidate_percentile < current_percentile

    candidate_score = candidate.get("score")
    current_score = current.get("score")
    if candidate_score is not None and current_score is None:
        return True
    if candidate_score is None and current_score is not None:
        return False
    if candidate_score is not None and current_score is not None:
        return candidate_score > current_score

    candidate_time = _parse_datetime(candidate.get("recordedAt"))
    current_time = _parse_datetime(current.get("recordedAt"))
    if candidate_time and current_time:
        return candidate_time > current_time

    return False


def _sort_best_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(entry: dict[str, Any]) -> tuple[int, float]:
        rank = entry.get("rank")
        if rank is not None:
            return (0, float(rank))
        recorded_at = _parse_datetime(entry.get("recordedAt"))
        if recorded_at is not None:
            return (1, -recorded_at.timestamp())
        return (2, 0.0)

    return sorted(results, key=sort_key)


def _list_persisted_mission_ids() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    try:
        mission_dirs = [path for path in CHECKPOINT_DIR.iterdir() if path.is_dir()]
    except OSError as exc:
        logfire.warning("mission_history_scan_failed", error=str(exc))
        return []

    mission_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in mission_dirs]


async def _load_best_results(limit: int | None = None) -> list[dict[str, Any]]:
    mission_ids = _list_persisted_mission_ids()
    if not mission_ids:
        return []

    results_by_competition: dict[str, dict[str, Any]] = {}
    for mission_id in mission_ids:
        persistence = create_persistence(mission_id)
        if not persistence.has_snapshots():
            continue
        try:
            state = await persistence.load_latest_state()
            result = await persistence.load_latest_result()
        except Exception as exc:
            logfire.warning("mission_history_load_failed", mission_id=mission_id, error=str(exc))
            continue

        best_result = _build_best_result(state, result)
        if not best_result:
            continue
        competition_id = best_result["competitionId"]
        existing = results_by_competition.get(competition_id)
        if existing is None or _is_better_best_result(best_result, existing):
            results_by_competition[competition_id] = best_result

    results = _sort_best_results(list(results_by_competition.values()))
    if limit and limit > 0:
        return results[:limit]
    return results


async def _search_competitions(request: CompetitionSearchRequest) -> list[dict[str, Any]]:  # noqa: C901
    adapter = _build_kaggle_adapter()
    competitions: list[Competition] = []
    seen: set[str] = set()
    categories = [competition_type.value for competition_type in request.competition_types]
    keywords = [domain.replace('_', ' ') for domain in request.domains]
    min_prize = request.min_prize
    if request.paid_only and (min_prize is None or min_prize == 0):
        min_prize = 1

    async with adapter:
        if not categories:
            async for competition in adapter.search_competitions(
                categories=None, keywords=keywords or None, min_prize=min_prize, active_only=True
            ):
                if competition.id in seen:
                    continue
                if competition.days_remaining < request.min_days_remaining:
                    continue
                if request.paid_only and (competition.prize_pool or 0) <= 0:
                    continue
                if not _matches_domains(competition, request.domains):
                    continue
                seen.add(competition.id)
                competitions.append(competition)
                if len(competitions) >= MAX_COMPETITION_RESULTS:
                    break
        else:
            for category in categories:
                async for competition in adapter.search_competitions(
                    categories=[category], keywords=keywords or None, min_prize=min_prize, active_only=True
                ):
                    if competition.id in seen:
                        continue
                    if competition.days_remaining < request.min_days_remaining:
                        continue
                    if request.paid_only and (competition.prize_pool or 0) <= 0:
                        continue
                    if not _matches_domains(competition, request.domains):
                        continue
                    seen.add(competition.id)
                    competitions.append(competition)
                    if len(competitions) >= MAX_COMPETITION_RESULTS:
                        break
                if len(competitions) >= MAX_COMPETITION_RESULTS:
                    break

    return [_serialize_competition(competition) for competition in competitions]


async def _fetch_competition(url: str) -> Competition:
    adapter = _build_kaggle_adapter()
    async with adapter:
        return await adapter.get_competition_by_url(url)


# =============================================================================
# Entry Point
# =============================================================================
app: Final = create_app()


def main() -> None:
    """Run the AG-UI server."""
    import uvicorn

    port = int(os.environ.get('PORT', str(DEFAULT_PORT)))
    host = os.environ.get('HOST', DEFAULT_HOST)

    uvicorn.run(APP_MODULE, host=host, port=port, reload=os.environ.get('RELOAD', 'false').lower() == 'true')


def _heuristic_parse(text_lower: str) -> MissionCriteria | None:
    """Fallback heuristic parser for mission criteria."""
    criteria_dict: dict[str, Any] = {}

    if 'featured' in text_lower:
        criteria_dict['target_competition_types'] = [CompetitionType.FEATURED]
    elif 'research' in text_lower:
        criteria_dict['target_competition_types'] = [CompetitionType.RESEARCH]
    elif 'playground' in text_lower:
        criteria_dict['target_competition_types'] = [CompetitionType.PLAYGROUND]

    prize_match = _PRIZE_PATTERN.search(text_lower)
    if prize_match:
        amount_str = prize_match.group(1).replace(',', '')
        multiplier_suffix = prize_match.group(2)
        amount = float(amount_str)
        if multiplier_suffix in ['k', 'thousand']:
            amount *= 1000
        elif multiplier_suffix in ['m', 'million']:
            amount *= 1_000_000
        criteria_dict['min_prize_pool'] = int(amount)

    days_match = _DAYS_PATTERN.search(text_lower)
    if days_match:
        days = int(days_match.group(1))
        if 'week' in days_match.group(2):
            days *= 7
        criteria_dict['min_days_remaining'] = days

    domains = []
    if 'computer vision' in text_lower or 'cv' in text_lower or 'image' in text_lower:
        domains.append('computer_vision')
    if 'nlp' in text_lower or 'natural language' in text_lower or 'text' in text_lower:
        domains.append('nlp')
    if 'tabular' in text_lower:
        domains.append('tabular')
    if domains:
        criteria_dict['target_domains'] = domains

    percentile_match = _PERCENTILE_PATTERN.search(text_lower)
    if percentile_match:
        criteria_dict['target_leaderboard_percentile'] = int(percentile_match.group(1)) / 100

    return MissionCriteria(**criteria_dict) if criteria_dict else MissionCriteria()


if __name__ == '__main__':
    main()
