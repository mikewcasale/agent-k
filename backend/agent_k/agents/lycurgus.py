"""Lycurgus orchestrator - mission coordination for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import inspect
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final

# Third-party (alphabetical)
import httpx
import logfire
from pydantic import Field
from pydantic_graph import Graph
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.adapters.openevolve import OpenEvolveAdapter
from agent_k.agents.evolver import evolver_agent
from agent_k.agents.lobbyist import lobbyist_agent
from agent_k.agents.prompts import LYCURGUS_SYSTEM_PROMPT
from agent_k.agents.scientist import scientist_agent
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.core.exceptions import CompetitionNotFoundError
from agent_k.core.models import MissionCriteria
from agent_k.mission.nodes import DiscoveryNode, EvolutionNode, PrototypeNode, ResearchNode, SubmissionNode
from agent_k.mission.state import GraphContext, MissionResult, MissionState
from agent_k.ui.ag_ui import EventEmitter

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai import Agent

    from agent_k.core.protocols import PlatformAdapter

__all__ = (
    'LycurgusDeps',
    'LycurgusOrchestrator',
    'LycurgusSettings',
    'LYCURGUS_SYSTEM_PROMPT',
    'MissionStatus',
    'SCHEMA_VERSION',
    'orchestrate',
    'validate_mission_result',
)

SCHEMA_VERSION: Final[str] = '1.0.0'


class LycurgusSettings(BaseSettings):
    """Settings for the Lycurgus orchestrator."""

    model_config = SettingsConfigDict(env_prefix='LYCURGUS_', env_file='.env', extra='ignore', validate_default=True)
    default_model: str = Field(default=DEFAULT_MODEL, description='Default model spec for mission orchestration')
    max_evolution_rounds: int = Field(default=100, ge=1, description='Maximum evolution rounds for missions')

    @classmethod
    def from_file(cls, path: Path) -> LycurgusSettings:
        """Create settings from JSON file."""
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(
            default_model=data.get('default_model', cls().default_model),
            max_evolution_rounds=data.get('max_evolution_rounds', cls().max_evolution_rounds),
        )

    @classmethod
    def with_devstral(cls, base_url: str | None = None) -> LycurgusSettings:
        """Create settings using Devstral model."""
        model = f'devstral:{base_url}' if base_url else 'devstral:local'
        return cls(default_model=model)


@dataclass
class LycurgusDeps:
    """Dependencies for the Lycurgus orchestrator."""

    event_emitter: EventEmitter
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter


@dataclass
class MissionStatus:
    """Mission status snapshot."""

    phase: str
    progress: float
    metrics: dict[str, Any]
    ABORTED: ClassVar[str] = 'aborted'


class LycurgusOrchestrator:
    """Orchestration agent coordinating the multi-agent Kaggle competition system.

    LYCURGUS (Multi-agent Evolutionary Learning Engine for Neural Competition
    Optimization and Leaderboard Intelligence Advancement) coordinates the
    Lobbyist, Scientist, and Evolver agents to autonomously compete in Kaggle.

    The orchestrator implements a state machine using pydantic-graph to manage
    the competition lifecycle from discovery through submission.

    Attributes:
        state: Current mission state.
        agents: Dictionary of specialized agents.
        graph: State machine graph for orchestration.
    """

    # =========================================================================
    # Class Variables (ClassVar annotations)
    # =========================================================================
    _default_model: ClassVar[str] = 'anthropic:claude-sonnet-4-5'
    _max_evolution_rounds: ClassVar[int] = 100
    _supported_competition_types: ClassVar[frozenset[str]] = frozenset({'featured', 'research', 'playground'})

    # =========================================================================
    # Instance Variable Annotations (slots if applicable)
    # =========================================================================
    __slots__ = (
        '_state',
        '_agents',
        '_graph',
        '_config',
        '_logger',
        '_event_emitter',
        '_http_client',
        '_platform_adapter',
        '_owns_http_client',
        '_owns_platform_adapter',
        '_paused',
        '_entered',
        '_resources_ready',
    )

    def __init__(
        self,
        *,
        config: LycurgusSettings | None = None,
        model: str | None = None,
        event_emitter: EventEmitter | None = None,
        http_client: httpx.AsyncClient | None = None,
        platform_adapter: PlatformAdapter | None = None,
    ) -> None:
        """Initialize the LYCURGUS orchestrator.

        Args:
            config: Configuration for orchestration behavior.
            model: Override default model for all agents.
            event_emitter: Event emitter for AG-UI streaming.
            http_client: Shared HTTP client for research tools.
            platform_adapter: Adapter for platform operations.
        """
        self._config = config or LycurgusSettings()
        if model is not None:
            self._config.default_model = model
        self._logger = logfire  # Use logfire directly, service name can be set in spans
        self._event_emitter = event_emitter
        self._http_client = http_client
        self._platform_adapter = platform_adapter
        self._owns_http_client = http_client is None
        self._owns_platform_adapter = platform_adapter is None
        self._paused = False
        self._entered = False
        self._resources_ready = False
        self._agents = self._initialize_agents()
        self._graph = self._build_orchestration_graph()
        self._state: MissionState | None = None

    # =========================================================================
    # Other Dunder Methods (alphabetical)
    # =========================================================================
    def __repr__(self) -> str:
        return f'{type(self).__name__}(state={self._state!r}, agents={list(self._agents.keys())!r})'

    def __str__(self) -> str:
        status = 'active' if self._state else 'idle'
        return f'LYCURGUS Orchestrator ({status})'

    async def __aenter__(self) -> LycurgusOrchestrator:
        """Async context manager entry for resource management."""
        await self._initialize_resources()
        self._entered = True
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Async context manager exit for cleanup."""
        self._entered = False
        await self._cleanup_resources()

    # =========================================================================
    # Class Methods (constructors and factories)
    # =========================================================================
    @classmethod
    def from_config_file(cls, path: Path) -> LycurgusOrchestrator:
        """Create orchestrator from configuration file.

        Args:
            path: Path to YAML or JSON configuration file.

        Returns:
            Configured LycurgusOrchestrator instance.
        """
        config = LycurgusSettings.from_file(path)
        return cls(config=config)

    @classmethod
    def with_custom_agents(cls, agents: dict[str, Agent[Any, Any]]) -> LycurgusOrchestrator:
        """Create orchestrator with custom agent implementations.

        Args:
            agents: Dictionary mapping agent names to implementations.

        Returns:
            Orchestrator with custom agents.
        """
        instance = cls()
        instance._agents.update(agents)
        return instance

    # =========================================================================
    # Static Methods
    # =========================================================================
    @staticmethod
    def validate_competition_id(competition_id: str) -> bool:
        """Validate Kaggle competition identifier format.

        Args:
            competition_id: Competition identifier to validate.

        Returns:
            True if valid, False otherwise.
        """
        pattern = r'^[a-z0-9-]+$'
        return bool(re.match(pattern, competition_id))

    # =========================================================================
    # Properties (read-only first, then read-write)
    # =========================================================================
    @property
    def state(self) -> MissionState | None:
        """Current mission state, or None if no mission active."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the orchestrator has an active mission."""
        return self._state is not None

    @property
    def current_phase(self) -> str | None:
        """Current phase of the active mission."""
        return None if self._state is None else self._state.current_phase

    @property
    def config(self) -> LycurgusSettings:
        """Orchestrator configuration (read-write)."""
        return self._config

    @config.setter
    def config(self, value: LycurgusSettings) -> None:
        """Update orchestrator configuration.

        Args:
            value: New configuration to apply.

        Raises:
            RuntimeError: If mission is active during reconfiguration.
        """
        if self.is_active:
            raise RuntimeError('Cannot reconfigure during active mission')
        self._config = value

    # =========================================================================
    # Public Instance Methods (alphabetical)
    # =========================================================================
    async def abort_mission(self, reason: str) -> None:
        """Abort the current mission.

        Args:
            reason: Reason for aborting the mission.

        Raises:
            RuntimeError: If no mission is active.
        """
        if not self.is_active:
            raise RuntimeError('No active mission to abort')

        with self._logger.span('abort_mission', reason=reason):
            await self._transition_to_aborted(reason)
            self._state = None

    async def execute_mission(
        self,
        competition_id: str | None,
        *,
        criteria: MissionCriteria | None = None,
        event_emitter: EventEmitter | None = None,
        http_client: httpx.AsyncClient | None = None,
        platform_adapter: PlatformAdapter | None = None,
    ) -> MissionResult:
        """Execute a full competition mission.

        This method orchestrates the complete competition lifecycle:
        1. Discovery and validation via Lobbyist
        2. Research and analysis via Scientist
        3. Solution evolution via Evolver
        4. Submission to Kaggle

        Args:
            competition_id: Target competition identifier (optional for discovery).
            criteria: Optional criteria constraining the mission.
            event_emitter: Event emitter for streaming events.
            http_client: Shared HTTP client for research tools.
            platform_adapter: Adapter for platform operations.

        Returns:
            MissionResult containing outcomes and metrics.

        Raises:
            CompetitionNotFoundError: If competition doesn't exist.
            MissionExecutionError: If mission fails during execution.
        """
        with self._logger.span('execute_mission', competition_id=competition_id):
            if competition_id and not self.validate_competition_id(competition_id):
                raise CompetitionNotFoundError(competition_id)

            if event_emitter is not None:
                self._event_emitter = event_emitter
            if http_client is not None:
                self._http_client = http_client
                self._owns_http_client = False
            if platform_adapter is not None:
                self._platform_adapter = platform_adapter
                self._owns_platform_adapter = False

            mission_id = str(uuid.uuid4())
            self._state = MissionState(
                mission_id=mission_id, competition_id=competition_id, criteria=criteria or MissionCriteria()
            )

            initialized_here = False
            if not self._resources_ready:
                await self._initialize_resources()
                initialized_here = True

            try:
                context = GraphContext(
                    event_emitter=self._event_emitter,
                    http_client=self._http_client,
                    platform_adapter=self._platform_adapter,
                )
                return await self._run_graph(context)
            finally:
                if initialized_here and not self._entered:
                    await self._cleanup_resources()
                self._state = None

    async def get_mission_status(self) -> MissionStatus:
        """Get current mission status.

        Returns:
            Current status of active mission.

        Raises:
            RuntimeError: If no mission is active.
        """
        if not self.is_active:
            raise RuntimeError('No active mission')

        state = self._state
        if state is None:
            raise RuntimeError('No active mission')
        progress = self._calculate_progress(state)
        metrics = {
            'phases_completed': list(state.phases_completed),
            'competitions_found': len(state.discovered_competitions),
            'current_phase': state.current_phase,
            'generations': (len(state.evolution_state.generation_history) if state.evolution_state else 0),
        }
        return MissionStatus(phase=state.current_phase, progress=progress, metrics=metrics)

    async def pause_mission(self) -> None:
        """Pause the current mission for later resumption."""
        if not self.is_active:
            raise RuntimeError('No active mission')

        state = self._state
        if state is None:
            raise RuntimeError('No active mission')

        if self._paused:
            return

        self._paused = True
        if self._event_emitter:
            await self._event_emitter.emit(
                'phase-error', {'phase': state.current_phase, 'error': 'mission_paused', 'recoverable': True}
            )
        self._logger.info('mission_paused', mission_id=state.mission_id)

    async def resume_mission(self) -> None:
        """Resume a previously paused mission."""
        if not self.is_active:
            raise RuntimeError('No active mission')

        state = self._state
        if state is None:
            raise RuntimeError('No active mission')

        if not self._paused:
            return

        self._paused = False
        if self._event_emitter:
            await self._event_emitter.emit('recovery-attempt', {'phase': state.current_phase, 'strategy': 'resume'})
        self._logger.info('mission_resumed', mission_id=state.mission_id)

    # =========================================================================
    # Protected Methods (for subclass use)
    # =========================================================================
    def _initialize_agents(self) -> dict[str, Agent[Any, Any]]:
        """Initialize specialized agent singletons."""
        return {'lobbyist': lobbyist_agent, 'scientist': scientist_agent, 'evolver': evolver_agent}

    def _build_orchestration_graph(self) -> Graph[MissionState, GraphContext, MissionResult]:
        """Build the state machine graph for orchestration."""
        return Graph(
            nodes=(DiscoveryNode, ResearchNode, PrototypeNode, EvolutionNode, SubmissionNode), state_type=MissionState
        )

    async def _run_graph(self, context: GraphContext) -> MissionResult:
        """Execute the orchestration graph to completion."""
        if self._state is None:
            raise RuntimeError('No mission state initialized')
        node = DiscoveryNode(lobbyist_agent=self._agents['lobbyist'])
        result = await self._graph.run(node, state=self._state, deps=context)
        self._state = result.state
        return result.output

    # =========================================================================
    # Private Methods (internal implementation)
    # =========================================================================
    def _calculate_progress(self, state: MissionState) -> float:
        """Calculate mission progress from the current state."""
        phases = ('discovery', 'research', 'prototype', 'evolution', 'submission')
        completed = float(len(state.phases_completed))
        if state.current_phase in phases and state.current_phase not in state.phases_completed:
            completed += 0.5
        return round(min(completed / len(phases), 1.0), 3)

    def _create_platform_adapter(self) -> PlatformAdapter:
        """Create a platform adapter based on available credentials."""
        username = os.getenv('KAGGLE_USERNAME')
        api_key = os.getenv('KAGGLE_KEY')
        if username and api_key:
            return KaggleAdapter(KaggleSettings(username=username, api_key=api_key))
        return OpenEvolveAdapter()

    async def _maybe_enter(self, adapter: PlatformAdapter) -> None:
        """Enter adapter context or authenticate when required."""
        enter = getattr(adapter, '__aenter__', None)
        if callable(enter):
            result = enter()
            if inspect.isawaitable(result):
                await result
            return
        await adapter.authenticate()

    async def _maybe_exit(self, adapter: PlatformAdapter) -> None:
        """Exit adapter context manager when supported."""
        exit_fn = getattr(adapter, '__aexit__', None)
        if callable(exit_fn):
            result = exit_fn(None, None, None)
            if inspect.isawaitable(result):
                await result

    async def _initialize_resources(self) -> None:
        """Initialize async resources."""
        if self._resources_ready:
            return

        if self._event_emitter is None:
            self._event_emitter = EventEmitter()

        if self._http_client is None:
            self._http_client = httpx.AsyncClient()

        if self._platform_adapter is None:
            self._platform_adapter = self._create_platform_adapter()
            self._owns_platform_adapter = True

        await self._maybe_enter(self._platform_adapter)
        self._resources_ready = True

    async def _cleanup_resources(self) -> None:
        """Clean up async resources."""
        if not self._resources_ready:
            return

        if self._owns_platform_adapter and self._platform_adapter:
            await self._maybe_exit(self._platform_adapter)

        if self._owns_http_client and self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._resources_ready = False

    async def _transition_to_aborted(self, reason: str) -> None:
        """Handle transition to aborted state."""
        if self._event_emitter and self._state:
            await self._event_emitter.emit(
                'phase-error', {'phase': self._state.current_phase, 'error': reason, 'recoverable': False}
            )
        self._logger.warning('mission_aborted', reason=reason)


async def orchestrate(
    orchestrator: LycurgusOrchestrator, competition_id: str, criteria: MissionCriteria | None = None
) -> MissionResult:
    """Convenience helper to execute a mission."""
    return await orchestrator.execute_mission(competition_id, criteria=criteria)


def validate_mission_result(result: MissionResult) -> MissionResult:
    """Validate mission result payload."""
    return result
