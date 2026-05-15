"""Lycurgus orchestrator - mission coordination for AGENT-K.

@notice: |
    Lycurgus orchestrator - mission coordination for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.agents.lycurgus
    provides:
        - agent_k.agents.lycurgus:LycurgusOrchestrator
        - agent_k.agents.lycurgus:LycurgusSettings
        - agent_k.agents.lycurgus:LycurgusDeps
        - agent_k.agents.lycurgus:MissionStatus
        - agent_k.agents.lycurgus:orchestrate
        - agent_k.agents.lycurgus:validate_mission_result
    consumes:
        - agent_k.mission.nodes:DiscoveryNode
        - agent_k.mission.nodes:ResearchNode
        - agent_k.mission.nodes:PrototypeNode
        - agent_k.mission.nodes:EvolutionNode
        - agent_k.mission.nodes:SubmissionNode
        - agent_k.ui.agui:EventEmitter
    pattern: orchestrator

@similar:
    - id: agent_k.mission.nodes
        when: "Mission nodes define phases; this module orchestrates them."

@agent-guidance:
    do:
        - "Use agent_k.agents.lycurgus as the canonical home for this capability."
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

import inspect
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final

import httpx
import logfire
from pydantic import Field
from pydantic_graph import Graph
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.adapters.openevolve import OpenEvolveAdapter
from agent_k.agents.evolver import evolver_agent
from agent_k.agents.lobbyist import lobbyist_agent
from agent_k.agents.prompts import LYCURGUS_SYSTEM_PROMPT
from agent_k.agents.scientist import scientist_agent
from agent_k.core.constants import DEFAULT_MODEL, MAX_MISSION_EVOLUTION_ROUNDS
from agent_k.core.exceptions import CompetitionNotFoundError
from agent_k.core.models import MissionCriteria
from agent_k.core.sage import Doc
from agent_k.mission.nodes import DiscoveryNode, EvolutionNode, PrototypeNode, ResearchNode, SubmissionNode
from agent_k.mission.persistence import MissionPersistence, create_persistence
from agent_k.mission.state import GraphContext, MissionResult, MissionState
from agent_k.ui.agui import EventEmitter

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai import Agent

    from agent_k.core.protocols import PlatformAdapter

__all__ = (
    "LycurgusDeps",
    "LycurgusOrchestrator",
    "LycurgusSettings",
    "LYCURGUS_SYSTEM_PROMPT",
    "MissionStatus",
    "SCHEMA_VERSION",
    "build_research_http_client",
    "orchestrate",
    "validate_mission_result",
)

SCHEMA_VERSION: Final[str] = "1.0.0"

RESEARCH_HTTP_TIMEOUT_SECONDS: Final[float] = 30.0
"""Read/write/pool timeout for the shared research HTTP client."""

RESEARCH_HTTP_CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
"""Connect timeout so dead hosts fail fast instead of stalling a mission."""

RESEARCH_HTTP_MAX_CONNECTIONS: Final[int] = 20
"""Connection pool ceiling for the shared research HTTP client."""


def build_research_http_client(
    timeout_seconds: Annotated[float, Doc("Read/write/pool timeout in seconds.")] = RESEARCH_HTTP_TIMEOUT_SECONDS,
    max_connections: Annotated[int, Doc("Maximum simultaneous connections.")] = RESEARCH_HTTP_MAX_CONNECTIONS,
) -> httpx.AsyncClient:
    """Build the shared HTTP client used by research toolsets.

    @dev: |
        httpx defaults to a 5-second timeout on every phase of a request, which
        is too aggressive for fetching competition pages, scholarly results, and
        leaderboard CSVs and causes flaky mid-mission failures. This applies a
        generous read/write timeout while keeping a short connect timeout so
        genuinely unreachable hosts still fail fast.

        @notice: |
            Returns an httpx.AsyncClient with explicit timeouts and pool limits.

        @factory-for:
            id: httpx:AsyncClient
            rationale: "Centralizes research HTTP client configuration."
            singleton: false
            cache-key: timeout_seconds

        @canonical-home:
            for:
                - "research HTTP client construction"
            notes: "Use build_research_http_client for orchestrator HTTP clients."
    """
    timeout = httpx.Timeout(timeout_seconds, connect=min(RESEARCH_HTTP_CONNECT_TIMEOUT_SECONDS, timeout_seconds))
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max(1, max_connections // 2))
    return httpx.AsyncClient(timeout=timeout, limits=limits)


class LycurgusSettings(BaseSettings):
    """Settings for the Lycurgus orchestrator.

    @notice: |
        Settings for the Lycurgus orchestrator.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: settings
            rationale: "Centralizes orchestration configuration."
            violations: "Per-run overrides lead to inconsistent missions."
    """

    model_config = SettingsConfigDict(env_prefix="LYCURGUS_", env_file=".env", extra="ignore", validate_default=True)
    default_model: str = Field(default=DEFAULT_MODEL, description="Default model spec for mission orchestration")
    max_evolution_rounds: int = Field(
        default=100, ge=1, le=MAX_MISSION_EVOLUTION_ROUNDS, description="Maximum evolution rounds for missions"
    )
    http_timeout_seconds: float = Field(
        default=RESEARCH_HTTP_TIMEOUT_SECONDS,
        gt=0,
        description="Read/write timeout (seconds) for the shared research HTTP client",
    )
    http_max_connections: int = Field(
        default=RESEARCH_HTTP_MAX_CONNECTIONS,
        ge=1,
        description="Maximum simultaneous connections for the shared research HTTP client",
    )

    @classmethod
    def from_file(cls, path: Annotated[Path, Doc("Path to JSON settings file.")]) -> LycurgusSettings:
        """Create settings from JSON file.

        @notice: |
            Loads orchestration settings from a JSON file.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        defaults = cls()
        return cls(
            default_model=data.get("default_model", defaults.default_model),
            max_evolution_rounds=data.get("max_evolution_rounds", defaults.max_evolution_rounds),
            http_timeout_seconds=data.get("http_timeout_seconds", defaults.http_timeout_seconds),
            http_max_connections=data.get("http_max_connections", defaults.http_max_connections),
        )

    @classmethod
    def with_devstral(
        cls, base_url: Annotated[str | None, Doc("Optional Devstral base URL.")] = None
    ) -> LycurgusSettings:
        """Create settings using Devstral model.

        @notice: |
            Sets the default model to Devstral (local or custom URL).
        """
        model = f"devstral:{base_url}" if base_url else "devstral:local"
        return cls(default_model=model)


@dataclass
class LycurgusDeps:
    """Dependencies for the Lycurgus orchestrator.

    @notice: |
        Dependencies for the Lycurgus orchestrator.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: dependency-container
            rationale: "Groups runtime services for orchestration."
            violations: "Hidden globals make orchestration brittle."

        @collaborators:
            required:
                - agent_k.ui.agui:EventEmitter
                - httpx:AsyncClient
                - agent_k.core.protocols:PlatformAdapter
            injection: constructor
            lifecycle: "Allocated per orchestrator run."
    """

    event_emitter: EventEmitter
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter


@dataclass
class MissionStatus:
    """Mission status snapshot.

    @notice: |
        Mission status snapshot.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: output-model
            rationale: "Stable status payload for UI updates."
            violations: "Ad-hoc status dicts hinder UI consumers."
    """

    phase: str
    progress: float
    metrics: dict[str, Any]
    ABORTED: ClassVar[str] = "aborted"


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

    @notice: |
        Coordinates discovery, research, prototype, evolution, and submission phases.

    @dev: |
        Uses pydantic-graph nodes to drive mission lifecycle.

    @pattern:
        name: orchestrator
        rationale: "Centralized coordinator for multi-agent mission state."
        violations: "Decentralized orchestration leads to inconsistent transitions."

    @collaborators:
        required:
            - agent_k.mission.nodes:DiscoveryNode
            - agent_k.mission.nodes:ResearchNode
            - agent_k.mission.nodes:PrototypeNode
            - agent_k.mission.nodes:EvolutionNode
            - agent_k.mission.nodes:SubmissionNode
        optional:
            - agent_k.adapters.kaggle:KaggleAdapter
            - agent_k.adapters.openevolve:OpenEvolveAdapter
        injection: constructor
        lifecycle: "Long-lived during mission runs."

    @concurrency:
        model: asyncio
        safe: false
        reason: "Mutates orchestration state and shared resources."

    @invariants:
        - "self._graph is initialized before orchestration runs."
        - "self._state is None when idle."
    """

    _default_model: ClassVar[str] = DEFAULT_MODEL
    _max_evolution_rounds: ClassVar[int] = 100
    _supported_competition_types: ClassVar[frozenset[str]] = frozenset({"featured", "research", "playground"})

    __slots__ = (
        "_state",
        "_agents",
        "_graph",
        "_config",
        "_logger",
        "_event_emitter",
        "_http_client",
        "_platform_adapter",
        "_owns_http_client",
        "_owns_platform_adapter",
        "_paused",
        "_entered",
        "_resources_ready",
    )

    def __init__(
        self,
        *,
        config: Annotated[LycurgusSettings | None, Doc("Optional orchestrator settings.")] = None,
        model: Annotated[str | None, Doc("Override default model spec for agents.")] = None,
        event_emitter: Annotated[EventEmitter | None, Doc("Event emitter for UI streaming.")] = None,
        http_client: Annotated[httpx.AsyncClient | None, Doc("Shared HTTP client for research tools.")] = None,
        platform_adapter: Annotated[PlatformAdapter | None, Doc("Adapter for platform operations.")] = None,
    ) -> None:
        """Initialize the LYCURGUS orchestrator.

        @notice: |
            Initializes orchestration state and prepares agents/graph.

        @dev: |
            Lazily initializes adapters if not supplied.

        @state-changes:
            - self._agents
            - self._graph
            - self._state
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(state={self._state!r}, agents={list(self._agents.keys())!r})"

    def __str__(self) -> str:
        status = "active" if self._state else "idle"
        return f"LYCURGUS Orchestrator ({status})"

    async def __aenter__(self) -> LycurgusOrchestrator:
        """Async context manager entry for resource management."""
        await self._initialize_resources()
        self._entered = True
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Async context manager exit for cleanup."""
        self._entered = False
        await self._cleanup_resources()

    @classmethod
    def from_config_file(cls, path: Annotated[Path, Doc("Path to JSON configuration file.")]) -> LycurgusOrchestrator:
        """Create orchestrator from configuration file.

        @notice: |
            Builds an orchestrator from a JSON config file.
        """
        config = LycurgusSettings.from_file(path)
        return cls(config=config)

    @classmethod
    def with_custom_agents(
        cls, agents: Annotated[dict[str, Agent[Any, Any]], Doc("Custom agent registry overrides.")]
    ) -> LycurgusOrchestrator:
        """Create orchestrator with custom agent implementations.

        @notice: |
            Injects custom agent instances for orchestration.
        """
        instance = cls()
        instance._agents.update(agents)
        return instance

    @staticmethod
    def validate_competition_id(competition_id: Annotated[str, Doc("Competition identifier to validate.")]) -> bool:
        """Validate Kaggle competition identifier format.

        @notice: |
            Returns true for lowercase alphanumeric slugs with dashes.
        """
        pattern = r"^[a-z0-9-]+$"
        return bool(re.match(pattern, competition_id))

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
            raise RuntimeError("Cannot reconfigure during active mission")
        self._config = value

    async def abort_mission(self, reason: str) -> None:
        """Abort the current mission.

        Args:
            reason: Reason for aborting the mission.

        Raises:
            RuntimeError: If no mission is active.
        """
        if not self.is_active:
            raise RuntimeError("No active mission to abort")

        with self._logger.span("abort_mission", reason=reason):
            await self._transition_to_aborted(reason)
            self._state = None

    async def execute_mission(
        self,
        competition_id: str | None,
        *,
        mission_id: str | None = None,
        criteria: MissionCriteria | None = None,
        event_emitter: EventEmitter | None = None,
        http_client: httpx.AsyncClient | None = None,
        platform_adapter: PlatformAdapter | None = None,
        persistence: MissionPersistence | None = None,
    ) -> MissionResult:
        """Execute a full competition mission.

        This method orchestrates the complete competition lifecycle:
        1. Discovery and validation via Lobbyist
        2. Research and analysis via Scientist
        3. Solution evolution via Evolver
        4. Submission to Kaggle

        Args:
            competition_id: Target competition identifier (optional for discovery).
            mission_id: Optional mission identifier (generated if omitted).
            criteria: Optional criteria constraining the mission.
            event_emitter: Event emitter for streaming events.
            http_client: Shared HTTP client for research tools.
            platform_adapter: Adapter for platform operations.
            persistence: Optional persistence store for mission snapshots.

        Returns:
            MissionResult containing outcomes and metrics.

        Raises:
            CompetitionNotFoundError: If competition doesn't exist.
            MissionExecutionError: If mission fails during execution.
        """
        with self._logger.span("execute_mission", competition_id=competition_id):
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

            mission_id = mission_id or str(uuid.uuid4())
            persistence = persistence or create_persistence(mission_id)
            if persistence.has_snapshots():
                self._logger.warning("mission_persistence_exists", mission_id=mission_id)
                return await self.resume_persisted_mission(
                    mission_id,
                    event_emitter=self._event_emitter,
                    http_client=self._http_client,
                    platform_adapter=self._platform_adapter,
                    persistence=persistence,
                )

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
                    agents=self._agents,
                )
                return await self._run_graph(context, persistence=persistence, resume=False)
            finally:
                if initialized_here and not self._entered:
                    await self._cleanup_resources()
                self._state = None

    async def resume_persisted_mission(
        self,
        mission_id: str,
        *,
        event_emitter: EventEmitter | None = None,
        http_client: httpx.AsyncClient | None = None,
        platform_adapter: PlatformAdapter | None = None,
        persistence: MissionPersistence | None = None,
    ) -> MissionResult:
        """Resume a persisted mission from snapshots."""
        if self.is_active:
            raise RuntimeError("Cannot resume while mission is active")

        with self._logger.span("resume_persisted_mission", mission_id=mission_id):
            if event_emitter is not None:
                self._event_emitter = event_emitter
            if http_client is not None:
                self._http_client = http_client
                self._owns_http_client = False
            if platform_adapter is not None:
                self._platform_adapter = platform_adapter
                self._owns_platform_adapter = False

            persistence = persistence or create_persistence(mission_id)
            if not persistence.has_snapshots():
                raise RuntimeError("No persisted mission state to resume")

            initialized_here = False
            if not self._resources_ready:
                await self._initialize_resources()
                initialized_here = True

            self._paused = False

            try:
                context = GraphContext(
                    event_emitter=self._event_emitter,
                    http_client=self._http_client,
                    platform_adapter=self._platform_adapter,
                    agents=self._agents,
                )
                return await self._run_graph(context, persistence=persistence, resume=True)
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
            raise RuntimeError("No active mission")

        state = self._state
        if state is None:
            raise RuntimeError("No active mission")
        progress = self._calculate_progress(state)
        metrics = {
            "phases_completed": list(state.phases_completed),
            "competitions_found": len(state.discovered_competitions),
            "current_phase": state.current_phase,
            "generations": (len(state.evolution_state.generation_history) if state.evolution_state else 0),
        }
        if state.evolution_state and state.evolution_state.failure_summary:
            metrics["failure_summary"] = dict(state.evolution_state.failure_summary)
        return MissionStatus(phase=state.current_phase, progress=progress, metrics=metrics)

    async def pause_mission(self) -> None:
        """Pause the current mission for later resumption."""
        if not self.is_active:
            raise RuntimeError("No active mission")

        state = self._state
        if state is None:
            raise RuntimeError("No active mission")

        if self._paused:
            return

        self._paused = True
        if self._event_emitter:
            await self._event_emitter.emit(
                "phase-error", {"phase": state.current_phase, "error": "mission_paused", "recoverable": True}
            )
        self._logger.info("mission_paused", mission_id=state.mission_id)

    async def resume_mission(self) -> None:
        """Resume a previously paused mission."""
        if not self.is_active:
            raise RuntimeError("No active mission")

        state = self._state
        if state is None:
            raise RuntimeError("No active mission")

        if not self._paused:
            return

        self._paused = False
        if self._event_emitter:
            await self._event_emitter.emit("recovery-attempt", {"phase": state.current_phase, "strategy": "resume"})
        self._logger.info("mission_resumed", mission_id=state.mission_id)

    def _initialize_agents(self) -> dict[str, Agent[Any, Any]]:
        """Initialize specialized agent singletons."""
        return {"lobbyist": lobbyist_agent, "scientist": scientist_agent, "evolver": evolver_agent}

    def _build_orchestration_graph(self) -> Graph[MissionState, GraphContext, MissionResult]:
        """Build the state machine graph for orchestration."""
        return Graph(
            nodes=(DiscoveryNode, ResearchNode, PrototypeNode, EvolutionNode, SubmissionNode), state_type=MissionState
        )

    async def _run_graph(
        self, context: GraphContext, *, persistence: MissionPersistence, resume: bool
    ) -> MissionResult:
        """Execute the orchestration graph to completion."""
        existing_result = await persistence.load_latest_result()
        if existing_result is not None:
            self._state = await persistence.load_latest_state()
            return existing_result

        if not resume:
            if self._state is None:
                raise RuntimeError("No mission state initialized")
            await self._graph.initialize(DiscoveryNode(), persistence, state=self._state)

        async with self._graph.iter_from_persistence(persistence, deps=context) as graph_run:
            self._state = graph_run.state
            async for _node in graph_run:
                pass

        result = graph_run.result
        assert result is not None, "GraphRun should have a result"
        self._state = result.state
        return result.output

    def _calculate_progress(self, state: MissionState) -> float:
        """Calculate mission progress from the current state."""
        phases = ("discovery", "research", "prototype", "evolution", "submission")
        completed = float(len(state.phases_completed))
        if state.current_phase in phases and state.current_phase not in state.phases_completed:
            completed += 0.5
        return round(min(completed / len(phases), 1.0), 3)

    def _create_platform_adapter(self) -> PlatformAdapter:
        """Create a platform adapter based on available credentials."""
        username = os.getenv("KAGGLE_USERNAME")
        api_key = os.getenv("KAGGLE_KEY")
        if username and api_key:
            return KaggleAdapter(KaggleSettings(username=username, api_key=api_key))
        return OpenEvolveAdapter()

    async def _maybe_enter(self, adapter: PlatformAdapter) -> None:
        """Enter adapter context or authenticate when required."""
        enter = getattr(adapter, "__aenter__", None)
        if callable(enter):
            result = enter()
            if inspect.isawaitable(result):
                await result
            return
        await adapter.authenticate()

    async def _maybe_exit(self, adapter: PlatformAdapter) -> None:
        """Exit adapter context manager when supported."""
        exit_fn = getattr(adapter, "__aexit__", None)
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
            self._http_client = build_research_http_client(
                timeout_seconds=self._config.http_timeout_seconds, max_connections=self._config.http_max_connections
            )
            self._logger.info(
                "research_http_client_created",
                timeout_seconds=self._config.http_timeout_seconds,
                max_connections=self._config.http_max_connections,
            )

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
                "phase-error", {"phase": self._state.current_phase, "error": reason, "recoverable": False}
            )
        self._logger.warning("mission_aborted", reason=reason)


async def orchestrate(
    orchestrator: Annotated[LycurgusOrchestrator, Doc("Orchestrator instance to run.")],
    competition_id: Annotated[str, Doc("Competition identifier (slug).")],
    criteria: Annotated[MissionCriteria | None, Doc("Optional mission criteria override.")] = None,
) -> MissionResult:
    """Convenience helper to execute a mission.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Executes a mission end-to-end using the orchestrator.

        @effects:
            io:
                - Kaggle API requests
    """
    return await orchestrator.execute_mission(competition_id, criteria=criteria)


def validate_mission_result(result: Annotated[MissionResult, Doc("Mission result payload.")]) -> MissionResult:
    """Validate mission result payload.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Passthrough validator for mission results.
    """
    return result
