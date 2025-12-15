"""Orchestration agent coordinating the multi-agent Kaggle competition system."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import logfire
from pydantic_ai import Agent
from pydantic_graph import Graph

from ...core.exceptions import CompetitionNotFoundError
from ...core.models import MissionCriteria, MissionResult, MissionState
from ..evolver import EvolverAgent
from ..lobbyist import LobbyistAgent
from ..scientist import ScientistAgent
from ...graph.nodes import DiscoveryNode, EvolutionNode, PrototypeNode, ResearchNode, SubmissionNode

__all__ = ['LycurgusOrchestrator', 'OrchestratorConfig', 'MissionStatus']


@dataclass
class OrchestratorConfig:
    """Configuration for LYCURGUS orchestrator.
    
    The default_model supports multiple formats:
        - Standard pydantic-ai strings: 'anthropic:claude-sonnet-4-5'
        - Devstral local: 'devstral:local'
        - Devstral with custom URL: 'devstral:http://localhost:1234/v1'
    """
    
    default_model: str = 'anthropic:claude-sonnet-4-5'
    max_evolution_rounds: int = 100
    
    @classmethod
    def from_file(cls, path: Path) -> OrchestratorConfig:
        """Create orchestrator config from JSON/YAML file."""
        data = json.loads(path.read_text())
        return cls(
            default_model=data.get('default_model', cls.default_model),
            max_evolution_rounds=data.get('max_evolution_rounds', cls.max_evolution_rounds),
        )
    
    @classmethod
    def with_devstral(cls, base_url: str | None = None) -> OrchestratorConfig:
        """Create orchestrator config using Devstral model.
        
        Args:
            base_url: Optional custom base URL for LM Studio server.
                     If None, uses default: http://192.168.105.1:1234/v1
        
        Returns:
            OrchestratorConfig configured for Devstral.
        """
        if base_url:
            model = f'devstral:{base_url}'
        else:
            model = 'devstral:local'
        return cls(default_model=model)


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
    # Section 1: Class Variables (ClassVar annotations)
    # =========================================================================
    _default_model: ClassVar[str] = 'anthropic:claude-sonnet-4-5'
    _max_evolution_rounds: ClassVar[int] = 100
    _supported_competition_types: ClassVar[frozenset[str]] = frozenset({
        'featured', 'research', 'playground',
    })
    
    # =========================================================================
    # Section 2: Instance Variable Annotations (slots if applicable)
    # =========================================================================
    __slots__ = ('_state', '_agents', '_graph', '_config', '_logger')
    
    # =========================================================================
    # Section 3: __init__ and __new__
    # =========================================================================
    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the LYCURGUS orchestrator.
        
        Args:
            config: Configuration for orchestration behavior.
            model: Override default model for all agents.
        """
        self._config = config or OrchestratorConfig()
        self._logger = logfire  # Use logfire directly, service name can be set in spans

        model = model or self._default_model
        self._agents = self._initialize_agents(model)
        self._graph = self._build_orchestration_graph()
        self._state: MissionState | None = None
    
    # =========================================================================
    # Section 4: Other Dunder Methods (alphabetical)
    # =========================================================================
    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
            f'state={self._state!r}, '
            f'agents={list(self._agents.keys())!r})'
        )
    
    def __str__(self) -> str:
        status = 'active' if self._state else 'idle'
        return f'LYCURGUS Orchestrator ({status})'
    
    async def __aenter__(self) -> LycurgusOrchestrator:
        """Async context manager entry for resource management."""
        await self._initialize_resources()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit for cleanup."""
        await self._cleanup_resources()
    
    # =========================================================================
    # Section 5: Class Methods (constructors and factories)
    # =========================================================================
    @classmethod
    def from_config_file(cls, path: Path) -> LycurgusOrchestrator:
        """Create orchestrator from configuration file.
        
        Args:
            path: Path to YAML or JSON configuration file.
        
        Returns:
            Configured LycurgusOrchestrator instance.
        """
        config = OrchestratorConfig.from_file(path)
        return cls(config=config)
    
    @classmethod
    def with_custom_agents(
        cls,
        agents: dict[str, Agent[Any, Any]],
    ) -> LycurgusOrchestrator:
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
    # Section 6: Static Methods
    # =========================================================================
    @staticmethod
    def validate_competition_id(competition_id: str) -> bool:
        """Validate Kaggle competition identifier format.
        
        Args:
            competition_id: Competition identifier to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        import re
        pattern = r'^[a-z0-9-]+$'
        return bool(re.match(pattern, competition_id))
    
    # =========================================================================
    # Section 7: Properties (read-only first, then read-write)
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
        if self._state is None:
            return None
        return self._state.current_phase
    
    @property
    def config(self) -> OrchestratorConfig:
        """Orchestrator configuration (read-write)."""
        return self._config
    
    @config.setter
    def config(self, value: OrchestratorConfig) -> None:
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
    # Section 8: Public Instance Methods (alphabetical)
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
        competition_id: str,
        *,
        criteria: MissionCriteria | None = None,
    ) -> MissionResult:
        """Execute a full competition mission.
        
        This method orchestrates the complete competition lifecycle:
        1. Discovery and validation via Lobbyist
        2. Research and analysis via Scientist
        3. Solution evolution via Evolver
        4. Submission to Kaggle
        
        Args:
            competition_id: Target Kaggle competition identifier.
            criteria: Optional criteria constraining the mission.
        
        Returns:
            MissionResult containing outcomes and metrics.
        
        Raises:
            CompetitionNotFoundError: If competition doesn't exist.
            MissionExecutionError: If mission fails during execution.
        """
        with self._logger.span('execute_mission', competition_id=competition_id):
            self._state = MissionState(
                competition_id=competition_id,
                criteria=criteria or MissionCriteria(),
            )
            
            try:
                result = await self._run_graph()
                return result
            finally:
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
        
        return MissionStatus(
            phase=self._state.current_phase,
            progress=self._state.progress,
            metrics=self._state.metrics,
        )
    
    async def pause_mission(self) -> None:
        """Pause the current mission for later resumption."""
        ...
    
    async def resume_mission(self) -> None:
        """Resume a previously paused mission."""
        ...
    
    # =========================================================================
    # Section 9: Protected Methods (for subclass use)
    # =========================================================================
    def _initialize_agents(self, model: str) -> dict[str, Agent[Any, Any]]:
        """Initialize specialized agents.
        
        Args:
            model: Model identifier for agents.
        
        Returns:
            Dictionary of initialized agents.
        """
        return {
            'lobbyist': LobbyistAgent(model=model),
            'scientist': ScientistAgent(model=model),
            'evolver': EvolverAgent(model=model),
        }
    
    def _build_orchestration_graph(self) -> Graph[MissionState, MissionResult]:
        """Build the state machine graph for orchestration."""
        return Graph(
            nodes=(
                DiscoveryNode,
                ResearchNode,
                PrototypeNode,
                EvolutionNode,
                SubmissionNode,
            ),
            state_type=MissionState,
        )
    
    async def _run_graph(self) -> MissionResult:
        """Execute the orchestration graph to completion."""
        node = DiscoveryNode()
        return await self._graph.run(node, state=self._state)
    
    # =========================================================================
    # Section 10: Private Methods (internal implementation)
    # =========================================================================
    async def _initialize_resources(self) -> None:
        """Initialize async resources."""
        for agent in self._agents.values():
            await agent.__aenter__()
    
    async def _cleanup_resources(self) -> None:
        """Clean up async resources."""
        for agent in self._agents.values():
            await agent.__aexit__(None, None, None)
    
    async def _transition_to_aborted(self, reason: str) -> None:
        """Handle transition to aborted state."""
        self._state.status = MissionStatus.ABORTED  # type: ignore[attr-defined]
        self._state.abort_reason = reason  # type: ignore[attr-defined]
