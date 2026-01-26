"""Agent registry and exports.

@notice: |
    Agent registry and exports.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.agents
    provides:
        - agent_k.agents:AGENT_REGISTRY
        - agent_k.agents:register_agent
        - agent_k.agents:get_agent
    consumes:
        - agent_k.agents.evolver:EvolverAgent
        - agent_k.agents.lobbyist:LobbyistAgent
        - agent_k.agents.scientist:ScientistAgent
        - agent_k.agents.lycurgus:LycurgusOrchestrator
    pattern: agent-registry

@similar:
    - id: agent_k.mission.nodes
        when: "Mission nodes orchestrate phases; this module only registers agents."

@agent-guidance:
    do:
        - "Use agent_k.agents as the canonical home for this capability."
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

from typing import TYPE_CHECKING, Annotated, Any, Final

from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from agent_k.agents.evolver import EvolverAgent as EvolverAgent, evolver_agent as evolver_agent
    from agent_k.agents.lobbyist import LobbyistAgent as LobbyistAgent, lobbyist_agent as lobbyist_agent
    from agent_k.agents.lycurgus import (
        LycurgusOrchestrator as LycurgusOrchestrator,
        LycurgusSettings as LycurgusSettings,
        MissionStatus as MissionStatus,
    )
    from agent_k.agents.scientist import ScientistAgent as ScientistAgent, scientist_agent as scientist_agent

__all__ = (
    "AGENT_REGISTRY",
    "get_agent",
    "register_agent",
    "evolver_agent",
    "lobbyist_agent",
    "scientist_agent",
    "EvolverAgent",
    "LobbyistAgent",
    "ScientistAgent",
    "LycurgusOrchestrator",
    "LycurgusSettings",
    "MissionStatus",
)

AGENT_REGISTRY: Final[dict[str, Agent[Any, Any]]] = {}


def register_agent(
    name: Annotated[str, Doc("Registry key for the agent singleton.")],
    agent: Annotated[Agent[Any, Any], Doc("Agent instance to register.")],
) -> None:
    """Register an agent singleton by name.

    @notice: |
        Adds a singleton agent instance to the registry.

    @dev: |
        Enforces uniqueness by name; raises if already registered.

    @effects:
        state:
            - AGENT_REGISTRY

    @errors:
        terminal:
            - ValueError
    """
    if name in AGENT_REGISTRY:
        raise ValueError(f"Agent {name!r} already registered")
    AGENT_REGISTRY[name] = agent


def get_agent(name: Annotated[str, Doc("Registry key for the agent singleton.")]) -> Agent[Any, Any]:
    """Return a registered agent by name.

    @notice: |
        Retrieves a registered agent singleton.

    @dev: |
        Raises if the agent name is unknown to avoid silent misconfiguration.

    @errors:
        terminal:
            - KeyError
    """
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]


def __getattr__(name: str) -> Any:
    """Lazy import agents to avoid requiring API keys at import time."""
    if name == "evolver_agent":
        from agent_k.agents.evolver import evolver_agent

        return evolver_agent
    if name == "EvolverAgent":
        from agent_k.agents.evolver import EvolverAgent

        return EvolverAgent
    if name == "lobbyist_agent":
        from agent_k.agents.lobbyist import lobbyist_agent

        return lobbyist_agent
    if name == "LobbyistAgent":
        from agent_k.agents.lobbyist import LobbyistAgent

        return LobbyistAgent
    if name == "scientist_agent":
        from agent_k.agents.scientist import scientist_agent

        return scientist_agent
    if name == "ScientistAgent":
        from agent_k.agents.scientist import ScientistAgent

        return ScientistAgent
    if name == "LycurgusOrchestrator":
        from agent_k.agents.lycurgus import LycurgusOrchestrator

        return LycurgusOrchestrator
    if name == "LycurgusSettings":
        from agent_k.agents.lycurgus import LycurgusSettings

        return LycurgusSettings
    if name == "MissionStatus":
        from agent_k.agents.lycurgus import MissionStatus

        return MissionStatus
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
