"""Agent registry and exports.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from typing import TYPE_CHECKING, Any, Final

# Third-party (alphabetical)
if TYPE_CHECKING:
    from pydantic_ai import Agent

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "AGENT_REGISTRY",
    "get_agent",
    "register_agent",
    "evolver_agent",
    "lobbyist_agent",
    "scientist_agent",
    "LycurgusOrchestrator",
    "LycurgusSettings",
    "MissionStatus",
)

# =============================================================================
# Section 3: Constants
# =============================================================================
AGENT_REGISTRY: Final[dict[str, Agent[Any, Any]]] = {}


# =============================================================================
# Section 12: Functions
# =============================================================================
def register_agent(name: str, agent: Agent[Any, Any]) -> None:
    """Register an agent singleton by name."""
    if name in AGENT_REGISTRY:
        raise ValueError(f"Agent {name!r} already registered")
    AGENT_REGISTRY[name] = agent


def get_agent(name: str) -> Agent[Any, Any]:
    """Return a registered agent by name."""
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]


# =============================================================================
# Section 13: Agent Imports (registration side effects)
# =============================================================================
from agent_k.agents.evolver import evolver_agent  # noqa: E402
from agent_k.agents.lobbyist import lobbyist_agent  # noqa: E402
from agent_k.agents.lycurgus import LycurgusOrchestrator, LycurgusSettings, MissionStatus  # noqa: E402
from agent_k.agents.scientist import scientist_agent  # noqa: E402
