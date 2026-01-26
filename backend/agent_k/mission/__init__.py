"""Mission graph components.

@notice: |
    Mission graph components.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.mission
    provides:
        - agent_k.mission
    pattern: mission-package

@agent-guidance:
    do:
        - "Use agent_k.mission as the canonical home for this capability."
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

from typing import TYPE_CHECKING

from .state import GraphContext, MissionResult, MissionState

if TYPE_CHECKING:
    from .nodes import DiscoveryNode, EvolutionNode, PrototypeNode, ResearchNode, SubmissionNode

__all__ = (
    "MissionResult",
    "MissionState",
    "GraphContext",
    "DiscoveryNode",
    "ResearchNode",
    "PrototypeNode",
    "EvolutionNode",
    "SubmissionNode",
)


def __getattr__(name: str) -> object:
    """Lazy-load graph nodes to avoid circular imports."""
    if name in {"DiscoveryNode", "ResearchNode", "PrototypeNode", "EvolutionNode", "SubmissionNode"}:
        from . import nodes

        return getattr(nodes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
