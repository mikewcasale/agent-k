"""Mission graph components.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from typing import TYPE_CHECKING

# Local imports (core first, then alphabetical)
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
    if name in {
        "DiscoveryNode",
        "ResearchNode",
        "PrototypeNode",
        "EvolutionNode",
        "SubmissionNode",
    }:
        from . import nodes

        return getattr(nodes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
