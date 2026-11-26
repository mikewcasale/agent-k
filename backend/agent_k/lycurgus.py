"""Public Pydantic Graph assembly for the Lycurgus mission orchestrator.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pydantic_graph import Graph

from .mission import (
    DiscoveryNode,
    EvolutionNode,
    MissionContext,
    MissionState,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
)
from .stores import MissionRunResult

__all__ = ("mission_graph",)

mission_graph = Graph[MissionState, MissionContext, MissionRunResult](
    nodes=[DiscoveryNode, ResearchNode, PrototypeNode, EvolutionNode, SubmissionNode],
    state_type=MissionState,
    run_end_type=MissionRunResult,
    name="lycurgus",
)
