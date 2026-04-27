"""Shared dependency containers.

@notice: |
    Shared dependency containers.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.deps
    provides:
        - agent_k.core.deps:BaseDeps
        - agent_k.core.deps:KaggleDeps
    pattern: dependency-container

@agent-guidance:
    do:
        - "Use agent_k.core.deps as the canonical home for this capability."
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from agent_k.adapters.kaggle import KaggleAdapter
    from agent_k.ui.agui import EventEmitter

__all__ = ("BaseDeps", "KaggleDeps")


@dataclass(kw_only=True)
class BaseDeps:
    """Base dependencies shared across agents.

    @notice: |
        Base dependencies shared across agents.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: dependency-container
            rationale: "Shared DI container for agent executions."
            violations: "Hidden globals make runs nondeterministic."

        @collaborators:
            required:
                - agent_k.ui.agui:EventEmitter
            optional:
                - httpx:AsyncClient
            injection: constructor
            lifecycle: "Allocated per agent run."
    """

    event_emitter: EventEmitter
    http_client: httpx.AsyncClient | None = None
    correlation_id: str | None = None


@dataclass(kw_only=True)
class KaggleDeps(BaseDeps):
    """Dependencies for Kaggle toolsets.

    @notice: |
        Dependencies for Kaggle toolsets.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: dependency-container
            rationale: "Groups Kaggle adapter and toolset settings."
            violations: "Direct adapter access bypasses shared settings."

        @collaborators:
            required:
                - agent_k.adapters.kaggle:KaggleAdapter
                - agent_k.ui.agui:EventEmitter
            optional:
                - httpx:AsyncClient
            injection: constructor
            lifecycle: "Allocated per agent run."
    """

    kaggle_adapter: KaggleAdapter
    max_results: int = 50
    search_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def platform_adapter(self) -> KaggleAdapter:
        """Expose the Kaggle adapter as a platform adapter."""
        return self.kaggle_adapter
