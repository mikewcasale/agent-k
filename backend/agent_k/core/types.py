"""Type aliases and type variables for AGENT-K.

@notice: |
    Type aliases and type variables for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.types
    provides:
        - agent_k.core.types
    pattern: type-definitions

@agent-guidance:
    do:
        - "Use agent_k.core.types as the canonical home for this capability."
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

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

AgentDepsT = TypeVar("AgentDepsT")
"""Type variable for agent dependencies."""

OutputT = TypeVar("OutputT")
"""Type variable for agent outputs."""

StateT = TypeVar("StateT")
"""Type variable for mission state."""

OutputT_co = TypeVar("OutputT_co", covariant=True)
"""Covariant type variable for outputs."""

InputT_contra = TypeVar("InputT_contra", contravariant=True)
"""Contravariant type variable for inputs."""

P = ParamSpec("P")
"""Parameter specification for decorator typing."""

R = TypeVar("R")
"""Type variable for decorator return values."""

__all__ = (
    "CompetitionId",
    "MissionId",
    "TaskId",
    "LeaderboardRank",
    "Score",
    "FitnessScore",
    "JsonDict",
    "MessageHistory",
    "ToolResult",
    "MetricDirection",
    "MissionPhase",
    "TaskStatus",
    "TaskPriority",
    "ToolType",
    "MemoryScope",
    "ErrorCategory",
    "RecoveryStrategy",
    "AgentDepsT",
    "OutputT",
    "StateT",
    "OutputT_co",
    "InputT_contra",
    "P",
    "R",
    "AsyncCallback",
    "EventCallback",
)

type CompetitionId = str
type MissionId = str
type TaskId = str
type LeaderboardRank = int
type Score = float
type FitnessScore = float

type JsonDict = dict[str, Any]
type MessageHistory = list["ModelMessage"]
type ToolResult = str | dict[str, Any]

type MetricDirection = Literal["maximize", "minimize"]
type MissionPhase = Literal["discovery", "research", "prototype", "evolution", "submission"]
type TaskStatus = Literal["pending", "in_progress", "completed", "failed", "blocked", "skipped"]
type TaskPriority = Literal["critical", "high", "medium", "low"]
type ToolType = Literal["web_search", "kaggle_mcp", "code_executor", "memory", "browser"]
type MemoryScope = Literal["session", "persistent", "global"]
type ErrorCategory = Literal["transient", "recoverable", "fatal"]
type RecoveryStrategy = Literal["retry", "fallback", "skip", "replan", "abort"]

type AsyncCallback = Callable[[str], Awaitable[None]]
type EventCallback = Callable[[str, JsonDict], Awaitable[None]]
