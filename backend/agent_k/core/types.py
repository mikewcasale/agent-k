"""Type aliases and type variables for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

# Third-party (alphabetical)
from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

# =============================================================================
# Section 2: Module Type Variables
# =============================================================================
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

# =============================================================================
# Section 3: Module Exports
# =============================================================================
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

# =============================================================================
# Section 4: Type Aliases
# =============================================================================
type CompetitionId = str
type MissionId = str
type TaskId = str
type LeaderboardRank = int
type Score = float
type FitnessScore = float

JsonDict = TypeAliasType("JsonDict", dict[str, Any])
MessageHistory = TypeAliasType("MessageHistory", list["ModelMessage"])
ToolResult = TypeAliasType("ToolResult", str | dict[str, Any])

MetricDirection = TypeAliasType("MetricDirection", Literal["maximize", "minimize"])
MissionPhase = TypeAliasType(
    "MissionPhase",
    Literal[
        "discovery",
        "research",
        "prototype",
        "evolution",
        "submission",
    ],
)
TaskStatus = TypeAliasType(
    "TaskStatus",
    Literal[
        "pending",
        "in_progress",
        "completed",
        "failed",
        "blocked",
        "skipped",
    ],
)
TaskPriority = TypeAliasType(
    "TaskPriority",
    Literal["critical", "high", "medium", "low"],
)
ToolType = TypeAliasType(
    "ToolType",
    Literal[
        "web_search",
        "kaggle_mcp",
        "code_executor",
        "memory",
        "browser",
    ],
)
MemoryScope = TypeAliasType(
    "MemoryScope",
    Literal["session", "persistent", "global"],
)
ErrorCategory = TypeAliasType(
    "ErrorCategory",
    Literal["transient", "recoverable", "fatal"],
)
RecoveryStrategy = TypeAliasType(
    "RecoveryStrategy",
    Literal["retry", "fallback", "skip", "replan", "abort"],
)

AsyncCallback = TypeAliasType(
    "AsyncCallback",
    Callable[[str], Awaitable[None]],
)
EventCallback = TypeAliasType(
    "EventCallback",
    Callable[[str, JsonDict], Awaitable[None]],
)
