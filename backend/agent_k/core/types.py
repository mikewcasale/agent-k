"""Type aliases and type variables for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from collections.abc import Awaitable, Callable

# Third-party (alphabetical)
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeAlias, TypeVar

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

CompetitionId: TypeAlias = str
MissionId: TypeAlias = str
TaskId: TypeAlias = str
LeaderboardRank: TypeAlias = int
Score: TypeAlias = float
FitnessScore: TypeAlias = float

JsonDict: TypeAlias = "dict[str, Any]"
MessageHistory: TypeAlias = 'list["ModelMessage"]'
ToolResult: TypeAlias = "str | dict[str, Any]"

MetricDirection: TypeAlias = 'Literal["maximize", "minimize"]'
MissionPhase: TypeAlias = """Literal[
    "discovery",
    "research",
    "prototype",
    "evolution",
    "submission",
]"""
TaskStatus: TypeAlias = """Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "blocked",
    "skipped",
]"""
TaskPriority: TypeAlias = 'Literal["critical", "high", "medium", "low"]'
ToolType: TypeAlias = """Literal[
    "web_search",
    "kaggle_mcp",
    "code_executor",
    "memory",
    "browser",
]"""
MemoryScope: TypeAlias = 'Literal["session", "persistent", "global"]'
ErrorCategory: TypeAlias = 'Literal["transient", "recoverable", "fatal"]'
RecoveryStrategy: TypeAlias = 'Literal["retry", "fallback", "skip", "replan", "abort"]'

AsyncCallback: TypeAlias = "Callable[[str], Awaitable[None]]"
EventCallback: TypeAlias = "Callable[[str, JsonDict], Awaitable[None]]"
