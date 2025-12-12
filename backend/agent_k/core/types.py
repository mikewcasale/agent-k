"""Type aliases and type variables for AGENT-K.

Type aliases provide semantic meaning to primitive types and enable
static analysis across the codebase.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

# =============================================================================
# Section 1: Module Exports
# =============================================================================
__all__ = [
    # Simple type aliases
    'CompetitionId',
    'MissionId',
    'TaskId',
    'LeaderboardRank',
    'Score',
    'FitnessScore',
    # Complex type aliases
    'JsonDict',
    'MessageHistory',
    'ToolResult',
    # Literal types
    'MetricDirection',
    'MissionPhase',
    'TaskStatus',
    'TaskPriority',
    'ToolType',
    'MemoryScope',
    'ErrorCategory',
    'RecoveryStrategy',
    # Type variables
    'AgentDepsT',
    'OutputT',
    'StateT',
    'P',
    'R',
    # Callbacks
    'AsyncCallback',
    'EventCallback',
]

# =============================================================================
# Section 2: Simple Type Aliases
# =============================================================================
CompetitionId: TypeAlias = str
MissionId: TypeAlias = str
TaskId: TypeAlias = str
LeaderboardRank: TypeAlias = int
Score: TypeAlias = float
FitnessScore: TypeAlias = float

# =============================================================================
# Section 3: Complex Type Aliases
# =============================================================================
JsonDict: TypeAlias = dict[str, Any]
MessageHistory: TypeAlias = 'list[ModelMessage]'
ToolResult: TypeAlias = str | dict[str, Any]

# =============================================================================
# Section 4: Literal Types (Constrained)
# =============================================================================
MetricDirection: TypeAlias = Literal['maximize', 'minimize']

MissionPhase: TypeAlias = Literal[
    'discovery',
    'research',
    'prototype',
    'evolution',
    'submission',
]

TaskStatus: TypeAlias = Literal[
    'pending',
    'in_progress',
    'completed',
    'failed',
    'blocked',
    'skipped',
]

TaskPriority: TypeAlias = Literal['critical', 'high', 'medium', 'low']

ToolType: TypeAlias = Literal[
    'web_search',
    'kaggle_mcp',
    'code_executor',
    'memory',
    'browser',
]

MemoryScope: TypeAlias = Literal['session', 'persistent', 'global']

ErrorCategory: TypeAlias = Literal['transient', 'recoverable', 'fatal']

RecoveryStrategy: TypeAlias = Literal['retry', 'fallback', 'skip', 'replan', 'abort']

# =============================================================================
# Section 5: Type Variables
# =============================================================================
AgentDepsT = TypeVar('AgentDepsT')
OutputT = TypeVar('OutputT')
StateT = TypeVar('StateT', bound='BaseState')
OutputT_co = TypeVar('OutputT_co', covariant=True)
InputT_contra = TypeVar('InputT_contra', contravariant=True)

# ParamSpec for decorator typing
P = ParamSpec('P')
R = TypeVar('R')

# =============================================================================
# Section 6: Callback Type Aliases
# =============================================================================
AsyncCallback: TypeAlias = Callable[[str], Awaitable[None]]
EventCallback: TypeAlias = Callable[[str, JsonDict], Awaitable[None]]
