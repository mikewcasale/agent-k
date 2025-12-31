"""Exception hierarchy for AGENT-K system.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ErrorCategory, RecoveryStrategy

__all__ = (
    'AgentKError',
    'AgentError',
    'AgentExecutionError',
    'ToolExecutionError',
    'OutputValidationError',
    'AdapterError',
    'PlatformConnectionError',
    'AuthenticationError',
    'RateLimitError',
    'CompetitionError',
    'CompetitionNotFoundError',
    'SubmissionError',
    'DeadlinePassedError',
    'EvolutionError',
    'ConvergenceError',
    'PopulationExtinctError',
    'FitnessEvaluationError',
    'MemoryError',
    'CheckpointError',
    'MemoryCapacityError',
    'GraphError',
    'StateTransitionError',
    'PhaseTimeoutError',
    'classify_error',
)


class AgentKError(Exception):
    """Base exception for all AGENT-K errors.

    All exceptions in the system inherit from this class, enabling
    catch-all handling at application boundaries.

    Attributes:
        context: Additional context for debugging.
        recoverable: Whether the error can potentially be recovered.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None, recoverable: bool = True) -> None:
        self.context = context or {}
        self.recoverable = recoverable
        super().__init__(message)


# =============================================================================
# Agent Exceptions
# =============================================================================
class AgentError(AgentKError):
    """Base exception for agent-related errors."""


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""

    def __init__(
        self, agent_name: str, message: str, *, cause: Exception | None = None, context: dict[str, Any] | None = None
    ) -> None:
        self.agent_name = agent_name
        self.cause = cause
        ctx = context or {}
        ctx['agent_name'] = agent_name
        if cause:
            ctx['cause_type'] = type(cause).__name__
        super().__init__(f'[{agent_name}] {message}', context=ctx)


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str, *, args: dict[str, Any] | None = None) -> None:
        self.tool_name = tool_name
        self.tool_args = args or {}
        super().__init__(
            f'Tool {tool_name} failed: {message}', context={'tool_name': tool_name, 'args': self.tool_args}
        )


class OutputValidationError(AgentError):
    """Raised when agent output fails validation."""

    def __init__(self, agent_name: str, validation_errors: list[str]) -> None:
        self.agent_name = agent_name
        self.validation_errors = validation_errors
        super().__init__(
            f'[{agent_name}] Output validation failed: {validation_errors}',
            context={'validation_errors': validation_errors},
        )


# =============================================================================
# Adapter Exceptions
# =============================================================================
class AdapterError(AgentKError):
    """Base exception for adapter-related errors."""


class PlatformConnectionError(AdapterError):
    """Raised when connection to platform fails."""

    def __init__(self, platform: str, message: str) -> None:
        self.platform = platform
        super().__init__(f'[{platform}] Connection failed: {message}', context={'platform': platform})


class AuthenticationError(AdapterError):
    """Raised when platform authentication fails."""

    def __init__(self, platform: str, message: str = 'Authentication failed') -> None:
        self.platform = platform
        super().__init__(f'[{platform}] {message}', context={'platform': platform}, recoverable=False)


class RateLimitError(AdapterError):
    """Raised when platform rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retry.
    """

    def __init__(self, platform: str, message: str, *, retry_after: int | None = None) -> None:
        self.platform = platform
        self.retry_after = retry_after
        super().__init__(f'[{platform}] {message}', context={'platform': platform, 'retry_after': retry_after})


# =============================================================================
# Competition Exceptions
# =============================================================================
class CompetitionError(AgentKError):
    """Base exception for competition-related errors."""


class CompetitionNotFoundError(CompetitionError):
    """Raised when competition does not exist."""

    def __init__(self, competition_id: str) -> None:
        self.competition_id = competition_id
        super().__init__(
            f'Competition not found: {competition_id}', context={'competition_id': competition_id}, recoverable=False
        )


class SubmissionError(CompetitionError):
    """Raised when submission fails."""

    def __init__(self, competition_id: str, message: str, *, submission_id: str | None = None) -> None:
        self.competition_id = competition_id
        self.submission_id = submission_id
        super().__init__(
            f'Submission to {competition_id} failed: {message}',
            context={'competition_id': competition_id, 'submission_id': submission_id},
        )


class DeadlinePassedError(CompetitionError):
    """Raised when competition deadline has passed."""

    def __init__(self, competition_id: str, deadline: str) -> None:
        self.competition_id = competition_id
        self.deadline = deadline
        super().__init__(
            f'Competition {competition_id} deadline passed: {deadline}',
            context={'competition_id': competition_id, 'deadline': deadline},
            recoverable=False,
        )


# =============================================================================
# Evolution Exceptions
# =============================================================================
class EvolutionError(AgentKError):
    """Base exception for evolution-related errors."""


class ConvergenceError(EvolutionError):
    """Raised when evolution fails to converge within limits."""

    def __init__(self, generations_completed: int, best_fitness: float, reason: str) -> None:
        self.generations_completed = generations_completed
        self.best_fitness = best_fitness
        self.reason = reason
        super().__init__(
            f'Evolution did not converge after {generations_completed} generations: {reason}',
            context={'generations_completed': generations_completed, 'best_fitness': best_fitness, 'reason': reason},
        )


class PopulationExtinctError(EvolutionError):
    """Raised when all population members fail fitness evaluation."""

    def __init__(self, generation: int, last_error: str) -> None:
        self.generation = generation
        self.last_error = last_error
        super().__init__(
            f'Population extinct at generation {generation}: {last_error}',
            context={'generation': generation, 'last_error': last_error},
            recoverable=False,
        )


class FitnessEvaluationError(EvolutionError):
    """Raised when fitness evaluation fails."""

    def __init__(self, solution_id: str, message: str, *, execution_error: str | None = None) -> None:
        self.solution_id = solution_id
        self.execution_error = execution_error
        super().__init__(
            f'Fitness evaluation failed for {solution_id}: {message}',
            context={'solution_id': solution_id, 'execution_error': execution_error},
        )


# =============================================================================
# Memory Exceptions
# =============================================================================
class MemoryError(AgentKError):
    """Base exception for memory-related errors."""


class CheckpointError(MemoryError):
    """Raised when checkpoint operations fail."""

    def __init__(self, checkpoint_name: str, operation: str, message: str) -> None:
        self.checkpoint_name = checkpoint_name
        self.operation = operation
        super().__init__(
            f'Checkpoint {operation} failed for {checkpoint_name}: {message}',
            context={'checkpoint_name': checkpoint_name, 'operation': operation},
        )


class MemoryCapacityError(MemoryError):
    """Raised when memory capacity is exceeded."""

    def __init__(self, current_size: int, max_size: int) -> None:
        self.current_size = current_size
        self.max_size = max_size
        super().__init__(
            f'Memory capacity exceeded: {current_size} / {max_size} bytes',
            context={'current_size': current_size, 'max_size': max_size},
        )


# =============================================================================
# Graph Exceptions
# =============================================================================
class GraphError(AgentKError):
    """Base exception for graph-related errors."""


class StateTransitionError(GraphError):
    """Raised when state transition is invalid."""

    def __init__(self, from_state: str, to_state: str, reason: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f'Invalid transition from {from_state} to {to_state}: {reason}',
            context={'from_state': from_state, 'to_state': to_state, 'reason': reason},
        )


class PhaseTimeoutError(GraphError):
    """Raised when a phase exceeds its timeout."""

    def __init__(self, phase: str, timeout_seconds: int, elapsed_seconds: float) -> None:
        self.phase = phase
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f'Phase {phase} timed out after {elapsed_seconds:.1f}s (limit: {timeout_seconds}s)',
            context={'phase': phase, 'timeout_seconds': timeout_seconds, 'elapsed_seconds': elapsed_seconds},
        )


def classify_error(exc: Exception) -> tuple[ErrorCategory, RecoveryStrategy]:
    """Classify errors into recovery categories and strategies."""
    if isinstance(exc, RateLimitError):
        return 'recoverable', 'retry'
    if isinstance(exc, AuthenticationError):
        return 'fatal', 'abort'
    if isinstance(exc, CompetitionNotFoundError):
        return 'fatal', 'abort'
    if isinstance(exc, AgentKError):
        return ('recoverable', 'retry') if exc.recoverable else ('fatal', 'abort')
    return 'transient', 'retry'
