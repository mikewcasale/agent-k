"""Exception hierarchy for AGENT-K system.

@notice: |
    Exception hierarchy for AGENT-K system.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.exceptions
    provides:
        - agent_k.core.exceptions
    pattern: exception-hierarchy

@agent-guidance:
    do:
        - "Use agent_k.core.exceptions as the canonical home for this capability."
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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ErrorCategory, RecoveryStrategy

__all__ = (
    "AgentKError",
    "AgentError",
    "AgentExecutionError",
    "ToolExecutionError",
    "OutputValidationError",
    "AdapterError",
    "PlatformConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "CompetitionError",
    "CompetitionNotFoundError",
    "CompetitionRulesNotAcceptedError",
    "SubmissionError",
    "DeadlinePassedError",
    "EvolutionError",
    "ConvergenceError",
    "PopulationExtinctError",
    "FitnessEvaluationError",
    "MemoryError",
    "CheckpointError",
    "MemoryCapacityError",
    "GraphError",
    "StateTransitionError",
    "PhaseTimeoutError",
    "classify_error",
)


class AgentKError(Exception):
    """Base exception for all AGENT-K errors.

        All exceptions in the system inherit from this class, enabling
        catch-all handling at application boundaries.

    @notice: |
        Base exception for all AGENT-K errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-base
            rationale: "Root exception with context and recoverability metadata."

    Attributes:
            context: Additional context for debugging.
            recoverable: Whether the error can potentially be recovered.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None, recoverable: bool = True) -> None:
        self.context = context or {}
        self.recoverable = recoverable
        super().__init__(message)


class AgentError(AgentKError):
    """Base exception for agent-related errors.

    @notice: |
        Base exception for agent-related errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-category
            rationale: "Category base for agent execution errors."
    """


class AgentExecutionError(AgentError):
    """Raised when agent execution fails.

    @notice: |
        Raised when agent execution fails.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures agent name and cause for debugging."
    """

    def __init__(
        self, agent_name: str, message: str, *, cause: Exception | None = None, context: dict[str, Any] | None = None
    ) -> None:
        self.agent_name = agent_name
        self.cause = cause
        ctx = context or {}
        ctx["agent_name"] = agent_name
        if cause:
            ctx["cause_type"] = type(cause).__name__
        super().__init__(f"[{agent_name}] {message}", context=ctx)


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails.

    @notice: |
        Raised when a tool execution fails.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures tool name and arguments for debugging."
    """

    def __init__(self, tool_name: str, message: str, *, args: dict[str, Any] | None = None) -> None:
        self.tool_name = tool_name
        self.tool_args = args or {}
        super().__init__(
            f"Tool {tool_name} failed: {message}", context={"tool_name": tool_name, "args": self.tool_args}
        )


class OutputValidationError(AgentError):
    """Raised when agent output fails validation.

    @notice: |
        Raised when agent output fails validation.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures validation errors for structured output failures."
    """

    def __init__(self, agent_name: str, validation_errors: list[str]) -> None:
        self.agent_name = agent_name
        self.validation_errors = validation_errors
        super().__init__(
            f"[{agent_name}] Output validation failed: {validation_errors}",
            context={"validation_errors": validation_errors},
        )


class AdapterError(AgentKError):
    """Base exception for adapter-related errors.

    @notice: |
        Base exception for adapter-related errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-category
            rationale: "Category base for platform adapter errors."
    """


class PlatformConnectionError(AdapterError):
    """Raised when connection to platform fails.

    @notice: |
        Raised when connection to platform fails.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures platform name for connection failures."
    """

    def __init__(self, platform: str, message: str) -> None:
        self.platform = platform
        super().__init__(f"[{platform}] Connection failed: {message}", context={"platform": platform})


class AuthenticationError(AdapterError):
    """Raised when platform authentication fails.

    @notice: |
        Raised when platform authentication fails.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Non-recoverable authentication failure."
    """

    def __init__(self, platform: str, message: str = "Authentication failed") -> None:
        self.platform = platform
        super().__init__(f"[{platform}] {message}", context={"platform": platform}, recoverable=False)


class RateLimitError(AdapterError):
    """Raised when platform rate limit is exceeded.

    @notice: |
        Raised when platform rate limit is exceeded.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Recoverable with retry_after hint for backoff."

    Attributes:
            retry_after: Seconds to wait before retry.
    """

    def __init__(self, platform: str, message: str, *, retry_after: int | None = None) -> None:
        self.platform = platform
        self.retry_after = retry_after
        super().__init__(f"[{platform}] {message}", context={"platform": platform, "retry_after": retry_after})


class CompetitionError(AgentKError):
    """Base exception for competition-related errors.

    @notice: |
        Base exception for competition-related errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-category
            rationale: "Category base for Kaggle competition errors."
    """


class CompetitionNotFoundError(CompetitionError):
    """Raised when competition does not exist.

    @notice: |
        Raised when competition does not exist.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Non-recoverable missing competition."
    """

    def __init__(self, competition_id: str) -> None:
        self.competition_id = competition_id
        super().__init__(
            f"Competition not found: {competition_id}", context={"competition_id": competition_id}, recoverable=False
        )


class CompetitionRulesNotAcceptedError(CompetitionError):
    """Raised when competition rules have not been accepted.

    @notice: |
        Raised when competition rules have not been accepted.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Non-recoverable until user accepts rules manually."
    """

    def __init__(self, competition_id: str) -> None:
        self.competition_id = competition_id
        rules_url = f"https://www.kaggle.com/competitions/{competition_id}/rules"
        message = (
            f"Competition rules not accepted for {competition_id}. "
            f"Open {rules_url} and accept the rules before downloading data."
        )
        super().__init__(message, context={"competition_id": competition_id, "rules_url": rules_url}, recoverable=False)


class SubmissionError(CompetitionError):
    """Raised when submission fails.

    @notice: |
        Raised when submission fails.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures submission context for retry logic."
    """

    def __init__(self, competition_id: str, message: str, *, submission_id: str | None = None) -> None:
        self.competition_id = competition_id
        self.submission_id = submission_id
        super().__init__(
            f"Submission to {competition_id} failed: {message}",
            context={"competition_id": competition_id, "submission_id": submission_id},
        )


class DeadlinePassedError(CompetitionError):
    """Raised when competition deadline has passed.

    @notice: |
        Raised when competition deadline has passed.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Non-recoverable deadline expiration."
    """

    def __init__(self, competition_id: str, deadline: str) -> None:
        self.competition_id = competition_id
        self.deadline = deadline
        super().__init__(
            f"Competition {competition_id} deadline passed: {deadline}",
            context={"competition_id": competition_id, "deadline": deadline},
            recoverable=False,
        )


class EvolutionError(AgentKError):
    """Base exception for evolution-related errors.

    @notice: |
        Base exception for evolution-related errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-category
            rationale: "Category base for evolutionary algorithm errors."
    """


class ConvergenceError(EvolutionError):
    """Raised when evolution fails to converge within limits.

    @notice: |
        Raised when evolution fails to converge within limits.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures evolution progress for analysis."
    """

    def __init__(self, generations_completed: int, best_fitness: float, reason: str) -> None:
        self.generations_completed = generations_completed
        self.best_fitness = best_fitness
        self.reason = reason
        super().__init__(
            f"Evolution did not converge after {generations_completed} generations: {reason}",
            context={"generations_completed": generations_completed, "best_fitness": best_fitness, "reason": reason},
        )


class PopulationExtinctError(EvolutionError):
    """Raised when all population members fail fitness evaluation.

    @notice: |
        Raised when all population members fail fitness evaluation.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Non-recoverable population extinction."
    """

    def __init__(self, generation: int, last_error: str) -> None:
        self.generation = generation
        self.last_error = last_error
        super().__init__(
            f"Population extinct at generation {generation}: {last_error}",
            context={"generation": generation, "last_error": last_error},
            recoverable=False,
        )


class FitnessEvaluationError(EvolutionError):
    """Raised when fitness evaluation fails.

    @notice: |
        Raised when fitness evaluation fails.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures solution and execution context for debugging."
    """

    def __init__(self, solution_id: str, message: str, *, execution_error: str | None = None) -> None:
        self.solution_id = solution_id
        self.execution_error = execution_error
        super().__init__(
            f"Fitness evaluation failed for {solution_id}: {message}",
            context={"solution_id": solution_id, "execution_error": execution_error},
        )


class MemoryError(AgentKError):
    """Base exception for memory-related errors.

    @notice: |
        Base exception for memory-related errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-category
            rationale: "Category base for memory and checkpoint errors."
    """


class CheckpointError(MemoryError):
    """Raised when checkpoint operations fail.

    @notice: |
        Raised when checkpoint operations fail.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures checkpoint name and operation for debugging."
    """

    def __init__(self, checkpoint_name: str, operation: str, message: str) -> None:
        self.checkpoint_name = checkpoint_name
        self.operation = operation
        super().__init__(
            f"Checkpoint {operation} failed for {checkpoint_name}: {message}",
            context={"checkpoint_name": checkpoint_name, "operation": operation},
        )


class MemoryCapacityError(MemoryError):
    """Raised when memory capacity is exceeded.

    @notice: |
        Raised when memory capacity is exceeded.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures size metrics for capacity planning."
    """

    def __init__(self, current_size: int, max_size: int) -> None:
        self.current_size = current_size
        self.max_size = max_size
        super().__init__(
            f"Memory capacity exceeded: {current_size} / {max_size} bytes",
            context={"current_size": current_size, "max_size": max_size},
        )


class GraphError(AgentKError):
    """Base exception for graph-related errors.

    @notice: |
        Base exception for graph-related errors.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-category
            rationale: "Category base for state machine errors."
    """


class StateTransitionError(GraphError):
    """Raised when state transition is invalid.

    @notice: |
        Raised when state transition is invalid.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures transition context for state machine debugging."
    """

    def __init__(self, from_state: str, to_state: str, reason: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f"Invalid transition from {from_state} to {to_state}: {reason}",
            context={"from_state": from_state, "to_state": to_state, "reason": reason},
        )


class PhaseTimeoutError(GraphError):
    """Raised when a phase exceeds its timeout.

    @notice: |
        Raised when a phase exceeds its timeout.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: exception-specific
            rationale: "Captures timing metrics for timeout analysis."
    """

    def __init__(self, phase: str, timeout_seconds: int, elapsed_seconds: float) -> None:
        self.phase = phase
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Phase {phase} timed out after {elapsed_seconds:.1f}s (limit: {timeout_seconds}s)",
            context={"phase": phase, "timeout_seconds": timeout_seconds, "elapsed_seconds": elapsed_seconds},
        )


def classify_error(exc: Exception) -> tuple[ErrorCategory, RecoveryStrategy]:
    """Classify errors into recovery categories and strategies.

    @notice: |
        Maps exceptions to error categories and recovery strategies.

    @dev: |
        Used by error handlers to determine retry vs abort behavior.
        Returns (category, strategy) tuple for the exception type.
    """
    if isinstance(exc, RateLimitError):
        return "recoverable", "retry"
    if isinstance(exc, AuthenticationError):
        return "fatal", "abort"
    if isinstance(exc, CompetitionNotFoundError):
        return "fatal", "abort"
    if isinstance(exc, AgentKError):
        return ("recoverable", "retry") if exc.recoverable else ("fatal", "abort")
    return "transient", "retry"
