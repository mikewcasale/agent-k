"""Tool-call timeout enforcement for AGENT-K toolsets.

@notice: |
    Tool-call timeout enforcement for AGENT-K toolsets.

@dev: |
    Pydantic-AI treats `ToolDefinition.metadata` as opaque passthrough data and
    never enforces a timeout from it, so a wedged tool call blocks the agent
    turn (and therefore the mission phase) forever. `TimeoutToolset` is the
    enforcement layer: it wraps a toolset and runs every `call_tool` under
    `asyncio.timeout`, resolving the budget from tool metadata, then policy
    overrides, then the policy default.

@graph:
    id: agent_k.toolsets.limits
    provides:
        - agent_k.toolsets.limits:ToolTimeoutPolicy
        - agent_k.toolsets.limits:TimeoutToolset
        - agent_k.toolsets.limits:TOOL_TIMEOUT_MARGIN_SECONDS
        - agent_k.toolsets.limits:resolve_tool_timeout
        - agent_k.toolsets.limits:retry_allowed_on_timeout
    consumes:
        - agent_k.core.exceptions:ToolExecutionError
    pattern: toolset-wrapper

@similar:
    - id: agent_k.toolsets
        when: "Composing production toolsets; this module only bounds call duration."

@agent-guidance:
    do:
        - "Use agent_k.toolsets.limits as the canonical home for this capability."
        - "Declare per-tool budgets with metadata={'timeout': seconds} at registration."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Assume pydantic-ai honors metadata['timeout'] on its own."

@human-review:
    last-verified: 2026-10-05
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Annotated, Any, Final

import logfire
from pydantic_ai import ModelRetry, WrapperToolset
from pydantic_ai.tools import AgentDepsT

from agent_k.core.exceptions import ToolExecutionError
from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic_ai import RunContext, ToolDefinition
    from pydantic_ai.toolsets import ToolsetTool

__all__ = (
    "DEFAULT_TOOL_TIMEOUT_SECONDS",
    "RETRY_METADATA_KEY",
    "TOOL_TIMEOUT_MARGIN_SECONDS",
    "TIMEOUT_METADATA_KEY",
    "TimeoutToolset",
    "ToolTimeoutPolicy",
    "resolve_tool_timeout",
    "retry_allowed_on_timeout",
)

DEFAULT_TOOL_TIMEOUT_SECONDS: Final[float] = 120.0
TOOL_TIMEOUT_MARGIN_SECONDS: Final[float] = 60.0
TIMEOUT_METADATA_KEY: Final[str] = "timeout"
RETRY_METADATA_KEY: Final[str] = "retry_on_timeout"


@dataclass(frozen=True, slots=True)
class ToolTimeoutPolicy:
    """Per-tool wall-clock budget for tool calls.

    @notice: |
        Per-tool wall-clock budget for tool calls.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: policy-model
            rationale: "Keeps call-duration budgets declarative and shared across agents."
            violations: "Inline sleeps and ad-hoc waits leave tool calls unbounded."
    """

    default_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
    overrides: Mapping[str, float] = field(default_factory=dict)
    non_retryable: frozenset[str] = frozenset()


def _coerce_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    seconds = float(value)
    if not isfinite(seconds) or seconds <= 0.0:
        return None
    return seconds


def resolve_tool_timeout(
    name: Annotated[str, Doc("Tool name as the model called it.")],
    tool_def: Annotated[ToolDefinition | None, Doc("Tool definition carrying optional metadata.")],
    policy: Annotated[ToolTimeoutPolicy, Doc("Timeout policy to apply.")],
) -> float | None:
    """Resolve the wall-clock budget for a tool call.

    @notice: |
        Returns the timeout in seconds, or None when the call is unbounded.

    @dev: |
        Precedence is metadata, then policy overrides, then the policy default.
        Non-positive, non-finite, and non-numeric values mean "no limit".
    """
    metadata = (tool_def.metadata if tool_def is not None else None) or {}
    if TIMEOUT_METADATA_KEY in metadata:
        return _coerce_seconds(metadata[TIMEOUT_METADATA_KEY])
    if name in policy.overrides:
        return _coerce_seconds(policy.overrides[name])
    return _coerce_seconds(policy.default_seconds)


def retry_allowed_on_timeout(
    name: Annotated[str, Doc("Tool name as the model called it.")],
    tool_def: Annotated[ToolDefinition | None, Doc("Tool definition carrying optional metadata.")],
    policy: Annotated[ToolTimeoutPolicy, Doc("Timeout policy to apply.")],
) -> bool:
    """Report whether a timed-out tool call may be retried by the model.

    @notice: |
        Returns False for tools whose side effects make a blind retry unsafe.

    @dev: |
        Tools declare this with metadata={'retry_on_timeout': False}; the policy
        `non_retryable` set covers tools whose registration cannot be changed.
    """
    metadata = (tool_def.metadata if tool_def is not None else None) or {}
    declared = metadata.get(RETRY_METADATA_KEY)
    if isinstance(declared, bool):
        return declared
    return name not in policy.non_retryable


@dataclass
class TimeoutToolset(WrapperToolset[AgentDepsT]):
    """Toolset wrapper that bounds every tool call with `asyncio.timeout`.

    @notice: |
        Toolset wrapper that bounds every tool call with `asyncio.timeout`.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: toolset-wrapper
            rationale: "Built-in WrapperToolset is the supported extension point for call interception."
            violations: "Per-tool asyncio.timeout calls duplicate policy and drift."

        @concurrency:
            model: asyncio
            safe: true
            reason: "Holds immutable policy state and only wraps the delegated call."

        @errors:
            recoverable:
                - pydantic_ai.ModelRetry
            terminal:
                - agent_k.core.exceptions:ToolExecutionError
    """

    policy: ToolTimeoutPolicy = field(default_factory=ToolTimeoutPolicy)

    async def call_tool(
        self,
        name: Annotated[str, Doc("Tool name as the model called it.")],
        tool_args: Annotated[dict[str, Any], Doc("Validated tool arguments.")],
        ctx: Annotated[RunContext[AgentDepsT], Doc("Run context for the tool call.")],
        tool: Annotated[ToolsetTool[AgentDepsT], Doc("Tool being invoked.")],
    ) -> Any:
        """Call the wrapped tool under its resolved timeout.

        @notice: |
            Cancels the call once its budget expires and reports it to the model.

        @effects:
            io:
                - delegated tool execution
                - logfire telemetry

        @errors:
            recoverable:
                - pydantic_ai.ModelRetry
            terminal:
                - agent_k.core.exceptions:ToolExecutionError
        """
        timeout = resolve_tool_timeout(name, tool.tool_def, self.policy)
        if timeout is None:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        try:
            async with asyncio.timeout(timeout):
                return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        except TimeoutError:
            retryable = retry_allowed_on_timeout(name, tool.tool_def, self.policy)
            logfire.warn("tool_call_timeout", tool_name=name, timeout_seconds=timeout, retryable=retryable)
            message = f"timed out after {timeout:g}s and was cancelled"
            if not retryable:
                raise ToolExecutionError(name, f"{message}; its side effects are unknown") from None
            raise ModelRetry(
                f"Tool {name!r} {message}. Retry with a narrower request or use a different tool."
            ) from None
