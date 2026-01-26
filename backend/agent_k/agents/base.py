"""Base agent patterns for AGENT-K.

@notice: |
    Base agent patterns for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.agents.base
    provides:
        - agent_k.agents.base:AgentDeps
        - agent_k.agents.base:BaseAgentMixin
        - agent_k.agents.base:MemoryMixin
        - agent_k.agents.base:universal_tool_preparation
        - agent_k.agents.base:prepare_output_tools_strict
    consumes:
        - agent_k.toolsets.memory:AgentKMemoryTool
        - agent_k.toolsets.memory:create_memory_backend
        - agent_k.toolsets.memory:register_memory_tool
    pattern: agent-base

@agent-guidance:
    do:
        - "Use agent_k.agents.base as the canonical home for this capability."
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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, Generic

from ..core.sage import Doc
from ..core.types import AgentDepsT, OutputT
from ..toolsets import AgentKMemoryTool, create_memory_backend, register_memory_tool

if TYPE_CHECKING:
    import httpx
    from pydantic_ai import Agent, RunContext, ToolDefinition

__all__ = ("BaseAgentMixin", "MemoryMixin", "AgentDeps", "prepare_output_tools_strict", "universal_tool_preparation")


@dataclass
class AgentDeps:
    """Base dependency container for all agents.

    Per spec Section 7.1, dependencies are injected via dataclass containers.

    @pattern:
        name: dependency-container
        rationale: "Standardizes injected services for agent runs."
        violations: "Ad-hoc deps hide runtime requirements."

    @collaborators:
        required:
            - httpx:AsyncClient
        optional:
            - agent_k.ui.agui:EventEmitter
        injection: constructor
        lifecycle: "Allocated per agent run."
    """

    http_client: httpx.AsyncClient
    event_emitter: Any = None  # Will be EventEmitter type
    memory_store: dict[str, Any] = field(default_factory=dict)


class BaseAgentMixin(ABC, Generic[AgentDepsT, OutputT]):
    """Base mixin providing common agent functionality.

    Per spec Section 3.2, class structure follows visibility-based ordering.

    @pattern:
        name: agent-base
        rationale: "Shared behavior for agent wrappers around pydantic-ai Agents."
        violations: "Duplicated run logic across agents."

    @concurrency:
        model: asyncio
        safe: false
        reason: "Subclasses typically manage shared state per instance."
    """

    _default_model: str = "anthropic:claude-sonnet-4-5"

    async def run(
        self,
        prompt: Annotated[str, Doc("Natural language instruction for the agent.")],
        *,
        deps: Annotated[AgentDepsT, Doc("Dependency container for the agent run.")],
    ) -> OutputT:
        """Execute the agent with the given prompt.

        Per spec, all public methods include comprehensive docstrings.

        @notice: |
            Execute the agent with the provided prompt and dependencies.

        @dev: |
            Subclasses should override to provide concrete run behavior.

        @effects:
            io:
                - model invocation
            state:
                - agent runtime state

        @errors:
            terminal:
                - NotImplementedError
        """
        raise NotImplementedError("BaseAgentMixin.run must be implemented by subclasses.")

    @abstractmethod
    def _create_agent(self, model: str) -> Agent[AgentDepsT, OutputT]:
        """Create the underlying Pydantic-AI agent."""
        ...

    def _get_agent_name(self) -> str:
        """Return agent name for logging."""
        return self.__class__.__name__


class MemoryMixin:
    """Mixin providing memory backend initialization for agents.

    @pattern:
        name: mixin
        rationale: "Shares memory initialization across agents without inheritance chains."
        violations: "Duplicating memory setup logic in each agent."

    @concurrency:
        model: asyncio
        safe: false
        reason: "Mutates agent instance state during setup."
    """

    _memory_backend: AgentKMemoryTool | None
    _agent: Agent[Any, Any]

    def _init_memory_backend(self) -> AgentKMemoryTool | None:
        try:
            return create_memory_backend()
        except RuntimeError:  # pragma: no cover - optional dependency
            return None

    def _setup_memory(self) -> None:
        """Set up memory tool if available."""
        if self._memory_backend is None:
            return
        register_memory_tool(self._agent, self._memory_backend)


async def universal_tool_preparation(
    ctx: Annotated[RunContext[AgentDepsT], Doc("Run context for tool preparation.")],
    tool_defs: Annotated[list[ToolDefinition], Doc("Tool definitions to prepare.")],
) -> list[ToolDefinition]:
    """Apply universal tool configuration across agents.

    @notice: |
        Normalizes tool definitions for strictness and timeouts.

    @dev: |
        Sets strict mode for OpenAI tool calls and applies timeouts to risky tools.

    @effects:
        state:
            - tool_defs metadata
    """
    result: list[ToolDefinition] = []

    for tool_def in tool_defs:
        if ctx.model.system == "openai":
            tool_def = replace(tool_def, strict=True)

        if "submit" in tool_def.name or "evaluate" in tool_def.name:
            metadata = dict(tool_def.metadata or {})
            metadata["timeout"] = 120.0
            tool_def = replace(tool_def, metadata=metadata)

        result.append(tool_def)

    return result


async def prepare_output_tools_strict(
    ctx: Annotated[RunContext[AgentDepsT], Doc("Run context for tool preparation.")],
    tool_defs: Annotated[list[ToolDefinition], Doc("Tool definitions to update.")],
) -> list[ToolDefinition]:
    """Enable strict mode on output tools for OpenAI models.

    @notice: |
        Marks output tools as strict for OpenAI models.

    @dev: |
        Leaves tool definitions unchanged for non-OpenAI providers.

    @effects:
        state:
            - tool_defs strict flag
    """
    if ctx.model.system == "openai":
        return [replace(td, strict=True) for td in tool_defs]
    return tool_defs
