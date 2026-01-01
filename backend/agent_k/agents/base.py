"""Base agent patterns for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic

# Local imports (core first, then alphabetical)
from ..core.types import AgentDepsT, OutputT
from ..toolsets import AgentKMemoryTool, create_memory_backend, register_memory_tool

if TYPE_CHECKING:
    import httpx
    from pydantic_ai import Agent, RunContext, ToolDefinition

__all__ = ('BaseAgentMixin', 'MemoryMixin', 'AgentDeps', 'prepare_output_tools_strict', 'universal_tool_preparation')


@dataclass
class AgentDeps:
    """Base dependency container for all agents.

    Per spec Section 7.1, dependencies are injected via dataclass containers.
    """

    http_client: httpx.AsyncClient
    event_emitter: Any = None  # Will be EventEmitter type
    memory_store: dict[str, Any] = field(default_factory=dict)


class BaseAgentMixin(ABC, Generic[AgentDepsT, OutputT]):
    """Base mixin providing common agent functionality.

    Per spec Section 3.2, class structure follows visibility-based ordering.
    """

    # =========================================================================
    # Class Variables
    # =========================================================================
    _default_model: str = 'anthropic:claude-sonnet-4-5'

    # =========================================================================
    # Public Methods
    # =========================================================================
    async def run(self, prompt: str, *, deps: AgentDepsT) -> OutputT:
        """Execute the agent with the given prompt.

        Per spec, all public methods include comprehensive docstrings.

        Args:
            prompt: Natural language instruction for the agent.
            deps: Dependency container with required services.

        Returns:
            Agent output of type OutputT.
        """
        raise NotImplementedError('BaseAgentMixin.run must be implemented by subclasses.')

    # =========================================================================
    # Protected Methods
    # =========================================================================
    @abstractmethod
    def _create_agent(self, model: str) -> Agent[AgentDepsT, OutputT]:
        """Create the underlying Pydantic-AI agent."""
        ...

    def _get_agent_name(self) -> str:
        """Return agent name for logging."""
        return self.__class__.__name__


class MemoryMixin:
    """Mixin providing memory backend initialization for agents."""

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
    ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Apply universal tool configuration across agents."""
    result: list[ToolDefinition] = []

    for tool_def in tool_defs:
        if ctx.model.system == 'openai':
            tool_def = replace(tool_def, strict=True)

        if 'submit' in tool_def.name or 'evaluate' in tool_def.name:
            metadata = dict(tool_def.metadata or {})
            metadata['timeout'] = 120.0
            tool_def = replace(tool_def, metadata=metadata)

        result.append(tool_def)

    return result


async def prepare_output_tools_strict(
    ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Enable strict mode on output tools for OpenAI models."""
    if ctx.model.system == 'openai':
        return [replace(td, strict=True) for td in tool_defs]
    return tool_defs
