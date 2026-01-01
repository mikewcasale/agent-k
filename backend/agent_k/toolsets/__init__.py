"""Toolsets and built-in tool helpers for AGENT-K agents.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from dataclasses import replace
from typing import Any, TypeVar

# Third-party (alphabetical)
from pydantic_ai import RunContext, ToolDefinition  # noqa: TC002
from pydantic_ai.toolsets import (
    AbstractToolset,
    CombinedToolset,
    FunctionToolset,
)

# Local imports (core first, then alphabetical)
from .code import code_toolset, create_code_execution_tool, prepare_code_execution_tool
from .kaggle import kaggle_toolset
from .memory import (
    AgentKMemoryTool,
    create_memory_backend,
    prepare_memory_tool,
    register_memory_tool,
)
from .search import (
    build_kaggle_search_query,
    build_scholarly_query,
    create_web_fetch_tool,
    create_web_search_tool,
    prepare_web_fetch,
    prepare_web_search,
)

DepsT = TypeVar("DepsT")
"""Type variable for toolset dependencies."""

__all__ = (
    "AgentKMemoryTool",
    "build_kaggle_search_query",
    "build_scholarly_query",
    "code_toolset",
    "compose_toolsets",
    "create_code_execution_tool",
    "create_memory_backend",
    "create_production_toolset",
    "create_web_fetch_tool",
    "create_web_search_tool",
    "kaggle_toolset",
    "prepare_code_execution_tool",
    "prepare_memory_tool",
    "prepare_web_fetch",
    "prepare_web_search",
    "register_memory_tool",
    "TOOLSET_REGISTRY",
)

TOOLSET_REGISTRY: dict[str, FunctionToolset[Any]] = {
    "kaggle": kaggle_toolset,
    "code": code_toolset,
}


def compose_toolsets(names: list[str], *, prefix: bool = True) -> AbstractToolset:
    """Compose multiple toolsets into one.

    Args:
        names: Toolset registry names to compose.
        prefix: Whether to prefix tools with the toolset name.

    Returns:
        Combined toolset ready for agent use.
    """
    toolsets: list[AbstractToolset] = []
    for name in names:
        if name not in TOOLSET_REGISTRY:
            raise KeyError(f"Unknown toolset: {name}. Available: {list(TOOLSET_REGISTRY)}")
        toolset: AbstractToolset = TOOLSET_REGISTRY[name]
        if prefix:
            toolset = toolset.prefixed(f"{name}_")
        toolsets.append(toolset)

    return CombinedToolset(toolsets)


def create_production_toolset(
    toolsets: list[FunctionToolset[DepsT]],
    *,
    require_approval_for: list[str] | None = None,
    prefix: str | None = None,
) -> AbstractToolset[DepsT]:
    """Create production-ready toolset with wrappers.

    Applies:
    - Prefixing for namespace isolation
    - Approval requirements for selected tools
    - Strict mode for OpenAI tool calls
    """
    combined: AbstractToolset[DepsT] = CombinedToolset(toolsets)

    if prefix:
        combined = combined.prefixed(f"{prefix}_")

    if require_approval_for:
        combined = combined.approval_required(
            lambda _ctx, tool_def, _args: tool_def.name in require_approval_for
        )

    async def prepare_for_model(
        ctx: RunContext[DepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        if ctx.model.system == "openai":
            return [replace(td, strict=True) for td in tool_defs]
        return tool_defs

    return combined.prepared(prepare_for_model)
