"""Toolsets and built-in tool helpers for AGENT-K agents.

@notice: |
    Toolsets and built-in tool helpers for AGENT-K agents.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.toolsets
    provides:
        - agent_k.toolsets:TOOLSET_REGISTRY
        - agent_k.toolsets:compose_toolsets
        - agent_k.toolsets:create_production_toolset
    pattern: toolset-registry

@similar:
    - id: agent_k.toolsets.kaggle
        when: "Specific Kaggle tool implementations."

@agent-guidance:
    do:
        - "Use agent_k.toolsets as the canonical home for this capability."
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

from dataclasses import replace
from typing import Annotated, Any, TypeVar

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset

from agent_k.core.sage import Doc

from .code import code_toolset, create_code_execution_tool, prepare_code_execution_tool
from .kaggle import kaggle_toolset
from .memory import AgentKMemoryTool, create_memory_backend, prepare_memory_tool, register_memory_tool
from .search import (
    build_kaggle_search_query,
    build_scholarly_query,
    create_web_fetch_tool,
    create_web_search_tool,
    prepare_web_fetch,
    prepare_web_search,
)
from .tracking import tracking_toolset

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
    "tracking_toolset",
    "TOOLSET_REGISTRY",
)

TOOLSET_REGISTRY: dict[str, FunctionToolset[Any]] = {
    "kaggle": kaggle_toolset,
    "code": code_toolset,
    "tracking": tracking_toolset,
}


def compose_toolsets(
    names: Annotated[list[str], Doc("Toolset registry names to compose.")],
    *,
    prefix: Annotated[bool, Doc("Whether to prefix tools with the toolset name.")] = True,
) -> AbstractToolset:
    """Compose multiple toolsets into one.

    @notice: |
        Combines toolsets and optionally prefixes tool names.

    @errors:
        terminal:
            - KeyError
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
    toolsets: Annotated[list[FunctionToolset[DepsT]], Doc("Toolsets to combine.")],
    *,
    require_approval_for: Annotated[list[str] | None, Doc("Tool names requiring approval.")] = None,
    prefix: Annotated[str | None, Doc("Prefix for tool names.")] = None,
) -> AbstractToolset[DepsT]:
    """Create production-ready toolset with wrappers.

    Applies:
    - Prefixing for namespace isolation
    - Approval requirements for selected tools
    - Strict mode for OpenAI tool calls

    @factory-for:
        id: agent_k.toolsets:CombinedToolset
        rationale: "Centralizes production wrapper policy."
        singleton: false
        cache-key: prefix

    @canonical-home:
        for:
            - "production toolset composition"
        notes: "Use create_production_toolset for standardized behavior."
    """
    combined: AbstractToolset[DepsT] = CombinedToolset(toolsets)

    if prefix:
        combined = combined.prefixed(f"{prefix}_")

    if require_approval_for:
        combined = combined.approval_required(lambda _ctx, tool_def, _args: tool_def.name in require_approval_for)

    async def prepare_for_model(
        ctx: Annotated[RunContext[DepsT], Doc("Run context for tool preparation.")],
        tool_defs: Annotated[list[ToolDefinition], Doc("Tool definitions to update.")],
    ) -> list[ToolDefinition]:
        if ctx.model.system == "openai":
            return [replace(td, strict=True) for td in tool_defs]
        return tool_defs

    return combined.prepared(prepare_for_model)
