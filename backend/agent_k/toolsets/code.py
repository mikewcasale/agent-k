"""Code execution tool helpers.

@notice: |
    Code execution tool helpers.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.toolsets.code
    provides:
        - agent_k.toolsets.code:code_toolset
        - agent_k.toolsets.code:create_code_execution_tool
        - agent_k.toolsets.code:prepare_code_execution_tool
    pattern: toolset

@agent-guidance:
    do:
        - "Use agent_k.toolsets.code as the canonical home for this capability."
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

from typing import Annotated, Any

from pydantic_ai import RunContext
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.sage import Doc

try:  # pragma: no cover - optional dependency
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:  # pragma: no cover - optional dependency
    OpenAIChatModel = None  # type: ignore[misc,assignment]

__all__ = ("code_toolset", "create_code_execution_tool", "prepare_code_execution_tool")

code_toolset: FunctionToolset[Any] = FunctionToolset(id="code")


def create_code_execution_tool() -> CodeExecutionTool:
    """Create a CodeExecutionTool instance.

    @notice: |
        Returns a configured code execution tool.
    """
    return CodeExecutionTool()


async def prepare_code_execution_tool(
    ctx: Annotated[RunContext[Any], Doc("Run context for tool preparation.")],
) -> CodeExecutionTool | None:
    """Enable CodeExecutionTool only for supported providers.

    @notice: |
        Returns a tool only for OpenAI-compatible providers.
    """
    if ctx.model.system != "openai":
        return None
    if OpenAIChatModel is not None and isinstance(ctx.model, OpenAIChatModel):
        return None
    return CodeExecutionTool()
