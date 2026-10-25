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
from agent_k.infra.providers import supports_code_execution_builtin

__all__ = ("code_toolset", "create_code_execution_tool", "prepare_code_execution_tool")

code_toolset: FunctionToolset[Any] = FunctionToolset(id="code")


def create_code_execution_tool() -> CodeExecutionTool:
    """Create a CodeExecutionTool instance.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns a configured code execution tool.
    """
    return CodeExecutionTool()


async def prepare_code_execution_tool(
    ctx: Annotated[RunContext[Any], Doc("Run context for tool preparation.")],
) -> CodeExecutionTool | None:
    """Enable CodeExecutionTool only for models that implement it.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns a tool only for models that map it onto a provider builtin.
    """
    if not supports_code_execution_builtin(ctx.model):
        return None
    return CodeExecutionTool()
