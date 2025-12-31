"""Code execution tool helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from typing import Any

# Third-party (alphabetical)
from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.toolsets import FunctionToolset

try:  # pragma: no cover - optional dependency
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:  # pragma: no cover - optional dependency
    OpenAIChatModel = None  # type: ignore[misc,assignment]

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "code_toolset",
    "create_code_execution_tool",
    "prepare_code_execution_tool",
)

# =============================================================================
# Toolset Definition
# =============================================================================
code_toolset: FunctionToolset[Any] = FunctionToolset(id="code")

# =============================================================================
# Section 12: Functions
# =============================================================================


def create_code_execution_tool() -> CodeExecutionTool:
    """Create a CodeExecutionTool instance."""
    return CodeExecutionTool()


async def prepare_code_execution_tool(ctx: RunContext[Any]) -> CodeExecutionTool | None:
    """Enable CodeExecutionTool only for supported providers."""
    if ctx.model.system not in {"anthropic", "openai", "google"}:
        return None
    if OpenAIChatModel is not None and isinstance(ctx.model, OpenAIChatModel):
        return None
    return CodeExecutionTool()
