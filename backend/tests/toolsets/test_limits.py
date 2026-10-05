"""Tests for tool-call timeout enforcement.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.exceptions import ToolExecutionError
from agent_k.toolsets import DEFAULT_TOOL_TIMEOUT_SECONDS, TimeoutToolset, ToolTimeoutPolicy, create_production_toolset
from agent_k.toolsets.limits import resolve_tool_timeout, retry_allowed_on_timeout

__all__ = ()


def _build_toolset() -> FunctionToolset[None]:
    toolset: FunctionToolset[None] = FunctionToolset(id="limits_test")

    @toolset.tool(metadata={"timeout": 0.05})
    async def hangs(ctx: RunContext[None]) -> str:
        """Block for longer than its declared budget."""
        await asyncio.sleep(30)
        return "never"

    @toolset.tool(metadata={"timeout": 0.05, "retry_on_timeout": False})
    async def hangs_unsafe(ctx: RunContext[None]) -> str:
        """Block for longer than its budget with unknown side effects."""
        await asyncio.sleep(30)
        return "never"

    @toolset.tool
    async def responds(ctx: RunContext[None]) -> str:
        """Return immediately."""
        return "fast"

    return toolset


def _single_call_model(tool_name: str) -> FunctionModel:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(model_fn)


async def test_timed_out_tool_is_reported_to_the_model() -> None:
    agent: Agent[None, str] = Agent(
        _single_call_model("hangs"), toolsets=[create_production_toolset([_build_toolset()])], retries=1
    )

    result = await agent.run("go")

    assert result.output == "done"
    retry_prompts = [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
    ]
    assert any("timed out after 0.05s" in str(content) for content in retry_prompts)


async def test_timed_out_tool_without_retry_raises() -> None:
    agent: Agent[None, str] = Agent(
        _single_call_model("hangs_unsafe"), toolsets=[create_production_toolset([_build_toolset()])], retries=1
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await agent.run("go")

    assert exc_info.value.tool_name == "hangs_unsafe"
    assert "side effects are unknown" in str(exc_info.value)


async def test_fast_tool_is_unaffected() -> None:
    agent: Agent[None, str] = Agent(
        _single_call_model("responds"), toolsets=[create_production_toolset([_build_toolset()])], retries=1
    )

    result = await agent.run("go")

    assert result.output == "done"
    tool_returns = [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert tool_returns == ["fast"]


async def test_default_budget_applies_without_metadata() -> None:
    toolset: FunctionToolset[None] = FunctionToolset(id="default_budget")
    observed: list[float | None] = []

    @toolset.tool
    async def plain(ctx: RunContext[None]) -> str:
        """Return immediately."""
        return "ok"

    class Recording(TimeoutToolset[None]):
        async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: Any) -> Any:
            observed.append(resolve_tool_timeout(name, tool.tool_def, self.policy))
            return await super().call_tool(name, tool_args, ctx, tool)

    agent: Agent[None, str] = Agent(_single_call_model("plain"), toolsets=[Recording(toolset)])
    await agent.run("go")

    assert observed == [DEFAULT_TOOL_TIMEOUT_SECONDS]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, 30.0),
        ({}, 30.0),
        ({"timeout": 5.0}, 5.0),
        ({"timeout": 0}, None),
        ({"timeout": -1.0}, None),
        ({"timeout": float("inf")}, None),
        ({"timeout": None}, None),
        ({"timeout": "90"}, None),
        ({"timeout": True}, None),
    ],
)
def test_resolve_tool_timeout_precedence(metadata: dict[str, Any] | None, expected: float | None) -> None:
    from pydantic_ai import ToolDefinition

    policy = ToolTimeoutPolicy(default_seconds=30.0, overrides={"slow": 600.0})
    tool_def = ToolDefinition(name="plain", metadata=metadata)

    assert resolve_tool_timeout("plain", tool_def, policy) == expected
    assert resolve_tool_timeout("slow", ToolDefinition(name="slow"), policy) == 600.0
    assert resolve_tool_timeout("plain", None, policy) == 30.0


def test_retry_allowed_on_timeout_sources() -> None:
    from pydantic_ai import ToolDefinition

    policy = ToolTimeoutPolicy(non_retryable=frozenset({"submit"}))

    assert retry_allowed_on_timeout("plain", ToolDefinition(name="plain"), policy) is True
    assert retry_allowed_on_timeout("submit", ToolDefinition(name="submit"), policy) is False
    assert (
        retry_allowed_on_timeout("submit", ToolDefinition(name="submit", metadata={"retry_on_timeout": True}), policy)
        is True
    )
    assert (
        retry_allowed_on_timeout("plain", ToolDefinition(name="plain", metadata={"retry_on_timeout": False}), policy)
        is False
    )


async def test_unbounded_policy_skips_the_timeout() -> None:
    toolset: FunctionToolset[None] = FunctionToolset(id="unbounded")

    @toolset.tool
    async def quick(ctx: RunContext[None]) -> str:
        """Return immediately."""
        return "ok"

    wrapped = TimeoutToolset(toolset, policy=ToolTimeoutPolicy(default_seconds=0.0))
    agent: Agent[None, str] = Agent(_single_call_model("quick"), toolsets=[wrapped])

    result = await agent.run("go")

    assert result.output == "done"


async def test_outer_cancellation_is_not_swallowed() -> None:
    toolset: FunctionToolset[None] = FunctionToolset(id="cancellation")

    @toolset.tool(metadata={"timeout": 30.0})
    async def slow(ctx: RunContext[None]) -> str:
        """Block until cancelled."""
        await asyncio.sleep(30)
        return "never"

    wrapped = TimeoutToolset(toolset, policy=ToolTimeoutPolicy())
    agent: Agent[None, str] = Agent(_single_call_model("slow"), toolsets=[wrapped])

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await agent.run("go")
