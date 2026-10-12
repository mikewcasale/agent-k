"""Tests for adaptive output-token headroom on agent runs.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_k.infra.headroom import MAX_TOKENS_CEILING, escalate_max_tokens, hit_token_ceiling, run_with_token_headroom

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

__all__ = ()


class SolutionPayload(BaseModel):
    """Structured output standing in for a code-bearing agent result."""

    code: str


def _truncated_then_complete(budgets: list[int | None], *, threshold: int) -> FunctionModel:
    """Build a model that truncates until the token budget clears a threshold."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        budget = (info.model_settings or {}).get("max_tokens")
        budgets.append(budget)
        tool_name = info.output_tools[0].name
        if budget is None or budget < threshold:
            return ModelResponse(parts=[ToolCallPart(tool_name, '{"code": "import lightgb')], finish_reason="length")
        return ModelResponse(parts=[ToolCallPart(tool_name, '{"code": "import lightgbm"}')], finish_reason="tool_call")

    return FunctionModel(respond)


class TestHitTokenCeiling:
    """Tests for hit_token_ceiling."""

    def test_detects_length_finish_reason(self) -> None:
        """A response stopped by the token budget should be detected."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart("evolve")]),
            ModelResponse(parts=[TextPart("partial")], finish_reason="length"),
        ]
        assert hit_token_ceiling(messages) is True

    def test_ignores_other_finish_reasons(self) -> None:
        """A normally completed response should not be reported as truncated."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart("evolve")]),
            ModelResponse(parts=[TextPart("done")], finish_reason="stop"),
        ]
        assert hit_token_ceiling(messages) is False

    def test_empty_history(self) -> None:
        """An empty history should not be reported as truncated."""
        assert hit_token_ceiling([]) is False


class TestEscalateMaxTokens:
    """Tests for escalate_max_tokens."""

    def test_multiplies_current_budget(self) -> None:
        """An explicit budget should grow by the configured factor."""
        escalated = escalate_max_tokens({"max_tokens": 4096}, ceiling=32768, factor=3)
        assert escalated is not None
        assert escalated["max_tokens"] == 12288

    def test_preserves_other_settings(self) -> None:
        """Unrelated model settings should survive escalation."""
        escalated = escalate_max_tokens({"max_tokens": 1024, "temperature": 0.7})
        assert escalated is not None
        assert escalated["temperature"] == 0.7

    def test_clamps_to_ceiling(self) -> None:
        """Escalation should never exceed the configured ceiling."""
        escalated = escalate_max_tokens({"max_tokens": 20000}, ceiling=32768, factor=3)
        assert escalated is not None
        assert escalated["max_tokens"] == 32768

    def test_returns_none_at_ceiling(self) -> None:
        """A budget already at the ceiling should not escalate further."""
        assert escalate_max_tokens({"max_tokens": MAX_TOKENS_CEILING}) is None

    def test_escalates_from_unset_base(self) -> None:
        """An unset budget should escalate from the provider-default base."""
        escalated = escalate_max_tokens(None, ceiling=32768, factor=3, unset_base=8192)
        assert escalated is not None
        assert escalated["max_tokens"] == 24576


class TestRunWithTokenHeadroom:
    """Tests for run_with_token_headroom."""

    async def test_retries_truncated_output_with_more_headroom(self) -> None:
        """A truncated structured output should be retried with a larger budget."""
        budgets: list[int | None] = []
        agent = Agent(
            _truncated_then_complete(budgets, threshold=10000),
            output_type=SolutionPayload,
            model_settings={"max_tokens": 4096},
            output_retries=0,
        )

        result = await run_with_token_headroom(agent, "evolve", deps=None, phase="evolution")

        assert result.output.code == "import lightgbm"
        assert budgets == [4096, 12288]

    async def test_leaves_agent_settings_untouched(self) -> None:
        """Escalation must not mutate the shared agent singleton settings."""
        budgets: list[int | None] = []
        agent = Agent(
            _truncated_then_complete(budgets, threshold=10000),
            output_type=SolutionPayload,
            model_settings={"max_tokens": 4096},
            output_retries=0,
        )

        await run_with_token_headroom(agent, "evolve", deps=None, phase="evolution")

        assert agent.model_settings == {"max_tokens": 4096}

    async def test_gives_up_after_escalation_budget(self) -> None:
        """Truncation that never clears should surface the model error."""
        budgets: list[int | None] = []
        agent = Agent(
            _truncated_then_complete(budgets, threshold=MAX_TOKENS_CEILING * 4),
            output_type=SolutionPayload,
            model_settings={"max_tokens": 4096},
            output_retries=0,
        )

        with pytest.raises(UnexpectedModelBehavior):
            await run_with_token_headroom(agent, "evolve", deps=None, phase="evolution", escalations=1)

        assert budgets == [4096, 12288]

    async def test_surfaces_truncation_when_provider_rejects_headroom(self) -> None:
        """A provider that caps output below the escalated budget keeps the real error."""
        budgets: list[int | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            budget = (info.model_settings or {}).get("max_tokens")
            budgets.append(budget)
            if budget is not None and budget > 4096:
                raise ModelHTTPError(status_code=400, model_name="small", body="max_tokens too large")
            tool_name = info.output_tools[0].name
            return ModelResponse(parts=[ToolCallPart(tool_name, '{"code": "import lightgb')], finish_reason="length")

        agent = Agent(
            FunctionModel(respond), output_type=SolutionPayload, model_settings={"max_tokens": 4096}, output_retries=0
        )

        with pytest.raises(UnexpectedModelBehavior) as exc_info:
            await run_with_token_headroom(agent, "evolve", deps=None, phase="evolution")

        assert isinstance(exc_info.value.__cause__, ModelHTTPError)
        assert budgets == [4096, 12288]

    async def test_propagates_unrelated_failures_without_retrying(self) -> None:
        """A failure that is not truncation should not spend an escalation."""
        calls: list[int | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append((info.model_settings or {}).get("max_tokens"))
            return ModelResponse(parts=[TextPart("no tool call here")], finish_reason="stop")

        agent = Agent(
            FunctionModel(respond), output_type=SolutionPayload, model_settings={"max_tokens": 4096}, output_retries=0
        )

        with pytest.raises(UnexpectedModelBehavior):
            await run_with_token_headroom(agent, "evolve", deps=None, phase="evolution")

        assert calls == [4096]
