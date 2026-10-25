"""Tests for capability-driven tool preparation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from agent_k.agents.base import universal_tool_preparation
from agent_k.infra.providers import get_model
from agent_k.toolsets import create_production_toolset, prepare_code_execution_tool

__all__ = ()

_STRICT_READY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}
_OPEN_DICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"record": {"type": "object", "additionalProperties": True}},
    "required": ["record"],
    "additionalProperties": False,
}
_OPTIONAL_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "rounds": {"type": "integer", "default": 3}},
    "required": ["code"],
    "additionalProperties": False,
}


def _context(model: Model) -> RunContext[None]:
    """Build a run context mirroring how agents receive an instrumented model."""
    return RunContext(deps=None, model=InstrumentedModel(model), usage=RunUsage())


def _tool_def(name: str = "evaluate_fitness", schema: dict[str, Any] | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Score a candidate.",
        parameters_json_schema=json.loads(json.dumps(schema or _STRICT_READY_SCHEMA)),
    )


def _openrouter_model() -> OpenAIChatModel:
    return OpenAIChatModel("openai/gpt-oss-120b:free", provider=OpenRouterProvider())


def _resolve(model: Model, tool_def: ToolDefinition) -> ToolDefinition:
    """Run a tool definition through the model's own schema customization."""
    params = ModelRequestParameters(function_tools=[tool_def])
    return next(iter(model.customize_request_parameters(params).tool_defs.values()))


def _providers() -> dict[str, Model]:
    devstral = get_model("devstral:local")
    assert not isinstance(devstral, str)
    return {
        "openrouter": _openrouter_model(),
        "devstral": devstral,
        "openai": infer_model("openai:gpt-4o"),
        "anthropic": infer_model("anthropic:claude-sonnet-4-5"),
    }


class TestUniversalToolPreparation:
    """Tests for universal_tool_preparation."""

    async def test_leaves_strictness_to_pydantic_ai(self) -> None:
        """Preparation must not pin strictness, so pydantic-ai can infer it per schema."""
        for name, model in _providers().items():
            prepared = await universal_tool_preparation(_context(model), [_tool_def()])

            assert [td.strict for td in prepared] == [None], name

    async def test_applies_timeout_metadata(self) -> None:
        """Timeout metadata is still applied to risky tools."""
        prepared = await universal_tool_preparation(_context(_openrouter_model()), [_tool_def()])

        assert (prepared[0].metadata or {})["timeout"] == 120.0

    async def test_leaves_safe_tools_without_timeout(self) -> None:
        """Tools outside the risky set keep their metadata untouched."""
        prepared = await universal_tool_preparation(_context(_openrouter_model()), [_tool_def("sample_elites")])

        assert prepared[0].metadata is None


class TestInferredStrictness:
    """Tests that pydantic-ai's inferred strictness survives preparation."""

    async def test_compatible_schema_still_becomes_strict(self) -> None:
        """A strict-ready schema must still be sent as strict on every OpenAI-protocol provider."""
        for name in ("openrouter", "devstral", "openai"):
            model = _providers()[name]
            prepared = await universal_tool_preparation(_context(model), [_tool_def()])

            assert _resolve(model, prepared[0]).strict is True, name

    async def test_open_dict_parameter_keeps_accepting_keys(self) -> None:
        """Forcing strict collapsed open dicts to objects accepting no keys; it must stay open."""
        for name, model in _providers().items():
            prepared = await universal_tool_preparation(_context(model), [_tool_def(schema=_OPEN_DICT_SCHEMA)])
            resolved = _resolve(model, prepared[0])

            assert resolved.parameters_json_schema["properties"]["record"]["additionalProperties"] is True, name

    async def test_optional_parameter_stays_optional(self) -> None:
        """Forcing strict marked every parameter required; optional ones must stay optional."""
        for name, model in _providers().items():
            prepared = await universal_tool_preparation(_context(model), [_tool_def(schema=_OPTIONAL_PARAM_SCHEMA)])
            resolved = _resolve(model, prepared[0])

            assert resolved.parameters_json_schema["required"] == ["code"], name


class TestProductionToolset:
    """Tests for create_production_toolset."""

    async def test_leaves_strictness_to_pydantic_ai(self) -> None:
        """The production wrapper must not pin strictness either."""
        toolset: FunctionToolset[Any] = FunctionToolset(id="probe")

        @toolset.tool
        def probe(value: str, limit: int = 5) -> str:
            """Probe tool."""
            return f"{value}{limit}"

        for name, model in _providers().items():
            tools = await create_production_toolset([toolset]).get_tools(_context(model))

            assert [tool.tool_def.strict for tool in tools.values()] == [None], name

    async def test_still_applies_approval_requirements(self) -> None:
        """Approval wrapping must survive removal of the strictness wrapper."""
        toolset: FunctionToolset[Any] = FunctionToolset(id="probe")

        @toolset.tool
        def submit_to_kaggle(value: str) -> str:
            """Submit tool."""
            return value

        combined = create_production_toolset([toolset], require_approval_for=["submit_to_kaggle"])
        tools = await combined.get_tools(_context(_openrouter_model()))

        assert set(tools) == {"submit_to_kaggle"}

    async def test_still_applies_prefix(self) -> None:
        """Prefixing must survive removal of the strictness wrapper."""
        toolset: FunctionToolset[Any] = FunctionToolset(id="probe")

        @toolset.tool
        def probe(value: str) -> str:
            """Probe tool."""
            return value

        tools = await create_production_toolset([toolset], prefix="evo").get_tools(_context(_openrouter_model()))

        # pydantic-ai's `prefixed` adds its own separator, so the configured "evo_" yields "evo__".
        assert set(tools) == {"evo__probe"}


class TestPrepareCodeExecutionTool:
    """Tests for prepare_code_execution_tool."""

    async def test_refuses_openai_chat_model(self) -> None:
        """OpenAIChatModel raises UserError for the tool, so it must be withheld."""
        assert await prepare_code_execution_tool(_context(infer_model("openai:gpt-4o"))) is None

    async def test_refuses_devstral_model(self) -> None:
        """The documented devstral:local spec cannot run the tool."""
        model = get_model("devstral:local")
        assert not isinstance(model, str)

        assert await prepare_code_execution_tool(_context(model)) is None

    async def test_refuses_openrouter_model(self) -> None:
        """OpenRouter is served over Chat Completions and cannot run the tool."""
        assert await prepare_code_execution_tool(_context(_openrouter_model())) is None

    async def test_allows_openai_responses_model(self) -> None:
        """The Responses API maps the tool onto code_interpreter."""
        tool = await prepare_code_execution_tool(_context(infer_model("openai-responses:gpt-4o")))

        assert isinstance(tool, CodeExecutionTool)

    async def test_allows_anthropic_model(self) -> None:
        """Anthropic maps the tool onto its code-execution beta."""
        tool = await prepare_code_execution_tool(_context(infer_model("anthropic:claude-sonnet-4-5")))

        assert isinstance(tool, CodeExecutionTool)
