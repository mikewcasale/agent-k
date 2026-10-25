"""Tests for provider capability detection.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pydantic_ai.models import infer_model
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from agent_k.infra.providers import (
    _CODE_EXECUTION_SPEC_CACHE,
    get_model,
    model_spec_supports_code_execution,
    supports_code_execution_builtin,
    unwrap_model,
)

__all__ = ()


def _openrouter_model() -> OpenAIChatModel:
    return OpenAIChatModel("openai/gpt-oss-120b:free", provider=OpenRouterProvider())


class TestUnwrapModel:
    """Tests for unwrap_model."""

    def test_returns_plain_model_unchanged(self) -> None:
        """An unwrapped model should be returned as-is."""
        model = _openrouter_model()

        assert unwrap_model(model) is model

    def test_strips_instrumentation_wrapper(self) -> None:
        """Instrumentation wrappers should be removed."""
        model = _openrouter_model()

        assert unwrap_model(InstrumentedModel(model)) is model

    def test_strips_nested_wrappers(self) -> None:
        """Nested wrappers should all be removed."""
        model = _openrouter_model()

        assert unwrap_model(InstrumentedModel(InstrumentedModel(model))) is model


class TestSupportsCodeExecutionBuiltin:
    """Tests for supports_code_execution_builtin."""

    def test_openai_chat_model_is_unsupported(self) -> None:
        """OpenAIChatModel raises UserError for the tool, so it must be refused."""
        assert supports_code_execution_builtin(infer_model("openai:gpt-4o")) is False

    def test_openai_responses_model_is_supported(self) -> None:
        """The Responses API maps the tool onto code_interpreter."""
        assert supports_code_execution_builtin(infer_model("openai-responses:gpt-4o")) is True

    def test_anthropic_model_is_supported(self) -> None:
        """Anthropic maps the tool onto its code-execution beta."""
        assert supports_code_execution_builtin(infer_model("anthropic:claude-sonnet-4-5")) is True

    def test_openrouter_chat_model_is_unsupported(self) -> None:
        """OpenRouter is served over Chat Completions, which cannot run the tool."""
        assert supports_code_execution_builtin(_openrouter_model()) is False

    def test_devstral_model_is_unsupported(self) -> None:
        """Local OpenAI-compatible endpoints cannot run the tool."""
        model = get_model("devstral:local")

        assert not isinstance(model, str)
        assert supports_code_execution_builtin(model) is False

    def test_detection_survives_instrumentation(self) -> None:
        """A wrapped Responses model is still capable."""
        responses_model = infer_model("openai-responses:gpt-4o")

        assert isinstance(responses_model, OpenAIResponsesModel)
        assert supports_code_execution_builtin(InstrumentedModel(responses_model)) is True


class TestModelSpecSupportsCodeExecution:
    """Tests for model_spec_supports_code_execution."""

    def test_openai_chat_spec_is_unsupported(self) -> None:
        """The openai: prefix resolves to Chat Completions, which cannot run code."""
        assert model_spec_supports_code_execution("openai:gpt-4o") is False

    def test_openai_responses_spec_is_supported(self) -> None:
        """The openai-responses: prefix resolves to a capable model."""
        assert model_spec_supports_code_execution("openai-responses:gpt-4o") is True

    def test_anthropic_spec_is_supported(self) -> None:
        """Anthropic specs resolve to a capable model."""
        assert model_spec_supports_code_execution("anthropic:claude-sonnet-4-5") is True

    def test_devstral_spec_is_unsupported(self) -> None:
        """Local Devstral endpoints cannot run the tool."""
        assert model_spec_supports_code_execution("devstral:local") is False

    def test_unresolvable_spec_is_unsupported(self) -> None:
        """A spec that cannot be resolved cannot run code either."""
        assert model_spec_supports_code_execution("not-a-provider:nope") is False

    def test_unresolvable_spec_is_not_cached(self) -> None:
        """A resolution failure must not be memoised, so configuration can land later."""
        model_spec_supports_code_execution("not-a-provider:nope")

        assert "not-a-provider:nope" not in _CODE_EXECUTION_SPEC_CACHE

    def test_resolved_verdict_is_cached(self) -> None:
        """Resolved verdicts are memoised so candidate executions stay cheap."""
        _CODE_EXECUTION_SPEC_CACHE.pop("devstral:local", None)

        assert model_spec_supports_code_execution("devstral:local") is False
        assert _CODE_EXECUTION_SPEC_CACHE["devstral:local"] is False
