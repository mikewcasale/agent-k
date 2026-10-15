"""Tests for the model factory infrastructure.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from agent_k.core.constants import DEFAULT_MODEL
from agent_k.infra.providers import (
    DEVSTRAL_BASE_URL,
    DEVSTRAL_MODEL_ID,
    OPENROUTER_FREE_MODELS,
    create_devstral_model,
    create_openrouter_model,
    get_model,
    is_devstral_model,
    is_openai_compatible_spec,
)

__all__ = ()


class TestDevstralConstants:
    """Tests for Devstral constants."""

    def test_devstral_model_id(self) -> None:
        """Devstral model ID should be set."""
        assert DEVSTRAL_MODEL_ID == "mistralai/devstral-small-2-2512"

    def test_devstral_base_url_default(self) -> None:
        """Default base URL should be LM Studio."""
        assert "1234" in DEVSTRAL_BASE_URL  # LM Studio default port


class TestCreateDevstralModel:
    """Tests for create_devstral_model function."""

    def test_creates_openai_model(self) -> None:
        """Should create an OpenAIChatModel."""
        model = create_devstral_model()

        assert isinstance(model, OpenAIChatModel)

    def test_custom_base_url(self) -> None:
        """Should accept custom base URL."""
        model = create_devstral_model(base_url="http://localhost:5000/v1")

        assert isinstance(model, OpenAIChatModel)

    def test_custom_model_id(self) -> None:
        """Should accept custom model ID."""
        model = create_devstral_model(model_id="custom-model")

        assert isinstance(model, OpenAIChatModel)


class TestCreateOpenRouterModel:
    """Tests for create_openrouter_model function."""

    def test_creates_openai_model(self) -> None:
        """Should create an OpenAIChatModel."""
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        model = create_openrouter_model("mistralai/devstral-small-2505")

        assert isinstance(model, OpenAIChatModel)


class TestGetModel:
    """Tests for get_model function."""

    def test_anthropic_model(self) -> None:
        """Anthropic model spec should return string."""
        result = get_model("anthropic:claude-3-haiku-20240307")

        assert result == "anthropic:claude-3-haiku-20240307"

    def test_openai_model(self) -> None:
        """OpenAI model spec should return string."""
        result = get_model("openai:gpt-4o")

        assert result == "openai:gpt-4o"

    def test_devstral_local(self) -> None:
        """Devstral local should return OpenAIChatModel."""
        result = get_model("devstral:local")

        assert isinstance(result, OpenAIChatModel)

    def test_devstral_custom_url(self) -> None:
        """Devstral with custom URL should return OpenAIChatModel."""
        result = get_model("devstral:http://localhost:8080/v1")

        assert isinstance(result, OpenAIChatModel)

    def test_devstral_custom_model_id(self) -> None:
        """Devstral with custom model ID should return OpenAIChatModel."""
        result = get_model("devstral:custom-model-id")

        assert isinstance(result, OpenAIChatModel)

    def test_openrouter_model(self) -> None:
        """OpenRouter model spec should return OpenAIChatModel."""
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        result = get_model("openrouter:mistralai/devstral-small-2505")

        assert isinstance(result, OpenAIChatModel)


class TestIsDevstralModel:
    """Tests for is_devstral_model function."""

    def test_devstral_local(self) -> None:
        """devstral:local should be identified as Devstral."""
        assert is_devstral_model("devstral:local") is True

    def test_devstral_custom_url(self) -> None:
        """Devstral with custom URL should be identified as Devstral."""
        assert is_devstral_model("devstral:http://localhost:8080/v1") is True

    def test_anthropic_not_devstral(self) -> None:
        """Anthropic models should not be identified as Devstral."""
        assert is_devstral_model("anthropic:claude-3-haiku-20240307") is False

    def test_openai_not_devstral(self) -> None:
        """OpenAI models should not be identified as Devstral."""
        assert is_devstral_model("openai:gpt-4o") is False

    def test_openrouter_not_devstral(self) -> None:
        """OpenRouter models should not be identified as Devstral."""
        assert is_devstral_model("openrouter:mistralai/devstral-small-2505") is False


class TestOpenRouterFreeModels:
    """Tests for the free OpenRouter model registry."""

    def test_registry_is_populated(self) -> None:
        """Registry should expose at least one free model."""
        assert OPENROUTER_FREE_MODELS

    def test_all_entries_are_openrouter_specs(self) -> None:
        """Every registered free model should be an OpenRouter spec."""
        for spec in OPENROUTER_FREE_MODELS.values():
            assert spec.startswith("openrouter:")
            assert spec.endswith(":free")

    def test_all_entries_are_openai_compatible(self) -> None:
        """Free models are used as fallbacks, so each must be callable."""
        for spec in OPENROUTER_FREE_MODELS.values():
            assert is_openai_compatible_spec(spec)

    def test_default_model_is_registered_or_openai_compatible(self) -> None:
        """The configured default must be usable by OpenAI-compatible clients."""
        assert is_openai_compatible_spec(DEFAULT_MODEL)


class TestIsOpenAICompatibleSpec:
    """Tests for is_openai_compatible_spec function."""

    @pytest.mark.parametrize(
        "spec",
        [
            "openrouter:openai/gpt-oss-120b:free",
            "openai:gpt-4o",
            "devstral:local",
            "devstral:http://localhost:1234/v1",
            "gpt-4o",
        ],
    )
    def test_compatible_specs(self, spec: str) -> None:
        """OpenAI-compatible and unprefixed specs should be accepted."""
        assert is_openai_compatible_spec(spec) is True

    @pytest.mark.parametrize(
        "spec",
        [
            "anthropic:claude-sonnet-4-5",
            "google:gemini-2.5-pro",
            "google-gla:gemini-2.0-flash",
            "groq:llama-3.3-70b-versatile",
            "bedrock:anthropic.claude-3-sonnet",
        ],
    )
    def test_native_sdk_specs(self, spec: str) -> None:
        """Specs bound to a native provider SDK should be rejected."""
        assert is_openai_compatible_spec(spec) is False
