"""Tests for the model factory infrastructure.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from agent_k.infra.providers import (
    DEVSTRAL_BASE_URL,
    DEVSTRAL_MODEL_ID,
    OPENROUTER_FREE_MODELS,
    create_devstral_model,
    create_openrouter_model,
    get_model,
    is_devstral_model,
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


class TestOpenRouterFreeModels:
    """Regression tests for the OPENROUTER_FREE_MODELS catalog."""

    def test_catalog_not_empty(self) -> None:
        """At least one free OpenRouter model must be advertised."""
        assert OPENROUTER_FREE_MODELS, "OPENROUTER_FREE_MODELS must not be empty"

    def test_entries_use_openrouter_free_format(self) -> None:
        """Every entry must be an openrouter: spec with a :free tier suffix."""
        for key, spec in OPENROUTER_FREE_MODELS.items():
            assert spec.startswith("openrouter:"), (
                f"OPENROUTER_FREE_MODELS[{key!r}] must use openrouter: prefix, got {spec!r}"
            )
            assert spec.endswith(":free"), f"OPENROUTER_FREE_MODELS[{key!r}] must target a :free tier, got {spec!r}"

    def test_excludes_known_dead_slugs(self) -> None:
        """Slugs verified dead in prior live tests must not return to the catalog."""
        dead_slugs = ("mistralai/devstral-2512:free", "kwaipilot/kat-coder-pro:free", "qwen/qwen3-coder:free")
        for spec in OPENROUTER_FREE_MODELS.values():
            for slug in dead_slugs:
                assert slug not in spec, f"Dead OpenRouter slug {slug!r} reappeared in {spec!r}"


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
