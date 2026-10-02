"""Tests for OpenEvolve LLM parameter propagation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import pytest
from openevolve.config import LLMConfig
from openevolve.llm.openai import OpenAILLM

from agent_k.adapters.openevolve import (
    _LLM_REQUEST_FALLBACKS,
    _SHARED_LLM_PARAM_NAMES,
    OpenEvolveRunner,
    _shared_llm_params,
)
from agent_k.infra.providers import DEVSTRAL_BASE_URL

__all__ = ()

pytestmark = pytest.mark.anyio


def _configured_llm_config(model_specs: list[str]) -> LLMConfig:
    """Build the LLM config an OpenEvolve run would actually execute with."""
    runner = OpenEvolveRunner(model_specs=model_specs)
    config = runner._configure_llm(runner._load_config())
    return config.llm


class TestSharedLlmParams:
    """Tests for _shared_llm_params."""

    def test_reads_configured_values(self) -> None:
        """Configured generation and request parameters are passed through."""
        llm_config = LLMConfig(temperature=0.55, top_p=0.9, max_tokens=8192, timeout=120, retries=5, retry_delay=2)

        params = _shared_llm_params(llm_config)

        assert params["temperature"] == 0.55
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 8192
        assert params["timeout"] == 120
        assert params["retries"] == 5
        assert params["retry_delay"] == 2

    def test_covers_every_shared_param_name(self) -> None:
        """Every shared parameter name is present in the result."""
        params = _shared_llm_params(LLMConfig())

        assert set(params) == set(_SHARED_LLM_PARAM_NAMES)

    @pytest.mark.parametrize("param", sorted(_LLM_REQUEST_FALLBACKS))
    def test_falls_back_when_request_param_is_none(self, param: str) -> None:
        """A None request parameter is replaced, never forwarded."""
        llm_config = LLMConfig()
        setattr(llm_config, param, None)

        params = _shared_llm_params(llm_config)

        assert params[param] == _LLM_REQUEST_FALLBACKS[param]


class TestConfigureLlm:
    """Tests for OpenEvolveRunner._configure_llm."""

    def test_models_inherit_generation_params(self) -> None:
        """Generation parameters reach the per-spec model configs."""
        llm_config = _configured_llm_config(["openrouter:openai/gpt-oss-120b:free"])

        model = llm_config.models[0]
        assert model.temperature == llm_config.temperature
        assert model.top_p == llm_config.top_p
        assert model.max_tokens == llm_config.max_tokens

    def test_models_inherit_request_params(self) -> None:
        """Request parameters reach the per-spec model configs."""
        llm_config = _configured_llm_config(["openrouter:openai/gpt-oss-120b:free"])

        for model in llm_config.models:
            assert model.timeout is not None
            assert model.retries is not None
            assert model.retry_delay is not None

    def test_evaluator_models_inherit_request_params(self) -> None:
        """The evaluator ensemble is configured alongside the evolution ensemble."""
        llm_config = _configured_llm_config(["openrouter:openai/gpt-oss-120b:free"])

        assert llm_config.evaluator_models
        for model in llm_config.evaluator_models:
            assert model.timeout is not None
            assert model.retries is not None

    def test_every_model_in_an_ensemble_is_configured(self) -> None:
        """Multi-model ensembles are configured per model, not just the first."""
        llm_config = _configured_llm_config(
            ["openrouter:openai/gpt-oss-120b:free", "openrouter:openai/gpt-oss-20b:free"]
        )

        assert len(llm_config.models) == 2
        for model in llm_config.models:
            assert model.max_tokens is not None
            assert model.retries is not None

    def test_per_spec_api_routing_is_preserved(self) -> None:
        """Propagating shared params must not overwrite per-spec endpoint routing."""
        llm_config = _configured_llm_config(["devstral:local"])

        model = llm_config.models[0]
        assert model.api_base == DEVSTRAL_BASE_URL
        assert model.api_key == "not-required"

    def test_system_message_still_overwrites(self) -> None:
        """The prompt system message is applied on top of the shared params."""
        llm_config = _configured_llm_config(["openrouter:openai/gpt-oss-120b:free"])

        assert llm_config.models[0].system_message


class TestOpenAiLlmFromConfiguredModel:
    """End-to-end checks against the OpenEvolve LLM client."""

    def _llm(self, monkeypatch: pytest.MonkeyPatch) -> OpenAILLM:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        llm_config = _configured_llm_config(["openrouter:openai/gpt-oss-120b:free"])
        return OpenAILLM(llm_config.models[0])

    def test_http_client_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The OpenAI client gets an explicit timeout and retry budget."""
        llm = self._llm(monkeypatch)

        assert llm.client.timeout is not None
        assert llm.client.max_retries > 0

    async def test_generate_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generation call completes instead of raising on `retries + 1`."""
        llm = self._llm(monkeypatch)
        sent: dict[str, Any] = {}

        async def _call_api(params: dict[str, Any]) -> str:
            sent.update(params)
            return "mutated code"

        monkeypatch.setattr(llm, "_call_api", _call_api)

        result = await llm.generate("improve this solution")

        assert result == "mutated code"
        assert sent["max_tokens"] is not None
        assert sent["temperature"] is not None
