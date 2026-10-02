"""Tests for the model factory infrastructure.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError, UserError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

from agent_k.infra.providers import (
    DEVSTRAL_BASE_URL,
    DEVSTRAL_MODEL_ID,
    create_devstral_model,
    create_fallback_model,
    create_openrouter_model,
    get_model,
    is_devstral_model,
    split_model_specs,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ()


def _completion_payload(content: str) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible chat completion body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "stub-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@dataclass
class _StubEndpoint:
    """Live OpenAI-compatible endpoint used to exercise real failover behavior."""

    base_url: str
    _counter: list[int] = field(default_factory=lambda: [0])

    @property
    def request_count(self) -> int:
        """Number of chat-completion requests the endpoint received."""
        return self._counter[0]


@asynccontextmanager
async def _chat_completions_server(*, status: int, body: dict[str, Any]) -> AsyncIterator[_StubEndpoint]:
    """Serve a fixed chat-completions response on a loopback port."""
    counter = [0]
    encoded = json.dumps(body).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            counter[0] += 1
            length = int(self.headers.get("content-length", 0))
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield _StubEndpoint(base_url=f"http://127.0.0.1:{port}/v1", _counter=counter)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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


class TestSplitModelSpecs:
    """Tests for split_model_specs function."""

    def test_single_spec(self) -> None:
        """A single spec should return a one-element chain."""
        assert split_model_specs("openai:gpt-4o") == ("openai:gpt-4o",)

    def test_strips_and_drops_blanks(self) -> None:
        """Padding and empty entries should be ignored."""
        assert split_model_specs(" openai:gpt-4o , , devstral:local ,") == ("openai:gpt-4o", "devstral:local")

    def test_preserves_order(self) -> None:
        """Chain order should be preserved so the primary model stays first."""
        assert split_model_specs("a:1,b:2,c:3") == ("a:1", "b:2", "c:3")

    def test_empty_spec_raises(self) -> None:
        """An empty spec should raise rather than build a model-less agent."""
        with pytest.raises(UserError):
            split_model_specs("   ,  ")


class TestCreateFallbackModel:
    """Tests for create_fallback_model function."""

    def test_builds_chain_in_order(self) -> None:
        """All resolvable specs should be kept in declaration order."""
        model = create_fallback_model(["devstral:first-model", "devstral:second-model"])

        assert isinstance(model, FallbackModel)
        assert [member.model_name for member in model.models] == ["first-model", "second-model"]

    def test_skips_specs_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A chain should survive members whose provider credentials are missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        model = create_fallback_model(["anthropic:claude-3-haiku-20240307", "devstral:local"])

        assert [member.model_name for member in model.models] == [DEVSTRAL_MODEL_ID]

    def test_raises_when_no_spec_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An entirely unusable chain should raise instead of failing at request time."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(UserError, match="failover chain"):
            create_fallback_model(["anthropic:claude-3-haiku-20240307"])


class TestGetModelFallbackChain:
    """Tests for failover chains resolved through get_model."""

    def test_chain_returns_fallback_model(self) -> None:
        """A multi-spec string should resolve to a FallbackModel."""
        result = get_model("devstral:primary-model,devstral:backup-model")

        assert isinstance(result, FallbackModel)
        assert [member.model_name for member in result.models] == ["primary-model", "backup-model"]

    def test_single_spec_is_unchanged(self) -> None:
        """A single spec should keep returning the bare pydantic-ai spec string."""
        assert get_model("openai:gpt-4o") == "openai:gpt-4o"

    async def test_fails_over_to_next_model_on_rate_limit(self) -> None:
        """A rate-limited primary endpoint should hand the request to the next model."""
        async with _chat_completions_server(status=429, body={"error": {"message": "rate limit exceeded"}}) as primary:
            async with _chat_completions_server(status=200, body=_completion_payload("fallback answer")) as backup:
                model = get_model(f"devstral:{primary.base_url},devstral:{backup.base_url}")
                agent = Agent(model=model, output_type=str, retries=0)

                result = await agent.run("ping")

        assert result.output == "fallback answer"
        assert primary.request_count >= 1
        assert backup.request_count == 1

    async def test_raises_group_when_every_model_fails(self) -> None:
        """Exhausting the chain should surface every underlying provider error."""
        async with _chat_completions_server(status=404, body={"error": {"message": "model not found"}}) as primary:
            async with _chat_completions_server(status=402, body={"error": {"message": "out of credits"}}) as backup:
                model = get_model(f"devstral:{primary.base_url},devstral:{backup.base_url}")
                agent = Agent(model=model, output_type=str, retries=0)

                with pytest.raises(FallbackExceptionGroup) as exc_info:
                    await agent.run("ping")

        statuses = sorted(error.status_code for error in exc_info.value.exceptions if isinstance(error, ModelHTTPError))
        assert statuses == [402, 404]


class TestIsDevstralModelChain:
    """Tests for is_devstral_model with failover chains."""

    def test_chain_with_devstral_member(self) -> None:
        """A chain containing Devstral should be identified as Devstral."""
        assert is_devstral_model("openrouter:openai/gpt-oss-20b:free,devstral:local") is True

    def test_chain_without_devstral_member(self) -> None:
        """A chain without Devstral should not be identified as Devstral."""
        assert is_devstral_model("openai:gpt-4o,anthropic:claude-3-haiku-20240307") is False

    def test_blank_spec(self) -> None:
        """A blank spec should not be identified as Devstral."""
        assert is_devstral_model("   ") is False
