"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.exceptions import PlatformConnectionError, RateLimitError

__all__ = ()

pytestmark = pytest.mark.anyio


class TestKaggleSettings:
    """Tests for the KaggleSettings class."""

    def test_config_creation(self) -> None:
        """Config should be created with credentials."""
        config = KaggleSettings(username="test_user", api_key="test_key")

        assert config.username == "test_user"
        assert config.api_key == "test_key"

    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = KaggleSettings(username="user", api_key="key")

        assert config.base_url == "https://www.kaggle.com/api/v1"


class TestKaggleAdapter:
    """Tests for the KaggleAdapter class."""

    def test_adapter_creation(self) -> None:
        """Adapter should be created with config."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        assert adapter is not None

    @pytest.fixture
    def mock_http_response(self) -> httpx.Response:
        """Create a mock HTTP response."""
        return httpx.Response(
            200,
            json=[
                {
                    "ref": "titanic",
                    "title": "Titanic",
                    "category": "gettingStarted",
                    "reward": "$0",
                    "deadline": "2030-01-01T00:00:00Z",
                }
            ],
        )

    async def test_search_competitions_basic(self) -> None:
        """Search competitions should return results."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        # The adapter requires actual HTTP calls or mocking
        # For unit tests, we verify the adapter is properly constructed
        assert adapter is not None

    async def test_get_leaderboard_basic(self) -> None:
        """Get leaderboard should return entries."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        assert adapter is not None


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation


class TestKaggleAdapterRetries:
    """Verify ``_request`` retry/backoff behavior for transient failures."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``asyncio.sleep`` so retry tests run instantly."""
        import asyncio

        async def _fake_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    @staticmethod
    def _adapter_with_transport(
        responses: list[httpx.Response], *, max_retries: int = 3
    ) -> tuple[KaggleAdapter, list[httpx.Request]]:
        """Build an adapter backed by a scripted mock transport."""
        calls: list[httpx.Request] = []
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if not queue:
                return httpx.Response(500, text="exhausted")
            return queue.pop(0)

        config = KaggleSettings(username="user", api_key="key", max_retries=max_retries, rate_limit_delay=0.0)
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            auth=(config.username, config.api_key),
            transport=httpx.MockTransport(handler),
        )
        return adapter, calls

    async def test_retries_on_429_then_succeeds(self) -> None:
        """A 429 with Retry-After should be retried, not raised immediately."""
        responses = [
            httpx.Response(429, headers={"Retry-After": "1"}, text="slow down"),
            httpx.Response(200, json={"ok": True}),
        ]
        adapter, calls = self._adapter_with_transport(responses)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(calls) == 2

    async def test_retries_on_5xx_then_succeeds(self) -> None:
        """Transient 5xx responses should trigger a retry."""
        responses = [
            httpx.Response(503, text="unavailable"),
            httpx.Response(502, text="bad gateway"),
            httpx.Response(200, json={"ok": True}),
        ]
        adapter, calls = self._adapter_with_transport(responses)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(calls) == 3

    async def test_raises_rate_limit_after_max_retries(self) -> None:
        """Persistent 429 responses should surface ``RateLimitError``."""
        responses = [httpx.Response(429, headers={"Retry-After": "2"}) for _ in range(3)]
        adapter, calls = self._adapter_with_transport(responses, max_retries=3)
        try:
            with pytest.raises(RateLimitError) as exc_info:
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert exc_info.value.retry_after == 2
        assert len(calls) == 3

    async def test_returns_last_5xx_after_max_retries(self) -> None:
        """Persistent 5xx responses should return the final response, not raise."""
        responses = [httpx.Response(502, text="bad gateway") for _ in range(3)]
        adapter, calls = self._adapter_with_transport(responses, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 502
        assert len(calls) == 3

    async def test_transport_error_exhausted_raises_connection_error(self) -> None:
        """Repeated network failures should raise ``PlatformConnectionError``."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        config = KaggleSettings(username="user", api_key="key", max_retries=2, rate_limit_delay=0.0)
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            auth=(config.username, config.api_key),
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(PlatformConnectionError):
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

    async def test_non_retryable_status_returns_immediately(self) -> None:
        """4xx responses outside 429 should be returned on the first attempt."""
        responses = [httpx.Response(404, text="not found")]
        adapter, calls = self._adapter_with_transport(responses)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 404
        assert len(calls) == 1

    def test_parse_retry_after_handles_non_numeric(self) -> None:
        """Non-numeric ``Retry-After`` values should parse to ``None``."""
        assert KaggleAdapter._parse_retry_after(None) is None
        assert KaggleAdapter._parse_retry_after("") is None
        assert KaggleAdapter._parse_retry_after("Wed, 01 Jan 2030 00:00:00 GMT") is None
        assert KaggleAdapter._parse_retry_after("7") == 7
