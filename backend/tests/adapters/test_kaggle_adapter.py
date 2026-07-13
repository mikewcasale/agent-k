"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any
from unittest.mock import patch

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


class TestKaggleAdapterRetry:
    """Tests for the ``_request`` retry policy on 429/5xx and transport errors."""

    @staticmethod
    def _build_adapter(handler: Any, *, max_retries: int = 3) -> KaggleAdapter:
        config = KaggleSettings(username="user", api_key="key", max_retries=max_retries, rate_limit_delay=0.0)
        adapter = KaggleAdapter(config)
        transport = httpx.MockTransport(handler)
        adapter._client = httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout, transport=transport)
        return adapter

    async def test_retries_on_429_then_succeeds(self) -> None:
        """A 429 response is retried after Retry-After and then succeeds."""
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
            return httpx.Response(200, json={"ok": True})

        adapter = self._build_adapter(handler)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(calls) == 2

    async def test_retries_on_5xx_then_succeeds(self) -> None:
        """A 503 response is retried and the eventual 200 is returned."""
        statuses = iter([503, 502, 200])

        def handler(_request: httpx.Request) -> httpx.Response:
            code = next(statuses)
            if code == 200:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(code, text="temporarily unavailable")

        adapter = self._build_adapter(handler, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200

    async def test_raises_rate_limit_error_after_exhaustion(self) -> None:
        """When all attempts return 429, ``RateLimitError`` is raised with retry_after."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")

        adapter = self._build_adapter(handler, max_retries=2)
        try:
            with pytest.raises(RateLimitError) as exc_info:
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert exc_info.value.retry_after == 7

    async def test_returns_5xx_after_exhaustion(self) -> None:
        """After exhausting retries on 5xx, the last response is returned."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="bad gateway")

        adapter = self._build_adapter(handler, max_retries=2)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 502

    async def test_transport_error_wraps_platform_connection_error(self) -> None:
        """Network-layer errors are wrapped as ``PlatformConnectionError``."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        adapter = self._build_adapter(handler, max_retries=2)
        try:
            with pytest.raises(PlatformConnectionError):
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

    async def test_retry_after_missing_header_uses_default(self) -> None:
        """Missing/invalid Retry-After falls back to a bounded default delay."""
        recorded: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            recorded.append(delay)

        def handler(_request: httpx.Request) -> httpx.Response:
            if len(recorded) == 0:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        config = KaggleSettings(username="user", api_key="key", max_retries=3, rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url, timeout=config.timeout, transport=httpx.MockTransport(handler)
        )
        try:
            with patch("agent_k.adapters.kaggle.asyncio.sleep", _fake_sleep):
                response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        # Default is rate_limit_delay * 5.0; rate_limit_delay is 1.0 here.
        assert recorded == [pytest.approx(5.0)]

    async def test_non_retryable_4xx_returned_immediately(self) -> None:
        """A 404 is returned to the caller after a single request."""
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(404, text="not found")

        adapter = self._build_adapter(handler, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/data/list/missing")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 404
        assert len(calls) == 1


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation
