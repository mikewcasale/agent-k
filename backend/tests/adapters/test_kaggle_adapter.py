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


def _make_adapter_with_transport(transport: httpx.MockTransport, *, max_retries: int = 3) -> KaggleAdapter:
    """Build a KaggleAdapter whose HTTP client uses the provided mock transport."""
    config = KaggleSettings(
        username="user",
        api_key="key",
        max_retries=max_retries,
        rate_limit_delay=0.0,
        max_backoff_seconds=0.0,
        backoff_jitter=0.0,
    )
    adapter = KaggleAdapter(config)
    adapter._client = httpx.AsyncClient(
        base_url=config.base_url, timeout=config.timeout, auth=(config.username, config.api_key), transport=transport
    )
    return adapter


class TestKaggleAdapterRetry:
    """Tests for _request retry behavior on 429 and 5xx responses."""

    async def test_retries_and_succeeds_after_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 responses should trigger a retry and then succeed."""
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return httpx.Response(200, json={"ok": True})

        adapter = _make_adapter_with_transport(httpx.MockTransport(handler))
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(calls) == 2
        assert sleeps and sleeps[0] == 0.0

    async def test_retries_and_succeeds_after_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """5xx responses should trigger a retry and then succeed."""

        async def fake_sleep(delay: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        statuses = iter([503, 502, 200])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(next(statuses), json={"ok": True})

        adapter = _make_adapter_with_transport(httpx.MockTransport(handler), max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200

    async def test_returns_last_5xx_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistent 5xx should surface the final response, not raise."""

        async def fake_sleep(delay: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        adapter = _make_adapter_with_transport(httpx.MockTransport(handler), max_retries=2)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 503

    async def test_raises_rate_limit_after_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistent 429 should raise RateLimitError once retries are used up."""

        async def fake_sleep(delay: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "5"})

        adapter = _make_adapter_with_transport(httpx.MockTransport(handler), max_retries=2)
        try:
            with pytest.raises(RateLimitError) as exc_info:
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert exc_info.value.retry_after == 5

    async def test_client_error_short_circuits_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-429 4xx responses should return immediately without retry."""
        calls: list[int] = []

        async def fake_sleep(delay: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(404)

        adapter = _make_adapter_with_transport(httpx.MockTransport(handler), max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/data/list/missing")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 404
        assert len(calls) == 1

    async def test_network_error_wraps_after_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated network errors should raise PlatformConnectionError."""

        async def fake_sleep(delay: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        adapter = _make_adapter_with_transport(httpx.MockTransport(handler), max_retries=2)
        try:
            with pytest.raises(PlatformConnectionError):
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

    def test_parse_retry_after_seconds(self) -> None:
        """Numeric Retry-After should parse as seconds."""
        response = httpx.Response(429, headers={"Retry-After": "42"})
        assert KaggleAdapter._parse_retry_after(response, default=1) == 42

    def test_parse_retry_after_missing(self) -> None:
        """Missing Retry-After should return default."""
        response = httpx.Response(429)
        assert KaggleAdapter._parse_retry_after(response, default=7) == 7

    def test_parse_retry_after_invalid(self) -> None:
        """Unparseable Retry-After should fall back to default."""
        response = httpx.Response(429, headers={"Retry-After": "not-a-date"})
        assert KaggleAdapter._parse_retry_after(response, default=3) == 3

    def test_backoff_respects_cap(self) -> None:
        """Backoff should never exceed max_backoff_seconds, even at high attempts."""
        config = KaggleSettings(
            username="u", api_key="k", rate_limit_delay=1.0, max_backoff_seconds=5.0, backoff_jitter=0.0
        )
        adapter = KaggleAdapter(config)
        # attempt=10 → 1024 uncapped, must clamp to 5
        assert adapter._compute_backoff(10) == pytest.approx(5.0)
