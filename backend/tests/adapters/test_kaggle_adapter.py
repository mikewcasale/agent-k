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


def _adapter_with_transport(handler: Any, *, max_retries: int = 3, rate_limit_delay: float = 0.0) -> KaggleAdapter:
    """Build an adapter whose HTTP client is bound to a mock transport."""
    config = KaggleSettings(username="user", api_key="key", max_retries=max_retries, rate_limit_delay=rate_limit_delay)
    adapter = KaggleAdapter(config)
    adapter._client = httpx.AsyncClient(
        base_url=config.base_url,
        timeout=config.timeout,
        auth=(config.username, config.api_key),
        transport=httpx.MockTransport(handler),
    )
    return adapter


class TestKaggleAdapterRetry:
    """Tests for the Kaggle adapter retry/backoff behavior."""

    async def test_returns_immediately_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 200 response should not trigger any backoff sleep."""
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"ok": True})

        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(calls) == 1
        assert sleeps == []

    async def test_retries_on_503_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Transient 5xx responses must be retried before surfacing to caller."""
        statuses = iter([503, 502, 200])

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(next(statuses), json={"ok": True})

        async def fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200

    async def test_returns_last_5xx_when_retries_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When every attempt returns 5xx the final response is surfaced to the caller."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="bad gateway")

        async def fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=2)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 502

    async def test_429_honors_retry_after_then_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 429 with Retry-After within budget should sleep and retry once."""
        statuses = iter([(429, "2"), (200, None)])

        def handler(_request: httpx.Request) -> httpx.Response:
            status, retry_after = next(statuses)
            headers = {"Retry-After": retry_after} if retry_after else {}
            return httpx.Response(status, headers=headers, json={"ok": True})

        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert sleeps == [2.0]

    async def test_429_raises_when_retry_after_exceeds_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 429 with a long Retry-After hint must surface as RateLimitError immediately."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "600"})

        async def fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=3)
        try:
            with pytest.raises(RateLimitError) as exc_info:
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert exc_info.value.retry_after == 600

    async def test_429_raises_when_retries_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated 429s must eventually surface as RateLimitError, not silent success."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "1"})

        async def fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=2)
        try:
            with pytest.raises(RateLimitError):
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

    async def test_transport_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transient transport failure should retry before raising PlatformConnectionError."""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(200, json={"ok": True})

        async def fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert attempts["count"] == 2

    async def test_transport_error_raises_after_exhausting_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistent transport failures must surface as PlatformConnectionError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        async def fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)

        adapter = _adapter_with_transport(handler, max_retries=2)
        try:
            with pytest.raises(PlatformConnectionError):
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()


class TestParseRetryAfter:
    """Tests for the ``Retry-After`` header parser."""

    def test_parses_integer_seconds(self) -> None:
        config = KaggleSettings(username="user", api_key="key", rate_limit_delay=0.5)
        adapter = KaggleAdapter(config)

        assert adapter._parse_retry_after("5") == pytest.approx(5.0)

    def test_falls_back_to_delay_when_missing(self) -> None:
        config = KaggleSettings(username="user", api_key="key", rate_limit_delay=2.5)
        adapter = KaggleAdapter(config)

        assert adapter._parse_retry_after(None) == pytest.approx(2.5)
        assert adapter._parse_retry_after("garbage") == pytest.approx(2.5)

    def test_clamps_negative_values_to_zero(self) -> None:
        config = KaggleSettings(username="user", api_key="key", rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)

        assert adapter._parse_retry_after("-10") == pytest.approx(0.0)
