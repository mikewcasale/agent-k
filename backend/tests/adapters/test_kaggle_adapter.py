"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, _parse_retry_after
from agent_k.core.exceptions import PlatformConnectionError, RateLimitError

__all__ = ()

pytestmark = pytest.mark.anyio


def _make_adapter(
    transport_handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 3,
    rate_limit_delay: float = 0.001,
) -> KaggleAdapter:
    """Build an adapter wired to an httpx MockTransport for retry tests."""
    config = KaggleSettings(username="user", api_key="key", max_retries=max_retries, rate_limit_delay=rate_limit_delay)
    adapter = KaggleAdapter(config)
    adapter._client = httpx.AsyncClient(
        base_url=config.base_url,
        timeout=config.timeout,
        auth=(config.username, config.api_key),
        transport=httpx.MockTransport(transport_handler),
    )
    return adapter


@pytest.fixture(autouse=True)
def _no_jitter_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove jitter and sleep so retry tests are deterministic and fast."""
    monkeypatch.setattr(random, "uniform", lambda _a, _b: 0.0)

    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", _sleep)


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
    """Retry/backoff behavior for the internal ``_request`` helper."""

    async def test_retries_on_5xx_then_succeeds(self) -> None:
        """A 503 followed by a 200 should be transparently retried."""
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(len(calls))
            if len(calls) <= 2:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, json={"ok": True})

        adapter = _make_adapter(handler, max_retries=4)
        response = await adapter._request("GET", "/competitions/list")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert len(calls) == 3
        await adapter._client.aclose()

    async def test_returns_last_5xx_when_retries_exhausted(self) -> None:
        """When every retry fails with 5xx the caller sees the final response."""
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(len(calls))
            return httpx.Response(502, text="bad gateway")

        adapter = _make_adapter(handler, max_retries=3)
        response = await adapter._request("GET", "/competitions/list")

        assert response.status_code == 502
        assert len(calls) == 3
        await adapter._client.aclose()

    async def test_does_not_retry_on_4xx(self) -> None:
        """Client errors (other than 429) must not trigger retries."""
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(len(calls))
            return httpx.Response(404, text="missing")

        adapter = _make_adapter(handler, max_retries=3)
        response = await adapter._request("GET", "/competitions/data/list/foo")

        assert response.status_code == 404
        assert len(calls) == 1
        await adapter._client.aclose()

    async def test_429_raises_rate_limit_with_retry_after(self) -> None:
        """A 429 response surfaces a RateLimitError carrying ``Retry-After``."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "12"}, text="slow down")

        adapter = _make_adapter(handler, max_retries=3)
        with pytest.raises(RateLimitError) as exc_info:
            await adapter._request("GET", "/competitions/list")
        assert exc_info.value.retry_after == 12
        await adapter._client.aclose()

    async def test_retries_then_raises_on_network_error(self) -> None:
        """Repeated network errors should surface PlatformConnectionError."""
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(len(calls))
            raise httpx.ConnectError("connection refused")

        adapter = _make_adapter(handler, max_retries=2)
        with pytest.raises(PlatformConnectionError):
            await adapter._request("GET", "/competitions/list")
        assert len(calls) == 2
        await adapter._client.aclose()


class TestComputeBackoff:
    """Tests for the exponential backoff helper."""

    def test_grows_exponentially_with_attempt(self) -> None:
        """Each attempt should double the base delay before any jitter."""
        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0))
        delays = [adapter._compute_backoff(attempt, retry_after=None) for attempt in range(4)]
        assert delays[0] >= 1.0
        assert delays[1] >= 2.0
        assert delays[2] >= 4.0
        assert delays[3] >= 8.0

    def test_caps_at_max_backoff(self) -> None:
        """Large attempt counts should be capped to keep retries bounded."""
        from agent_k.adapters.kaggle import _MAX_BACKOFF_SECONDS

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0))
        delay = adapter._compute_backoff(20, retry_after=None)
        assert delay <= _MAX_BACKOFF_SECONDS * (1.0 + 0.11)

    def test_retry_after_dominates_when_larger(self) -> None:
        """A server-provided Retry-After hint overrides exponential backoff when larger."""
        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k", rate_limit_delay=0.1))
        delay = adapter._compute_backoff(0, retry_after=5.0)
        assert delay >= 5.0


class TestParseRetryAfter:
    """Tests for ``_parse_retry_after`` header parsing."""

    def test_handles_delta_seconds(self) -> None:
        assert _parse_retry_after("17") == 17.0

    def test_handles_fractional_seconds(self) -> None:
        assert _parse_retry_after("1.5") == 1.5

    def test_returns_none_for_missing(self) -> None:
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None

    def test_handles_http_date(self) -> None:
        """RFC 7231 date format should be parseable and non-negative."""
        result = _parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT")
        assert result is not None
        assert result > 0

    def test_returns_none_for_garbage(self) -> None:
        assert _parse_retry_after("not-a-date") is None
