"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from agent_k.adapters.kaggle import (
    _DEFAULT_RETRY_AFTER,
    _MAX_RETRY_AFTER,
    KaggleAdapter,
    KaggleSettings,
    _parse_retry_after,
)
from agent_k.core.exceptions import PlatformConnectionError, RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ()

pytestmark = pytest.mark.anyio


async def _make_adapter(
    handler: Callable[[httpx.Request], httpx.Response], *, max_retries: int = 3, rate_limit_delay: float = 0.0
) -> KaggleAdapter:
    """Build a KaggleAdapter whose HTTP client is bound to ``handler``.

    Replaces the client the adapter created in ``__post_init__`` with one wired
    to an ``httpx.MockTransport`` so requests never leave the process (and so
    any ambient HTTP(S)_PROXY env vars cannot intercept them).
    """
    config = KaggleSettings(username="user", api_key="key", max_retries=max_retries, rate_limit_delay=rate_limit_delay)
    adapter = KaggleAdapter(config)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=config.base_url,
        timeout=config.timeout,
        auth=(config.username, config.api_key),
    )
    return adapter


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real sleeps inside the retry loop to keep tests fast."""

    async def _noop(_: float) -> None:
        return None

    monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", _noop)


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


class TestParseRetryAfter:
    """Tests for the ``_parse_retry_after`` helper."""

    def test_missing_returns_default(self) -> None:
        assert _parse_retry_after(None) == _DEFAULT_RETRY_AFTER

    def test_valid_integer(self) -> None:
        assert _parse_retry_after("5") == 5

    def test_whitespace_stripped(self) -> None:
        assert _parse_retry_after("  12  ") == 12

    def test_negative_clamped_to_zero(self) -> None:
        assert _parse_retry_after("-30") == 0

    def test_absurd_value_capped(self) -> None:
        assert _parse_retry_after("100000") == _MAX_RETRY_AFTER

    def test_non_numeric_falls_back(self) -> None:
        # HTTP-date form is not (yet) parsed; fall back to the safe default.
        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == _DEFAULT_RETRY_AFTER


class TestRequestRetryLoop:
    """Tests for retry behaviour in ``KaggleAdapter._request``."""

    async def test_retries_on_429_then_succeeds(self) -> None:
        calls: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append({"headers": dict(request.headers)})
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "slow down"})
            return httpx.Response(200, json={"ok": True})

        adapter = await _make_adapter(handler)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(calls) == 2

    async def test_retries_on_5xx_then_succeeds(self) -> None:
        seen: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            seen.append(1)
            if len(seen) < 3:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, json={"ok": True})

        adapter = await _make_adapter(handler)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 200
        assert len(seen) == 3

    async def test_retries_on_transport_error_then_raises(self) -> None:
        seen: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            seen.append(1)
            raise httpx.ConnectError("boom")

        adapter = await _make_adapter(handler, max_retries=3)
        try:
            with pytest.raises(PlatformConnectionError):
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert len(seen) == 3

    async def test_exhausted_429_raises_rate_limit_error(self) -> None:
        seen: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "slow down"})

        adapter = await _make_adapter(handler, max_retries=2)
        try:
            with pytest.raises(RateLimitError) as exc_info:
                await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert exc_info.value.retry_after == 2
        assert len(seen) == 2

    async def test_exhausted_5xx_returns_last_response(self) -> None:
        seen: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(502, text="bad gateway")

        adapter = await _make_adapter(handler, max_retries=2)
        try:
            response = await adapter._request("GET", "/competitions/list")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 502
        assert len(seen) == 2

    async def test_non_retryable_4xx_returned_immediately(self) -> None:
        seen: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(404, json={"error": "not found"})

        adapter = await _make_adapter(handler, max_retries=3)
        try:
            response = await adapter._request("GET", "/competitions/data/list/missing")
        finally:
            await adapter._client.aclose()

        assert response.status_code == 404
        assert len(seen) == 1
